"""Этап 0 — фиксация окружения.

Собирает всё, без чего число производительности недействительно: коммит
llama.cpp, версии CUDA, драйвер, железо, версии Python-пакетов.

Этот блок обязан лежать рядом с любыми числами. Каждый прогон кладёт свой
env.json в свой каталог результатов.

    python -m common.env --out env.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import HostConfig, ConfigError  # noqa: E402

# Битовая маска clocks_throttle_reasons.active из nvidia-smi.
THROTTLE_REASONS = {
    0x0000000000000001: "GpuIdle",
    0x0000000000000002: "ApplicationsClocksSetting",
    0x0000000000000004: "SwPowerCap",
    0x0000000000000008: "HwSlowdown",
    0x0000000000000010: "SyncBoost",
    0x0000000000000020: "SwThermalSlowdown",
    0x0000000000000040: "HwThermalSlowdown",
    0x0000000000000080: "HwPowerBrakeSlowdown",
    0x0000000000000100: "DisplayClockSetting",
}

# Причины, которые реально портят замер. GpuIdle и DisplayClockSetting — нет.
HARMFUL_THROTTLE = {
    "SwPowerCap", "HwSlowdown", "SwThermalSlowdown",
    "HwThermalSlowdown", "HwPowerBrakeSlowdown",
}


def _run(cmd: list[str], timeout: int = 60) -> str | None:
    """Выполнить команду, вернуть stdout. None при любой неудаче."""
    if shutil.which(cmd[0]) is None and not Path(cmd[0]).is_file():
        return None
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                             encoding="utf-8", errors="replace")
    except (subprocess.SubprocessError, OSError):
        return None
    return res.stdout if res.returncode == 0 else None


def use_utf8_output() -> None:
    """Кириллица в консоли Windows иначе превращается в мусор."""
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        except (OSError, AttributeError):
            pass
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):
            pass


def decode_throttle(mask_raw: str) -> list[str]:
    """0x0000000000000001 -> ['GpuIdle']"""
    try:
        mask = int(mask_raw, 16) if mask_raw.lower().startswith("0x") else int(mask_raw)
    except (ValueError, AttributeError):
        return []
    return [name for bit, name in THROTTLE_REASONS.items() if mask & bit]


# --------------------------------------------------------------------------
# GPU
# --------------------------------------------------------------------------

GPU_FIELDS = [
    "name", "driver_version", "memory.total", "compute_cap",
    "clocks.max.sm", "clocks.max.memory", "clocks.current.sm",
    "clocks.current.memory", "power.max_limit", "power.draw",
    "temperature.gpu", "clocks_throttle_reasons.active", "pcie.link.gen.max",
    "pcie.link.width.max", "persistence_mode", "utilization.gpu",
]

# Ниже этой загрузки причины троттлинга ничего не значат: простаивающая карта
# сбрасывает частоты и рапортует SwPowerCap, хотя ничем не ограничена.
IDLE_UTILIZATION_PCT = 5.0


def collect_gpu() -> dict:
    out = _run(
        ["nvidia-smi", f"--query-gpu={','.join(GPU_FIELDS)}",
         "--format=csv,noheader,nounits"]
    )
    if out is None:
        return {"available": False, "error": "nvidia-smi недоступен"}

    row = [c.strip() for c in out.strip().splitlines()[0].split(",")]
    raw = dict(zip(GPU_FIELDS, row))

    def num(key):
        try:
            return float(raw[key])
        except (ValueError, KeyError):
            return None

    active = decode_throttle(raw.get("clocks_throttle_reasons.active", ""))
    gpu = {
        "available": True,
        "name": raw.get("name"),
        "driver_version": raw.get("driver_version"),
        "compute_capability": raw.get("compute_cap"),
        "vram_total_mib": num("memory.total"),
        "clocks_mhz": {
            "sm_max": num("clocks.max.sm"),
            "sm_current": num("clocks.current.sm"),
            "memory_max": num("clocks.max.memory"),
            "memory_current": num("clocks.current.memory"),
        },
        "power_w": {"limit": num("power.max_limit"), "draw": num("power.draw")},
        "temperature_c": num("temperature.gpu"),
        "pcie": {
            "gen_max": raw.get("pcie.link.gen.max"),
            "width_max": raw.get("pcie.link.width.max"),
        },
        "persistence_mode": raw.get("persistence_mode"),
        "utilization_pct": num("utilization.gpu"),
        "throttle_reasons_active": active,
        "throttle_harmful": sorted(set(active) & HARMFUL_THROTTLE),
    }
    # На простое RTX 5090 показывает SwPowerCap при 22 МГц и 25 Вт из 575 — это
    # энергосбережение, а не ограничение. Считать это троттлингом значит
    # заблокировать замеры на исправной машине.
    util = gpu["utilization_pct"]
    gpu["idle"] = util is not None and util < IDLE_UTILIZATION_PCT
    if gpu["idle"]:
        gpu["throttle_harmful"] = []

    # Максимальную CUDA драйвера отдаёт только шапка обычного вывода.
    header = _run(["nvidia-smi"])
    if header:
        m = re.search(r"CUDA Version:\s*([\d.]+)", header)
        if m:
            gpu["driver_max_cuda_version"] = m.group(1)
    return gpu


def theoretical_bandwidth_gbs(mem_clock_mhz: float, bus_width_bits: int) -> float:
    """Паспортная полоса. Только справочно — roofline считается по измеренной.

    GDDR передаёт по два слова за такт, отсюда множитель 2.
    """
    return mem_clock_mhz * 2 * bus_width_bits / 8 / 1000


# --------------------------------------------------------------------------
# CPU / память / ОС
# --------------------------------------------------------------------------

def collect_cpu() -> dict:
    info = {
        "model": platform.processor() or None,
        "arch": platform.machine(),
        "logical_cores": os.cpu_count(),
        "physical_cores": None,
    }

    if sys.platform == "win32":
        try:
            import winreg
            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            )
            with key:
                info["model"] = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
        out = _run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_Processor | "
             "Measure-Object -Property NumberOfCores -Sum).Sum"],
            timeout=45,
        )
        if out and out.strip().isdigit():
            info["physical_cores"] = int(out.strip())
    else:
        try:
            text = Path("/proc/cpuinfo").read_text(encoding="utf-8")
            m = re.search(r"^model name\s*:\s*(.+)$", text, re.MULTILINE)
            if m:
                info["model"] = m.group(1).strip()
            pairs = set(
                re.findall(r"physical id\s*:\s*(\d+)[\s\S]*?core id\s*:\s*(\d+)", text)
            )
            if pairs:
                info["physical_cores"] = len(pairs)
        except OSError:
            pass
    return info


def collect_memory() -> dict:
    total = None
    if sys.platform == "win32":
        import ctypes

        class MemStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        st = MemStatus()
        st.dwLength = ctypes.sizeof(MemStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
            total = st.ullTotalPhys
    else:
        try:
            m = re.search(
                r"^MemTotal:\s*(\d+)\s*kB",
                Path("/proc/meminfo").read_text(encoding="utf-8"),
                re.MULTILINE,
            )
            if m:
                total = int(m.group(1)) * 1024
        except OSError:
            pass
    return {"total_bytes": total,
            "total_gib": round(total / 1024**3, 2) if total else None}


def collect_os() -> dict:
    return {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "platform": platform.platform(),
        "hostname": platform.node(),
    }


# --------------------------------------------------------------------------
# llama.cpp
# --------------------------------------------------------------------------

def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def collect_llama_cpp(cfg: HostConfig, hash_binaries: bool = True) -> dict:
    info: dict = {"bin_dir": str(cfg.bin_dir)}

    # llama-bench не понимает --version: печатает usage и выходит с кодом 1.
    # Версию сообщают llama-server и llama-cli.
    text = ""
    for name in ("llama-server", "llama-cli", "llama-perplexity"):
        try:
            exe = cfg.exe(name)
        except ConfigError:
            continue
        try:
            res = subprocess.run([str(exe), "--version"], capture_output=True,
                                 text=True, timeout=120,
                                 encoding="utf-8", errors="replace")
        except (subprocess.SubprocessError, OSError):
            continue
        # Версия уходит в stderr вместе с логом загрузки бэкендов.
        candidate = (res.stdout or "") + (res.stderr or "")
        if re.search(r"version:\s*\d+\s*\(", candidate):
            text = candidate
            info["version_probe_binary"] = name
            break
    if not text:
        return {**info, "error": "ни один бинарь llama.cpp не сообщил версию"}

    m = re.search(r"version:\s*(\d+)\s*\(([0-9a-f]+)\)", text)
    if m:
        info["build_number"] = int(m.group(1))
        info["commit"] = m.group(2)
    m = re.search(r"built with (.+?) for (.+)", text)
    if m:
        info["compiler"] = m.group(1)
        info["target"] = m.group(2)

    # Бэкенды спрашиваем отдельной командой, а не выцарапываем из --version:
    # у b9006 они там печатались, у b10326 --version стал тихим и сообщает
    # только версию. --list-devices печатает их всегда и на обеих платформах —
    # на Windows это строки load_backend, на Linux со вкомпилированным CUDA
    # признаком служит инициализация устройств.
    devices_text = ""
    try:
        probe = subprocess.run([str(cfg.exe("llama-bench")), "--list-devices"],
                               capture_output=True, text=True, timeout=180,
                               encoding="utf-8", errors="replace")
        devices_text = (probe.stdout or "") + (probe.stderr or "")
    except (ConfigError, subprocess.SubprocessError, OSError):
        devices_text = text

    backends = sorted(set(re.findall(r"load_backend: loaded (\w+) backend",
                                     devices_text)))
    if not backends and re.search(r"ggml_cuda_init: found \d+ CUDA device",
                                  devices_text):
        backends = ["CUDA"]
    info["backends_loaded"] = backends
    info["devices"] = [
        {"id": m.group(1), "name": m.group(2).strip()}
        for m in re.finditer(r"^\s{2}(\w+): (.+?) \(", devices_text, re.MULTILINE)
    ]

    # Под какую CUDA собран бинарь: у релизных архивов это в имени каталога.
    m = re.search(r"cuda-([\d.]+)", cfg.bin_dir.name, re.IGNORECASE)
    if m:
        info["build_cuda_version"] = m.group(1)
        info["build_cuda_version_source"] = "имя каталога релизной сборки"
    else:
        # Сборка из исходников: имени каталога нет, и раньше здесь не
        # записывалось ничего. В отчёт тогда шла версия рантайма torch, которая
        # к компилятору llama.cpp отношения не имеет и совпадает с ним лишь по
        # случайности. Спрашиваем сам компилятор.
        for probe in (["nvcc", "--version"], ["/usr/local/cuda/bin/nvcc", "--version"]):
            try:
                r = subprocess.run(probe, capture_output=True, text=True, timeout=20)
            except (subprocess.SubprocessError, OSError):
                continue
            m = re.search(r"release ([\d.]+)", (r.stdout or "") + (r.stderr or ""))
            if m:
                info["build_cuda_version"] = m.group(1)
                info["build_cuda_version_source"] = "nvcc --version"
                break
        else:
            # Компилятора нет (например, бинари принесены с другой машины).
            # Пишем это явно, чтобы потом не выдавать догадку за измерение.
            info["build_cuda_version"] = None
            info["build_cuda_version_source"] = "не определена: nvcc недоступен"

    # Куда указывает символьная ссылка тулкита: помогает, когда в системе
    # несколько версий и nvcc в PATH не тот, которым собирали.
    for cand in ("/usr/local/cuda", "/opt/cuda"):
        try:
            p = Path(cand)
            if p.is_symlink():
                info["cuda_toolkit_link"] = str(p.resolve())
                break
            if p.is_dir():
                info["cuda_toolkit_link"] = str(p)
                break
        except OSError:
            continue

    declared = cfg.data["llama_cpp"].get("pinned_commit")
    info["pinned_commit"] = declared
    if declared and info.get("commit"):
        info["commit_matches_config"] = info["commit"].startswith(declared) or \
            declared.startswith(info["commit"])

    info["built_from_source"] = cfg.data["llama_cpp"].get("build_from_source", False)

    binaries = {}
    names = ["llama-bench", "llama-server", "llama-perplexity", "llama-cli"]
    for name in names:
        try:
            p = cfg.exe(name)
        except ConfigError:
            continue
        entry = {"size_bytes": p.stat().st_size}
        if hash_binaries:
            entry["sha256"] = sha256_file(p)
        binaries[name] = entry
    # Бэкенд CUDA — тот самый код, чью скорость мы меряем.
    for backend in ("ggml-cuda.dll", "libggml-cuda.so"):
        p = cfg.bin_dir / backend
        if p.is_file():
            entry = {"size_bytes": p.stat().st_size}
            if hash_binaries:
                entry["sha256"] = sha256_file(p)
            binaries[backend] = entry
    info["binaries"] = binaries
    return info


# --------------------------------------------------------------------------
# Python / харнесс
# --------------------------------------------------------------------------

TRACKED_PACKAGES = ["torch", "numpy", "gguf", "sentencepiece", "safetensors"]


def collect_python() -> dict:
    from importlib.metadata import PackageNotFoundError, version

    packages = {}
    for name in TRACKED_PACKAGES:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None

    info = {
        "version": platform.python_version(),
        "executable": sys.executable,
        "packages": packages,
    }

    # Рантайм CUDA, которым меряется полоса в Этапе 2 — он свой, не из тулкита.
    try:
        import torch
        info["torch_cuda"] = {
            "runtime_version": torch.version.cuda,
            "is_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["torch_cuda"]["device_0"] = {
                "name": props.name,
                "total_memory_mib": props.total_memory // 1024**2,
                "compute_capability": f"{props.major}.{props.minor}",
                "multi_processor_count": props.multi_processor_count,
            }
    except ImportError:
        info["torch_cuda"] = None
    return info


def collect_harness_git() -> dict:
    root = Path(__file__).resolve().parent.parent
    out = _run(["git", "-C", str(root), "rev-parse", "HEAD"])
    if out is None:
        return {"commit": None, "note": "харнесс не в git-репозитории или коммитов нет"}
    commit = out.strip()
    dirty = _run(["git", "-C", str(root), "status", "--porcelain"])
    return {
        "commit": commit,
        "dirty": bool(dirty and dirty.strip()),
    }


# --------------------------------------------------------------------------
# Сборка
# --------------------------------------------------------------------------

def collect_env(cfg: HostConfig, hash_binaries: bool = True) -> dict:
    gpu = collect_gpu()
    env = {
        "schema_version": 1,
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "host_id": cfg.host_id,
        "phase": cfg.phase,
        "config_file": str(cfg.path),
        "config_declared": {
            "cuda_arch": cfg.gpu.get("cuda_arch"),
            "vram_type": cfg.gpu.get("vram_type"),
            "memory_bus_width_bits": cfg.gpu.get("memory_bus_width_bits"),
            "spec_bandwidth_gbs": cfg.gpu.get("spec_bandwidth_gbs"),
        },
        "os": collect_os(),
        "cpu": collect_cpu(),
        "memory": collect_memory(),
        "gpu": gpu,
        "llama_cpp": collect_llama_cpp(cfg, hash_binaries=hash_binaries),
        "python": collect_python(),
        "harness": collect_harness_git(),
        "measured_bandwidth_gbs": None,  # заполняется Этапом 2
    }

    # Сверка паспортной полосы с частотой памяти, чтобы поймать опечатку в конфиге.
    bus = cfg.gpu.get("memory_bus_width_bits")
    mem_clock = gpu.get("clocks_mhz", {}).get("memory_max") if gpu.get("available") else None
    if bus and mem_clock:
        derived = theoretical_bandwidth_gbs(mem_clock, bus)
        env["config_declared"]["derived_spec_bandwidth_gbs"] = round(derived, 1)
    return env


def validate_env(env: dict, cfg: HostConfig) -> list[str]:
    """Проблемы, делающие замеры недействительными. Пустой список — всё чисто."""
    problems = []
    gpu = env["gpu"]

    if not gpu.get("available"):
        problems.append("GPU не виден через nvidia-smi")
        return problems

    expected = cfg.gpu.get("expected_name_substring", "")
    if expected and expected.lower() not in (gpu.get("name") or "").lower():
        problems.append(
            f"GPU '{gpu.get('name')}' не совпадает с конфигом '{cfg.host_id}' "
            f"(ожидалось '{expected}')"
        )

    cc = (gpu.get("compute_capability") or "").replace(".", "")
    arch = str(cfg.gpu.get("cuda_arch", ""))
    if cc and arch and cc != arch:
        problems.append(
            f"compute capability {gpu.get('compute_capability')} не совпадает "
            f"с cuda_arch={arch} из конфига"
        )

    llama = env["llama_cpp"]
    if "error" in llama:
        problems.append(f"llama.cpp: {llama['error']}")
    elif not llama.get("commit"):
        problems.append("не удалось определить коммит llama.cpp")
    elif llama.get("commit_matches_config") is False:
        problems.append(
            f"коммит бинаря {llama['commit']} != pinned_commit "
            f"{llama.get('pinned_commit')} из конфига"
        )
    if llama.get("backends_loaded") and "CUDA" not in llama["backends_loaded"]:
        problems.append(
            f"llama.cpp не загрузил CUDA-бэкенд (загружены: {llama['backends_loaded']})"
        )

    if gpu.get("throttle_harmful") and not gpu.get("idle"):
        problems.append(f"GPU троттлит уже сейчас: {gpu['throttle_harmful']}")

    max_temp = cfg.safety.get("max_temperature_c")
    if max_temp and gpu.get("temperature_c") and gpu["temperature_c"] >= max_temp:
        problems.append(
            f"температура GPU {gpu['temperature_c']}C >= порога {max_temp}C ещё до нагрузки"
        )

    if env["python"].get("torch_cuda") is None:
        problems.append("torch недоступен — Этап 2 (полоса памяти) не выполнится")
    elif not env["python"]["torch_cuda"].get("is_available"):
        problems.append("torch не видит CUDA — Этап 2 (полоса памяти) не выполнится")

    if env["python"]["packages"].get("gguf") is None:
        problems.append("пакет gguf не установлен — Этап 4 (байты/токен) не выполнится")

    return problems


def format_summary(env: dict, problems: list[str]) -> str:
    gpu, llama = env["gpu"], env["llama_cpp"]
    lines = [
        f"host        : {env['host_id']} (фаза {env['phase']})",
        f"os          : {env['os']['platform']}",
        f"cpu         : {env['cpu']['model']} "
        f"({env['cpu']['physical_cores']}C/{env['cpu']['logical_cores']}T)",
        f"ram         : {env['memory']['total_gib']} GiB",
    ]
    if gpu.get("available"):
        lines += [
            f"gpu         : {gpu['name']} (sm_{(gpu['compute_capability'] or '').replace('.', '')})",
            f"vram        : {gpu['vram_total_mib']:.0f} MiB "
            f"{env['config_declared'].get('vram_type') or ''}".rstrip(),
            f"driver      : {gpu['driver_version']} (max CUDA {gpu.get('driver_max_cuda_version')})",
            f"clocks max  : sm {gpu['clocks_mhz']['sm_max']:.0f} MHz, "
            f"mem {gpu['clocks_mhz']['memory_max']:.0f} MHz",
            f"power limit : {gpu['power_w']['limit']} W",
            f"temp now    : {gpu['temperature_c']:.0f} C",
        ]
    decl = env["config_declared"]
    lines.append(
        f"bandwidth   : {decl.get('spec_bandwidth_gbs')} GB/s паспортная "
        f"(из частоты: {decl.get('derived_spec_bandwidth_gbs')}) — измеренная в Этапе 2"
    )
    lines += [
        f"llama.cpp   : b{llama.get('build_number')} ({llama.get('commit')}) "
        f"CUDA {llama.get('build_cuda_version')} "
        f"[{llama.get('build_cuda_version_source', 'источник не записан')}], "
        f"{llama.get('compiler')}",
        f"backends    : {', '.join(llama.get('backends_loaded') or [])}",
        f"python      : {env['python']['version']}",
        f"torch       : {env['python']['packages'].get('torch')} "
        f"(CUDA {(env['python'].get('torch_cuda') or {}).get('runtime_version')})",
        f"gguf        : {env['python']['packages'].get('gguf')}",
        f"harness git : {env['harness'].get('commit') or env['harness'].get('note')}",
    ]

    if problems:
        lines.append("")
        lines.append(f"ПРОБЛЕМЫ ({len(problems)}):")
        lines += [f"  ! {p}" for p in problems]
    else:
        lines.append("")
        lines.append("проверки окружения пройдены")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Этап 0 — сбор и проверка окружения")
    ap.add_argument("--host", help="id конфига или путь к нему; по умолчанию — по GPU")
    ap.add_argument("--out", type=Path, help="куда писать env.json")
    ap.add_argument("--no-hash", action="store_true",
                    help="не считать sha256 бинарей (быстрее)")
    ap.add_argument("--strict", action="store_true",
                    help="ненулевой код возврата при любой проблеме")
    args = ap.parse_args()

    use_utf8_output()

    try:
        cfg = HostConfig.load(args.host)
    except ConfigError as err:
        print(f"ошибка конфигурации: {err}", file=sys.stderr)
        return 2

    env = collect_env(cfg, hash_binaries=not args.no_hash)
    problems = validate_env(env, cfg)
    env["validation"] = {"ok": not problems, "problems": problems}

    out = args.out or (cfg.results_dir / "env.json")
    out.parent.mkdir(parents=True, exist_ok=True)

    # Полоса — свойство машины, а не прогона: она меряется один раз Этапом 2 и
    # проставляется сюда. Перезаписать env.json начисто значит потерять её и
    # обрушить разложение времени, которое без неё считать не по чему.
    if out.is_file():
        try:
            prev = json.loads(out.read_text(encoding="utf-8"))
            if prev.get("measured_bandwidth_gbs") and not env.get("measured_bandwidth_gbs"):
                env["measured_bandwidth_gbs"] = prev["measured_bandwidth_gbs"]
                env["bandwidth"] = prev.get("bandwidth")
        except (OSError, json.JSONDecodeError):
            pass

    out.write_text(json.dumps(env, indent=2, ensure_ascii=False), encoding="utf-8")

    print(format_summary(env, problems))
    print(f"\nзаписано: {out}")
    return 1 if (problems and args.strict) else 0


if __name__ == "__main__":
    raise SystemExit(main())
