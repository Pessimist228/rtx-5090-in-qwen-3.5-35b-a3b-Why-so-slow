"""Этап 3 — обвязка над llama-bench.

Гоняет матрицу конфигураций из config/<host>.json, складывает по строке JSONL
на точку и считает медиану с разбросом по сырым повторам.

Три вещи, ради которых обвязка вообще нужна.

Частичная выгрузка. На 8 ГБ слой может уехать на CPU, и llama-bench честно
измерит получившееся, не сказав ни слова. Число будет тихо испорченным, а не
явно неверным, поэтому размещение слоёв разбирается из подробного лога, и любой
слой на CPU means падение точки, а не пометка.

Частоты под нагрузкой. Снимать nvidia-smi до и после бесполезно: после прогона
карта уже сбросила частоты, и замер покажет простой. Частоты снимаются потоком
во время прогона, иначе тепловой троттлинг мобильной карты попадёт в остаток и
будет неотличим от накладных CUDA-бэкенда, то есть от искомого.

Медиана, а не среднее. llama-bench печатает avg и stddev, но отдаёт сырые
samples_ns — по ним и считаем.

    python measure/bench.py --model Qwen3.5-9B-Q4_0.gguf
    python measure/bench.py --model Qwen3.5-9B-Q4_0.gguf --quick
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import HostConfig, ConfigError  # noqa: E402
from common.env import (  # noqa: E402
    HARMFUL_THROTTLE, collect_env, decode_throttle, use_utf8_output, validate_env,
)


class OffloadError(RuntimeError):
    """Модель уехала на CPU — замер недействителен."""


# --------------------------------------------------------------------------
# Опрос частот во время прогона
# --------------------------------------------------------------------------

class ClockSampler(threading.Thread):
    """Снимает частоты и температуру, пока идёт прогон.

    После прогона карта сбрасывает частоты за доли секунды, поэтому замер
    до/после не показывает ничего о том, что было под нагрузкой.
    """

    FIELDS = ("clocks.current.sm,clocks.current.memory,temperature.gpu,"
              "power.draw,clocks_throttle_reasons.active")

    def __init__(self, interval: float = 0.5):
        super().__init__(daemon=True)
        self.interval = interval
        self.samples: list[dict] = []
        # Не _stop: так называется внутренний метод threading.Thread.
        self._halt = threading.Event()

    def run(self) -> None:
        while not self._halt.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", f"--query-gpu={self.FIELDS}",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=10,
                ).stdout.strip()
            except (subprocess.SubprocessError, OSError):
                out = ""
            if out:
                parts = [p.strip() for p in out.splitlines()[0].split(",")]
                if len(parts) >= 5:
                    try:
                        self.samples.append({
                            "sm_mhz": float(parts[0]),
                            "mem_mhz": float(parts[1]),
                            "temp_c": float(parts[2]),
                            "power_w": float(parts[3]),
                            "throttle": decode_throttle(parts[4]),
                        })
                    except ValueError:
                        pass
            self._halt.wait(self.interval)

    def stop(self) -> dict:
        self._halt.set()
        self.join(timeout=5)
        if not self.samples:
            return {"samples": 0}
        sm = [s["sm_mhz"] for s in self.samples]
        harmful = sorted({r for s in self.samples for r in s["throttle"]}
                         & HARMFUL_THROTTLE)
        return {
            "samples": len(self.samples),
            "sm_mhz_median": statistics.median(sm),
            "sm_mhz_min": min(sm),
            "sm_mhz_max": max(sm),
            "mem_mhz_median": statistics.median(s["mem_mhz"] for s in self.samples),
            "temp_c_max": max(s["temp_c"] for s in self.samples),
            "power_w_median": statistics.median(s["power_w"] for s in self.samples),
            "power_w_max": max(s["power_w"] for s in self.samples),
            "throttle_harmful": harmful,
        }


# --------------------------------------------------------------------------
# Разбор подробного лога
# --------------------------------------------------------------------------

def parse_placement(log: str) -> dict:
    """Где оказались слои и что с CUDA-графами."""
    cuda_layers = len(re.findall(r"assigned to device CUDA\d", log))
    cpu_layers = len(re.findall(r"assigned to device CPU", log))

    buffers = {}
    for m in re.finditer(r"^load_tensors:\s+(\S+) model buffer size =\s+([\d.]+) MiB",
                         log, re.MULTILINE):
        buffers[m.group(1)] = float(m.group(2))

    # llama.cpp сообщает о срыве захвата графов отдельной строкой.
    graph_disabled = re.findall(
        r"(?:disabling CUDA graphs|CUDA graphs are not supported)[^\n]*", log)
    return {
        "layers_on_gpu": cuda_layers,
        "layers_on_cpu": cpu_layers,
        "model_buffers_mib": buffers,
        "cuda_graph_warmups": len(re.findall(r"CUDA graph warmup complete", log)),
        "cuda_graph_disabled_reasons": sorted(set(graph_disabled)),
        "cuda_graphs_used": bool(re.search(r"CUDA graph warmup complete", log))
                            and not graph_disabled,
        "recurrent_state_mib": _recurrent_mib(log),
        "kv_layers": len(re.findall(r"llama_kv_cache: layer\s+\d+: dev", log)),
        "gdn_layers": len(re.findall(r"llama_memory_recurrent, layer\s+\d+: dev", log)),
    }


def _recurrent_mib(log: str) -> float | None:
    """Состояние GDN: читается и пишется каждый токен, квантованием не режется."""
    m = re.search(r"llama_memory_recurrent: size =\s+([\d.]+) MiB", log)
    return float(m.group(1)) if m else None


def check_offload(placement: dict, cfg: HostConfig) -> None:
    if not cfg.safety.get("require_full_gpu_offload", True):
        return
    if placement["layers_on_cpu"] > 0:
        raise OffloadError(
            f"{placement['layers_on_cpu']} слоёв уехало на CPU — замер недействителен"
        )
    if placement["layers_on_gpu"] == 0:
        raise OffloadError("ни одного слоя на GPU — замер недействителен")


# --------------------------------------------------------------------------
# Матрица
# --------------------------------------------------------------------------

def build_matrix(cfg: HostConfig, quick: bool = False) -> list[dict]:
    """Точки матрицы. Префилл и декод разведены: у них разный смысл.

    ТЗ просит `-b 1` как «только single stream», но в llama-bench `-b` это
    размер батча префилла, а не число потоков. Проверено на Qwen3.5-0.8B:
    pp512 даёт 8901 т/с при -b 2048 и 218.7 т/с при -b 1, то есть в 41 раз
    меньше. С `-b 1` префилл перестаёт быть префиллом и становится цепочкой
    однотокенных проходов. Single stream у llama-bench и так единственный
    режим: он не сервер и параллельных запросов не делает. Поэтому размер
    батча берётся из конфига, а не единица.
    """
    m = cfg.bench_matrix
    fa_values = m["flash_attn"]
    kv_values = m["kv_cache_types"]
    if quick:
        fa_values, kv_values = [fa_values[-1]], [kv_values[0]]

    points = []
    prefills = m["prefill_tokens"][:1] if quick else m["prefill_tokens"]
    depths = m["decode_depths"][:1] if quick else m["decode_depths"]

    for fa in fa_values:
        for kv in kv_values:
            if skip_reason(fa, kv):
                continue
            for p in prefills:
                points.append({"kind": "prefill", "n_prompt": p, "n_gen": 0,
                               "n_depth": 0, "flash_attn": fa, "kv_type": kv})
            for d in depths:
                points.append({"kind": "decode", "n_prompt": 0,
                               "n_gen": m["decode_tokens"], "n_depth": d,
                               "flash_attn": fa, "kv_type": kv})
    return points


def skip_reason(fa: int, kv_type: str) -> str | None:
    """Почему сочетание невыполнимо.

    Матрица из ТЗ просит независимо перебрать fa 0/1 и KV f16/q8_0, но
    квантованный KV в llama.cpp требует flash attention: без него контекст
    просто не создаётся. Восемь из тридцати двух точек невыполнимы, и честнее
    отсеять их с причиной, чем восемь раз получить «failed to create context»
    и гадать, дело в нехватке VRAM или в конфигурации.
    """
    if kv_type != "f16" and not fa:
        return f"квантованный KV ({kv_type}) требует flash attention (-fa 1)"
    return None


def run_point(cfg: HostConfig, model_path: Path, point: dict,
              reps: int, extra_env: dict | None = None) -> dict:
    m = cfg.bench_matrix
    cmd = [
        str(cfg.exe("llama-bench")),
        "-m", str(model_path),
        "-p", str(point["n_prompt"]),
        "-n", str(point["n_gen"]),
        "-d", str(point["n_depth"]),
        "-r", str(reps),
        "-ngl", str(m["n_gpu_layers"]),
        "-sm", m["split_mode"],
        "-fa", str(point["flash_attn"]),
        "-ctk", point["kv_type"],
        "-ctv", point["kv_type"],
        "-t", str(m["threads"]),
        "-o", "jsonl",
        "-v",
    ]

    import os
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    sampler = ClockSampler()
    sampler.start()
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", env=env)
    wall = time.time() - t0
    clocks = sampler.stop()

    record = {
        "point": point,
        "command": cmd,
        "extra_env": extra_env or {},
        "wall_seconds": round(wall, 2),
        "clocks_under_load": clocks,
        "returncode": proc.returncode,
    }

    placement = parse_placement(proc.stderr)
    record["placement"] = placement

    if proc.returncode != 0:
        record["status"] = "failed"
        record["error"] = _tail_error(proc.stderr)
        return record

    # Выгрузка на CPU — не пометка, а падение: тихо испорченное число хуже
    # отсутствующего.
    check_offload(placement, cfg)

    lines = [l for l in proc.stdout.splitlines() if l.strip().startswith("{")]
    if not lines:
        record["status"] = "failed"
        record["error"] = "llama-bench не выдал JSONL"
        return record

    raw = json.loads(lines[0])
    ts, ns = raw["samples_ts"], raw["samples_ns"]
    n_tokens = raw["n_gen"] or raw["n_prompt"]
    median_ts = statistics.median(ts)
    spread = (max(ts) - min(ts)) / median_ts * 100 if median_ts else 0.0
    noise_limit = m.get("noise_threshold_pct", 3.0)

    record.update({
        "status": "ok",
        # Кладём весь конфиг, который сообщил llama-bench, а не выборку полей:
        # схема между релизами меняется (в b10326 пропал use_mmap), и белый
        # список ронял бы прогон на ровном месте. Сырые повторы вынесены
        # наверх, чтобы не дублировать их.
        "llama_bench": {k: v for k, v in raw.items()
                        if k not in ("samples_ns", "samples_ts")},
        "samples_ts": ts,
        "samples_ns": ns,
        "median_ts": round(median_ts, 3),
        "mean_ts": round(statistics.fmean(ts), 3),
        "median_ms_per_token": round(statistics.median(ns) / 1e6 / n_tokens, 4),
        "spread_pct": round(spread, 2),
        "noisy": spread > noise_limit,
        "throttled": bool(clocks.get("throttle_harmful")),
    })
    return record


def gpu_now() -> dict:
    """Температура и частота одним запросом."""
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=temperature.gpu,clocks.current.sm,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        p = [x.strip() for x in out.splitlines()[0].split(",")]
        return {"temp_c": float(p[0]), "sm_mhz": float(p[1]), "power_w": float(p[2])}
    except (subprocess.SubprocessError, OSError, ValueError, IndexError):
        return {}


def wait_for_cooldown(cfg: HostConfig, verbose: bool = True) -> dict:
    """Дождаться остывания до целевой температуры перед точкой.

    Без этого матрица меряет не конфигурации, а историю нагрева: замер подряд
    показал просадку SM с 2550 до 1650 МГц за семь точек, и «штраф за глубину
    контекста» оказался неотличим от того, что карта успела нагреться.
    Выравнивание стартовой температуры делает точки сопоставимыми.
    """
    m = cfg.bench_matrix
    target = m.get("cooldown_target_c")
    if not target:
        return {}
    max_wait = m.get("cooldown_max_wait_s", 120)

    t0 = time.time()
    start = gpu_now()
    while True:
        cur = gpu_now()
        temp = cur.get("temp_c")
        if temp is None or temp <= target:
            break
        if time.time() - t0 >= max_wait:
            break
        if verbose:
            print(f"\r    остывание {temp:.0f}C -> {target}C "
                  f"({time.time() - t0:.0f}/{max_wait}с)   ", end="", flush=True)
        time.sleep(3)
    waited = time.time() - t0
    if verbose and waited > 3:
        print("\r" + " " * 60 + "\r", end="")
    end = gpu_now()
    return {
        "waited_s": round(waited, 1),
        "temp_start_c": start.get("temp_c"),
        "temp_end_c": end.get("temp_c"),
        "target_c": target,
        "reached_target": bool(end.get("temp_c") is not None
                               and end["temp_c"] <= target),
    }


def order_points(points: list[dict], cfg: HostConfig) -> list[dict]:
    """Перемешать точки, чтобы остаточный дрейф не совпадал с конфигурацией.

    Даже с остыванием карта дрейфует. Если порядок совпадает с порядком
    конфигураций, дрейф неотличим от их эффекта. Перемешивание превращает
    систематическое смещение в шум. Seed берётся из конфига, порядок
    воспроизводим.
    """
    if not cfg.bench_matrix.get("randomize_order", False):
        return points
    import random

    rng = random.Random(cfg.bench_matrix.get("order_seed", 1))
    shuffled = points[:]
    rng.shuffle(shuffled)
    return shuffled


def inherit_bandwidth(env: dict, cfg: HostConfig) -> None:
    """Подтянуть измеренную полосу из результатов Этапа 2.

    collect_env() собирает окружение заново и про полосу ничего не знает, а
    она свойство машины, а не прогона: меряется один раз и должна лежать рядом
    с каждым числом, иначе разложение времени считать не по чему.
    """
    src = cfg.results_dir / "env.json"
    if not src.is_file():
        return
    try:
        prev = json.loads(src.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if prev.get("measured_bandwidth_gbs"):
        env["measured_bandwidth_gbs"] = prev["measured_bandwidth_gbs"]
        env["bandwidth"] = prev.get("bandwidth")


def _tail_error(stderr: str) -> str:
    lines = [l for l in stderr.splitlines() if l.strip()]
    keep = [l for l in lines if re.search(r"error|failed|not enough|out of memory",
                                          l, re.IGNORECASE)]
    return " | ".join((keep or lines)[-3:])[:500]


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Этап 3 — матрица прогонов llama-bench")
    ap.add_argument("--model", required=True, help="имя файла GGUF или путь")
    ap.add_argument("--host", help="id конфига; по умолчанию — по GPU")
    ap.add_argument("--quant", help="метка кванта для имени каталога; по умолчанию из имени файла")
    ap.add_argument("--reps", type=int, help="повторов на точку (по умолчанию из конфига)")
    ap.add_argument("--quick", action="store_true",
                    help="одна точка префилла и декода, fa=1, kv=f16 — для проверки обвязки")
    ap.add_argument("--graphs-off", action="store_true",
                    help="продублировать матрицу с GGML_CUDA_DISABLE_GRAPHS=1; "
                         "разница и есть стоимость запусков ядер")
    ap.add_argument("--out-dir", type=Path, help="каталог результатов (по умолчанию генерируется)")
    args = ap.parse_args()

    use_utf8_output()

    try:
        cfg = HostConfig.load(args.host)
        model_path = cfg.find_model(args.model)
    except ConfigError as err:
        print(f"ошибка конфигурации: {err}", file=sys.stderr)
        return 2

    env = collect_env(cfg, hash_binaries=False)
    inherit_bandwidth(env, cfg)

    # torch и gguf нужны Этапам 2 и 4, а не замерам: llama-bench про них не
    # знает. Если полоса уже измерена, их отсутствие не повод не мерить —
    # иначе сломанный torch останавливает работу на исправной машине.
    problems = validate_env(env, cfg)
    advisory = []
    if env.get("measured_bandwidth_gbs"):
        advisory = [p for p in problems if "torch" in p or "gguf" in p]
        problems = [p for p in problems if p not in advisory]

    if problems:
        print("окружение не прошло проверку — замеры не начаты:", file=sys.stderr)
        for p in problems:
            print(f"  ! {p}", file=sys.stderr)
        return 1
    for p in advisory:
        print(f"предупреждение: {p}", file=sys.stderr)
    if env.get("measured_bandwidth_gbs") is None:
        print("предупреждение: полоса не измерена, сначала measure/bandwidth.py",
              file=sys.stderr)

    quant = args.quant or _guess_quant(model_path.name)
    run_dir = args.out_dir or cfg.run_dir(model_path.stem, quant)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "env.json").write_text(
        json.dumps(env, indent=2, ensure_ascii=False), encoding="utf-8")

    reps = args.reps or cfg.bench_matrix.get("repetitions", 5)
    points = build_matrix(cfg, quick=args.quick)
    variants = [({}, "graphs_on")]
    if args.graphs_off:
        variants.append(({"GGML_CUDA_DISABLE_GRAPHS": "1"}, "graphs_off"))

    work = [{"point": p, "extra_env": e, "variant": v}
            for e, v in variants for p in points]
    work = order_points(work, cfg)

    # Якорь: одна и та же точка в начале и в конце. Если её результат уехал
    # больше порога шума, матрица не самосогласована — карта дрейфовала, и
    # сравнивать точки между собой нельзя.
    anchor = cfg.bench_matrix.get("anchor_check", False) and not args.quick
    if anchor:
        ap = {"kind": "decode", "n_prompt": 0, "n_gen": cfg.bench_matrix["decode_tokens"],
              "n_depth": 0, "flash_attn": 1, "kv_type": "f16"}
        work = ([{"point": ap, "extra_env": {}, "variant": "anchor_first"}] + work
                + [{"point": ap, "extra_env": {}, "variant": "anchor_last"}])

    total = len(work)
    print(f"модель   : {model_path}")
    print(f"каталог  : {run_dir}")
    print(f"точек    : {total} (повторов на точку: {reps})")
    print(f"полоса   : {env.get('measured_bandwidth_gbs')} ГБ/с")
    print(f"порядок  : {'перемешан, seed=' + str(cfg.bench_matrix.get('order_seed', 1)) if cfg.bench_matrix.get('randomize_order') else 'как задан'}")
    print(f"остывание: до {cfg.bench_matrix.get('cooldown_target_c')}C "
          f"перед каждой точкой\n")

    jsonl_path = run_dir / "bench.jsonl"
    ok = failed = noisy = 0
    anchors: dict[str, float] = {}

    with jsonl_path.open("w", encoding="utf-8") as fh:
        for i, item in enumerate(work, 1):
            point, variant = item["point"], item["variant"]
            label = (f"{point['kind']:<7} "
                     f"{'p' + str(point['n_prompt']) if point['kind'] == 'prefill' else 'n' + str(point['n_gen']) + '@d' + str(point['n_depth']):<11}"
                     f"fa={point['flash_attn']} kv={point['kv_type']:<5} {variant}")
            print(f"[{i}/{total}] {label} ... ", end="", flush=True)

            cooldown = wait_for_cooldown(cfg)
            try:
                rec = run_point(cfg, model_path, point, reps, item["extra_env"])
            except OffloadError as err:
                print(f"ВЫГРУЗКА: {err}")
                return 3
            rec["variant"] = variant
            rec["run_index"] = i
            rec["cooldown"] = cooldown
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()

            if rec["status"] != "ok":
                failed += 1
                print(f"ОШИБКА: {rec.get('error', '')[:120]}")
                continue
            ok += 1
            if variant.startswith("anchor"):
                anchors[variant] = rec["median_ts"]

            flags = []
            if rec["noisy"]:
                noisy += 1
                flags.append(f"ШУМ {rec['spread_pct']}%")
            if rec["throttled"]:
                flags.append(f"ТРОТТЛИНГ {rec['clocks_under_load']['throttle_harmful']}")
            sm = rec["clocks_under_load"].get("sm_mhz_median")
            line = (f"{rec['median_ts']:8.2f} т/с  "
                    f"{rec['median_ms_per_token']:7.3f} мс/тк")
            if sm:
                line += f"  sm={sm:.0f}МГц"
            if flags:
                line += "  " + " ".join(flags)
            print(line)

    print(f"\nготово: {ok} ок, {failed} с ошибкой, {noisy} шумных")

    verdict = None
    if len(anchors) == 2:
        first, last = anchors["anchor_first"], anchors["anchor_last"]
        drift = (last - first) / first * 100
        limit = cfg.bench_matrix.get("noise_threshold_pct", 3.0)
        verdict = {
            "anchor_first_ts": first,
            "anchor_last_ts": last,
            "drift_pct": round(drift, 2),
            "limit_pct": limit,
            "self_consistent": abs(drift) <= limit,
        }
        print(f"\nякорь    : {first:.2f} -> {last:.2f} т/с ({drift:+.2f}%)")
        if abs(drift) > limit:
            print(f"  ! дрейф больше порога {limit}% — точки матрицы "
                  f"НЕ сопоставимы между собой")
        else:
            print(f"  матрица самосогласована (порог {limit}%)")
        (run_dir / "anchor.json").write_text(
            json.dumps(verdict, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"записано: {jsonl_path}")
    return 0


def _guess_quant(filename: str) -> str:
    m = re.search(r"(UD-)?(IQ|Q)\d+[_A-Za-z0-9]*|BF16|F16", filename)
    return m.group(0) if m else "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
