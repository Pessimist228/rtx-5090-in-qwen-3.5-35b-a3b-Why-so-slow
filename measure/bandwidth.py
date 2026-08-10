"""Этап 2 — реальная полоса памяти GPU.

Паспортные 256 или 1792 ГБ/с брать нельзя: достижимое ниже, и весь roofline
считается по измеренному числу.

Что именно меряется. При декоде с батчем 1 каждый вес читается ровно один раз
на один матрично-векторный продукт и больше не переиспользуется, запись
ничтожна. Значит потолок задаёт **последовательное чтение**, и заголовочным
числом идёт оно. Копирование меряется рядом как перекрёстная проверка.

Отдельная забота — мобильная карта на 75 Вт. Короткий тест успевает отработать
на бусте, а настоящий прогон llama-bench идёт минутами и уже на сниженных
частотах. Полоса с буста завысила бы крышу и занизила остаток, поэтому
измерение держит нагрузку до выхода на установившийся режим, а просадка первой
выборки относительно последней пишется в отчёт.

    python measure/bandwidth.py
    python measure/bandwidth.py --buffer-gib 2 --samples 80
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import HostConfig, ConfigError  # noqa: E402
from common.env import collect_gpu, use_utf8_output  # noqa: E402

# Спецификации GPU считают в десятичных ГБ, не в гибибайтах. Смешать эти две
# шкалы — получить ошибку в 7%, что сравнимо с искомым остатком.
GB = 1_000_000_000

TARGET_SAMPLE_S = 0.2  # столько длится одна выборка, чтобы запуск ядра не считался


def _sync_time(fn, iters: int) -> float:
    """Время одного прохода в секундах, по событиям CUDA."""
    import torch

    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / 1000.0 / iters


def _pattern_read(buf):
    """Чистое чтение: полный проход по буферу, запись — один скаляр."""
    return lambda: buf.sum()


def _pattern_copy(src, dst):
    """Чтение плюс запись, по байту в обе стороны."""
    return lambda: dst.copy_(src)


def measure_pattern(name: str, fn, bytes_per_pass: int, samples: int) -> dict:
    """Гоняет выборки, возвращает статистику. Медиана, а не среднее."""
    # Прогрев и подбор числа проходов на выборку.
    _sync_time(fn, 1)
    t_one = _sync_time(fn, 3)
    iters = max(1, int(TARGET_SAMPLE_S / t_one)) if t_one > 0 else 1

    per_sample = []
    for _ in range(samples):
        t = _sync_time(fn, iters)
        per_sample.append(bytes_per_pass / t / GB)

    median = statistics.median(per_sample)
    spread = (max(per_sample) - min(per_sample)) / median * 100 if median else 0.0
    # Просадка от первой десятой выборок к последней — это и есть троттлинг.
    head = statistics.median(per_sample[: max(1, len(per_sample) // 10)])
    tail = statistics.median(per_sample[-max(1, len(per_sample) // 10):])
    return {
        "pattern": name,
        "bytes_per_pass": bytes_per_pass,
        "passes_per_sample": iters,
        "samples": len(per_sample),
        "gbs_median": round(median, 2),
        "gbs_min": round(min(per_sample), 2),
        "gbs_max": round(max(per_sample), 2),
        "spread_pct": round(spread, 2),
        "gbs_first_decile": round(head, 2),
        "gbs_last_decile": round(tail, 2),
        "sustained_drop_pct": round((head - tail) / head * 100, 2) if head else 0.0,
        "gbs_samples": [round(v, 2) for v in per_sample],
    }


def validate_measurement(buf_b: int, samples: int = 9) -> dict:
    """Перекрёстная проверка, что измерено именно упирание в память.

    Два независимых признака. Первый: полоса не должна зависеть от размера
    буфера — если бы работал кэш, малый буфер оказался бы быстрее большого.
    Второй: разные ядра редукции должны сойтись на одном числе — если они
    расходятся, мы померили ядро, а не память.

    Стоит полторы минуты и снимает главный риск Фазы 2: на sm_120 PyTorch
    выберет другие ядра, и проверять там будет уже некогда.
    """
    import torch

    # Размеры ниже L2 брать нельзя: они померяют кэш, а не память. У Blackwell
    # L2 равен 96 MiB, и буфер в 64 MiB помещается в него целиком — замер
    # показал 4285 ГБ/с при полосе памяти 1683. Порог в четыре кэша даёт запас.
    l2 = getattr(torch.cuda.get_device_properties(0), "L2_cache_size", 0) or 0
    floor = max(l2 * 4, 64 * 1024**2)

    sweep = {}
    skipped = []
    for mib in (64, 256, 1024, 2048):
        nb = mib * 1024**2
        if nb >= buf_b:
            continue
        if nb < floor:
            skipped.append(f"{mib}MiB")
            continue
        x = torch.empty(nb // 4, dtype=torch.float32, device="cuda").uniform_(-1, 1)
        sweep[f"{mib}MiB"] = round(
            measure_pattern("read", _pattern_read(x), nb, samples)["gbs_median"], 1
        )
        del x
        torch.cuda.empty_cache()

    x = torch.empty(buf_b // 4, dtype=torch.float32, device="cuda").uniform_(-1, 1)
    ops = {
        "sum_f32": measure_pattern("read", lambda: x.sum(), buf_b, samples)["gbs_median"],
        "max_f32": measure_pattern("read", lambda: x.max(), buf_b, samples)["gbs_median"],
        "sum_f16": measure_pattern(
            "read", lambda: x.view(torch.float16).sum(), buf_b, samples
        )["gbs_median"],
    }
    full = ops["sum_f32"]
    del x
    torch.cuda.empty_cache()

    problems = []
    spread = (max(ops.values()) - min(ops.values())) / full * 100
    if spread > 2.0:
        problems.append(
            f"ядра редукции расходятся на {spread:.1f}% ({ops}) — "
            f"измерено ядро, а не полоса памяти"
        )
    for label, gbs in sweep.items():
        if gbs > full * 1.05:
            problems.append(
                f"на буфере {label} полоса {gbs} выше, чем на полном ({full}) — "
                f"данные попадают в кэш, буфер надо увеличить"
            )
    return {
        "size_sweep_gbs": sweep,
        "size_sweep_skipped_below_l2": skipped,
        "l2_cache_bytes": l2,
        "sweep_floor_bytes": floor,
        "op_agreement_gbs": {k: round(v, 1) for k, v in ops.items()},
        "op_spread_pct": round(spread, 2),
        "problems": problems,
    }


def measure_bandwidth(cfg: HostConfig, buffer_gib: float, samples: int,
                      validate: bool = False) -> dict:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("torch не видит CUDA — полосу измерить нечем")

    torch.cuda.set_device(0)
    free_b, total_b = torch.cuda.mem_get_info()

    # Копированию нужно два буфера. Треть свободной памяти оставляет запас и
    # гарантирует, что буфер во много раз больше L2 и кэш роли не играет.
    want_b = int(buffer_gib * 1024**3)
    cap_b = int(free_b * 0.35)
    buf_b = min(want_b, cap_b)
    buf_b -= buf_b % (4 * 1024 * 1024)
    if buf_b < 64 * 1024**2:
        raise RuntimeError(
            f"свободной VRAM слишком мало: {free_b / 1024**3:.2f} GiB"
        )

    n = buf_b // 4  # float32
    gpu_before = collect_gpu()

    src = torch.empty(n, dtype=torch.float32, device="cuda").uniform_(-1, 1)
    results = {}

    results["read"] = measure_pattern("read", _pattern_read(src), buf_b, samples)

    dst = torch.empty_like(src)
    results["copy"] = measure_pattern(
        "copy", _pattern_copy(src, dst), buf_b * 2, samples
    )
    del dst
    torch.cuda.empty_cache()

    validation = validate_measurement(buf_b) if validate else None

    gpu_after = collect_gpu()

    read_gbs = results["read"]["gbs_median"]
    spec = cfg.gpu.get("spec_bandwidth_gbs")

    props = torch.cuda.get_device_properties(0)
    out = {
        "schema_version": 1,
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "host_id": cfg.host_id,
        "method": "torch: последовательное чтение (sum) и копирование device-to-device",
        "device": {
            "name": props.name,
            "l2_cache_bytes": getattr(props, "L2_cache_size", None),
            "vram_total_bytes": total_b,
            "vram_free_bytes_at_start": free_b,
        },
        "buffer_bytes": buf_b,
        "buffer_gib": round(buf_b / 1024**3, 3),
        "patterns": results,
        "measured_bandwidth_gbs": read_gbs,
        "spec_bandwidth_gbs": spec,
        "efficiency_pct": round(read_gbs / spec * 100, 1) if spec else None,
        "thermal": {
            "before": _thermal(gpu_before),
            "after": _thermal(gpu_after),
        },
        "validation": validation,
    }
    out["warnings"] = _warnings(out, cfg)
    if validation:
        out["warnings"] += validation["problems"]
    return out


def _thermal(gpu: dict) -> dict:
    if not gpu.get("available"):
        return {}
    return {
        "temperature_c": gpu.get("temperature_c"),
        "sm_clock_mhz": gpu.get("clocks_mhz", {}).get("sm_current"),
        "memory_clock_mhz": gpu.get("clocks_mhz", {}).get("memory_current"),
        "power_draw_w": gpu.get("power_w", {}).get("draw"),
        "throttle_reasons": gpu.get("throttle_reasons_active"),
        "throttle_harmful": gpu.get("throttle_harmful"),
    }


def _warnings(out: dict, cfg: HostConfig) -> list[str]:
    warns = []
    read = out["patterns"]["read"]
    noise = cfg.bench_matrix.get("noise_threshold_pct", 3.0)

    if read["spread_pct"] > noise:
        warns.append(
            f"разброс чтения {read['spread_pct']}% > порога {noise}% — прогон шумный"
        )
    if read["sustained_drop_pct"] > noise:
        warns.append(
            f"полоса просела на {read['sustained_drop_pct']}% к концу теста — "
            f"карта уходит в троттлинг, для roofline брать установившееся значение "
            f"{read['gbs_last_decile']} ГБ/с"
        )
    harmful = out["thermal"].get("after", {}).get("throttle_harmful") or []
    if harmful:
        warns.append(f"под нагрузкой сработал троттлинг: {harmful}")

    eff = out.get("efficiency_pct")
    if eff is not None:
        if eff > 100:
            warns.append(
                f"измеренная полоса {eff}% от паспортной — это невозможно, "
                f"проверьте spec_bandwidth_gbs в конфиге"
            )
        elif eff < 50:
            warns.append(
                f"измеренная полоса всего {eff}% от паспортной — подозрительно низко"
            )

    # Копирование двигает вдвое больше байт; сильно обогнать чтение оно не может.
    copy_gbs = out["patterns"]["copy"]["gbs_median"]
    if copy_gbs > read["gbs_median"] * 1.15:
        warns.append(
            f"копирование ({copy_gbs}) заметно быстрее чтения "
            f"({read['gbs_median']}) — тест чтения, похоже, не упёрся в память"
        )
    return warns


def update_env(env_path: Path, out: dict) -> bool:
    """Проставить measured_bandwidth_gbs в env.json — так требует Этап 2."""
    if not env_path.is_file():
        return False
    env = json.loads(env_path.read_text(encoding="utf-8"))
    env["measured_bandwidth_gbs"] = out["measured_bandwidth_gbs"]
    env["bandwidth"] = {
        "measured_gbs": out["measured_bandwidth_gbs"],
        "sustained_gbs": out["patterns"]["read"]["gbs_last_decile"],
        "spec_gbs": out["spec_bandwidth_gbs"],
        "efficiency_pct": out["efficiency_pct"],
        "measured_at": out["measured_at"],
    }
    env_path.write_text(json.dumps(env, indent=2, ensure_ascii=False), encoding="utf-8")
    return True


def format_summary(out: dict) -> str:
    r, c = out["patterns"]["read"], out["patterns"]["copy"]
    lines = [
        f"буфер       : {out['buffer_gib']} GiB "
        f"(L2 {(out['device']['l2_cache_bytes'] or 0) / 1024**2:.0f} MiB — кэш не влияет)",
        f"выборок     : {r['samples']} по {r['passes_per_sample']} проходов",
        "",
        f"чтение      : {r['gbs_median']:.1f} ГБ/с  "
        f"(мин {r['gbs_min']:.1f}, макс {r['gbs_max']:.1f}, разброс {r['spread_pct']:.1f}%)",
        f"копирование : {c['gbs_median']:.1f} ГБ/с  (читает и пишет, разброс {c['spread_pct']:.1f}%)",
        "",
        f"начало      : {r['gbs_first_decile']:.1f} ГБ/с",
        f"установилось: {r['gbs_last_decile']:.1f} ГБ/с  "
        f"(просадка {r['sustained_drop_pct']:.1f}%)",
    ]
    t_b, t_a = out["thermal"].get("before", {}), out["thermal"].get("after", {})
    if t_b and t_a:
        lines += [
            "",
            f"температура : {t_b.get('temperature_c')} -> {t_a.get('temperature_c')} C",
            f"частота sm  : {t_b.get('sm_clock_mhz')} -> {t_a.get('sm_clock_mhz')} MHz",
            f"мощность    : {t_b.get('power_draw_w')} -> {t_a.get('power_draw_w')} Вт",
        ]
    v = out.get("validation")
    if v:
        lines += [
            "",
            f"по размеру  : {v['size_sweep_gbs']} (кэш не помогает — число не растёт)",
            f"по ядрам    : {v['op_agreement_gbs']} (расхождение {v['op_spread_pct']}%)",
        ]
    lines += [
        "",
        f"ИЗМЕРЕНО    : {out['measured_bandwidth_gbs']:.1f} ГБ/с",
        f"паспортная  : {out['spec_bandwidth_gbs']} ГБ/с  "
        f"(достигнуто {out['efficiency_pct']}%)",
    ]
    if out["warnings"]:
        lines += ["", f"ПРЕДУПРЕЖДЕНИЯ ({len(out['warnings'])}):"]
        lines += [f"  ! {w}" for w in out["warnings"]]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Этап 2 — измерение полосы памяти")
    ap.add_argument("--host", help="id конфига; по умолчанию — по GPU")
    ap.add_argument("--buffer-gib", type=float, default=1.5,
                    help="размер буфера, GiB (обрезается по свободной VRAM)")
    ap.add_argument("--samples", type=int, default=60,
                    help="число выборок на паттерн; больше — виднее троттлинг")
    ap.add_argument("--validate", action="store_true",
                    help="перекрёстная проверка: развёртка по размеру буфера "
                         "и согласие разных ядер редукции (+~1.5 мин)")
    ap.add_argument("--out", type=Path, help="куда писать bandwidth.json")
    ap.add_argument("--env", type=Path,
                    help="env.json, куда проставить measured_bandwidth_gbs")
    args = ap.parse_args()

    use_utf8_output()

    try:
        cfg = HostConfig.load(args.host)
    except ConfigError as err:
        print(f"ошибка конфигурации: {err}", file=sys.stderr)
        return 2

    try:
        out = measure_bandwidth(cfg, args.buffer_gib, args.samples,
                                validate=args.validate)
    except RuntimeError as err:
        print(f"измерение не удалось: {err}", file=sys.stderr)
        return 3

    dest = args.out or (cfg.results_dir / "bandwidth.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    env_path = args.env or (cfg.results_dir / "env.json")
    patched = update_env(env_path, out)

    print(format_summary(out))
    print(f"\nзаписано: {dest}")
    print(f"env.json : {'обновлён' if patched else 'не найден, measured_bandwidth_gbs не проставлен'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
