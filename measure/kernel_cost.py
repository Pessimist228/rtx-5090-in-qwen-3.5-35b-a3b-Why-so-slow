"""Цена одного ядра: порог выхода на полосу, константа C, цена сериализации.

Зачем. Профиль Фазы 2 показал 1553 ядра на токен при среднем времени жизни
ядра около 2 мкс. Отсюда план оптимизации — укрупнять ядра. Но потолок этого
плана задаётся двумя числами, которых у нас нет:

    N_плато  — минимальный размер чтения, на котором ядро выходит на полосу.
               Слитое ядро, читающее меньше, полосу не получит.
    C        — фиксированная цена одного ядра, не зависящая от объёма работы.

Подстановка C в t = C*K + байты/полоса даёт допустимое число ядер на токен K
для желаемой утилизации. При C = 0.5 мкс девяносто процентов берутся сотнями
ядер, при C = 2 мкс — не берутся никак. Один вечер замеров отвечает на вопрос,
который иначе выяснится месяцем работы.

Что здесь меряется:

  size    развёртка размера чтения: полоса и время от N. Даёт N_плато и C как
          пересечение прямой t(N) с нулём.
  chain   развёртка числа запусков: T от M для цепочки тривиальных ядер.
          Наклон — цена одного ядра в чистом виде, без полезной работы.
  ceiling подстановка измеренного C в формулу: сколько ядер на токен допустимо.

Три режима запуска, разница между ними и есть искомое:

  stream  ядра идут в один поток встык, зависимости по данным нет
  chain   то же, но каждое ядро читает результат предыдущего (как реальный слой)
  graph   то же, но захвачено в CUDA-граф (так работает llama.cpp)

  chain - stream = цена сериализации, фузией не убирается: блоки идут по порядку
  stream - graph = что уже дал захват графа

Про кэш. Наивный свип 64 КиБ..64 МиБ померяет L2, а не память: у 4060 кэш
32 МиБ, у 5090 — 96 МиБ, весь диапазон помещается внутрь. Поэтому каждый
запуск читает новый срез большого пула: к моменту возврата к срезу он вытеснен.
Развёртка продлена вверх до 256 МиБ, иначе не с чем сравнивать плато.

    python measure/kernel_cost.py                 # всё, ~5 минут
    python measure/kernel_cost.py --only size
    python measure/kernel_cost.py --bytes-per-token 2.133e9 --kernels-per-token 1553
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

GB = 1_000_000_000
MiB = 1024 ** 2

# Развёртка размера. Нижняя граница — заведомо латентная область, верхняя
# заведомо за пределами любого L2, чтобы плато было видно как плато.
SIZE_STEPS_B = [
    64 * 1024, 128 * 1024, 256 * 1024, 512 * 1024,
    1 * MiB, 2 * MiB, 4 * MiB, 8 * MiB, 16 * MiB, 32 * MiB,
    64 * MiB, 128 * MiB, 256 * MiB, 512 * MiB, 1024 * MiB,
]
# Верх диапазона рассчитан на карту, которую труднее заполнить. У 4060 плато
# наступает к 32 МиБ при 24 SM, но у 5090 их 170, и порог обязан уехать вверх.
# Оборвать развёртку на 256 МиБ значит рискнуть принять склон за плато.

# Развёртка числа ядер. 1553 — измеренное число ядер на токен у MoE на 5090,
# оно должно попасть внутрь диапазона, а не на край.
CHAIN_STEPS = [50, 100, 200, 400, 800, 1200, 1600, 2000]

TARGET_BATCH_S = 0.05   # столько длится один замер, чтобы события CUDA не врали
SAMPLES = 9             # медиана по выборкам, среднее не годится

# Байты на токен по умолчанию для каждого хоста — из Этапа 4. Нужны, чтобы
# поймать подстановку чужой модели к полосе текущей карты.
DEFAULT_BPT_HOST = {"laptop-4060": 4.904e9, "rented-5090": 2.133e9}


# --------------------------------------------------------------------------
# примитивы запуска
# --------------------------------------------------------------------------

def _events():
    import torch
    return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)


def _time_stream(fns, repeats: int) -> float:
    """Секунд на один вызов: список ядер, запущенный repeats раз в текущий поток."""
    import torch

    start, end = _events()
    torch.cuda.synchronize()
    start.record()
    for _ in range(repeats):
        for f in fns:
            f()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / 1000.0 / (repeats * len(fns))


def _capture(fns):
    """Захватить список ядер в CUDA-граф. Прогрев на боковом потоке обязателен."""
    import torch

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            for f in fns:
                f()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for f in fns:
            f()
    return g


def _time_replay(g, nodes: int, repeats: int) -> float:
    """Секунд на один узел: готовый граф, воспроизведённый repeats раз."""
    import torch

    start, end = _events()
    torch.cuda.synchronize()
    start.record()
    for _ in range(repeats):
        g.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / 1000.0 / (repeats * nodes)


def _cpu_bound(fns) -> dict:
    """Успевает ли CPU кормить GPU. Без этого поточный замер меряет Python.

    Каждый вызов torch из Python стоит единицы-десятки микросекунд, что на
    порядок больше цены запуска ядра. Если после того, как CPU отпустил
    очередь, GPU дорабатывает считанные наносекунды, значит карта простаивала
    и померена скорость Python, а не железа. Воспроизведение графа этим не
    страдает: в его цикле Python нет.
    """
    import time
    import torch

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for f in fns:
        f()
    t1 = time.perf_counter()
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    cpu_us = (t1 - t0) / len(fns) * 1e6
    tail_us = (t2 - t1) / len(fns) * 1e6
    return {
        "cpu_us_per_launch": round(cpu_us, 3),
        "gpu_tail_us_per_launch": round(tail_us, 3),
        "cpu_bound": tail_us < cpu_us * 0.25,
    }


def _measure(fns, mode: str, samples: int = SAMPLES) -> dict:
    """Медиана времени на один вызов плюс разброс. Число повторов подбирается.

    Граф захватывается один раз на всю серию: пересборка на каждой выборке
    померяла бы цену захвата, а не цену воспроизведения.
    """
    import torch

    g = _capture(fns) if mode == "graph" else None
    runner = ((lambda _f, r: _time_replay(g, len(fns), r)) if g is not None
              else _time_stream)

    t_one = runner(fns, 1)
    per_pass = t_one * len(fns)
    repeats = max(1, min(2000, int(TARGET_BATCH_S / per_pass))) if per_pass > 0 else 1

    runner(fns, repeats)  # прогрев, результат отбрасывается
    vals = [runner(fns, repeats) for _ in range(samples)]
    med = statistics.median(vals)
    if g is not None:
        del g, runner
        torch.cuda.empty_cache()
    return {
        "s_per_launch": med,
        "us_per_launch": med * 1e6,
        "spread_pct": (max(vals) - min(vals)) / med * 100 if med else 0.0,
        "repeats": repeats,
        "launches": len(fns),
    }


# --------------------------------------------------------------------------
# замер 1 — развёртка размера чтения
# --------------------------------------------------------------------------

def _cold_readers(pool, nbytes: int, max_slices: int = 512):
    """Ядра чтения по N байт, каждое со своего среза — данные всегда холодные.

    Возвращает список замыканий и суммарный охват в байтах. Охват обязан
    заметно превышать L2, иначе меряется кэш.
    """
    import torch

    n_elem = nbytes // 4
    slices = min(max_slices, pool.numel() // n_elem)
    # Свой приёмник на срез: общий скаляр связал бы чтения записью в одну
    # ячейку и добавил бы к времени ядра сериализацию, которой здесь не место.
    outs = [torch.zeros((), dtype=torch.float32, device="cuda") for _ in range(slices)]
    views = [pool[i * n_elem:(i + 1) * n_elem] for i in range(slices)]
    fns = [(lambda v=v, o=o: torch.sum(v, dim=0, out=o)) for v, o in zip(views, outs)]
    return fns, slices * nbytes, outs


def sweep_size(pool, l2_bytes: int, modes=("stream", "graph")) -> dict:
    """Полоса и время от размера чтения. Даёт N_плато и C как пересечение."""
    import torch

    pool_b = pool.numel() * 4
    rows = []
    for nbytes in SIZE_STEPS_B:
        if nbytes * 2 > pool_b:
            continue
        fns, span, outs = _cold_readers(pool, nbytes)
        row = {
            "bytes": nbytes,
            "kib": nbytes // 1024,
            "slices": len(fns),
            "cold_span_bytes": span,
            "cold_span_over_l2": round(span / l2_bytes, 1) if l2_bytes else None,
        }
        for mode in modes:
            m = _measure(fns, mode)
            row[mode] = {
                "us": round(m["us_per_launch"], 3),
                "gbs": round(nbytes / m["s_per_launch"] / GB, 1),
                "spread_pct": round(m["spread_pct"], 2),
            }
        rows.append(row)
        del fns, outs
        torch.cuda.empty_cache()
    return {"rows": rows, "pool_bytes": pool_b}


def analyse_size(sweep: dict, mode: str) -> dict:
    """N_плато и C = пересечение прямой t(N) с нулём по линейному хвосту."""
    rows = [r for r in sweep["rows"] if mode in r]
    if len(rows) < 4:
        return {}

    gbs = [r[mode]["gbs"] for r in rows]
    peak = max(gbs)

    thresholds = {}
    for pct in (80, 90, 95):
        hit = next((r["bytes"] for r in rows if r[mode]["gbs"] >= peak * pct / 100), None)
        thresholds[f"n_at_{pct}pct_bytes"] = hit

    # Прямую строим по хвосту, где режим уже линеен: t = C + N/B. Малые N
    # латентно-ограничены и в подгонку идти не должны.
    tail = [r for r in rows if r[mode]["gbs"] >= peak * 0.9]
    fit = {}
    if len(tail) >= 2:
        import numpy as np
        x = np.array([r["bytes"] for r in tail], dtype=float)
        y = np.array([r[mode]["us"] for r in tail], dtype=float) * 1e-6
        slope, intercept = np.polyfit(x, y, 1)
        fit = {
            "points": len(tail),
            "fit_bandwidth_gbs": round(1.0 / slope / GB, 1) if slope > 0 else None,
            "C_us": round(intercept * 1e6, 3),
            "C_from": "пересечение t(N) с нулём по линейному хвосту",
        }
    return {"peak_gbs": peak, **thresholds, "intercept": fit}


# --------------------------------------------------------------------------
# замер 2 — развёртка числа ядер, цена узла
# --------------------------------------------------------------------------

def _trivial(count: int, dependent: bool):
    """Тривиальные ядра: работы нет, остаётся чистая цена запуска.

    dependent=True — цепочка по данным на одном тензоре, как реальный слой:
    ядро не может стартовать, пока не дописало предыдущее.
    dependent=False — независимые тензоры, порядок задаёт только поток.
    """
    import torch

    if dependent:
        x = torch.zeros(1, device="cuda")
        return [(lambda: x.add_(1.0)) for _ in range(count)], x
    xs = [torch.zeros(1, device="cuda") for _ in range(count)]
    return [(lambda t=t: t.add_(1.0)) for t in xs], xs


def sweep_chain(modes=("stream", "chain", "graph")) -> dict:
    """T от числа ядер M. Наклон — цена одного ядра, свободная от работы."""
    import torch

    rows = []
    starved = {}
    for m in CHAIN_STEPS:
        row = {"kernels": m}
        for mode in modes:
            fns, keep = _trivial(m, dependent=(mode != "stream"))
            r = _measure(fns, "graph" if mode == "graph" else "stream")
            row[mode] = {
                "us_per_kernel": round(r["us_per_launch"], 4),
                "total_us": round(r["us_per_launch"] * m, 2),
                "spread_pct": round(r["spread_pct"], 2),
            }
            if mode != "graph":
                row[mode].update(_cpu_bound(fns))
                starved[mode] = starved.get(mode, 0) + int(row[mode]["cpu_bound"])
            del fns, keep
            torch.cuda.empty_cache()
        rows.append(row)
    return {"rows": rows, "cpu_bound_points": starved, "points": len(rows)}


def analyse_chain(sweep: dict, mode: str) -> dict:
    """Цена одного ядра как наклон T(M). Свободный член — цена самого запуска пачки."""
    import numpy as np

    rows = [r for r in sweep["rows"] if mode in r]
    if len(rows) < 3:
        return {}
    x = np.array([r["kernels"] for r in rows], dtype=float)
    y = np.array([r[mode]["total_us"] for r in rows], dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    return {
        "C_us": round(float(slope), 4),
        "batch_overhead_us": round(float(intercept), 2),
        "points": len(rows),
        "max_resid_us": round(float(np.max(np.abs(resid))), 2),
        "linear": bool(np.max(np.abs(resid)) < 0.05 * float(np.max(y))),
    }


# --------------------------------------------------------------------------
# замер 3 — подстановка в формулу
# --------------------------------------------------------------------------

def ceiling(c_us: float, bytes_per_token: float, bandwidth_gbs: float,
            kernels_now: float | None = None) -> dict:
    """Сколько ядер на токен допустимо для заданной утилизации.

        t = C*K + байты/полоса,  утилизация = (байты/полоса) / t

    Отсюда K_max = (байты/полоса) * (1/util - 1) / C.
    """
    t_mem_us = bytes_per_token / (bandwidth_gbs * GB) * 1e6
    out = {
        "C_us": c_us,
        "bytes_per_token": bytes_per_token,
        "bandwidth_gbs": bandwidth_gbs,
        "t_memory_us": round(t_mem_us, 1),
        "steps": [],
    }
    for util in (0.50, 0.60, 0.70, 0.80, 0.90, 0.95):
        k_max = t_mem_us * (1 / util - 1) / c_us if c_us > 0 else float("inf")
        out["steps"].append({
            "utilization": util,
            "max_kernels_per_token": int(k_max),
            "t_token_us": round(t_mem_us / util, 1),
            "tokens_per_s": round(1e6 / (t_mem_us / util), 1),
        })
    if kernels_now:
        out["kernels_now"] = kernels_now
        t_now_us = c_us * kernels_now + t_mem_us
        out["predicted_now"] = {
            "t_token_us": round(t_now_us, 1),
            "overhead_us": round(c_us * kernels_now, 1),
            "utilization_pct": round(100 * t_mem_us / t_now_us, 1),
            "tokens_per_s": round(1e6 / t_now_us, 1),
        }
    return out


# --------------------------------------------------------------------------

def run(cfg: HostConfig, args) -> dict:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("torch не видит CUDA — мерить нечем")
    torch.cuda.set_device(0)

    props = torch.cuda.get_device_properties(0)
    l2 = getattr(props, "L2_cache_size", 0) or 0
    free_b, _ = torch.cuda.mem_get_info()

    out = {
        "schema_version": 1,
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "host_id": cfg.host_id,
        "device": {
            "name": props.name,
            "sm_count": props.multi_processor_count,
            "l2_cache_bytes": l2,
            "l2_cache_mib": round(l2 / MiB, 1),
        },
        "gpu_before": _thermal(collect_gpu()),
        "warnings": [],
    }

    want = ("size", "chain", "ceiling") if args.only == "all" else (args.only,)

    pool = None
    if "size" in want:
        # Пул должен многократно перекрывать L2, иначе развёртка померяет кэш.
        pool_b = min(int(args.pool_gib * 1024 ** 3), int(free_b * 0.6))
        pool_b -= pool_b % (4 * MiB)
        if pool_b < l2 * 4:
            raise RuntimeError(
                f"пул {pool_b/MiB:.0f} МиБ не перекрывает L2 {l2/MiB:.0f} МиБ "
                f"с запасом — развёртка померяет кэш"
            )
        pool = torch.empty(pool_b // 4, dtype=torch.float32, device="cuda").uniform_(-1, 1)
        out["size"] = sweep_size(pool, l2)
        out["size_analysis"] = {m: analyse_size(out["size"], m) for m in ("stream", "graph")}
        del pool
        torch.cuda.empty_cache()

    if "chain" in want:
        out["chain"] = sweep_chain()
        out["chain_analysis"] = {
            m: analyse_chain(out["chain"], m) for m in ("stream", "chain", "graph")
        }
        ca = out["chain_analysis"]
        if ca.get("chain") and ca.get("stream"):
            out["serialization_us"] = round(ca["chain"]["C_us"] - ca["stream"]["C_us"], 4)
        if ca.get("stream") and ca.get("graph"):
            out["graph_saving_us"] = round(ca["stream"]["C_us"] - ca["graph"]["C_us"], 4)

    if "ceiling" in want:
        # Цена узла графа — то самое C: llama.cpp графы захватывает, значит
        # платит именно её, а не цену запуска из потока.
        c = args.c_us
        src = "задано ключом --c-us"
        node = (out.get("chain_analysis") or {}).get("graph") or {}
        if c is None and node:
            c = node["C_us"]
            src = "наклон T(M) для графа из тривиальных ядер"
        if c:
            bw = args.bandwidth_gbs or _env_bandwidth(cfg) or cfg.gpu.get("spec_bandwidth_gbs")
            out["ceiling"] = ceiling(c, args.bytes_per_token, bw, args.kernels_per_token)
            out["ceiling"]["C_source"] = src
            # Полоса берётся с текущей карты, а байты на токен — из аргумента.
            # Смешать полосу одной карты с моделью, снятой на другой, — получить
            # правдоподобную и бессмысленную таблицу.
            if args.bandwidth_gbs is None and args.bytes_per_token != DEFAULT_BPT_HOST.get(cfg.host_id):
                out["warnings"].append(
                    f"полоса {bw} ГБ/с взята с текущей карты ({cfg.host_id}), а байты на "
                    f"токен заданы вручную — убедитесь, что это одна и та же связка "
                    f"карта+модель, иначе потолок посчитан ни для чего"
                )

    out["gpu_after"] = _thermal(collect_gpu())
    out["warnings"] += _warnings(out, l2)
    return out


def _env_bandwidth(cfg: HostConfig):
    p = cfg.results_dir / "env.json"
    if p.is_file():
        return json.loads(p.read_text(encoding="utf-8")).get("measured_bandwidth_gbs")
    return None


def _thermal(gpu: dict) -> dict:
    if not gpu.get("available"):
        return {}
    return {
        "temperature_c": gpu.get("temperature_c"),
        "sm_clock_mhz": gpu.get("clocks_mhz", {}).get("sm_current"),
        "power_draw_w": gpu.get("power_w", {}).get("draw"),
        "throttle_harmful": gpu.get("throttle_harmful"),
    }


def _warnings(out: dict, l2: int) -> list[str]:
    warns = []
    for r in (out.get("size") or {}).get("rows", []):
        over = r.get("cold_span_over_l2")
        if over is not None and over < 4:
            warns.append(
                f"на {r['kib']} КиБ холодный охват всего {over}x от L2 — "
                f"точка может мерить кэш"
            )
    for mode, a in (out.get("chain_analysis") or {}).items():
        if a and not a.get("linear"):
            warns.append(
                f"T(M) для режима {mode} не линеен (макс. невязка {a['max_resid_us']} мкс) — "
                f"наклон как цену ядра брать нельзя"
            )
    starved = (out.get("chain") or {}).get("cpu_bound_points") or {}
    total = (out.get("chain") or {}).get("points", 0)
    for mode, n in starved.items():
        if n:
            warns.append(
                f"режим {mode}: на {n} из {total} точек GPU голодал, CPU не успевал "
                f"запускать — померена диспетчеризация Python, а не цена запуска ядра. "
                f"Ни C, ни цену сериализации отсюда брать нельзя, годится только граф"
            )
    harm = (out.get("gpu_after") or {}).get("throttle_harmful") or []
    if harm:
        warns.append(f"под нагрузкой сработал троттлинг: {harm} — числа занижены")
    return warns


def format_summary(out: dict) -> str:
    L = []
    d = out["device"]
    L.append(f"карта: {d['name']}, SM {d['sm_count']}, L2 {d['l2_cache_mib']} МиБ")

    if "size" in out:
        L += ["", "РАЗВЁРТКА РАЗМЕРА ЧТЕНИЯ (данные холодные, срезы прокручиваются)",
              f"{'размер':>10}{'срезов':>8}{'охват/L2':>10}"
              f"{'поток, мкс':>12}{'ГБ/с':>9}{'граф, мкс':>12}{'ГБ/с':>9}"]
        L.append("-" * 70)
        for r in out["size"]["rows"]:
            size = f"{r['kib']} КиБ" if r["kib"] < 1024 else f"{r['kib']//1024} МиБ"
            s, g = r.get("stream", {}), r.get("graph", {})
            L.append(f"{size:>10}{r['slices']:>8}{r['cold_span_over_l2']:>9.0f}x"
                     f"{s.get('us', 0):>12.2f}{s.get('gbs', 0):>9.1f}"
                     f"{g.get('us', 0):>12.2f}{g.get('gbs', 0):>9.1f}")
        for mode in ("stream", "graph"):
            a = out.get("size_analysis", {}).get(mode) or {}
            if not a:
                continue
            L.append("")
            L.append(f"  [{mode}] пик {a['peak_gbs']:.1f} ГБ/с")
            for pct in (80, 90, 95):
                v = a.get(f"n_at_{pct}pct_bytes")
                L.append(f"  [{mode}] {pct}% полосы с "
                         f"{(str(v // 1024) + ' КиБ') if v else 'не достигнуто'}")
            it = a.get("intercept") or {}
            if it:
                L.append(f"  [{mode}] C по пересечению = {it['C_us']} мкс "
                         f"(полоса по наклону {it['fit_bandwidth_gbs']} ГБ/с)")

    if "chain" in out:
        L += ["", "РАЗВЁРТКА ЧИСЛА ЯДЕР (тривиальные ядра, полезной работы нет)",
              f"{'ядер':>8}{'поток, мкс':>14}{'цепочка, мкс':>16}{'граф, мкс':>13}"]
        L.append("-" * 51)
        for r in out["chain"]["rows"]:
            L.append(f"{r['kernels']:>8}{r['stream']['us_per_kernel']:>14.3f}"
                     f"{r['chain']['us_per_kernel']:>16.3f}"
                     f"{r['graph']['us_per_kernel']:>13.3f}")
        L.append("")
        for mode, label in (("stream", "поток, без зависимости"),
                            ("chain", "поток, цепочка по данным"),
                            ("graph", "граф, цепочка по данным")):
            a = out["chain_analysis"].get(mode) or {}
            if a:
                L.append(f"  {label:<26} C = {a['C_us']:.3f} мкс/ядро"
                         f"{'' if a['linear'] else '  [НЕ ЛИНЕЕН]'}")
        if "serialization_us" in out:
            L.append(f"  цена сериализации          {out['serialization_us']:+.3f} мкс/ядро "
                     f"(фузией не убирается)")
        if "graph_saving_us" in out:
            L.append(f"  что дал захват графа       {out['graph_saving_us']:+.3f} мкс/ядро")

    c = out.get("ceiling")
    if c:
        L += ["", "ПОТОЛОК: сколько ядер на токен допустимо",
              f"  C = {c['C_us']:.3f} мкс ({c['C_source']})",
              f"  байт/токен {c['bytes_per_token']/GB:.3f} ГБ, полоса {c['bandwidth_gbs']} ГБ/с, "
              f"крыша по памяти {c['t_memory_us']:.0f} мкс",
              "",
              f"{'утилизация':>12}{'ядер/токен':>13}{'мс/токен':>11}{'т/с':>9}"]
        L.append("-" * 45)
        for s in c["steps"]:
            L.append(f"{s['utilization']*100:>11.0f}%{s['max_kernels_per_token']:>13}"
                     f"{s['t_token_us']/1000:>11.3f}{s['tokens_per_s']:>9.1f}")
        p = c.get("predicted_now")
        if p:
            L += ["", f"  сейчас {c['kernels_now']:.0f} ядер/токен -> накладные "
                      f"{p['overhead_us']/1000:.3f} мс, утилизация {p['utilization_pct']}%, "
                      f"{p['tokens_per_s']} т/с"]

    if out["warnings"]:
        L += ["", f"ПРЕДУПРЕЖДЕНИЯ ({len(out['warnings'])}):"]
        L += [f"  ! {w}" for w in out["warnings"]]
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description="Цена ядра: порог полосы, константа C, потолок")
    ap.add_argument("--host", help="id конфига; по умолчанию — по GPU")
    ap.add_argument("--only", default="all", choices=("all", "size", "chain", "ceiling"))
    ap.add_argument("--pool-gib", type=float, default=4.0,
                    help="пул для холодных срезов; обязан многократно перекрывать L2")
    ap.add_argument("--bytes-per-token", type=float, default=2.133e9,
                    help="байт на токен из Этапа 4 (по умолчанию MoE 35B-A3B)")
    ap.add_argument("--kernels-per-token", type=float, default=1553,
                    help="ядер на токен из Этапа 6")
    ap.add_argument("--bandwidth-gbs", type=float,
                    help="полоса; по умолчанию измеренная из env.json")
    ap.add_argument("--c-us", type=float, help="взять C готовым, не мерить")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    use_utf8_output()

    try:
        cfg = HostConfig.load(args.host)
    except ConfigError as err:
        print(f"ошибка конфигурации: {err}", file=sys.stderr)
        return 2

    try:
        out = run(cfg, args)
    except RuntimeError as err:
        print(f"замер не удался: {err}", file=sys.stderr)
        return 3

    dest = args.out or (cfg.results_dir / "kernel_cost.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print(format_summary(out))
    print(f"\nзаписано: {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
