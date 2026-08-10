"""Этап 6 — атрибуция остатка по трассе nsys.

Разложение говорит, сколько времени уходит мимо памяти. Здесь выясняется, куда.

Окно установившегося декода отбивается по графовым запускам, а не по доле
трассы: профиль включает загрузку модели и прогрев, и грубое «возьмём последние
две трети» смешивает их с замером. При активных CUDA-графах каждый токен — это
ровно один cudaGraphLaunch, поэтому границы токенов известны точно.

Считает: ядер на токен, время GPU под нагрузкой против дыр, топ ядер по
времени и по числу запусков.

    python analyze/attribute.py results-5090/profile
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.env import use_utf8_output  # noqa: E402


def _num(row: dict, *names):
    for n in names:
        v = row.get(n)
        if v not in (None, ""):
            try:
                return float(v)
            except ValueError:
                pass
    return None


def load_csv(directory: Path, suffix: str, match: str = "") -> list[dict]:
    """Читает CSV отчёта nsys.

    В каталоге лежат профили обоих режимов трассировки, и брать первый по
    алфавиту нельзя: у графового режима отдельные ядра внутри графа вообще не
    трассируются, и счёт ядер по нему даёт ноль. Режим выбирается явно.
    """
    hits = sorted(p for p in directory.glob(f"*{suffix}*.csv") if match in p.name)
    if not hits:
        return []
    with hits[0].open(encoding="utf-8", errors="replace") as fh:
        return list(csv.DictReader(fh))


def analyse(directory: Path, match: str = "node") -> dict:
    trace = load_csv(directory, "cuda_gpu_trace", match)
    api = load_csv(directory, "cuda_api_sum", match)
    kern = load_csv(directory, "cuda_gpu_kern_sum", match)
    if not trace:
        raise SystemExit(f"в {directory} нет cuda_gpu_trace CSV")

    graph_launches = 0
    kernel_launch_calls = 0
    for r in api:
        name = r.get("Name", "")
        calls = _num(r, "Num Calls", "NumCalls") or 0
        if "GraphLaunch" in name:
            graph_launches += int(calls)
        elif "LaunchKernel" in name:
            kernel_launch_calls += int(calls)

    events = []
    for r in trace:
        st = _num(r, "Start (ns)", "Start(ns)")
        du = _num(r, "Duration (ns)", "Duration(ns)")
        if st is None or du is None:
            continue
        events.append((st, du, r.get("Name") or r.get("Kernel Name") or ""))
    events.sort()

    total_kernels = len(events)
    # Ядра внутри графов: всё, что не запущено поштучно.
    graph_kernels = max(total_kernels - kernel_launch_calls, 0)
    per_token = graph_kernels / graph_launches if graph_launches else None

    # Окно установившегося декода: отбрасываем первую половину событий, где
    # сидят загрузка и прогрев, и режем по целому числу токенов.
    if per_token:
        window_tokens = max(int(graph_launches * 0.5), 1)
        take = int(window_tokens * per_token)
        tail = events[-take:]
    else:
        tail = events[len(events) // 2:]
        window_tokens = None

    span = tail[-1][0] + tail[-1][1] - tail[0][0]
    # Слияние перекрытий: параллельные ядра не должны считаться дважды.
    busy = 0.0
    cs, ce = tail[0][0], tail[0][0] + tail[0][1]
    for st, du, _ in tail[1:]:
        if st > ce:
            busy += ce - cs
            cs, ce = st, st + du
        else:
            ce = max(ce, st + du)
    busy += ce - cs

    by_name_time: dict[str, float] = defaultdict(float)
    by_name_count: dict[str, int] = defaultdict(int)
    for r in kern:
        name = (r.get("Name") or "").split("(")[0][:60]
        by_name_time[name] += _num(r, "Total Time (ns)") or 0
        by_name_count[name] += int(_num(r, "Instances") or 0)

    return {
        "directory": str(directory),
        "trace_mode": match,
        "graph_launches": graph_launches,
        "cuda_graphs_used": graph_launches > 0,
        "direct_kernel_launches": kernel_launch_calls,
        "kernels_total": total_kernels,
        "kernels_in_graphs": graph_kernels,
        "kernels_per_token": per_token,
        "window_tokens": window_tokens,
        "window_ms": span / 1e6,
        "gpu_busy_ms": busy / 1e6,
        "gpu_idle_ms": (span - busy) / 1e6,
        "gpu_busy_pct": busy / span * 100 if span else None,
        "ms_per_token_in_window": span / 1e6 / window_tokens if window_tokens else None,
        "top_by_time": sorted(by_name_time.items(), key=lambda kv: -kv[1])[:12],
        "top_by_count": sorted(by_name_count.items(), key=lambda kv: -kv[1])[:12],
        "kernel_time_total_ms": sum(by_name_time.values()) / 1e6,
        "distinct_kernels": len(by_name_time),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Этап 6 — атрибуция по трассе nsys")
    ap.add_argument("profile_dir", type=Path)
    ap.add_argument("--trace", default="node",
                    help="какой профиль разбирать: node (счёт ядер) или graph")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    use_utf8_output()
    a = analyse(args.profile_dir, args.trace)

    print(f"графовых запусков : {a['graph_launches']}  "
          f"(= токенов, при активных графах один граф на токен)")
    print(f"CUDA-графы        : {'ЗАХВАТЫВАЮТСЯ' if a['cuda_graphs_used'] else 'НЕ используются'}")
    print(f"ядер всего        : {a['kernels_total']:,} "
          f"({a['distinct_kernels']} различных)")
    print(f"  из них поштучно : {a['direct_kernel_launches']:,} (загрузка, прогрев)")
    print(f"  внутри графов   : {a['kernels_in_graphs']:,}")
    if a["kernels_per_token"]:
        print(f"ЯДЕР НА ТОКЕН     : {a['kernels_per_token']:.0f}")
    print()
    print(f"окно замера       : {a['window_tokens']} токенов, {a['window_ms']:.1f} мс")
    if a["ms_per_token_in_window"]:
        print(f"  мс на токен     : {a['ms_per_token_in_window']:.3f}")
    print(f"GPU под нагрузкой : {a['gpu_busy_ms']:.1f} мс ({a['gpu_busy_pct']:.1f}%)")
    print(f"GPU простаивает   : {a['gpu_idle_ms']:.1f} мс ({100 - a['gpu_busy_pct']:.1f}%)")

    print("\nтоп ядер по суммарному времени:")
    for name, ns in a["top_by_time"][:8]:
        print(f"  {ns/1e6:>8.1f} мс  {name}")
    print("\nтоп ядер по числу запусков:")
    for name, cnt in a["top_by_count"][:8]:
        per = cnt / a["graph_launches"] if a["graph_launches"] else 0
        print(f"  {cnt:>8,}  ({per:>6.1f} на токен)  {name}")

    if args.out:
        args.out.write_text(json.dumps(a, indent=2, ensure_ascii=False),
                            encoding="utf-8")
        print(f"\nзаписано: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
