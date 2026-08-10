"""Этап 8 — сборка отчёта.

Собирает report.md из того, что произвели предыдущие этапы. Ничего не считает
заново: если числа расходятся, значит расходятся исходники, и это надо видеть,
а не сглаживать пересчётом.

Порядок разделов задан ТЗ: окружение, замеры, байты на токен, разложение,
атрибуция из профиля, перплексия, команды воспроизведения. Перплексия идёт
рядом со скоростью не для красоты — правило ТЗ гласит, что цифра скорости без
цифры качества не публикуется, и отчёт это правило проводит.

    python analyze/report.py results/<прогон>/
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.env import use_utf8_output  # noqa: E402

GB = 1_000_000_000


def load(path: Path):
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    out = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                out.append(json.loads(line))
    return out


def section_env(env: dict) -> list[str]:
    if not env:
        return ["_env.json отсутствует — результаты недействительны._"]
    gpu, llama = env.get("gpu", {}), env.get("llama_cpp", {})
    cc = (gpu.get("compute_capability") or "").replace(".", "")
    rows = [
        ("хост", f"{env.get('host_id')} (фаза {env.get('phase')})"),
        ("снято", env.get("collected_at", "")[:19].replace("T", " ") + " UTC"),
        ("ОС", env.get("os", {}).get("platform")),
        ("CPU", f"{env.get('cpu', {}).get('model')} "
                f"({env.get('cpu', {}).get('logical_cores')} потоков)"),
        ("RAM", f"{env.get('memory', {}).get('total_gib')} GiB"),
        ("GPU", f"{gpu.get('name')} (sm_{cc})"),
        ("VRAM", f"{gpu.get('vram_total_mib'):.0f} MiB "
                 f"{env.get('config_declared', {}).get('vram_type', '')}"
                 if gpu.get("vram_total_mib") else None),
        ("драйвер", f"{gpu.get('driver_version')} (max CUDA "
                    f"{gpu.get('driver_max_cuda_version')})"),
        ("лимит мощности", f"{gpu.get('power_w', {}).get('limit')} Вт"),
        ("llama.cpp", f"b{llama.get('build_number')} ({llama.get('commit')}), "
                      f"{llama.get('compiler')}"),
        ("бэкенды", ", ".join(llama.get("backends_loaded") or [])),
        ("полоса измеренная", f"**{env.get('measured_bandwidth_gbs')} ГБ/с**"),
        ("полоса паспортная",
         f"{env.get('config_declared', {}).get('spec_bandwidth_gbs')} ГБ/с "
         f"({(env.get('bandwidth') or {}).get('efficiency_pct')}% достигнуто)"),
        ("python", env.get("python", {}).get("version")),
        ("харнесс", (env.get("harness") or {}).get("commit")),
    ]
    lines = ["| | |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in rows if v not in (None, "", "None")]
    return lines


def section_bench(records: list[dict], noise: float) -> list[str]:
    ok = [r for r in records if r.get("status") == "ok"]
    if not ok:
        return ["_замеров нет._"]

    lines = []
    for kind, title in (("decode", "Декод"), ("prefill", "Префилл")):
        rows = [r for r in ok if r["point"]["kind"] == kind]
        if not rows:
            continue
        rows.sort(key=lambda r: (r["point"]["flash_attn"], r["point"]["kv_type"],
                                 r["point"]["n_depth"], r["point"]["n_prompt"]))
        lines += [f"\n### {title}", "",
                  "| fa | KV | точка | медиана, т/с | мс/токен | разброс | SM, МГц | флаги |",
                  "|---|---|---|---|---|---|---|---|"]
        for r in rows:
            p = r["point"]
            point = (f"p{p['n_prompt']}" if kind == "prefill"
                     else f"n{p['n_gen']}@d{p['n_depth']}")
            sm = (r.get("clocks_under_load") or {}).get("sm_mhz_median")
            flags = []
            if r.get("noisy"):
                flags.append(f"шум {r['spread_pct']}%")
            if r.get("throttled"):
                flags.append("троттлинг")
            if r.get("variant", "").startswith("anchor"):
                flags.append(r["variant"])
            lines.append(
                f"| {p['flash_attn']} | {p['kv_type']} | {point} | "
                f"{r['median_ts']:.2f} | {r['median_ms_per_token']:.3f} | "
                f"{r['spread_pct']:.2f}% | {sm:.0f} | {', '.join(flags) or '—'} |"
                if sm else
                f"| {p['flash_attn']} | {p['kv_type']} | {point} | "
                f"{r['median_ts']:.2f} | {r['median_ms_per_token']:.3f} | "
                f"{r['spread_pct']:.2f}% | — | {', '.join(flags) or '—'} |")

    noisy = [r for r in ok if r.get("noisy")]
    failed = [r for r in records if r.get("status") != "ok"]
    lines += ["", f"Всего точек: {len(records)}, успешных {len(ok)}, "
                  f"с ошибкой {len(failed)}, шумных {len(noisy)} "
                  f"(порог разброса {noise}%)."]
    if failed:
        lines += ["", "Не выполнились:", ""]
        seen = set()
        for r in failed:
            msg = (r.get("error") or "")[:120]
            if msg in seen:
                continue
            seen.add(msg)
            lines.append(f"- `{msg}`")
    return lines


def section_bytes(bpt: dict) -> list[str]:
    if not bpt:
        return ["_bytes_per_token не считался._"]
    lines = ["| категория | тензоров | в файле, ГБ | читается, ГБ | доля | примечание |",
             "|---|---|---|---|---|---|"]
    total = bpt.get("total_bytes", 1)
    for r in sorted(bpt.get("rows", []), key=lambda x: -x["read_bytes"]):
        lines.append(f"| {r['category']} | {r['tensors']} | "
                     f"{r['stored_bytes']/GB:.3f} | {r['read_bytes']/GB:.3f} | "
                     f"{r['read_bytes']/total*100:.1f}% | {r.get('note', '')} |")
    lines += [
        f"| **веса итого** | | | **{bpt['weights_read_bytes']/GB:.3f}** | "
        f"{bpt['weights_read_bytes']/total*100:.1f}% | |",
        f"| состояние GDN | | | {bpt['gdn_state_bytes_rw']/GB:.3f} | "
        f"{bpt['gdn_state_bytes_rw']/total*100:.1f}% | чтение+запись, не квантуется |",
        f"| KV-кэш | | | {bpt['kv_bytes']/GB:.3f} | "
        f"{bpt['kv_bytes']/total*100:.1f}% | {bpt['kv_bytes_per_ctx_token']:.0f} Б "
        f"на токен контекста |",
        f"| **ВСЕГО** | | | **{total/GB:.3f}** | | |",
    ]
    if bpt.get("expert_fraction", 1) < 1:
        lines += ["", f"Доля читаемых экспертов: {bpt['expert_fraction']:.4f}. "
                      f"Слоёв внимания {bpt['attention_layers']}, GDN {bpt['gdn_layers']}."]
    return lines


def section_decomposition(dec: dict) -> list[str]:
    if not dec:
        return ["_разложение не считалось._"]
    lines = [f"Полоса: **{dec['bandwidth_gbs']} ГБ/с** (измеренная).", "",
             "| fa | KV | глубина | байт/токен, ГБ | память, мс | замер, мс | "
             "остаток, мс | доля | утилизация |", "|---|---|---|---|---|---|---|---|---|"]
    for p in dec.get("points", []):
        pt = p["point"]
        lines.append(
            f"| {pt['flash_attn']} | {pt['kv_type']} | {pt['n_depth']} | "
            f"{p['total_b']/GB:.3f} | {p['t_memory_ms']:.2f} | "
            f"{p['t_measured_ms']:.2f} | **{p['t_residual_ms']:.2f}** | "
            f"{p['residual_pct']:.1f}% | {p['utilization_pct']:.1f}% |")

    groups: dict[tuple, list[float]] = {}
    for p in dec.get("points", []):
        key = (p["point"]["flash_attn"], p["point"]["kv_type"])
        groups.setdefault(key, []).append(p["t_residual_ms"])
    lines += ["", "Постоянство остатка по глубине контекста:", ""]
    for (fa, kv), vals in sorted(groups.items()):
        lines.append(f"- `fa={fa} kv={kv}`: медиана **{statistics.median(vals):.2f} мс**, "
                     f"от {min(vals):.2f} до {max(vals):.2f}, "
                     f"размах {max(vals) - min(vals):.2f} мс")
    return lines


def section_profile(attr: dict, util: dict | None) -> list[str]:
    if not attr:
        return ["_профиль не снимался._"]
    lines = [
        f"- CUDA-графы: **{'захватываются' if attr['cuda_graphs_used'] else 'НЕ используются'}**, "
        f"{attr['graph_launches']} запусков на столько же токенов",
        f"- ядер на токен: **{attr['kernels_per_token']:.0f}**" if attr.get("kernels_per_token")
        else "- ядер на токен: не определено",
        f"- различных ядер: {attr['distinct_kernels']}",
    ]
    if util:
        lines.append(f"- занятость SM в чистом прогоне (без профилировщика): "
                     f"**{util['sm_median_pct']}%**")
    lines += ["", "Топ ядер по числу запусков:", "",
              "| запусков | на токен | ядро |", "|---|---|---|"]
    for name, cnt in attr.get("top_by_count", [])[:8]:
        per = cnt / attr["graph_launches"] if attr.get("graph_launches") else 0
        lines.append(f"| {cnt:,} | {per:.1f} | `{name}` |")
    return lines


def section_quality(ppls: list[dict]) -> list[str]:
    if not ppls:
        return ["_перплексия не мерялась. По правилу ТЗ цифры скорости из этого "
                "отчёта публиковать нельзя._"]
    lines = ["| модель | перплексия | ± | контекст | фрагментов | датасет |",
             "|---|---|---|---|---|---|"]
    for p in ppls:
        lines.append(f"| {p['model']} | **{p['perplexity']:.4f}** | {p['stderr']:.4f} | "
                     f"{p['context']} | {p['chunks']} | {p['dataset']} |")
    return lines


def build(run_dir: Path, extra: dict) -> str:
    env = load(run_dir / "env.json")
    records = load_jsonl(run_dir / "bench.jsonl")
    dec = load(run_dir / "decomposition.json")
    anchor = load(run_dir / "anchor.json")
    bpt = extra.get("bpt")
    attr = extra.get("attr")
    util = extra.get("util")
    ppls = extra.get("ppl", [])
    noise = 3.0
    cmds = sorted({" ".join(r["command"]) for r in records if r.get("command")})

    out = [f"# Отчёт: {run_dir.name}", ""]

    out += ["## Окружение", ""] + section_env(env) + [""]

    if anchor:
        verdict = ("самосогласована" if anchor["self_consistent"]
                   else "**НЕ самосогласована**")
        out += [f"Якорная точка: {anchor['anchor_first_ts']:.2f} → "
                f"{anchor['anchor_last_ts']:.2f} т/с ({anchor['drift_pct']:+.2f}%), "
                f"матрица {verdict} при пороге {anchor['limit_pct']}%.", ""]

    out += ["## Замеры"] + section_bench(records, noise) + [""]
    out += ["## Байты на токен", ""] + section_bytes(bpt) + [""]
    out += ["## Разложение времени", ""] + section_decomposition(dec) + [""]
    out += ["## Атрибуция из профиля", ""] + section_profile(attr, util) + [""]
    out += ["## Качество", ""] + section_quality(ppls) + [""]

    out += ["## Воспроизведение", "",
            "Команды llama-bench, породившие таблицу замеров:", "", "```"]
    out += cmds[:6]
    if len(cmds) > 6:
        out.append(f"... и ещё {len(cmds) - 6} вариаций по fa/kv/глубине")
    out += ["```", ""]
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description="Этап 8 — сборка отчёта")
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--bytes-json", type=Path, help="вывод bytes_per_token.py")
    ap.add_argument("--attribution-json", type=Path, help="вывод attribute.py")
    ap.add_argument("--util-json", type=Path, help="занятость SM из чистого прогона")
    ap.add_argument("--ppl-json", type=Path, nargs="*", default=[],
                    help="выводы perplexity.sh")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    use_utf8_output()
    extra = {
        "bpt": load(args.bytes_json) if args.bytes_json else None,
        "attr": load(args.attribution_json) if args.attribution_json else None,
        "util": load(args.util_json) if args.util_json else None,
        "ppl": [p for p in (load(x) for x in args.ppl_json) if p],
    }
    text = build(args.run_dir, extra)
    out = args.out or (args.run_dir / "report.md")
    out.write_text(text, encoding="utf-8")
    print(text[:1500])
    print(f"\n... записано полностью: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
