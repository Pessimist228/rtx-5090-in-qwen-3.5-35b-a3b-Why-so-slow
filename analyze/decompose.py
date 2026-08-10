"""Этап 5 — разложение времени на токен.

    t_измеренное = байты_на_токен / полоса_измеренная + t_остаток

Байты берутся не из размера файла, а из того, что реально читается на токен:

- веса: буфер CUDA0, который llama.cpp выделила под них. Это не размер файла:
  token_embd.weight остаётся на CPU, потому что из него читается одна строка,
  а не вся матрица. Для Qwen3.5-9B Q4_0 разница 545.62 MiB, то есть 10.7%.
- состояние GDN: читается И пишется каждый токен, квантованием не режется,
  поэтому идёт отдельной строкой и с множителем два.
- KV-кэш: только на слоях внимания. У Qwen3.5 это каждый четвёртый слой,
  остальные — Gated DeltaNet без KV. Наивный расчёт по всем слоям завысил бы
  вклад вчетверо.

    python analyze/decompose.py results/<прогон>/
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.env import use_utf8_output  # noqa: E402
from analyze.bytes_per_token import (  # noqa: E402
    bytes_per_token as bpt_compute, read_model,
)

MIB = 1024 ** 2
GB = 1_000_000_000

# Байт на элемент KV-кэша. У квантованных типов блок несёт ещё и множитель:
# q8_0 это 32 значения по байту плюс fp16 на блок, то есть 34 байта на 32.
KV_BYTES_PER_ELEM = {
    "f16": 2.0, "bf16": 2.0, "f32": 4.0,
    "q8_0": 34 / 32, "q5_1": 24 / 32, "q5_0": 22 / 32, "q4_1": 20 / 32,
    "q4_0": 18 / 32,
}


def load_run(run_dir: Path) -> tuple[list[dict], dict]:
    records = []
    with (run_dir / "bench.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                records.append(json.loads(line))
    env = json.loads((run_dir / "env.json").read_text(encoding="utf-8"))
    return records, env


def kv_geometry(model_path: Path) -> dict:
    """Геометрия KV из метаданных GGUF."""
    from gguf import GGUFReader

    r = GGUFReader(str(model_path))

    def field(name):
        f = r.fields.get(name)
        return f.contents() if f is not None else None

    arch = field("general.architecture")
    return {
        "arch": arch,
        "n_head_kv": field(f"{arch}.attention.head_count_kv"),
        "key_length": field(f"{arch}.attention.key_length"),
        "value_length": field(f"{arch}.attention.value_length"),
        "block_count": field(f"{arch}.block_count"),
    }


def bytes_per_token(rec: dict, geom: dict, model: dict) -> dict:
    """Что читается и пишется на один токен декода.

    Веса берутся разбором тензоров, а не размером буфера CUDA0. Для плотной
    модели это одно и то же, а для MoE — нет: в буфере лежат все 256 экспертов,
    а на токен читаются 8. У Qwen3.5-35B-A3B буфер равен 18.7 ГиБ, тогда как
    читается 2.0 ГБ — завышение в девять раз.

    Состояние GDN берётся из рантайма: llama.cpp печатает его точный размер, а
    аналитика недосчитывает свёрточную часть.
    """
    place = rec["placement"]
    point = rec["point"]
    elem = KV_BYTES_PER_ELEM.get(point["kv_type"], 2.0)

    avg_depth = point["n_depth"] + point["n_gen"] / 2
    calc = bpt_compute(model, avg_depth, elem)

    weights = calc["weights_read_bytes"]
    gdn = ((place.get("recurrent_state_mib") or 0.0) * MIB * 2
           or calc["gdn_state_bytes_rw"])

    # Слои внимания считаем по рантайму: он один знает, что MTP-голова кэша
    # не получает.
    n_attn = place.get("kv_layers") or calc["attention_layers"]
    per_ctx_token = n_attn * geom["n_head_kv"] * (
        geom["key_length"] + geom["value_length"]) * elem
    kv = per_ctx_token * avg_depth

    total = weights + gdn + kv
    return {
        "weights_b": weights,
        "gdn_state_b": gdn,
        "kv_cache_b": kv,
        "total_b": total,
        "kv_bytes_per_ctx_token": per_ctx_token,
        "attention_layers": n_attn,
        "avg_depth": avg_depth,
        "expert_fraction": calc["expert_fraction"],
    }


def decompose(rec: dict, geom: dict, bw_gbs: float, model: dict) -> dict:
    b = bytes_per_token(rec, geom, model)
    t_measured = rec["median_ms_per_token"]
    t_memory = b["total_b"] / (bw_gbs * GB) * 1000
    residual = t_measured - t_memory
    return {
        **b,
        "t_measured_ms": t_measured,
        "t_memory_ms": t_memory,
        "t_residual_ms": residual,
        "residual_pct": residual / t_measured * 100,
        "utilization_pct": t_memory / t_measured * 100,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Этап 5 — разложение времени")
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--model", type=Path, help="GGUF для геометрии KV")
    args = ap.parse_args()

    use_utf8_output()
    records, env = load_run(args.run_dir)
    bw = env.get("measured_bandwidth_gbs")
    if not bw:
        print("в env.json нет measured_bandwidth_gbs — сначала Этап 2", file=sys.stderr)
        return 2

    model_name = next((r["llama_bench"]["model_filename"] for r in records
                       if r["status"] == "ok"), None)
    model_path = args.model
    if model_path is None:
        from common.config import HostConfig
        model_path = HostConfig.load().find_model(model_name)
    geom = kv_geometry(model_path)
    model = read_model(model_path)

    decodes = [r for r in records
               if r["status"] == "ok" and r["point"]["kind"] == "decode"]
    decodes.sort(key=lambda r: (r["point"]["flash_attn"], r["point"]["kv_type"],
                                r["point"]["n_depth"]))

    print(f"модель  : {model_name}")
    print(f"полоса  : {bw} ГБ/с (измеренная)")
    print(f"слоёв   : {geom['block_count']} всего, "
          f"{decodes[0]['placement']['kv_layers']} с вниманием, "
          f"{decodes[0]['placement']['gdn_layers']} GDN")
    print(f"KV      : n_head_kv={geom['n_head_kv']}, "
          f"k={geom['key_length']}, v={geom['value_length']}\n")

    hdr = (f"{'fa':>2} {'kv':<5} {'глубина':>8} │ {'веса':>7} {'GDN':>6} {'KV':>8} "
           f"{'всего':>7} │ {'память':>7} {'замер':>7} {'остаток':>8} {'доля':>6} {'утил':>6}")
    print(hdr)
    print("─" * len(hdr))

    residuals = []
    for r in decodes:
        d = decompose(r, geom, bw, model)
        p = r["point"]
        residuals.append((p["flash_attn"], p["kv_type"], p["n_depth"],
                          d["t_residual_ms"]))
        print(f"{p['flash_attn']:>2} {p['kv_type']:<5} {p['n_depth']:>8} │ "
              f"{d['weights_b']/GB:>6.3f}Г {d['gdn_state_b']/1e6:>5.0f}М "
              f"{d['kv_cache_b']/1e6:>7.0f}М {d['total_b']/GB:>6.3f}Г │ "
              f"{d['t_memory_ms']:>6.2f}м {d['t_measured_ms']:>6.2f}м "
              f"{d['t_residual_ms']:>7.2f}м {d['residual_pct']:>5.1f}% "
              f"{d['utilization_pct']:>5.1f}%")

    print()
    for fa, kv in sorted({(f, k) for f, k, _, _ in residuals}):
        vals = [v for f, k, _, v in residuals if (f, k) == (fa, kv)]
        spread = max(vals) - min(vals)
        print(f"  fa={fa} kv={kv:<5}: остаток {statistics.median(vals):.2f} мс "
              f"(от {min(vals):.2f} до {max(vals):.2f}, размах {spread:.2f} мс "
              f"на четырёхкратном росте глубины)")

    out = args.run_dir / "decomposition.json"
    out.write_text(json.dumps(
        {"bandwidth_gbs": bw, "geometry": geom,
         "points": [{"point": r["point"], **decompose(r, geom, bw, model)}
                    for r in decodes]},
        indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nзаписано: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
