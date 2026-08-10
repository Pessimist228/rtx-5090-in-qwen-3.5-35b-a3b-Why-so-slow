"""Этап 4 — байты на токен, аналитически из метаданных GGUF.

Фундамент всех дальнейших выводов, поэтому считается по фактическим типам
тензоров, а не по размеру файла, и раскладывается по категориям так, чтобы
каждую строку можно было проверить на бумаге.

Что здесь принципиально и почему нельзя брать размер файла.

**Эмбеддинг не читается.** `token_embd.weight` — это одна строка на токен,
а не вся матрица. llama.cpp это подтверждает делом: при `-ngl 99` он оставляет
эмбеддинг на CPU. У Qwen3.5-9B Q4_0 это 545.62 MiB, то есть 10.7% файла.

**Голова читается целиком.** `output.weight` даёт логиты по всему словарю
каждый токен. У Qwen3.5 эмбеддинги не связаны, так что это отдельный тензор.

**У MoE читается не всё.** Тензоры экспертов лежат в файле целиком, но на
токен активируются только `expert_used_count` из `expert_count`. Для
Qwen3.5-35B-A3B это 8 из 256 — разница в 32 раза, и списать её нельзя.
Общий эксперт читается всегда и идёт отдельной строкой.

**Состояние GDN — не веса.** Оно читается И пишется каждый токен, не
уменьшается квантованием и не лежит в файле как тензор. Считается из
геометрии и выносится отдельной строкой, как требует ТЗ.

    python analyze/bytes_per_token.py /workspace/models/model.gguf --depth 4096
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.env import use_utf8_output  # noqa: E402

GB = 1_000_000_000

# Категории по имени тензора. Порядок важен: первое совпадение выигрывает,
# потому что ffn_gate_exps должен попасть в экспертов, а не в плотный FFN.
CATEGORIES = [
    ("embedding",      r"token_embd"),
    ("output_head",    r"^output\.weight$|^output_norm"),
    ("moe_router",     r"ffn_gate_inp"),
    ("moe_shared",     r"_shexp"),
    ("moe_routed",     r"_exps"),
    ("attention",      r"attn_"),
    ("gdn_ssm",        r"ssm_|time_mix|linear_attn|\.conv"),
    ("ffn_dense",      r"ffn_(gate|up|down)"),
    ("norms",          r"_norm"),
]


def classify(name: str) -> str:
    for label, pattern in CATEGORIES:
        if re.search(pattern, name):
            return label
    return "other"


def read_model(path: Path) -> dict:
    from gguf import GGUFReader
    from gguf.constants import GGML_QUANT_SIZES

    r = GGUFReader(str(path))

    def field(name):
        f = r.fields.get(name)
        if f is None:
            return None
        try:
            return f.contents()
        except Exception:
            return None

    arch = field("general.architecture")
    meta = {
        "arch": arch,
        "block_count": field(f"{arch}.block_count"),
        "embedding_length": field(f"{arch}.embedding_length"),
        "head_count": field(f"{arch}.attention.head_count"),
        "head_count_kv": field(f"{arch}.attention.head_count_kv"),
        "key_length": field(f"{arch}.attention.key_length"),
        "value_length": field(f"{arch}.attention.value_length"),
        "expert_count": field(f"{arch}.expert_count"),
        "expert_used_count": field(f"{arch}.expert_used_count"),
        "expert_shared_count": field(f"{arch}.expert_shared_count"),
        "ssm_state_size": field(f"{arch}.ssm.state_size"),
        "ssm_conv_kernel": field(f"{arch}.ssm.conv_kernel"),
        "ssm_inner_size": field(f"{arch}.ssm.inner_size"),
        "ssm_time_step_rank": field(f"{arch}.ssm.time_step_rank"),
        "nextn_predict_layers": field(f"{arch}.nextn_predict_layers"),
    }

    # Хвостовые блоки MTP-головы при обычном декоде не выполняются: они нужны
    # только спекулятивному декодированию. Веса у них полновесные (внимание,
    # эксперты, общий эксперт), поэтому не исключить их — завысить байты на
    # токен. Подтверждается рантаймом: llama.cpp заводит KV-кэш на 10 слоёв,
    # а слоёв с attn_k в файле 11 — одиннадцатый и есть MTP-голова.
    n_nextn = meta.get("nextn_predict_layers") or 0
    n_blocks = meta.get("block_count") or 0
    mtp_blocks = set(range(n_blocks - n_nextn, n_blocks)) if n_nextn else set()

    tensors = []
    for t in r.tensors:
        block, tsize = GGML_QUANT_SIZES[t.tensor_type]
        m = re.match(r"blk\.(\d+)\.", t.name)
        is_mtp = bool(m) and int(m.group(1)) in mtp_blocks
        tensors.append({
            "name": t.name,
            "type": t.tensor_type.name,
            "elements": int(t.n_elements),
            "bytes": int(t.n_elements) // block * tsize,
            "shape": [int(x) for x in t.shape],
            "category": "mtp_head" if is_mtp else classify(t.name),
        })
    return {"meta": meta, "tensors": tensors, "file_bytes": path.stat().st_size,
            "mtp_blocks": sorted(mtp_blocks)}


def bytes_per_token(model: dict, depth: int, kv_bytes_per_elem: float = 2.0) -> dict:
    meta = model["meta"]
    by_cat: dict[str, dict] = defaultdict(lambda: {"bytes": 0, "count": 0})
    for t in model["tensors"]:
        c = by_cat[t["category"]]
        c["bytes"] += t["bytes"]
        c["count"] += 1

    n_exp = meta.get("expert_count") or 0
    n_used = meta.get("expert_used_count") or 0
    # Доля экспертов, реально читаемая на токен. У плотной модели экспертов нет
    # и множитель не применяется.
    expert_fraction = (n_used / n_exp) if (n_exp and n_used) else 1.0

    rows = []
    total = 0
    for cat in ("attention", "gdn_ssm", "ffn_dense", "moe_shared", "moe_routed",
                "moe_router", "norms", "output_head", "other"):
        stored = by_cat[cat]["bytes"]
        if not stored:
            continue
        if cat == "moe_routed":
            read = stored * expert_fraction
            note = f"{n_used} из {n_exp} экспертов ({expert_fraction:.4f})"
        else:
            read = stored
            note = ""
        total += read
        rows.append({"category": cat, "tensors": by_cat[cat]["count"],
                     "stored_bytes": stored, "read_bytes": read, "note": note})

    # Из эмбеддинга на токен читается одна строка длиной в hidden_size —
    # единицы килобайт против сотен мегабайт хранения. На фоне остального это
    # ноль, но строку в таблице оставляем, чтобы разница с размером файла
    # сходилась и её не пришлось искать.
    emb = by_cat["embedding"]["bytes"]
    hidden = meta.get("embedding_length") or 0
    emb_total_elems = sum(t["elements"] for t in model["tensors"]
                          if t["category"] == "embedding")
    emb_row = (emb / (emb_total_elems / hidden)) if (hidden and emb_total_elems) else 0
    rows.append({"category": "embedding", "tensors": by_cat["embedding"]["count"],
                 "stored_bytes": emb, "read_bytes": emb_row,
                 "note": f"одна строка на токен, ~{emb_row/1024:.1f} КиБ"})
    total += emb_row

    if by_cat["mtp_head"]["bytes"]:
        rows.append({"category": "mtp_head", "tensors": by_cat["mtp_head"]["count"],
                     "stored_bytes": by_cat["mtp_head"]["bytes"], "read_bytes": 0,
                     "note": f"блоки {model['mtp_blocks']}, при обычном декоде не "
                             f"выполняются — исключены"})

    # Состояние GDN: у слоёв без KV живёт рекуррентное состояние, оно
    # читается и записывается каждый токен и квантованием не уменьшается.
    n_layers = meta.get("block_count") or 0
    n_attn = _count_attention_layers(model)
    n_gdn = n_layers - n_attn
    state_bytes = 0
    if meta.get("ssm_state_size") and meta.get("ssm_inner_size"):
        # r+s состояние на слой, f32
        state_bytes = n_gdn * meta["ssm_inner_size"] * meta["ssm_state_size"] * 4

    # KV-кэш только на слоях внимания.
    kv_per_ctx = (n_attn * (meta.get("head_count_kv") or 0)
                  * ((meta.get("key_length") or 0) + (meta.get("value_length") or 0))
                  * kv_bytes_per_elem)
    kv_bytes = kv_per_ctx * depth

    return {
        "rows": rows,
        "weights_read_bytes": total,
        "gdn_state_bytes_rw": state_bytes * 2,
        "gdn_layers": n_gdn,
        "attention_layers": n_attn,
        "kv_bytes_per_ctx_token": kv_per_ctx,
        "kv_bytes": kv_bytes,
        "depth": depth,
        "total_bytes": total + state_bytes * 2 + kv_bytes,
        "expert_fraction": expert_fraction,
    }


def _count_attention_layers(model: dict) -> int:
    """Сколько слоёв несут внимание: у гибридов это не все слои."""
    layers = set()
    for t in model["tensors"]:
        m = re.match(r"blk\.(\d+)\.attn_(k|v)\.weight", t["name"])
        if m:
            layers.add(int(m.group(1)))
    return len(layers)


def main() -> int:
    ap = argparse.ArgumentParser(description="Этап 4 — байты на токен из GGUF")
    ap.add_argument("model", type=Path)
    ap.add_argument("--depth", type=int, default=0, help="глубина контекста")
    ap.add_argument("--kv-type", default="f16", help="тип KV-кэша")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    use_utf8_output()
    kv_elem = {"f16": 2.0, "bf16": 2.0, "f32": 4.0,
               "q8_0": 34 / 32, "q5_1": 24 / 32, "q4_0": 18 / 32}.get(args.kv_type, 2.0)

    model = read_model(args.model)
    res = bytes_per_token(model, args.depth, kv_elem)
    meta = model["meta"]

    print(f"модель   : {args.model.name}")
    print(f"arch     : {meta['arch']}, слоёв {meta['block_count']} "
          f"({res['attention_layers']} внимания, {res['gdn_layers']} GDN)")
    if meta.get("expert_count"):
        print(f"MoE      : {meta['expert_used_count']} из {meta['expert_count']} "
              f"экспертов на токен, общих {meta.get('expert_shared_count')}")
    print(f"файл     : {model['file_bytes'] / GB:.3f} ГБ")
    print(f"глубина  : {args.depth} токенов, KV {args.kv_type}\n")

    print(f"{'категория':<14} {'тнз':>4} {'в файле':>10} {'читается':>10} {'доля':>7}  примечание")
    print("─" * 88)
    for r in sorted(res["rows"], key=lambda x: -x["read_bytes"]):
        pct = r["read_bytes"] / res["total_bytes"] * 100
        print(f"{r['category']:<14} {r['tensors']:>4} "
              f"{r['stored_bytes']/GB:>9.3f}Г {r['read_bytes']/GB:>9.3f}Г "
              f"{pct:>6.1f}%  {r['note']}")
    print("─" * 88)
    print(f"{'веса итого':<14} {'':>4} {'':>10} {res['weights_read_bytes']/GB:>9.3f}Г "
          f"{res['weights_read_bytes']/res['total_bytes']*100:>6.1f}%")
    print(f"{'состояние GDN':<14} {'':>4} {'':>10} {res['gdn_state_bytes_rw']/GB:>9.3f}Г "
          f"{res['gdn_state_bytes_rw']/res['total_bytes']*100:>6.1f}%  чтение+запись, не квантуется")
    print(f"{'KV-кэш':<14} {'':>4} {'':>10} {res['kv_bytes']/GB:>9.3f}Г "
          f"{res['kv_bytes']/res['total_bytes']*100:>6.1f}%  "
          f"{res['kv_bytes_per_ctx_token']:.0f} Б на токен контекста")
    print("═" * 88)
    print(f"{'ВСЕГО':<14} {'':>4} {'':>10} {res['total_bytes']/GB:>9.3f}Г")

    if args.out:
        args.out.write_text(json.dumps({"meta": meta, **res}, indent=2,
                                       ensure_ascii=False), encoding="utf-8")
        print(f"\nзаписано: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
