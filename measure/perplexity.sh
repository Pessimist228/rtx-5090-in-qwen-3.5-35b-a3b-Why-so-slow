#!/usr/bin/env bash
# Этап 7 — качество как гейт.
#
# ТЗ: «цифра скорости без цифры качества не публикуется». Смысл в том, что
# любой квант можно сделать быстрее, ухудшив его, и без перплексии рядом
# скорость ничего не значит.
#
# Датасет один и тот же для всех квантов, иначе числа несравнимы. Контекст
# тоже фиксирован: перплексия сильно от него зависит, и сравнивать замеры с
# разным -c бессмысленно.
#
#   ./perplexity.sh --model model.gguf --data wiki.test.raw --bin .../llama-perplexity
#
# Прогон долгий. На 35B по wikitext-2 при -c 4096 это порядка десяти минут,
# поэтому по умолчанию берётся ограниченное число фрагментов: для гейта важна
# сопоставимость между квантами, а не третий знак.

set -euo pipefail

MODEL=""
DATA=""
BIN=""
OUT=""
CTX=4096
CHUNKS=64
NGL=99
FA=1
# Тип KV-кэша влияет и на скорость, и на качество, поэтому он часть замера, а
# не умолчание: сравнивать перплексию при разном -ctk/-ctv между собой можно,
# а вот публиковать скорость на q8_0 рядом с качеством на f16 нельзя.
KV="f16"

usage() { sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)  MODEL="$2"; shift 2 ;;
        --data)   DATA="$2";  shift 2 ;;
        --bin)    BIN="$2";   shift 2 ;;
        --out)    OUT="$2";   shift 2 ;;
        --ctx)    CTX="$2";   shift 2 ;;
        --chunks) CHUNKS="$2"; shift 2 ;;
        --ngl)    NGL="$2";   shift 2 ;;
        --fa)     FA="$2";    shift 2 ;;
        --kv)     KV="$2";    shift 2 ;;
        -h|--help) usage 0 ;;
        *) echo "неизвестный аргумент: $1" >&2; usage 1 ;;
    esac
done

[[ -n "$MODEL" ]] || { echo "ошибка: нужен --model" >&2; exit 1; }
[[ -f "$MODEL" ]] || { echo "ошибка: модель не найдена: $MODEL" >&2; exit 1; }
[[ -n "$DATA"  ]] || { echo "ошибка: нужен --data (одинаковый для всех квантов)" >&2; exit 1; }
[[ -f "$DATA"  ]] || { echo "ошибка: датасет не найден: $DATA" >&2; exit 1; }
[[ -n "$BIN"   ]] || { echo "ошибка: нужен --bin (llama-perplexity)" >&2; exit 1; }
[[ -x "$BIN"   ]] || { echo "ошибка: бинарь не исполняем: $BIN" >&2; exit 1; }

NAME="$(basename "$MODEL" .gguf)"
[[ -n "$OUT" ]] || OUT="./ppl_${NAME}.json"
LOG="${OUT%.json}.log"

echo "модель   : $NAME"
echo "датасет  : $(basename "$DATA")"
echo "контекст : $CTX, фрагментов: $CHUNKS, KV: $KV, fa: $FA"
echo

"$BIN" -m "$MODEL" -f "$DATA" -c "$CTX" --chunks "$CHUNKS" \
       -ngl "$NGL" -fa "$FA" -ctk "$KV" -ctv "$KV" -sm none > "$LOG" 2>&1 || {
    echo "llama-perplexity завершился с ошибкой, см. $LOG" >&2
    tail -15 "$LOG" >&2
    exit 1
}

# Итоговая строка выглядит как «Final estimate: PPL = 7.1234 +/- 0.04567».
PPL=$(grep -oE "Final estimate: PPL = [0-9.]+" "$LOG" | tail -1 | grep -oE "[0-9.]+$" || true)
ERR=$(grep -oE "PPL = [0-9.]+ \+/- [0-9.]+" "$LOG" | tail -1 | grep -oE "[0-9.]+$" || true)

if [[ -z "$PPL" ]]; then
    echo "не удалось разобрать перплексию из $LOG" >&2
    tail -5 "$LOG" >&2
    exit 1
fi

python3 - "$OUT" "$NAME" "$PPL" "${ERR:-0}" "$CTX" "$CHUNKS" "$(basename "$DATA")" "$KV" "$FA" <<'PY'
import json, sys
out, name, ppl, err, ctx, chunks, data, kv, fa = sys.argv[1:10]
json.dump({"model": name, "perplexity": float(ppl), "stderr": float(err),
           "context": int(ctx), "chunks": int(chunks), "dataset": data,
           "kv_type": kv, "flash_attn": int(fa)},
          open(out, "w"), indent=2, ensure_ascii=False)
PY

echo "перплексия: $PPL +/- ${ERR:-?}"
echo "записано  : $OUT"
