#!/usr/bin/env bash
# Критерий приёмки 1 — весь цикл одной командой, без ручного вмешательства.
#
#   ./run_all.sh --model Qwen_Qwen3.5-35B-A3B-Q4_0.gguf
#   ./run_all.sh --model model.gguf --quick          # проверка обвязки, ~10 мин
#   ./run_all.sh --model model.gguf --data wiki.test.raw
#
# Между машинами меняется только config/<host>.json, код одинаковый — это
# критерий приёмки 8. Хост определяется по имени GPU, поэтому команда на ноуте
# и на арендованной карте буквально одна и та же.
#
# Этапы, которых не на чем выполнить, пропускаются с пометкой, а не роняют
# прогон: на ноуте может не быть датасета для перплексии, на поде — nsys.
# Но Этап 0 идёт со --strict: начинать замеры на машине, где уже что-то не
# так, нельзя.

set -uo pipefail

HOST=""
MODEL=""
DATA=""
QUICK=0
GRAPHS_OFF="--graphs-off"
DO_PROFILE=1
DO_PPL=1
PPL_CHUNKS=48
PPL_CTX=4096

usage() { sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --host)   HOST="$2"; shift 2 ;;
        --model)  MODEL="$2"; shift 2 ;;
        --data)   DATA="$2"; shift 2 ;;
        --quick)  QUICK=1; GRAPHS_OFF=""; shift ;;
        --no-graphs-off) GRAPHS_OFF=""; shift ;;
        --skip-profile)  DO_PROFILE=0; shift ;;
        --skip-ppl)      DO_PPL=0; shift ;;
        --ppl-chunks)    PPL_CHUNKS="$2"; shift 2 ;;
        -h|--help) usage 0 ;;
        *) echo "неизвестный аргумент: $1" >&2; usage 1 ;;
    esac
done

[[ -n "$MODEL" ]] || { echo "ошибка: нужен --model" >&2; exit 1; }

cd "$(dirname "$0")"
HOST_ARG=(); [[ -n "$HOST" ]] && HOST_ARG=(--host "$HOST")

SKIPPED=()
FAILED=()
step() { echo; echo "==================== $* ===================="; }
skip() { echo "  ПРОПУЩЕН: $*"; SKIPPED+=("$*"); }
fail() { echo "  ОШИБКА: $*" >&2; FAILED+=("$*"); }

# --- Этап 0 ---------------------------------------------------------------
step "Этап 0 — окружение"
if ! python3 -m common.env "${HOST_ARG[@]}" --strict; then
    echo "окружение не прошло проверку — дальше идти нельзя" >&2
    exit 2
fi

RESULTS="$(python3 - "${HOST:-}" <<'PY'
import sys
sys.path.insert(0, ".")
from common.config import HostConfig
print(HostConfig.load(sys.argv[1] or None).results_dir)
PY
)"
[[ -n "$RESULTS" ]] || { echo "не удалось определить каталог результатов" >&2; exit 2; }
mkdir -p "$RESULTS"
echo "каталог результатов: $RESULTS"

# Путь к модели ищется теми же правилами, что у остальных этапов.
MODEL_PATH="$(python3 - "${HOST:-}" "$MODEL" <<'PY'
import sys
sys.path.insert(0, ".")
from common.config import HostConfig
try:
    print(HostConfig.load(sys.argv[1] or None).find_model(sys.argv[2]))
except Exception:
    print(sys.argv[2])
PY
)"
echo "модель             : $MODEL_PATH"
[[ -f "$MODEL_PATH" ]] || { echo "модель не найдена: $MODEL_PATH" >&2; exit 2; }

# --- Этап 2 ---------------------------------------------------------------
step "Этап 2 — полоса памяти"
python3 measure/bandwidth.py "${HOST_ARG[@]}" --validate || fail "bandwidth.py"

# --- Этап 3 ---------------------------------------------------------------
step "Этап 3 — матрица llama-bench"
RUN_DIR="${RESULTS}/run_$(date -u +%Y%m%dT%H%M%SZ)"
QUICK_ARG=(); [[ $QUICK == 1 ]] && QUICK_ARG=(--quick)
GO_ARG=();    [[ -n "$GRAPHS_OFF" ]] && GO_ARG=("$GRAPHS_OFF")
if ! python3 measure/bench.py "${HOST_ARG[@]}" --model "$MODEL" \
        --out-dir "$RUN_DIR" "${QUICK_ARG[@]}" "${GO_ARG[@]}"; then
    fail "bench.py — без матрицы отчёт бессмысленен"
    exit 3
fi

# --- Этап 4 ---------------------------------------------------------------
step "Этап 4 — байты на токен"
BPT="${RUN_DIR}/bytes_per_token.json"
python3 analyze/bytes_per_token.py "$MODEL_PATH" --depth 0 --kv-type f16 --out "$BPT" \
    || { fail "bytes_per_token.py"; BPT=""; }

# --- Этап 5 ---------------------------------------------------------------
step "Этап 5 — разложение времени"
python3 analyze/decompose.py "$RUN_DIR" --model "$MODEL_PATH" || fail "decompose.py"

# --- Этап 6 ---------------------------------------------------------------
step "Этап 6 — профиль и атрибуция"
ATTR=""
if [[ $DO_PROFILE == 0 ]]; then
    skip "профиль отключён ключом --skip-profile"
elif ! command -v nsys >/dev/null 2>&1; then
    skip "nsys не найден в PATH — Этап 6 не выполняется"
else
    BENCH_BIN="$(python3 - "${HOST:-}" <<'PY'
import sys
sys.path.insert(0, ".")
from common.config import HostConfig
print(HostConfig.load(sys.argv[1] or None).exe("llama-bench"))
PY
)"
    PROF_DIR="${RUN_DIR}/profile"
    if ./measure/profile.sh --model "$MODEL_PATH" --bin "$BENCH_BIN" \
            --out "$PROF_DIR" --n-gen 128 --depth 0 --fa 1 --kv f16 \
            --graph-trace node; then
        ATTR="${RUN_DIR}/attribution.json"
        python3 analyze/attribute.py "$PROF_DIR" --trace node --out "$ATTR" \
            || { fail "attribute.py"; ATTR=""; }
    else
        fail "profile.sh"
    fi
fi

# --- Этап 7 ---------------------------------------------------------------
step "Этап 7 — перплексия (гейт качества)"
PPL_JSONS=()
if [[ $DO_PPL == 0 ]]; then
    skip "перплексия отключена ключом --skip-ppl"
elif [[ -z "$DATA" || ! -f "$DATA" ]]; then
    skip "датасет не задан или не найден — скорость из отчёта публиковать нельзя"
else
    PBIN="$(python3 - "${HOST:-}" <<'PY'
import sys
sys.path.insert(0, ".")
from common.config import HostConfig
print(HostConfig.load(sys.argv[1] or None).exe("llama-perplexity"))
PY
)"
    for KV in f16 q8_0; do
        OUT="${RUN_DIR}/ppl_kv-${KV}.json"
        if ./measure/perplexity.sh --model "$MODEL_PATH" --data "$DATA" \
                --bin "$PBIN" --ctx "$PPL_CTX" --chunks "$PPL_CHUNKS" \
                --kv "$KV" --out "$OUT"; then
            PPL_JSONS+=("$OUT")
        else
            fail "perplexity.sh kv=$KV"
        fi
        [[ $QUICK == 1 ]] && break
    done
fi

# --- Этап 8 ---------------------------------------------------------------
step "Этап 8 — отчёт"
REPORT_ARGS=("$RUN_DIR" --out "${RUN_DIR}/report.md")
[[ -n "$BPT"  ]] && REPORT_ARGS+=(--bytes-json "$BPT")
[[ -n "$ATTR" ]] && REPORT_ARGS+=(--attribution-json "$ATTR")
[[ ${#PPL_JSONS[@]} -gt 0 ]] && REPORT_ARGS+=(--ppl-json "${PPL_JSONS[@]}")
python3 analyze/report.py "${REPORT_ARGS[@]}" || fail "report.py"

# --- итог -----------------------------------------------------------------
echo
echo "==================== ИТОГ ===================="
echo "прогон : $RUN_DIR"
[[ -f "${RUN_DIR}/report.md" ]] && echo "отчёт  : ${RUN_DIR}/report.md"
if [[ ${#SKIPPED[@]} -gt 0 ]]; then
    echo "пропущено (${#SKIPPED[@]}):"; printf '  - %s\n' "${SKIPPED[@]}"
fi
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "с ошибками (${#FAILED[@]}):"; printf '  ! %s\n' "${FAILED[@]}"
    exit 4
fi
echo "все этапы отработали"
