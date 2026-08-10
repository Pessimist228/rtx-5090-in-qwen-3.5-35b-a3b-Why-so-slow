#!/usr/bin/env bash
# Этап 6 — атрибуция остатка через nsys.
#
# Разложение времени говорит, СКОЛЬКО времени уходит мимо памяти. Профиль
# говорит, КУДА именно. Отвечает на два критерия приёмки:
#
#   6. сколько запусков ядер приходится на один токен
#   7. захватываются ли CUDA-графы — да/нет
#
# Почему nsys, а не ncu. ncu профилирует ядра поштучно и ради точных счётчиков
# сериализует их, разрушая ровно то, что мы измеряем — промежутки между
# запусками. Вопрос «сколько времени вне ядер» ncu не отвечает в принципе.
# nsys снимает временную шкалу целиком и показывает дыры.
#
#   ./profile.sh --model /workspace/models/model.gguf --out /workspace/results/prof
#
# Прогон намеренно короткий: батч 1, 128 токенов декода. Профиль на всю
# матрицу не нужен, нужна одна чистая точка.

set -euo pipefail

MODEL=""
OUT="./profile"
BIN=""
NGEN=128
DEPTH=0
FA=1
KV="f16"
GRAPHS_OFF=0
# node — видно каждое ядро внутри графа, но nsys добавляет около 70% времени,
# и деление «GPU занят / простаивает» по такой трассе недостоверно.
# graph — граф идёт одним узлом, накладные малы, зато поимённого состава нет.
# Нужны оба: первым считаем ядра, вторым меряем время.
GRAPH_TRACE="node"

# osrt — трассировка блокирующих вызовов ОС, она есть только у линуксовой
# сборки nsys: на Windows тот же ключ обрывает прогон с Illegal --trace
# argument. Список подсистем поэтому выбирается по платформе, а не задаётся
# намертво, иначе Фаза 1 на ноуте и Фаза 2 на арендованной карте требовали бы
# разных команд — это ровно то, что запрещает критерий приёмки 8.
case "${OSTYPE:-$(uname -s 2>/dev/null || echo unknown)}" in
    msys*|cygwin*|win32*|MINGW*|MSYS*|CYGWIN*) TRACE="cuda,nvtx" ;;
    *)                                         TRACE="cuda,nvtx,osrt" ;;
esac

usage() { sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)  MODEL="$2"; shift 2 ;;
        --out)    OUT="$2";   shift 2 ;;
        --bin)    BIN="$2";   shift 2 ;;
        --n-gen)  NGEN="$2";  shift 2 ;;
        --depth)  DEPTH="$2"; shift 2 ;;
        --fa)     FA="$2";    shift 2 ;;
        --kv)     KV="$2";    shift 2 ;;
        --graphs-off) GRAPHS_OFF=1; shift ;;
        --graph-trace) GRAPH_TRACE="$2"; shift 2 ;;
        --trace)  TRACE="$2"; shift 2 ;;
        -h|--help) usage 0 ;;
        *) echo "неизвестный аргумент: $1" >&2; usage 1 ;;
    esac
done

[[ -n "$MODEL" ]] || { echo "ошибка: нужен --model" >&2; exit 1; }
[[ -f "$MODEL" ]] || { echo "ошибка: модель не найдена: $MODEL" >&2; exit 1; }

command -v nsys >/dev/null 2>&1 || {
    echo "ошибка: nsys не найден. Этап 6 без него не выполняется." >&2
    echo "  Ubuntu с репозиторием NVIDIA: apt-get install -y cuda-nsight-systems-12-8" >&2
    exit 1
}

if [[ -z "$BIN" ]]; then
    BIN="$(dirname "$0")/../../llama.cpp/build/bin/llama-bench"
fi
[[ -x "$BIN" ]] || { echo "ошибка: llama-bench не найден: $BIN" >&2; exit 1; }

mkdir -p "$OUT"

# Суффикс собирается через if, а не через $(... && echo ...): при ложном
# условии подстановка возвращает код 1, присваивание наследует его, и set -e
# убивает скрипт молча, ещё до первой строчки вывода.
SUFFIX=""
if [[ $GRAPHS_OFF == 1 ]]; then
    SUFFIX="_nographs"
fi
REPORT="${OUT}/decode_d${DEPTH}_fa${FA}_${KV}_${GRAPH_TRACE}${SUFFIX}"

# sed, а не head: под pipefail head закрывает канал раньше времени, источник
# получает SIGPIPE, конвейер возвращает ненулевой код и set -e убивает скрипт.
echo "nsys        : $(nsys --version 2>&1 | sed -n '1p')"
echo "трассировка : $TRACE (graph-trace=$GRAPH_TRACE)"
echo "бинарь      : $BIN"
echo "точка       : декод ${NGEN} токенов, глубина ${DEPTH}, fa=${FA}, kv=${KV}"
if [[ $GRAPHS_OFF == 1 ]]; then echo "CUDA-графы  : ПРИНУДИТЕЛЬНО ВЫКЛЮЧЕНЫ"; fi
echo

ENV_PREFIX=()
if [[ $GRAPHS_OFF == 1 ]]; then ENV_PREFIX=(env GGML_CUDA_DISABLE_GRAPHS=1); fi

# -r 1 и без прогрева: профилируем один чистый проход, а не статистику.
# --cuda-graph-trace=node нужен, чтобы ядра внутри графа были видны поимённо,
# иначе весь граф схлопнется в одну строку и вопрос «сколько ядер на токен»
# останется без ответа.
"${ENV_PREFIX[@]}" nsys profile \
    --trace="$TRACE" \
    --cuda-graph-trace="$GRAPH_TRACE" \
    --sample=none \
    --force-overwrite=true \
    --output="$REPORT" \
    "$BIN" -m "$MODEL" -p 0 -n "$NGEN" -d "$DEPTH" -r 1 \
           -ngl 99 -sm none -fa "$FA" -ctk "$KV" -ctv "$KV" \
    > "${REPORT}.bench.log" 2>&1 || {
        echo "nsys profile завершился с ошибкой, см. ${REPORT}.bench.log" >&2
        tail -20 "${REPORT}.bench.log" >&2
        exit 1
    }

echo "профиль снят: ${REPORT}.nsys-rep"
echo

# Статистику разбираем в CSV: их читает analyze/, а не человек.
for rep in cuda_gpu_kern_sum cuda_api_sum cuda_gpu_trace; do
    nsys stats --report "$rep" --format csv --force-export=true \
        --output "${REPORT}_${rep}" "${REPORT}.nsys-rep" >/dev/null 2>&1 \
        && echo "  ${rep}: ok" || echo "  ${rep}: не собрался"
done

echo
echo "=== топ-15 ядер по суммарному времени ==="
nsys stats --report cuda_gpu_kern_sum --format table "${REPORT}.nsys-rep" 2>/dev/null \
    | sed -n '1,22p' || echo '  не удалось'

echo
echo "=== вызовы CUDA API (здесь видно, запускаются ядра поштучно или графом) ==="
nsys stats --report cuda_api_sum --format table "${REPORT}.nsys-rep" 2>/dev/null \
    | sed -n '1,18p' || echo '  не удалось'

echo
echo "=== захват CUDA-графов ==="
grep -icE "cudaGraphLaunch|cudaGraphInstantiate" "${REPORT}_cuda_api_sum.csv" 2>/dev/null \
    | sed 's/^/  строк с графовыми вызовами: /' || true
grep -iE "CUDA graph" "${REPORT}.bench.log" | sort | uniq -c | sed 's/^/  /' || \
    echo "  llama.cpp ничего не сказал про графы"

echo
echo "готово. Дальше: analyze/attribute.py ${OUT}"
