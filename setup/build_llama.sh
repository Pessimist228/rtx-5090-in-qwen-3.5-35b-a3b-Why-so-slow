#!/usr/bin/env bash
# Сборка llama.cpp с CUDA под заданную архитектуру.
#
# Фаза 1 (ноут, sm_89) этим скриптом НЕ пользуется: там взят официальный
# release-архив b9006, он сам сообщает свой коммит, а ТЗ запрещает править
# llama.cpp — собирать из исходников нечего.
#
# Скрипт нужен Фазе 2: Blackwell (sm_120) требует CUDA >= 12.8, готовых
# бинарей под него может не оказаться. На арендованной карте он должен
# отработать с первого раза — отладки там быть не должно.
#
#   ./build_llama.sh --arch 120 --commit c5a3bc39b --prefix ~/llama.cpp
#   ./build_llama.sh --arch 120 --commit b10326 --with-tests
#
# Архитектура и коммит берутся из config/<host>.json, руками не правятся.
#
# --with-tests добавляет test-backend-ops. Он гоняет настоящие ядра ggml на
# заданных формах и печатает достигнутую пропускную способность, то есть даёт
# кривую полосы для реального mul_mat_vec_q без модели и без ncu. На поде это
# единственный доступный путь: контейнер идёт без CAP_SYS_ADMIN, и счётчики
# Nsight Compute там закрыты наглухо.

set -euo pipefail

ARCH=""
COMMIT=""
PREFIX="${HOME}/llama.cpp"
REPO="https://github.com/ggml-org/llama.cpp.git"
JOBS="$(nproc 2>/dev/null || echo 4)"
BUILD_COMMIT=""
TESTS="OFF"

usage() {
    sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --arch)    ARCH="$2";   shift 2 ;;
        --commit)  COMMIT="$2"; shift 2 ;;
        --prefix)  PREFIX="$2"; shift 2 ;;
        --repo)    REPO="$2";   shift 2 ;;
        --jobs)    JOBS="$2";   shift 2 ;;
        --build-commit) BUILD_COMMIT="$2"; shift 2 ;;
        --with-tests) TESTS="ON"; shift ;;
        -h|--help) usage 0 ;;
        *) echo "неизвестный аргумент: $1" >&2; usage 1 ;;
    esac
done

[[ -n "$ARCH" ]]   || { echo "ошибка: нужен --arch (89 для 4060, 120 для 5090)" >&2; exit 1; }
[[ -n "$COMMIT" ]] || { echo "ошибка: нужен --commit — сборка без пина невоспроизводима" >&2; exit 1; }

for tool in git cmake nvcc; do
    command -v "$tool" >/dev/null 2>&1 || { echo "ошибка: '$tool' не найден в PATH" >&2; exit 1; }
done

CUDA_VER="$(nvcc --version | sed -n 's/.*release \([0-9.]*\).*/\1/p')"
echo "CUDA toolkit : ${CUDA_VER}"
echo "target arch  : sm_${ARCH}"
echo "commit       : ${COMMIT}"
echo "prefix       : ${PREFIX}"

# Blackwell не поддерживается тулкитами до 12.8 — лучше упасть здесь,
# чем получить непонятную ошибку компиляции через двадцать минут.
if [[ "$ARCH" == "120" ]]; then
    major="${CUDA_VER%%.*}"; minor="${CUDA_VER#*.}"; minor="${minor%%.*}"
    if (( major < 12 || (major == 12 && minor < 8) )); then
        echo "ошибка: sm_120 требует CUDA >= 12.8, найдена ${CUDA_VER}" >&2
        exit 1
    fi
fi

# Полный клон тянет около 400 МБ истории. На арендованных подах маршрут до
# GitHub бывает узким (замерено 91 кБ/с — это больше часа), а нужен ровно один
# коммит. Для тега берём поверхностный клон, для голого хеша приходится
# клонировать целиком: --depth по хешу не работает.
if [[ -f "${PREFIX}/CMakeLists.txt" && ! -d "${PREFIX}/.git" ]]; then
    # Исходники положены снаружи — обычно потому, что маршрут с пода до GitHub
    # узкий (замерено 45 кБ/с даже на поверхностном клоне), и архив быстрее
    # скачать на своей машине и залить по ssh. Коммит тогда не проверить, он
    # берётся из конфига на веру, а сходится ли он — скажет env.json по тому,
    # что сообщит собранный бинарь.
    echo "исходники уже в ${PREFIX} — git пропускаем"
elif [[ ! -d "${PREFIX}/.git" ]]; then
    if git ls-remote --tags --exit-code "$REPO" "refs/tags/${COMMIT}" >/dev/null 2>&1; then
        echo "поверхностный клон тега ${COMMIT}"
        git clone --depth 1 --branch "$COMMIT" "$REPO" "$PREFIX"
    else
        echo "коммит не тег — полный клон"
        git clone "$REPO" "$PREFIX"
        git -C "$PREFIX" checkout --detach "$COMMIT"
    fi
else
    git -C "$PREFIX" fetch --depth 1 origin "$COMMIT" || git -C "$PREFIX" fetch --all --tags
    git -C "$PREFIX" checkout --detach FETCH_HEAD 2>/dev/null || \
        git -C "$PREFIX" checkout --detach "$COMMIT"
fi

BUILD_DIR="${PREFIX}/build"
rm -rf "$BUILD_DIR"

# Без .git cmake ставит BUILD_NUMBER=0 и BUILD_COMMIT="unknown", и бинарь
# перестаёт сообщать, из чего собран. Для харнесса это критично: провенанс
# сверяется с pinned_commit, и «unknown» делает результат недействительным.
# CMakeLists уважает заданные снаружи значения (if NOT DEFINED), поэтому при
# сборке из архива проставляем их явно: номер из тега вида bNNNNN, хеш — из
# --build-commit.
VERSION_FLAGS=()
if [[ ! -d "${PREFIX}/.git" ]]; then
    if [[ "$COMMIT" =~ ^b([0-9]+)$ ]]; then
        VERSION_FLAGS+=("-DLLAMA_BUILD_NUMBER=${BASH_REMATCH[1]}")
    fi
    if [[ -n "$BUILD_COMMIT" ]]; then
        VERSION_FLAGS+=("-DLLAMA_BUILD_COMMIT=${BUILD_COMMIT}")
    else
        echo "предупреждение: сборка из архива без --build-commit, хеш будет unknown" >&2
    fi
fi

cmake -S "$PREFIX" -B "$BUILD_DIR" \
    "${VERSION_FLAGS[@]}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="$ARCH" \
    -DGGML_NATIVE=ON \
    -DLLAMA_BUILD_TESTS="$TESTS" \
    -DLLAMA_BUILD_EXAMPLES=ON \
    -DLLAMA_CURL=OFF

cmake --build "$BUILD_DIR" --config Release -j "$JOBS"

echo
echo "готово. проверка:"
"${BUILD_DIR}/bin/llama-bench" --version 2>&1 | grep -E '^version:' || true
echo
echo "пропишите в config/<host>.json:"
echo "  \"bin_dir\": \"${BUILD_DIR}/bin\""
