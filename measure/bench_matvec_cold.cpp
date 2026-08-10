// Кривая полосы для настоящего mul_mat_vec_q на холодных весах.
//
// Зачем своя программа. test-backend-ops для этого вопроса непригоден: в
// режиме perf он дублирует один и тот же узел n_runs раз, матрица оседает в L2
// (96 МиБ у 5090) и даёт 3654 ГБ/с при физических 1687. Заглушки на torch
// врут в другую сторону — они вдвое медленнее настоящего ядра.
//
// Здесь N различных матриц, рабочий набор в несколько ГиБ, поэтому к моменту
// возврата к матрице она вытеснена. Это ровно тот режим, в котором работает
// декод: каждый вес читается один раз за токен.
//
//   bench_matvec_cold [тип] [K] [рабочий_набор_ГиБ]
//
// тип — квант весов: q4_0, q8_0, q6_K, f16. Модель смешивает несколько, и
//       порог у них может отличаться: на байт веса приходится разный объём
//       распаковки.
//
// Всегда один вектор на вызов: замеряется голый single-stream декод, как
// требует ТЗ (-b 1). Пачки шире одного токена сюда не входят намеренно.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <vector>
#include <algorithm>

static double now_s() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

static ggml_type parse_type(const char * s) {
    if (!strcmp(s, "q4_0")) return GGML_TYPE_Q4_0;
    if (!strcmp(s, "q8_0")) return GGML_TYPE_Q8_0;
    if (!strcmp(s, "q6_K")) return GGML_TYPE_Q6_K;
    if (!strcmp(s, "q4_K")) return GGML_TYPE_Q4_K;
    if (!strcmp(s, "f16"))  return GGML_TYPE_F16;
    fprintf(stderr, "неизвестный тип: %s\n", s);
    exit(1);
}

int main(int argc, char ** argv) {
    const ggml_type WT = argc > 1 ? parse_type(argv[1]) : GGML_TYPE_Q4_0;
    int64_t K          = argc > 2 ? atoll(argv[2]) : 4096;
    double  target_gib = argc > 3 ? atof(argv[3])  : 4.0;

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) { fprintf(stderr, "CUDA-бэкенд не поднялся\n"); return 1; }

    const size_t row_b  = ggml_row_size(WT, K);
    const size_t want_b = (size_t)(target_gib * (1ull << 30));

    printf("# настоящее ядро ggml, веса холодные (N различных матриц)\n");
    printf("# тип=%s, K=%lld, строка=%zu Б, рабочий набор ~%.1f ГиБ\n",
           ggml_type_name(WT), (long long) K, row_b, target_gib);
    printf("%12s %9s %7s %11s %11s %12s\n",
           "МБ/ядро", "строк", "ядер", "мкс/ядро", "ГБ/с", "набор,ГиБ");
    fflush(stdout);

    const size_t MiB = 1u << 20;
    const size_t sizes[] = { 1, 2, 4, 8, 16, 32, 64, 128, 256, 512 };

    for (size_t si = 0; si < sizeof(sizes) / sizeof(sizes[0]); si++) {
        const size_t S = sizes[si] * MiB;
        const int64_t m = (int64_t)(S / row_b);
        if (m < 1) continue;

        const size_t bpk = (size_t) m * row_b;              // байт на одно ядро
        int N = (int) std::min<size_t>(1024, std::max<size_t>(4, want_b / bpk));

        ggml_init_params pw = { (size_t)(N + 8) * ggml_tensor_overhead(), NULL, true };
        ggml_init_params pc = { (size_t)(N + 8) * ggml_tensor_overhead()
                                + ggml_graph_overhead_custom(N + 8, false), NULL, true };
        ggml_context * ctxw = ggml_init(pw);
        ggml_context * ctxc = ggml_init(pc);

        std::vector<ggml_tensor *> ws(N);
        for (int i = 0; i < N; i++) ws[i] = ggml_new_tensor_2d(ctxw, WT, K, m);

        ggml_backend_buffer_t bw = ggml_backend_alloc_ctx_tensors(ctxw, backend);
        if (!bw) {
            printf("%12.1f %9lld %7d   не хватило VRAM\n", bpk / 1e6, (long long) m, N);
            ggml_free(ctxw); ggml_free(ctxc); continue;
        }
        ggml_backend_buffer_set_usage(bw, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        ggml_tensor * x = ggml_new_tensor_2d(ctxc, GGML_TYPE_F32, K, 1);
        std::vector<ggml_tensor *> ys(N);
        for (int i = 0; i < N; i++) ys[i] = ggml_mul_mat(ctxc, ws[i], x);

        ggml_backend_buffer_t bc = ggml_backend_alloc_ctx_tensors(ctxc, backend);
        if (!bc) {
            printf("%12.1f %9lld %7d   не хватило VRAM под результаты\n", bpk / 1e6, (long long) m, N);
            ggml_backend_buffer_free(bw); ggml_free(ctxw); ggml_free(ctxc); continue;
        }

        std::vector<float> xh(K, 0.01f);
        ggml_backend_tensor_set(x, xh.data(), 0, K * sizeof(float));

        ggml_cgraph * gf = ggml_new_graph_custom(ctxc, N + 8, false);
        for (int i = 0; i < N; i++) ggml_build_forward_expand(gf, ys[i]);

        // прогрев: первый проход тянет за собой захват графа и прогрев аллокатора
        ggml_backend_graph_compute(backend, gf);
        ggml_backend_synchronize(backend);

        double t0 = now_s();
        ggml_backend_graph_compute(backend, gf);
        ggml_backend_synchronize(backend);
        double one = now_s() - t0;

        int R = std::max(3, std::min(100, (int)(0.3 / std::max(one, 1e-6))));
        std::vector<double> per;
        for (int rep = 0; rep < 5; rep++) {
            t0 = now_s();
            for (int r = 0; r < R; r++) ggml_backend_graph_compute(backend, gf);
            ggml_backend_synchronize(backend);
            per.push_back((now_s() - t0) / R / N);
        }
        std::sort(per.begin(), per.end());
        const double s_per_kernel = per[per.size() / 2];

        printf("%12.1f %9lld %7d %11.2f %11.1f %12.2f\n",
               bpk / 1e6, (long long) m, N,
               s_per_kernel * 1e6, bpk / s_per_kernel / 1e9,
               (double) N * bpk / (1u << 30));
        fflush(stdout);

        ggml_backend_buffer_free(bc);
        ggml_backend_buffer_free(bw);
        ggml_free(ctxc);
        ggml_free(ctxw);
    }

    ggml_backend_free(backend);
    return 0;
}
