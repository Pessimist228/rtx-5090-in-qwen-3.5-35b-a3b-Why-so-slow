**Same GPU, same llama.cpp build: dense model pulls 72% of memory bandwidth, MoE pulls 41%. Routing is not why.**

RTX 5090, llama.cpp b10326, Q4_0, batch 1, empty context.

- Dense Qwen3.5-9B: 5.000 GB per token, 239.62 t/s, **72%** of measured bandwidth
- MoE Qwen3.5-35B-A3B: 2.133 GB per token, 317.45 t/s, **41%**

The MoE is faster in absolute terms. It just leaves about 1.4x of its own ceiling unused.

I checked the obvious suspects first and all three are dead. CUDA graphs are captured (127 launches per 128 tokens, no disable reason). Routing costs 7% of kernel time against 51% for matvecs. SM occupancy on a clean run is 97%, so there are no scheduling gaps. The card is busy almost all the time, just reading memory inefficiently.

**The cause is how many bytes one kernel reads.** I wrote a small ggml program that runs the real `mul_mat_vec_q` over N distinct matrices with a 4 GiB working set, so data is always cold, the way decode actually reads weights:

| MB per kernel | 1.0 | 4.2 | 8.4 | 16.8 | 33.6 | 134 | 537 |
|---|---|---|---|---|---|---|---|
| % of peak bandwidth | 23% | 53% | 70% | 83% | 91% | 98% | 100% |

MoE averages 4.6 MB per matvec, the dense model 22.2 MB. That is the entire gap.

The trace shows experts are already batched into one launch (`mul_mat_vec_q` takes an index array, 4.1 launches per layer, not 24). Eight experts out of 256 is simply few bytes even when gathered. Meanwhile the output head, one launch of 417 MB, hits 1601 GB/s or 96% of bandwidth. One model contains both.

**Checks.** The curve was fitted on the MoE, then used to predict the dense model cold: 248.6 t/s predicted against 239.62 measured, 3.7% error. It also predicts speed at 4096 / 16384 / 32768 context within 2.4% without any new measurement.

**Two things that will fool you if you try this.** `test-backend-ops perf` duplicates one graph node, so the tensor sits in L2 and it reports 3654 GB/s on a card that does 1683. And timing launches from Python measures Python: I got 9 to 18 µs per launch while the GPU finished 0.02 µs after the CPU released the queue. Only a captured CUDA graph gives a real number (0.94 µs per node).

**Unrelated but bigger than anything above:** `llama-bench` shows 317 t/s, a real `llama-server` request delivers 225. Breakdown per token: engine 3.150 ms, server loop with greedy sampling +0.279, a normal sampler (top_k 40, top_p 0.9, min_p, repeat_penalty) **+0.723**, HTTP and prefill +0.291. The sampler alone is a quarter of engine time, on CPU, over a 250k vocabulary. Streaming costs nothing.

Driver 570 to 580 gives 8% on decode with identical perplexity to four decimals, and the gain lives entirely in small kernels: +8.6% at 1 MB, +0.3% at 537 MB.

Full write-up, code and raw data: https://github.com/Pessimist228/rtx-5090-in-qwen-3.5-35b-a3b-Why-so-slow

Caveats up front: batch 1 only, no K quants, achieved bandwidth is bytes over time rather than an ncu counter (the rented container had no `CAP_SYS_ADMIN`), and no comparison against vLLM or SGLang. The fusion projection at the end of the post is a projection, I have not written a fused kernel.
