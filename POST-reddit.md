# Draft for r/LocalLLaMA

Paste-ready. Reddit markdown, tables render on new reddit and on old.reddit with the
markdown-mode editor. Flair: Discussion. Post the link inline, not as a link post,
otherwise the body is lost.

Title:

> MoE leaves half your memory bandwidth on the floor, and routing is not the reason. Measured on a 5090.

---

I spent a weekend measuring where a token actually goes on an RTX 5090 running llama.cpp, and one number kept bothering me.

Same card, same build, batch 1, empty context, Q4_0:

| | GB read per token | t/s | share of memory bandwidth |
|---|---|---|---|
| dense Qwen3.5-9B | 5.000 | 239.62 | **72%** |
| MoE Qwen3.5-35B-A3B | 2.133 | 317.45 | **41%** |

The MoE is faster in wall clock, obviously, it reads less than half the bytes. But it leaves about 1.4x of its own ceiling unused and the dense model does not. Measured bandwidth on this card is 1671 GB/s, 93% of the 1792 on the box.

I went after the three explanations everyone reaches for first, and all three are wrong.

CUDA graphs are captured, 127 launches per 128 tokens, no fallback reason in the log. Routing costs 7% of kernel time against 51% for the matvecs. SM occupancy on a clean uninstrumented run is 97%, so the card is busy almost all the time, just reading memory inefficiently.

## The cause is bytes per kernel

A kernel does not reach full bandwidth until it reads enough of them, so I wrote a small ggml program that runs the real `mul_mat_vec_q` over N distinct matrices with a 4 GiB working set. Nothing is ever in L2 and every launch touches weights it has not seen, which is how decode reads: once each, with no reuse.

| MB read per kernel | 1.0 | 4.2 | 8.4 | 16.8 | 33.6 | 134 | 537 |
|---|---|---|---|---|---|---|---|
| % of peak bandwidth | 23% | 53% | 70% | 83% | 91% | 98% | 100% |

The MoE averages 4.6 MB per matvec. The dense model averages 22.2 MB. Look both up on that curve and the whole gap is accounted for without any second effect.

And this is not a llama.cpp bug. The experts are already gathered into one launch: `mul_mat_vec_q` takes an index array and fires 4.1 times per layer, not 24. Eight active experts out of 256 is simply not many bytes even after you batch them. The same model contains the counterexample, the output head is one launch of 417 MB and it hits 1601 GB/s, which is 96% of the card.

## Does the curve hold up

It was fitted on the MoE, so I used it to predict the dense model before running it: 248.6 t/s predicted, 239.62 measured, 3.7% off. Same curve predicts speed at 4096, 16384 and 32768 context to within 2.4% with no new measurement.

## llama-bench and llama-server disagree by 29%

`llama-bench` says 317 t/s. A real request to `llama-server` gives you 225.

Per token, same pod, same model, 512 tokens generated:

| | ms/token | running total |
|---|---|---|
| engine, `llama-bench` | 3.150 | 3.150 |
| server loop, greedy sampling | +0.279 | 3.429 |
| a normal sampler (top_k 40, top_p 0.9, min_p 0.05, repeat_penalty 1.1) | **+0.723** | 4.152 |
| HTTP and per request prefill | +0.291 | 4.443 |

The sampler alone is 23% of engine time. It is CPU work over a 250k vocabulary and it happens between decode steps, so the GPU waits through it. Streaming, which I fully expected to cost something, costs nothing: 225.8 t/s without it, 225.1 with it.

So if you have been comparing your chat speed against a `llama-bench` screenshot and feeling cheated, you were not imagining it. That is a 29% haircut before anyone optimizes anything.

## Three things you can use today

Driver 570.195.03 to 580.159.03 is worth 8.0% on decode, median over 12 matched matrix points, and 5.8% on prefill. Perplexity is identical to four decimal places, 5.7732 ± 0.0434 on both, and kernels per token are identical too, so nothing got quietly dumber. The gain lives entirely in short kernels: +8.6% at 1 MB per kernel, +0.3% at 537 MB. Same reason turning graphs off costs 26.8% on 570 but only 14.0% on 580.

On this model, quantized KV loses on both axes. `-ctk q8_0 -ctv q8_0` gives 301.73 t/s against 317.45 for f16, and perplexity is very slightly worse. Only 10 of 41 layers are attention here, the KV cache is small, there is nothing to save, and you pay unpacking on every access. Check it on your own model before copying the flag from someone's config.

Flash attention matters more than usual with a long context. With `-fa 1` the drop from 0 to 32k is 14.5%. Without it, 32%.

## Caveats, since someone will ask

Batch 1 only, this is single stream decode and nothing here is a claim about serving throughput. No K quants tested. Achieved bandwidth is bytes divided by time, not an ncu `dram__throughput` counter, because the rented container had no `CAP_SYS_ADMIN` and Nsight Compute cannot read counters without it. No comparison against vLLM, SGLang or TensorRT-LLM, this is a llama.cpp measurement. Speculative decoding and the MTP head were deliberately turned off, I wanted the bare decode numbers.

The fusion projection at the end of the write-up, roughly 1.5x from merging matvecs and up to 2.3x if you also cut the 1142 small kernels per token, is a projection. I have not written a fused kernel. The pure memory roofline is 783 t/s, so 2.47x is the hard ceiling, and no amount of kernel work goes past it.

## Two traps if you go measuring yourself

`test-backend-ops perf` duplicates one graph node, so the tensor sits in L2 forever and it will tell you the card does 3654 GB/s. The cold curve above peaks at 1683, and a plain sequential read benchmark says 1671, which is the 1671 in the second paragraph. Two instruments, 0.7% apart. Neither of them is 3654.

Timing launches from Python measures Python. I got 9 to 18 µs per launch and believed it for an embarrassing while, until I checked and found the GPU finishing 0.02 µs after the CPU released the queue. It was dispatch overhead the whole time. A captured CUDA graph gives the real number, 0.94 µs per node.

Full write-up with the method, the raw JSON behind every number, and the ggml benchmark: https://github.com/Pessimist228/rtx-5090-in-qwen-3.5-35b-a3b-Why-so-slow

Happy to run things if someone wants a specific configuration checked, though the 5090 was rented and is gone, so anything new means renting again.
