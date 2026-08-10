# The harness: measuring and dissecting single stream llama.cpp inference

This documents the tool. The findings live in [POST-en.md](POST-en.md); numbers
quoted here are illustrative and the post is the source of truth for results.

The harness measures and diagnoses only. It writes no kernels, patches nothing
in llama.cpp, and optimizes nothing. Everything it produces is meant to be
checkable on paper:

```
t_measured = bytes_per_token / measured_bandwidth + t_residual
```

## The whole pipeline in one command

```bash
./run_all.sh --model Qwen_Qwen3.5-35B-A3B-Q4_0.gguf --data wiki.test.raw
./run_all.sh --model model.gguf --quick --ppl-chunks 4   # wiring check
```

Runs environment capture, bandwidth, the llama-bench matrix, bytes per token,
the roofline decomposition, nsys profiling with attribution, perplexity, and a
report, all into one run directory. The host is picked from the GPU name, so
the command is identical on a laptop and on a rented card.

Environment capture runs with `--strict`: a run must not start on a machine
where something is already wrong. Stages with nothing to run on are skipped and
noted in the summary rather than failing the run, because a laptop may have no
perplexity dataset and a rented pod may have no nsys. A `bench.py` failure does
stop the run, since a report without the matrix is meaningless.

## Environment capture

```bash
python -m common.env                      # host from GPU name
python -m common.env --host laptop-4060   # or explicitly
python -m common.env --strict             # nonzero exit on any problem
```

Prints a summary and writes `env.json`. A number without this block next to it
is not valid, so every run drops its own `env.json` into its own directory.

`validate_env()` catches what makes measurements meaningless: the GPU not
matching the config or its compute capability differing from `cuda_arch`,
llama.cpp not loading the CUDA backend or its commit diverging from
`pinned_commit`, the GPU already throttling or hot before any load, and missing
torch (no bandwidth stage) or missing gguf (no bytes per token stage).

Throttling is decoded from the `clocks_throttle_reasons.active` bitmask.
`GpuIdle` and `DisplayClockSetting` do not interfere with a measurement and are
not held against it, unlike `SwPowerCap`, `HwSlowdown` and thermal reasons. On a
laptop card with a 75 W limit that flag is the main source of noise.

## Memory bandwidth

```bash
python measure/bandwidth.py --validate
```

Writes `bandwidth.json` and sets `measured_bandwidth_gbs` in `env.json`.

The headline number is a **sequential read**. At batch 1 every weight is read
exactly once per matrix vector product and never reused, and writes are
negligible, so the ceiling is set by reading. Copy is measured alongside as a
sanity check: it moves twice the bytes and must not beat reading.

`--validate` adds two independent cross checks, because a suspiciously high
number is usually a cache artifact rather than good silicon:

- **buffer size sweep**: bandwidth must not fall as the buffer grows. If a
  cache were helping, the small buffer would be the fast one.
- **kernel agreement**: `sum` on f32, `max` on f32 and `sum` on f16 must land
  on the same number. If they disagree, a kernel was measured, not memory.

Specification numbers are never used for the roofline. On the cards here the
achievable read is 93% to 98% of spec, and dividing by the spec number moves
every downstream conclusion.

## The llama-bench matrix

```bash
python measure/bench.py --model model.gguf --graphs-off
```

Runs the configuration matrix from `config/<host>.json`, parses llama-bench
JSONL, and writes one line per point with the full configuration inside.

Three things in here matter more than they look:

**Cooldown between points.** A run without it showed SM clocks sagging from
2550 to 1650 MHz over seven points, so later points were slower because of heat
rather than because of their configuration.

**Shuffled order plus an anchor point.** Shuffling turns residual thermal drift
into noise instead of a systematic trend. The same point is measured first and
last, and if the two disagree by more than the noise threshold the whole matrix
is marked as not self consistent.

**Medians, not means, with spread recorded.** Points whose spread exceeds the
threshold are flagged noisy rather than silently averaged.

`--graphs-off` duplicates the whole matrix with `GGML_CUDA_DISABLE_GRAPHS=1`.
The difference is the price of kernel launches, which is worth knowing before
deciding whether launch overhead is worth attacking.

## Bytes per token

```bash
python analyze/bytes_per_token.py model.gguf --depth 0 --kv-type f16
```

Reads GGUF metadata and counts what a single decode step actually pulls from
memory, by category, in bytes and percent.

Counting is done from each tensor's real `ggml_type` rather than from the bits
per weight implied by the quant name. Quant files are mixtures: a file called
Q4_0 can hold Q4_0, Q6_K, Q5_K and Q4_1 tensors at once, and its effective bits
per weight is not 4.5. The per tensor sum agrees with the file size to within
0.2%, which is metadata and alignment.

Two details change the answer by more than ten percent if you get them wrong:

- `output.weight` is read in full every token and counts.
- `token_embd.weight` is a separate tensor of the same shape, but a token reads
  one row of it, so it does not count.

For MoE the routed experts are counted by their actual fraction (8 of 256, not
all of them), the shared expert counts in full, and the recurrent state is a
separate line because it is read and written every token and quantization does
not shrink it.

## Profiling and attribution

```bash
./measure/profile.sh --model model.gguf --bin llama-bench --graph-trace node
python analyze/attribute.py results/profile --trace node
```

Answers two questions that the roofline cannot: how many kernels run per token,
and whether CUDA graphs are captured.

`--cuda-graph-trace=node` is required. Without it a whole graph collapses into
one trace row and the kernel count is unobtainable. With it the profiler adds
time, and it adds it unevenly: long kernels are barely affected while kernels
near a microsecond can be inflated by about 2x. Time shares from such a trace
are usable, absolute durations are not. For the busy versus idle question,
`nvidia-smi dmon` on an uninstrumented run is the honest instrument.

## The bandwidth curve, `bench_matvec_cold.cpp`

This is the part worth stealing. It measures how much bandwidth a kernel
actually achieves as a function of how many bytes it reads, using the real ggml
kernel on cold data.

```bash
g++ -O2 -o bench_matvec_cold measure/bench_matvec_cold.cpp \
    -I$LLAMA/ggml/include -L$LLAMA/build/bin \
    -lggml -lggml-base -lggml-cuda -Wl,-rpath,$LLAMA/build/bin

./bench_matvec_cold q4_0 4096 4.0     # type, row length, working set in GiB
```

Three ways to get this wrong, all of which I hit before this program existed:

**A naive size sweep measures L2.** L2 is 96 MiB on a 5090 and 32 MiB on a
4060, so a 64 KB to 64 MB sweep fits entirely in cache and the plateau appears
within the first few kilobytes. Here every launch reads its own slice of a
multi gigabyte pool, so by the time the pool wraps around the data has been
evicted.

**Reusing one tensor measures cache.** `test-backend-ops perf` duplicates one
graph node, which is why it reports several times the physical bandwidth of the
card. In decode every weight is read once per token.

**Timing launches from Python measures Python.** A torch call costs single to
double digit microseconds, an order of magnitude more than a kernel launch, so
the GPU idles while the CPU dispatches. Only a captured CUDA graph gives a real
per launch number, because `replay()` has no Python in its loop.

`measure/kernel_cost.py` covers the same ground with torch instead of ggml. It
is useful for the graph node cost and for comparing cards with one kernel, but
its absolute thresholds sit well above those of the real matvec, so the two are
not interchangeable.

## Sampler cost

```bash
MODEL=model.gguf ./measure/sampler_cost.sh
MODEL=model.gguf THREADS=1 OUT=results/sampler_t1.jsonl ./measure/sampler_cost.sh
```

`server_overhead.sh` measures the whole wrapper around the engine and finds that
a normal sampler costs 0.723 ms per token against a 3.150 ms engine step. That
number bundles top_k, top_p, min_p, repeat_penalty and temperature together.
This script takes the bundle apart: one server, one request shape, one stage
enabled at a time through the `samplers` array, plus a sweep over k.

Two details decide whether the numbers mean anything. Every request sets
`ignore_eos`, so token counts are identical across configurations; without it a
run can stop early and its client side rate is meaningless, which happened once
in the `server_overhead.sh` data at 116 tokens instead of 512. And the baseline
for the per stage ladder is temperature only at 1.0 rather than greedy, because
at temperature 0 llama.cpp takes a greedy path that skips the chain entirely.

Compare deltas, not absolute rates. Across models the engine step differs, so
only `full_chain` minus `greedy` is comparable.

## Quality gate

```bash
./measure/perplexity.sh --model model.gguf --data wiki.test.raw --kv f16
```

The dataset and context length are fixed, otherwise numbers from different
quants are not comparable. KV cache type is part of the measurement rather than
a default, because it changes both speed and quality, and publishing a speed
taken at one KV type next to a quality number taken at another is exactly the
mistake this gate exists to prevent.

A speed number is not published without a quality number beside it.

## Configuration

Only the file in `config/` differs between machines; the code is identical. The
host is detected from the GPU name (`gpu.expected_name_substring`), so the
command line does not change either.

| file | machine |
|---|---|
| `config/laptop-4060.json` | laptop, RTX 4060 8 GB, sm_89, dense model |
| `config/rented-5090.json` | rented, RTX 5090 32 GB, sm_120, MoE model |

The run matrix, safety thresholds and hardware parameters live in the config,
not in the code.

## Layout

```
config/       machine configs, the only thing that differs between hosts
common/       config.py (loading, model lookup, run directories)
              env.py     (environment capture and validation)
setup/        build_llama.sh
measure/      bandwidth.py, bench.py, profile.sh, perplexity.sh,
              server_overhead.sh, sampler_cost.sh, kernel_cost.py,
              bench_matvec_cold.cpp
analyze/      bytes_per_token.py, decompose.py, attribute.py, report.py
data/         raw measurements behind the numbers in the posts
results/      <host>_<gpu>_<model>_<quant>_<timestamp>/
```

## Install

```bash
pip install -r requirements.txt
# torch is installed separately, with a wheel matching the target driver:
# pip install torch --index-url https://download.pytorch.org/whl/cu130
```

A CUDA toolkit is not required for most of the pipeline: bandwidth is measured
through PyTorch with its own runtime. Building llama.cpp and compiling
`bench_matvec_cold.cpp` do need one.

Nsight Systems and Nsight Compute install together and separately from the
toolkit (`winget install --id Nvidia.Nsight.Compute` on Windows). Neither puts
itself on PATH, and `profile.sh` expects `nsys` there.

On GeForce cards Nsight Compute cannot read performance counters without
permission. On Windows that is `RmProfilingAdminOnly = 0` under
`HKLM:\SOFTWARE\NVIDIA Corporation\Global\NVTweak` plus a reboot, since the
driver reads the key at initialization. In a container it needs `CAP_SYS_ADMIN`,
which a rented pod usually does not grant, and which cannot be added from
inside.

## Known gaps

`setup/fetch_models.sh` was never written; models were placed by hand.

Nsight Compute counters were never actually read, for the permission reason
above, so achieved bandwidth in the posts is bytes divided by time.

The small kernel cost used in the fusion projection is fitted at a single
point. It agrees with an independently measured graph node cost, but it has not
been tested at small kernel counts.
