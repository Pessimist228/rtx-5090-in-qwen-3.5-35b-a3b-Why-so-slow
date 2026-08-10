#!/usr/bin/env bash
# Per stage cost of the llama.cpp sampler chain.
#
# server_overhead.sh established that a normal sampler costs 0.723 ms/token on
# top of a 3.150 ms engine step, roughly a quarter of decode time, on CPU, while
# the GPU waits. That measurement bundled top_k, top_p, min_p, repeat_penalty
# and temperature together. This one takes the bundle apart.
#
# Method: one server, one request shape, one stage enabled at a time via the
# "samplers" array. The difference against the temperature-only baseline is the
# price of that stage. A separate sweep varies k, because the working hypothesis
# is that the cost is a partial sort over the whole vocabulary and should grow
# with both vocabulary size and k.
#
# Two fixes over server_overhead.sh, both of which bit that script:
#   ignore_eos, so every request generates exactly NPRED tokens. One run there
#   stopped at 116 and its client-side rate was meaningless.
#   The response goes to a file before python reads it, never through the shell.
#
# Usage:
#   MODEL=/workspace/models/foo.gguf ./sampler_cost.sh
#   MODEL=... THREADS=1 OUT=.../sampler_t1.jsonl ./sampler_cost.sh
set -uo pipefail

case "${OSTYPE:-$(uname -s 2>/dev/null || echo unknown)}" in
    msys*|cygwin*|win32*|MINGW*|MSYS*|CYGWIN*)
        BIN=${BIN:-D:/llama/llama-b10326-bin-win-cuda-12.4-x64/llama-server.exe}
        OUT=${OUT:-D:/llama/harness/results/sampler_cost.jsonl}
        PY=${PY:-python} ;;
    *)
        BIN=${BIN:-/workspace/llama.cpp-b10326/build/bin/llama-server}
        OUT=${OUT:-/workspace/harness/results/sampler_cost.jsonl}
        PY=${PY:-python3} ;;
esac
MODEL=${MODEL:?set MODEL to a .gguf path}
PORT=${PORT:-8080}
NPRED=${NPRED:-512}
REPS=${REPS:-3}
THREADS=${THREADS:-16}
LOG=${LOG:-./sampler_server.log}
TMP=${TMP:-./sampler_resp.json}
mkdir -p "$(dirname "$OUT")"

command -v pkill >/dev/null && { pkill -f "llama-server" 2>/dev/null; sleep 2; }
"$BIN" -m "$MODEL" --host 127.0.0.1 --port "$PORT" \
       -ngl 99 -fa 1 -ctk f16 -ctv f16 -c 4096 -np 1 -t "$THREADS" --no-warmup \
       > "$LOG" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT

for _ in $(seq 1 150); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  sleep 2
done
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null || { echo "server did not come up"; tail -20 "$LOG"; exit 1; }
echo "server up, threads=$THREADS, model=$(basename "$MODEL")"

# name | extra json for the request body
# Every stage runs with temperature 1.0 so the dist sampler is always the tail
# of the chain. temp0 is the separate greedy baseline the server short circuits.
CASES=(
  "greedy|\"temperature\":0.0,\"samplers\":[\"temperature\"]"
  "dist_only|\"temperature\":1.0,\"samplers\":[\"temperature\"]"
  "top_k40|\"temperature\":1.0,\"top_k\":40,\"samplers\":[\"top_k\",\"temperature\"]"
  "top_p90|\"temperature\":1.0,\"top_p\":0.9,\"samplers\":[\"top_p\",\"temperature\"]"
  "min_p05|\"temperature\":1.0,\"min_p\":0.05,\"samplers\":[\"min_p\",\"temperature\"]"
  "penalties|\"temperature\":1.0,\"repeat_penalty\":1.1,\"repeat_last_n\":64,\"samplers\":[\"penalties\",\"temperature\"]"
  "typ_p90|\"temperature\":1.0,\"typical_p\":0.9,\"samplers\":[\"typ_p\",\"temperature\"]"
  "full_chain|\"temperature\":0.7,\"top_k\":40,\"top_p\":0.9,\"min_p\":0.05,\"repeat_penalty\":1.1,\"repeat_last_n\":64,\"samplers\":[\"penalties\",\"top_k\",\"top_p\",\"min_p\",\"temperature\"]"
  "full_no_topk|\"temperature\":0.7,\"top_k\":0,\"top_p\":0.9,\"min_p\":0.05,\"repeat_penalty\":1.1,\"repeat_last_n\":64,\"samplers\":[\"penalties\",\"top_p\",\"min_p\",\"temperature\"]"
)

# k sweep: if the cost is a partial sort over the vocabulary, it moves with k.
for k in 1 40 200 1000 10000 100000; do
  CASES+=("k_$k|\"temperature\":1.0,\"top_k\":$k,\"samplers\":[\"top_k\",\"temperature\"]")
done

run() {                                   # name, extra json -> one jsonl line
  local name="$1" extra="$2"
  local body="{\"prompt\":\"Recite the alphabet slowly.\",\"n_predict\":$NPRED,\"stream\":false,\"cache_prompt\":false,\"ignore_eos\":true,$extra}"
  local t0 t1
  t0=$(date +%s.%N)
  curl -s -X POST "http://127.0.0.1:$PORT/completion" \
       -H 'Content-Type: application/json' -d "$body" > "$TMP"
  t1=$(date +%s.%N)
  "$PY" - "$name" "$t0" "$t1" "$THREADS" "$TMP" <<'PY'
import json, sys
name, t0, t1, threads, path = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
raw = open(path, encoding="utf-8", errors="replace").read()
try:
    tm = json.loads(raw).get("timings", {})
except Exception as e:
    print(json.dumps({"name": name, "error": str(e), "raw": raw[:160]}))
    raise SystemExit
n, wall = tm.get("predicted_n", 0), t1 - t0
print(json.dumps({
    "name": name, "threads": threads, "predicted_n": n,
    "server_tps": round(tm.get("predicted_per_second", 0), 2),
    "server_ms_per_token": round(tm.get("predicted_ms", 0) / n, 4) if n else None,
    "wall_s": round(wall, 4),
    "client_tps": round(n / wall, 2) if wall > 0 else None,
}))
PY
}

echo "warmup"; run warmup "\"temperature\":0.0,\"samplers\":[\"temperature\"]" >/dev/null

: > "$OUT"
for cfg in "${CASES[@]}"; do
  name=${cfg%%|*}; extra=${cfg#*|}
  echo "--- $name ---"
  for _ in $(seq 1 "$REPS"); do
    R=$(run "$name" "$extra")
    echo "  $R"
    echo "$R" >> "$OUT"
  done
done

kill $SRV 2>/dev/null; trap - EXIT
echo; echo "written: $OUT"; echo

"$PY" - "$OUT" <<'PY'
import json, sys, statistics
from collections import defaultdict
rows = defaultdict(list)
for line in open(sys.argv[1], encoding="utf-8"):
    line = line.strip()
    if not line:
        continue
    r = json.loads(line)
    if "error" in r or r.get("predicted_n", 0) == 0:
        continue
    rows[r["name"]].append(r)

def med(name, key):
    v = [r[key] for r in rows.get(name, []) if r.get(key) is not None]
    return statistics.median(v) if v else None

base = med("greedy", "server_ms_per_token")
print(f"{'case':16} {'ms/token':>9} {'vs greedy':>10} {'t/s':>8} {'n':>4}")
for name in rows:
    ms, tps = med(name, "server_ms_per_token"), med(name, "server_tps")
    n = len(rows[name])
    delta = f"{ms - base:+.4f}" if (ms is not None and base is not None) else "-"
    print(f"{name:16} {ms:9.4f} {delta:>10} {tps:8.2f} {n:4d}")
if base:
    print(f"\nbaseline greedy = {base:.4f} ms/token; deltas are the price of the chain")
PY
