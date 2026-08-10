#!/usr/bin/env bash
# Накладные llama-server поверх движка.
#
# llama-bench меряет чистый декод. Пользователь видит сервер, а тот сверху
# кладёт токенизацию, сэмплирование, детокенизацию, HTTP и, если включён,
# стриминг по токену. Считаем и то, и другое, чтобы отделить движок от обвязки.
#
# Сервер сам сообщает predicted_per_second — это его внутренняя скорость
# декода. Разница с llama-bench есть цена сэмплирования и слотов. Разница
# между внутренней скоростью и часами клиента есть цена HTTP и стриминга.
set -uo pipefail

BIN=/workspace/llama.cpp-b10326/build/bin/llama-server
MODEL=${MODEL:-/workspace/models/Qwen_Qwen3.5-35B-A3B-Q4_0.gguf}
PORT=8080
OUT=${OUT:-/workspace/harness/results/server_overhead.json}
NPRED=512
TMP=/tmp/srv_resp.json

pkill -f "llama-server" 2>/dev/null; sleep 2
"$BIN" -m "$MODEL" --host 127.0.0.1 --port $PORT \
       -ngl 99 -fa 1 -ctk f16 -ctv f16 -c 4096 -np 1 -t 16 --no-warmup \
       > /workspace/server.log 2>&1 &
SRV=$!
for i in $(seq 1 150); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  sleep 2
done
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null || { echo "сервер не поднялся"; tail -20 /workspace/server.log; exit 1; }
echo "сервер поднят"

run() {                       # имя, json-параметры сэмплера, stream
  local name="$1" sampler="$2" stream="$3"
  local body="{\"prompt\":\"Recite the alphabet slowly.\",\"n_predict\":$NPRED,\"stream\":$stream,\"cache_prompt\":false,$sampler}"
  local t0 t1
  t0=$(date +%s.%N)
  if [ "$stream" = "true" ]; then
    # последний кадр SSE несёт timings; убираем префикс data:
    curl -sN -X POST "http://127.0.0.1:$PORT/completion" \
         -H 'Content-Type: application/json' -d "$body" \
      | grep -a '"timings"' | tail -1 | sed 's/^data: //' > "$TMP"
  else
    curl -s -X POST "http://127.0.0.1:$PORT/completion" \
         -H 'Content-Type: application/json' -d "$body" > "$TMP"
  fi
  t1=$(date +%s.%N)
  python3 - "$name" "$t0" "$t1" "$stream" "$TMP" <<'PY'
import json, sys
name, t0, t1, stream, path = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4], sys.argv[5]
raw = open(path, encoding="utf-8", errors="replace").read()
try:
    tm = json.loads(raw).get("timings", {})
except Exception as e:
    print(json.dumps({"name": name, "error": str(e), "raw": raw[:160]}, ensure_ascii=False))
    raise SystemExit
n, wall = tm.get("predicted_n", 0), t1 - t0
print(json.dumps({
    "name": name, "stream": stream == "true",
    "predicted_n": n,
    "server_tps": round(tm.get("predicted_per_second", 0), 2),
    "server_ms": round(tm.get("predicted_ms", 0), 1),
    "prompt_n": tm.get("prompt_n"),
    "prompt_tps": round(tm.get("prompt_per_second", 0), 1),
    "wall_s": round(wall, 4),
    "client_tps": round(n / wall, 2) if wall > 0 else None,
}, ensure_ascii=False))
PY
}

GREEDY='"temperature":0.0,"top_k":1,"top_p":1.0,"min_p":0.0,"repeat_penalty":1.0'
TYPICAL='"temperature":0.7,"top_k":40,"top_p":0.9,"min_p":0.05,"repeat_penalty":1.1'

: > "$OUT"
for cfg in "greedy_nostream|$GREEDY|false" "greedy_stream|$GREEDY|true" \
           "typical_nostream|$TYPICAL|false" "typical_stream|$TYPICAL|true"; do
  IFS='|' read -r name sampler stream <<< "$cfg"
  echo "--- $name ---"
  for rep in 1 2 3; do
    R=$(run "$name" "$sampler" "$stream")
    echo "  $R"
    echo "$R" >> "$OUT"
  done
done

kill $SRV 2>/dev/null
echo "записано: $OUT (jsonl)"
