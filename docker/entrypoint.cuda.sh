#!/usr/bin/env bash
# Boot vLLM serving Qwen3-VL on :8001, then the FastAPI orchestrator on :8000.
# CUDA build — see entrypoint.sh for the AMD ROCm equivalent.
#
# Tunables (override via -e on docker run):
#   CRUCIBLE_VLM_MODEL          (default Qwen/Qwen3-VL-8B-Instruct)
#   CRUCIBLE_VLM_SERVED_NAME    (default crucible-vlm)
#   VLLM_PORT                   (default 8001)
#   VLLM_MAX_LEN                (default 32768 — enough for 16 imgs at 768px)
#   VLLM_GPU_UTIL               (default 0.90)
#   VLLM_MAX_IMAGES_PER_PROMPT  (default 16)
#   VLLM_MAX_NUM_SEQS           (default 16)
#   VLLM_DTYPE                  (default bfloat16)
#   VLLM_TENSOR_PARALLEL        (default 1; raise for multi-GPU)

set -euo pipefail

mkdir -p /var/log/crucible

VLLM_MODEL="${CRUCIBLE_VLM_MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
VLLM_SERVED_NAME="${CRUCIBLE_VLM_SERVED_NAME:-crucible-vlm}"
VLLM_PORT="${VLLM_PORT:-8001}"
VLLM_MAX_LEN="${VLLM_MAX_LEN:-32768}"
VLLM_GPU_UTIL="${VLLM_GPU_UTIL:-0.90}"
VLLM_MAX_IMAGES="${VLLM_MAX_IMAGES_PER_PROMPT:-16}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-16}"
VLLM_DTYPE="${VLLM_DTYPE:-bfloat16}"
VLLM_TP="${VLLM_TENSOR_PARALLEL:-1}"

echo "[entrypoint] starting vLLM ${VLLM_MODEL} as '${VLLM_SERVED_NAME}' on :${VLLM_PORT}"

# Notes on the chosen flags:
#   --guided-decoding-backend xgrammar  — Qwen3-VL's `json_object` mode is
#     known to emit stray backticks; xgrammar with json_schema is the
#     reliable path (vLLM issue #18819).
#   --limit-mm-per-prompt.image  N      — modern dot syntax (vLLM >= 0.11).
#   --mm-processor-kwargs               — caps image patch budget so 768px
#     client-side images pass through untouched.
#   --trust-remote-code                  — required by Qwen3-VL processor.
vllm serve "${VLLM_MODEL}" \
    --host 0.0.0.0 \
    --port "${VLLM_PORT}" \
    --served-model-name "${VLLM_SERVED_NAME}" \
    --tensor-parallel-size "${VLLM_TP}" \
    --dtype "${VLLM_DTYPE}" \
    --max-model-len "${VLLM_MAX_LEN}" \
    --gpu-memory-utilization "${VLLM_GPU_UTIL}" \
    --limit-mm-per-prompt.image "${VLLM_MAX_IMAGES}" \
    --limit-mm-per-prompt.video 0 \
    --max-num-seqs "${VLLM_MAX_NUM_SEQS}" \
    --trust-remote-code \
    --guided-decoding-backend xgrammar \
    --mm-processor-kwargs '{"min_pixels": 200704, "max_pixels": 802816}' \
    --disable-log-requests \
    > /var/log/crucible/vllm.log 2>&1 &
VLLM_PID=$!

echo "[entrypoint] vLLM pid ${VLLM_PID}; waiting for /v1/models on :${VLLM_PORT}"
ready=0
for i in $(seq 1 120); do
    if curl -fs "http://localhost:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
        echo "[entrypoint] vLLM is up after ${i} attempts"
        ready=1
        break
    fi
    sleep 5
done
if [[ "$ready" != "1" ]]; then
    echo "[entrypoint] vLLM did not become ready; tailing logs and exiting"
    tail -n 200 /var/log/crucible/vllm.log || true
    exit 1
fi

cleanup() {
    echo "[entrypoint] shutting down"
    kill -TERM "${VLLM_PID}" 2>/dev/null || true
    wait "${VLLM_PID}" 2>/dev/null || true
}
trap cleanup SIGINT SIGTERM EXIT

echo "[entrypoint] starting FastAPI orchestrator on :${CRUCIBLE_API_PORT}"
exec uvicorn src.api:app --host "${CRUCIBLE_API_HOST}" --port "${CRUCIBLE_API_PORT}"
