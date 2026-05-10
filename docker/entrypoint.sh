#!/usr/bin/env bash
# Launches vLLM serving Qwen3-VL on port 8001 and the FastAPI orchestrator on
# 8000. Logs go to /var/log/crucible.

set -euo pipefail

mkdir -p /var/log/crucible

VLLM_MODEL="${CRUCIBLE_VLM_MODEL:-Qwen/Qwen3-VL-32B-Instruct}"
VLLM_PORT="${VLLM_PORT:-8001}"
VLLM_MAX_LEN="${CRUCIBLE_MAX_MODEL_LEN:-65536}"
VLLM_GPU_UTIL="${VLLM_GPU_UTIL:-0.85}"

echo "[entrypoint] starting vLLM serving ${VLLM_MODEL} on :${VLLM_PORT}"
vllm serve "${VLLM_MODEL}" \
    --port "${VLLM_PORT}" \
    --max-model-len "${VLLM_MAX_LEN}" \
    --gpu-memory-utilization "${VLLM_GPU_UTIL}" \
    --trust-remote-code \
    > /var/log/crucible/vllm.log 2>&1 &
VLLM_PID=$!

echo "[entrypoint] vLLM pid ${VLLM_PID}; waiting for /v1/models on :${VLLM_PORT}"
for _ in $(seq 1 90); do
    if curl -fs "http://localhost:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
        echo "[entrypoint] vLLM is up"
        break
    fi
    sleep 5
done

cleanup() {
    echo "[entrypoint] shutting down"
    kill -TERM "${VLLM_PID}" 2>/dev/null || true
    wait "${VLLM_PID}" 2>/dev/null || true
}
trap cleanup SIGINT SIGTERM EXIT

echo "[entrypoint] starting FastAPI orchestrator on :${CRUCIBLE_API_PORT}"
exec uvicorn src.api:app --host "${CRUCIBLE_API_HOST}" --port "${CRUCIBLE_API_PORT}"
