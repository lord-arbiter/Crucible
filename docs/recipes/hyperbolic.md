# Recipe: Hyperbolic

## When to use

You want a hosted Qwen3-VL endpoint with the cheapest per-token price
on the market. Hyperbolic typically hosts Qwen3-VL-72B-Instruct at
extremely low rates, with an OpenAI-compatible API and reasonable
latency.

Best for: free / low-budget experimentation, validating Crucible
end-to-end before deciding whether to self-host.

## Cost

Verify current pricing at <https://hyperbolic.xyz/pricing>. Historical
pattern is ~$0.20 per million input tokens, ~$0.40 per million output
tokens for Qwen3-VL — among the cheapest hosted Qwen3-VL anywhere.

A full Crucible run on 25 episodes costs **~$0.50** at these rates.

## Setup

1. Create an account at <https://hyperbolic.xyz>.
2. Add a credit card (most accounts get a small free starting credit).
3. Generate an API key in the dashboard.
4. Find Qwen3-VL in the model catalog — note the exact model id.

```bash
export CRUCIBLE_VLM_ENDPOINT=https://api.hyperbolic.xyz/v1
export CRUCIBLE_VLM_MODEL=Qwen/Qwen3-VL-72B-Instruct
export CRUCIBLE_VLM_API_KEY=<your-key>
```

Then follow [hosted-api-quickstart.md](./hosted-api-quickstart.md)
sections 3–5.

## Known quirks

- **Model id format** uses HuggingFace-style namespace (`Qwen/Qwen3-VL-72B-Instruct`).
  If you copy from a different provider's docs, change the format.
- **`response_format=json_schema`** support varies by model — Crucible's
  three-tier fallback handles it. If you see `PARSE_ERROR`, the
  fallback didn't kick in fast enough; lower `CRUCIBLE_CRITIC_TEMPERATURE`
  to 0.1.
- **Context length** for Qwen3-VL on Hyperbolic is typically 32k or
  128k. Crucible only needs ~30k for 16 images at 768px + telemetry
  digest — well within either limit.
- **Rate limits** are gentle for paid accounts but tight on free trial.
  Set `CRUCIBLE_PARALLEL_CRITICS=false` if you hit 429s.

## Verification

```bash
python scripts/one_shot_test.py \
    --repo lerobot/aloha_static_cups_open \
    --critic visual \
    --frames 8
```

Pass: JSON parses, score ∈ [0, 10], verdict in
{EXCELLENT, ACCEPTABLE, MARGINAL, REJECT}, latency under 10 seconds.

If you want to validate all five critics:

```bash
for c in visual kinematic task strategy safety; do
  python scripts/one_shot_test.py --repo lerobot/aloha_static_cups_open --critic $c
done
```
