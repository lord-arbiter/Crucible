# Recipe: Together AI

## When to use

You want a hosted Qwen3-VL endpoint from a well-established US provider
with strong reliability and a generous free trial. Together AI typically
offers the full Qwen3-VL family (8B, 32B, 72B variants) with the
OpenAI-compatible API.

## Cost

Verify current pricing at <https://www.together.ai/pricing>. Together's
historical pattern is roughly twice Hyperbolic's per-token rates but with
better reliability and longer trial credits ($5–$25 starting credit
common).

A full Crucible run on 25 episodes costs **~$1** at these rates.

## Setup

1. Create an account at <https://www.together.ai>.
2. Note any free trial credit on your dashboard.
3. Generate an API key under Settings → API Keys.
4. Confirm the Qwen3-VL model id you want from the model catalog.

```bash
export CRUCIBLE_VLM_ENDPOINT=https://api.together.xyz/v1
export CRUCIBLE_VLM_MODEL=Qwen/Qwen3-VL-32B-Instruct
export CRUCIBLE_VLM_API_KEY=<your-key>
```

Then follow [hosted-api-quickstart.md](./hosted-api-quickstart.md)
sections 3–5.

## Known quirks

- **Model id format** matches HuggingFace namespacing. Recent Together
  releases also support `meta-llama/...` and `Qwen/...` prefixes.
- **`response_format=json_schema`** has been supported on Together for
  most large models, but Qwen3-VL coverage may lag — Crucible's
  fallback chain handles it.
- **Streaming** is supported but Crucible doesn't use it for critic
  calls (we want the full response before parsing).
- **Concurrency** on the free trial is capped low (typically 5 RPS).
  If you hit 429s, set `CRUCIBLE_PARALLEL_CRITICS=false`.

## Verification

Same as the Hyperbolic recipe — run `scripts/one_shot_test.py` for one
critic and confirm valid JSON output.
