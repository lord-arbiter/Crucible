# Recipe: Google Gemini (OpenAI-compatible endpoint)

## When to use

You want the **cheapest** hosted multimodal of any major provider, with
the largest context window of any model in the table — Gemini 2.5 Flash
sits at roughly $0.075 / M input tokens, with native long-context
(1M tokens) support. Best for batch curation jobs at scale.

Google ships an OpenAI-compatible endpoint for Gemini at
`https://generativelanguage.googleapis.com/v1beta/openai`, so no proxy
is needed.

## Cost

Verify current pricing at <https://ai.google.dev/pricing>.

- Gemini 2.5 Flash: ~$0.075 / M input, ~$0.30 / M output
- Gemini 2.5 Pro: ~$1.25 / M input, ~$5 / M output

A 25-episode validation run on Flash costs **~$0.10**, on Pro **~$1**.
For a 1000-episode dataset, Flash is **~$4**. The cheapest path of any
hosted model in the same quality tier.

## Setup

1. Create / sign in at <https://aistudio.google.com>.
2. Get a free-tier API key (no credit card required for the free tier;
   billing kicks in past free quota).
3. Configure Crucible:

```bash
export CRUCIBLE_VLM_ENDPOINT=https://generativelanguage.googleapis.com/v1beta/openai
export CRUCIBLE_VLM_MODEL=gemini-2.5-flash
# or:
export CRUCIBLE_VLM_MODEL=gemini-2.5-pro
export CRUCIBLE_VLM_API_KEY=<your-google-ai-studio-key>
```

Then follow [hosted-api-quickstart.md](./hosted-api-quickstart.md)
sections 3–5.

## Known quirks

- **JSON schema** is supported on the OpenAI-compat endpoint as of
  Gemini 2.5; Crucible's fallback chain covers older variants.
- **Image format** is base64 data URLs in `image_url.url` — same as
  other OpenAI-compat hosts.
- **Free-tier rate limits** are tight (15 RPM on Flash, 5 RPM on Pro).
  The free tier is enough for a 5-episode smoke test; for 25+ episodes
  you want a paid project. Set `CRUCIBLE_PARALLEL_CRITICS=false` on
  free tier.
- **Thinking** ("extended thinking" on Gemini 2.5 Pro) defaults to
  enabled and adds latency; pass `extra_body={"thinking_budget": 0}`
  to disable. Crucible doesn't auto-disable it (we keep behavior
  predictable across providers); document this for your deployment.
- **Long context** (1M tokens) is overkill for our 16-image + 1k text
  workload, but useful if you extend Crucible to score 100+ frames per
  episode for very long teleops.

## Verification

```bash
python scripts/one_shot_test.py \
    --repo lerobot/aloha_static_cups_open \
    --critic visual \
    --frames 8
```

Pass: JSON parses, score ∈ [0, 10], verdict in
{EXCELLENT, ACCEPTABLE, MARGINAL, REJECT}, latency under 5 seconds on
paid tier (10–20 seconds on free tier due to rate-limit waits).

## Why Gemini Flash vs other backends

For pure cost-per-curated-episode, Flash is currently the best deal
among hosted multimodal providers. It's about half the cost of
Hyperbolic's Qwen3-VL pricing and an order of magnitude cheaper than
GPT-4o.

Quality on the strategy / safety axes (informal testing) is competent
but slightly less nuanced than GPT-4o or Claude. For curation pipelines
where the verdict is the deliverable (not the rationale), this is fine
— the rubric structure does most of the work.
