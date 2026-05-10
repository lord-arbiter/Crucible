# Recipe: OpenAI GPT-4o

## When to use

You want the most reliable JSON-mode behavior of any backend — GPT-4o
honors `response_format=json_schema` strictly, which means Crucible's
fallback chain almost never kicks in. Best starting point if you're
evaluating Crucible itself rather than benchmarking models against
each other.

## Cost

Verify current pricing at <https://openai.com/api/pricing>. GPT-4o
historical pricing is roughly $2.50 / M input tokens, $10 / M output
tokens. GPT-4o-mini is ~$0.15 / M input, ~$0.60 / M output.

A 25-episode validation run on GPT-4o costs **~$2–4**; on GPT-4o-mini
costs **~$0.20**. Mini is fine for development and small datasets;
GPT-4o noticeably outperforms on the strategy + safety axes.

## Setup

1. Create / sign in at <https://platform.openai.com>.
2. Generate an API key under Settings → API keys.
3. Configure Crucible:

```bash
export CRUCIBLE_VLM_ENDPOINT=https://api.openai.com/v1
export CRUCIBLE_VLM_MODEL=gpt-4o
# or:
export CRUCIBLE_VLM_MODEL=gpt-4o-mini
export CRUCIBLE_VLM_API_KEY=sk-proj-<your-key>
```

Then follow [hosted-api-quickstart.md](./hosted-api-quickstart.md)
sections 3–5.

## Known quirks

- **JSON Schema is canonical here.** GPT-4o was the first OpenAI model
  to ship strict JSON schema honoring; the rest of the industry's
  json_schema implementations are modeled after this one. You'll see
  cleanest output of any backend.
- **Image input format** is identical to other OpenAI-compat hosts —
  base64 data URLs in `image_url.url`. Crucible's default works
  out-of-box.
- **Rate limits** scale with usage tier. New accounts cap at 500 RPM
  per model, which is plenty for sequential dataset scoring but may
  bite if you parallelize 5 critics × 5 episodes simultaneously. Set
  `CRUCIBLE_PARALLEL_CRITICS=false` if you hit 429s.
- **Image budget** at 768px max-dim is well under GPT-4o's 2048×2048
  hard limit; no client-side concerns.
- **Latency** is ~1–3 seconds per critic call on the OpenAI side.
  Faster than most self-hosted options.

## Verification

```bash
python scripts/one_shot_test.py \
    --repo lerobot/aloha_static_cups_open \
    --critic visual \
    --frames 8
```

Pass: JSON parses, score ∈ [0, 10], verdict in
{EXCELLENT, ACCEPTABLE, MARGINAL, REJECT}, latency under 5 seconds.

## Why GPT-4o vs hosted Qwen

| | GPT-4o | Hosted Qwen3-VL-72B |
|---|---|---|
| Per-million input tokens | $2.50 | ~$0.20 (Hyperbolic) |
| JSON schema reliability | excellent | good with xgrammar |
| Strategy axis quality (informal) | strong | strong |
| Latency | ~1–3 s | ~2–5 s |
| Open weights | no | yes |
| Privacy / on-prem path | no | self-host |

If cost dominates, pick Qwen3-VL hosted. If reliability dominates and
budget is loose, pick GPT-4o. Either passes the same tests.
