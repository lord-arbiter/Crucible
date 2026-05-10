# Recipe: Anthropic Claude (via LiteLLM proxy)

## When to use

You want Claude's strong reasoning on the **strategy** and **safety**
axes — informal testing shows Claude Sonnet / Opus produce more nuanced
rationales for these axes than other model families. Cost sits between
GPT-4o and hosted Qwen.

Claude's native API is **not** OpenAI-compatible. Crucible's transport
layer expects OpenAI-compat, so we use [LiteLLM](https://github.com/BerriAI/litellm)
as a thin proxy that translates OpenAI requests → Claude API and
responses back. This adds one extra process to your stack but no
material latency.

## Cost

Verify current pricing at <https://www.anthropic.com/pricing>. Claude
Sonnet 4.5 is roughly $3 / M input tokens, $15 / M output tokens.
Opus 4.5 is ~5× more.

A 25-episode validation run on Claude Sonnet costs **~$3–5**. On Opus,
~$15–25 — overkill for dataset scoring; reserve for the most uncertain
edge cases.

## Setup

1. Create / sign in at <https://console.anthropic.com>.
2. Generate an API key under Settings → API Keys.
3. Install LiteLLM:

```bash
pip install 'litellm[proxy]'
```

4. Run the LiteLLM proxy locally:

```bash
export ANTHROPIC_API_KEY=sk-ant-<your-key>
litellm --model claude-sonnet-4-5 --port 4000
```

This exposes Claude through an OpenAI-compatible endpoint at
`http://localhost:4000/v1`.

5. Point Crucible at the proxy:

```bash
export CRUCIBLE_VLM_ENDPOINT=http://localhost:4000/v1
export CRUCIBLE_VLM_MODEL=claude-sonnet-4-5
export CRUCIBLE_VLM_API_KEY=sk-1234   # any non-empty string; LiteLLM doesn't auth on localhost by default
```

Then follow [hosted-api-quickstart.md](./hosted-api-quickstart.md)
sections 3–5.

For production you'd run the proxy as a Docker container with proper
auth — see <https://docs.litellm.ai/docs/proxy/quick_start>.

## Known quirks

- **Image format** through LiteLLM is base64 in `image_url.url` (the
  proxy translates to Claude's native `type: "image"` form internally).
- **`response_format=json_schema`** is supported in modern Claude API
  versions through the OpenAI-compat layer; Crucible's three-tier
  fallback handles the alternative.
- **Rate limits** are the proxy's combined limit on the Anthropic
  account. Concurrent critic dispatch (5 per episode) is fine on
  paid accounts.
- **Thinking mode** (Claude's extended-thinking feature) is OFF by
  default through LiteLLM — leave it off for critic calls. Crucible's
  prompts work better without it.

## Verification

```bash
python scripts/one_shot_test.py \
    --repo lerobot/aloha_static_cups_open \
    --critic strategy \
    --frames 12
```

Pass: JSON parses, verdict in
{EXEMPLARY, GOOD, MEDIOCRE, POOR}, rationale shows nuanced multi-clause
reasoning (this is where Claude shines vs. other models).

## Why Claude on the strategy / safety axes

The other model families (GPT-4o, Qwen3-VL, Gemini) produce competent
strategy critiques. Claude tends to produce ones that **explain more
carefully why** an episode is good or bad — useful when the rationale
field is the actual deliverable for human curators reviewing the kept
set. If your downstream use case is "human reads the rationale and
decides," Claude is the strongest pick on a quality basis. If your
use case is "score → threshold → push, no human review," GPT-4o or
Qwen3-VL is more economical.
