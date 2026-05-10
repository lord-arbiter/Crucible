# Recipe: DashScope (Alibaba's official Qwen API)

## When to use

You want the official Qwen3-VL endpoint from the model authors. DashScope
is Alibaba Cloud's managed API, OpenAI-compatible, and typically gets
new Qwen3-VL features (longer context, better JSON-mode support) before
third-party hosts.

## Cost

Verify current pricing at <https://www.alibabacloud.com/help/en/model-studio/billing>
or the equivalent Chinese-region page. Pricing typically tracks
Alibaba Cloud's usage tier — pay-as-you-go is common, with a free trial
of ~1M tokens for new accounts.

## Setup

1. Create an Alibaba Cloud account at <https://www.alibabacloud.com>
   (international) or the regional equivalent.
2. Activate Model Studio / DashScope from the console.
3. Generate an API key under Workspaces → API-KEY.

```bash
# International endpoint
export CRUCIBLE_VLM_ENDPOINT=https://dashscope-intl.aliyuncs.com/compatible-mode/v1
export CRUCIBLE_VLM_MODEL=qwen3-vl-plus
export CRUCIBLE_VLM_API_KEY=sk-<your-key>

# Or for the China-region endpoint:
# export CRUCIBLE_VLM_ENDPOINT=https://dashscope.aliyuncs.com/compatible-mode/v1
```

Then follow [hosted-api-quickstart.md](./hosted-api-quickstart.md)
sections 3–5.

## Known quirks

- **Model ids** are not the HuggingFace-style namespace — they're Alibaba's
  short names (`qwen3-vl-plus`, `qwen3-vl-max`, etc.). Check the
  official model list before setting `CRUCIBLE_VLM_MODEL`.
- **`enable_thinking`** is a Qwen3-specific flag. If outputs include
  `<think>...</think>` blocks, append `/no_think` to the user prompt
  (Crucible already does this in `src/critics.py`). For DashScope you
  may also need to pass `extra_body={"enable_thinking": False}`.
  Crucible's `extra_body` plumbing supports this — see
  `src/critics.py` for how to wire it.
- **`response_format=json_schema`** support is the most reliable here
  since Qwen authors test against their own API.
- **OpenAI compatibility** has small gaps: streaming format may differ
  slightly, but Crucible doesn't use streaming for critic calls.
- **Region matters.** International endpoint is reachable from outside
  China; the China endpoint requires a Chinese cloud account.

## Verification

```bash
python scripts/one_shot_test.py \
    --repo lerobot/aloha_static_cups_open \
    --critic visual \
    --frames 8
```

Pass: JSON parses, score ∈ [0, 10], verdict in
{EXCELLENT, ACCEPTABLE, MARGINAL, REJECT}.
