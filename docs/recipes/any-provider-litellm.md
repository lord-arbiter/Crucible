# Recipe: Any provider — universal LiteLLM gateway

## When to use

You want to drop in **any API key from any provider** and have Crucible
just work. No proxy setup, no endpoint URL hunting, no provider-specific
docs. Set `CRUCIBLE_VLM_MODEL=anthropic/claude-sonnet-4-5` and
`CRUCIBLE_VLM_API_KEY=sk-ant-...`, run, done.

This recipe is the cleanest path when:
- You want to A/B different providers without code changes
- The provider doesn't expose an OpenAI-compatible endpoint natively
  (Anthropic, AWS Bedrock, Google Vertex AI, Cohere, Replicate, ...)
- You're scoring across multiple providers in one pipeline

It works because Crucible bundles [LiteLLM](https://github.com/BerriAI/litellm)
as an optional transport. LiteLLM is the de-facto OSS translator
between the OpenAI chat-completions API surface and 100+ provider
native APIs. Crucible auto-routes to LiteLLM when the model id has a
recognized provider prefix.

## Install

The pip default install is slim — LiteLLM is opt-in:

```bash
pip install 'crucible-curation[universal]'
```

Or use the Docker images, which bake LiteLLM in by default:

```bash
docker build -f docker/Dockerfile.cuda -t crucible:cuda .  # NVIDIA
docker build -f docker/Dockerfile.gpu  -t crucible:rocm .  # AMD
```

## Quickstart per provider

### Anthropic Claude (native, no proxy)

```bash
export CRUCIBLE_VLM_MODEL=anthropic/claude-sonnet-4-5
export CRUCIBLE_VLM_API_KEY=sk-ant-...
# CRUCIBLE_VLM_ENDPOINT must be empty/unset.

python scripts/one_shot_test.py --repo lerobot/aloha_static_cups_open --critic visual
```

### AWS Bedrock

Bedrock uses AWS IAM credentials, not API keys. LiteLLM picks them up
from the standard AWS env / `~/.aws/credentials` chain:

```bash
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_REGION=us-east-1   # or wherever the model is deployed
export CRUCIBLE_VLM_MODEL=bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0

python scripts/one_shot_test.py --repo lerobot/aloha_static_cups_open --critic visual
```

### Google Vertex AI

Vertex uses gcloud Application Default Credentials. Authenticate once,
then:

```bash
gcloud auth application-default login
export VERTEXAI_PROJECT=your-gcp-project-id
export VERTEXAI_LOCATION=us-central1
export CRUCIBLE_VLM_MODEL=vertex_ai/gemini-2.5-pro

python scripts/one_shot_test.py --repo lerobot/aloha_static_cups_open --critic visual
```

### Google AI Studio (Gemini API key)

Simpler than Vertex — uses the Google AI Studio API key:

```bash
export CRUCIBLE_VLM_MODEL=gemini/gemini-2.5-flash
export CRUCIBLE_VLM_API_KEY=...   # from aistudio.google.com
```

### Cohere

```bash
export CRUCIBLE_VLM_MODEL=cohere/command-a
export CRUCIBLE_VLM_API_KEY=...
```

### Groq, Replicate, Mistral, Fireworks, DeepInfra, OpenRouter, xAI, ...

Same pattern — `<provider>/<model>` plus the provider's API key:

```bash
export CRUCIBLE_VLM_MODEL=groq/llama-3.2-90b-vision-preview
export CRUCIBLE_VLM_MODEL=mistral/pixtral-large-latest
export CRUCIBLE_VLM_MODEL=fireworks_ai/accounts/fireworks/models/llama-v3p2-90b-vision-instruct
export CRUCIBLE_VLM_MODEL=openrouter/anthropic/claude-sonnet-4
export CRUCIBLE_VLM_MODEL=xai/grok-4-vision
```

LiteLLM's full provider list: <https://docs.litellm.ai/docs/providers>.

## How transport selection works

When you run a Crucible scoring job, `src/critics.py:_get_transport(cfg)`
picks one of three transports based on the config:

```
                 ┌─ cfg.vlm_endpoint set? ─ yes ──▶ OpenAI-compat client
config.py /     │
.env.example ──┤  no, but cfg.vlm_model has a LiteLLM prefix? ──▶ LiteLLM
                │
                 └─ no, but cfg.vlm_model is a known OpenAI name? ──▶ OpenAI direct
```

The selection is logged at INFO on every job:

```
INFO  crucible.transport=litellm model=anthropic/claude-sonnet-4-5 api_key_set=True
```

If you set both an endpoint URL **and** a LiteLLM-prefixed model id,
the endpoint URL wins (OpenAI-compat path). This lets you point at a
LiteLLM proxy you're running yourself if you want explicit control.

## Provider prefixes Crucible recognizes

The full list (in `src/critics.py:LITELLM_PROVIDER_PREFIXES`):

```
anthropic/        bedrock/          codestral/        databricks/
azure/            bedrock_converse/ cohere/           deepinfra/
azure_ai/         cerebras/         cohere_chat/      deepseek/
fireworks_ai/     gemini/           groq/             huggingface/
mistral/          ollama/           openrouter/       perplexity/
replicate/        sagemaker/        together_ai/      vertex_ai/
watsonx/          xai/              cloudflare/
```

If a provider's prefix isn't here yet, send a PR adding it to the
tuple — LiteLLM probably already supports it.

## Cost across providers (May 2026 ranges, per million input tokens)

| Provider | Cheapest VLM | $/M input |
|---|---|---|
| Google AI Studio (Gemini Flash) | `gemini/gemini-2.5-flash` | ~$0.075 |
| Hyperbolic (OpenAI-compat) | `Qwen/Qwen3-VL-72B-Instruct` | ~$0.20 |
| Together AI | `Qwen/Qwen3-VL-32B-Instruct` | ~$0.40 |
| OpenAI | `gpt-4o-mini` | ~$0.15 |
| Anthropic | `anthropic/claude-3-5-haiku` | ~$0.80 |
| Anthropic | `anthropic/claude-sonnet-4-5` | ~$3 |
| AWS Bedrock | varies by model | passes through |
| Vertex AI | varies by model | passes through |

Crucible's three-tier output mode (`json_schema` → `json_object` →
unconstrained) handles each provider's structured-output quirks. You
don't need to pick a provider that supports `json_schema` specifically.

## Verification

Every provider should pass this smoke:

```bash
# 1. Set env vars per provider (see Quickstart above)
# 2. Single critic call
python scripts/one_shot_test.py \
    --repo lerobot/aloha_static_cups_open \
    --critic visual \
    --frames 8

# Expected: JSON with score in [0,10], verdict in {EXCELLENT, ACCEPTABLE,
# MARGINAL, REJECT}, non-empty rationale, ≥1 evidence item.
```

If JSON parsing fails on a provider, Crucible's three-tier fallback
should catch it and the verdict will be `PARSE_ERROR` rather than a
hard error. Check the orchestrator logs for the actual model output.

## Known quirks

- **Vision support varies.** Not every model on every provider supports
  multimodal input. LiteLLM emits a clear error when you try; Crucible
  surfaces it. Stick to vision-capable models for the critics; the
  aggregator is text-only and works on any model.
- **Anthropic native API** does not officially support
  `response_format=json_schema` at the top level (May 2026); LiteLLM
  works around this with prompt engineering. If you see prose-wrapped
  JSON from Claude, lower `CRUCIBLE_CRITIC_TEMPERATURE` to 0.1.
- **Bedrock model availability** varies by region. Pre-provision the
  model in your AWS account before scoring.
- **Vertex AI** requires `gcloud auth application-default login` and a
  project with the Vertex AI API enabled. This is a one-time setup.
- **Rate limits** are per-provider. If you hit 429s on parallel critic
  dispatch, set `CRUCIBLE_PARALLEL_CRITICS=false` to serialize the 5
  critic calls per episode (5× per-episode latency, but no rate-limit
  hits).

## When to use this vs. the OpenAI-compat path

| | LiteLLM path | OpenAI-compat path |
|---|---|---|
| Setup | `pip install [universal]` + env vars | endpoint URL + model + key |
| Dep weight | +120 MB | tiny |
| Anthropic native | yes | needs LiteLLM proxy |
| Bedrock / Vertex | yes | needs proxy |
| Self-hosted vLLM | works (use `openai/<model>` prefix) | preferred — direct |
| Maximum portability | yes | no |
| Maximum performance | slightly higher overhead | direct, lowest latency |

For development and provider experimentation: LiteLLM. For high-volume
production self-host: OpenAI-compat to your own vLLM.
