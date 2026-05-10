# Compute options

Crucible runs against any OpenAI-compatible Qwen3-VL endpoint. This
doc surveys the realistic options across cost, control, and quality
tiers. For provisioning details see the per-option recipes in
[`docs/recipes/`](recipes/).

## Hosted multimodal APIs (no GPU to manage)

| Provider | Models | Pricing (per M input tok) | Recipe |
|---|---|---|---|
| OpenAI | GPT-4o, GPT-4o-mini | ~$2.50 / ~$0.15 | [openai-gpt4o.md](recipes/openai-gpt4o.md) |
| Anthropic (via LiteLLM proxy) | Claude Sonnet, Opus | ~$3 / ~$15 | [anthropic-claude.md](recipes/anthropic-claude.md) |
| Google AI Studio | Gemini 2.5 Flash, Pro | ~$0.075 / ~$1.25 | [google-gemini.md](recipes/google-gemini.md) |
| Hyperbolic | Qwen3-VL 72B / 32B | ~$0.20 | [hyperbolic.md](recipes/hyperbolic.md) |
| Together AI | Qwen3-VL family | ~$0.40 | [together-ai.md](recipes/together-ai.md) |
| DashScope (Alibaba) | qwen3-vl-plus / max (official) | Pay-as-you-go | [dashscope.md](recipes/dashscope.md) |

When to pick a hosted API:

- You're scoring fewer than ~100 episodes (cheaper than EC2).
- You don't want to manage a GPU box.
- You're validating Crucible end-to-end before committing to self-hosting.

## Self-hosted on cloud GPU

| Cloud | Instance | GPU | Best Qwen3-VL | $/hr | Recipe |
|---|---|---|---|---|---|
| AWS EC2 | `g6e.xlarge` | 1× L40S 48 GB | 8B fp16 | $1.86 | [aws-ec2.md](recipes/aws-ec2.md) |
| AWS EC2 | `g6e.12xlarge` | 4× L40S 48 GB | 32B tensor-parallel | $10.49 | (planned) |
| AWS EC2 | `p5.48xlarge` | 8× H100 80 GB | 32B / 72B | $98.32 | (overkill for v0.1) |
| AWS SageMaker | JumpStart Qwen3-VL endpoint | varies | varies | varies | [aws-sagemaker.md](recipes/aws-sagemaker.md) |
| AMD Cloud | MI300X | 1× MI300X 192 GB | 32B fp16, 65k context | free $100 credit / pay-as-you-go | [amd-mi300x.md](recipes/amd-mi300x.md) |
| RunPod community | MI300X | 1× MI300X | 32B fp16 | $0.50 | [amd-mi300x.md](recipes/amd-mi300x.md) |
| RunPod community | H100 PCIe 80 GB | 1× H100 | 32B fp16 (tight) | $1.99 | adapt aws-ec2.md |
| Hot Aisle | MI300X 1×/2×/4×/8× | MI300X | 32B / 72B | $1.99 | [amd-mi300x.md](recipes/amd-mi300x.md) |
| Lambda | H100 80 GB SXM | 1× H100 | 32B fp16 | $2.99 | adapt aws-ec2.md |

When to pick self-hosted:

- You're scoring more than ~100 episodes (per-call cost beats hosted
  APIs at scale).
- You need full control over vLLM flags (max-model-len, concurrency,
  guided decoding).
- You want a private deployment for proprietary teleop data.
- You want maximum throughput for batch curation.

## Local development

| Hardware | Model | Wall-time per critic | Recipe |
|---|---|---|---|
| Apple Silicon Mac (16+ GB) | Qwen3-VL-2B-4bit via MLX | ~30 s | [local-mac.md](recipes/local-mac.md) |
| NVIDIA workstation (24 GB+) | Qwen3-VL-8B via vLLM | ~1 s | adapt aws-ec2.md |

When to develop locally:

- You're iterating on Crucible code itself (prompts, schemas, I/O
  layer) and want fast feedback without cloud round-trips.
- You don't have cloud accounts set up yet.
- You're on a flight.

Quality caveat: Qwen3-VL-2B is meaningfully weaker than 8B/32B/72B.
Use local-Mac for development; validate against a larger model before
drawing conclusions about a curation result.

## Decision tree

```
Are you scoring fewer than 100 episodes?
├── Yes: Hosted API (Hyperbolic / Together / DashScope)
└── No, more:
    Do you have AMD MI300X access?
    ├── Yes: AMD MI300X (192 GB headroom for 32B at full context)
    └── No:
        Do you have AWS credits?
        ├── Yes: AWS EC2 g6e.xlarge (Qwen3-VL-8B)
        │       OR g6e.12xlarge (Qwen3-VL-32B tensor-parallel)
        │       OR SageMaker JumpStart (managed)
        └── No: RunPod community MI300X $0.50/hr
                OR Lambda H100 $2.99/hr
                OR Hyperbolic at scale (still cheaper than self-host
                   up to ~1000 episodes/month)
```

## Cost worked example

Curating a single LeRobot dataset (50 episodes) with Crucible at
default settings (16 frames per episode, 5 critics + aggregator):

- Hosted API: ~$0.30–$1.00 total spend, no infrastructure to manage.
- AWS EC2 g6e.xlarge: ~$2 (1 hour bring-up + 30 min curation).
- AMD MI300X via Dev Cloud credit: $0 (under the $100 free tier).

Curating a fleet (1000 episodes/month):

- Hosted API: ~$10–$50/month, scaling linearly.
- AWS EC2 g6e.xlarge running 24/7: $1340/month.
- AWS EC2 g6e.xlarge running 4 hr/day: $223/month.
- Self-hosted MI300X on Hot Aisle 4 hr/day: $239/month.

Self-hosting wins at high steady volume, hosted APIs win for bursty
or one-off workloads. Crucible doesn't care which you pick.

## Verifying current pricing

All numbers above are May 2026 estimates. Current source-of-truth:

- AWS EC2: <https://aws.amazon.com/ec2/instance-types/g6e/>
- AWS SageMaker: <https://aws.amazon.com/sagemaker/pricing/>
- RunPod: <https://www.runpod.io/pricing>
- Lambda: <https://lambdalabs.com/service/gpu-cloud>
- Hot Aisle: <https://hotaisle.xyz>
- Hyperbolic: <https://hyperbolic.xyz/pricing>
- Together AI: <https://www.together.ai/pricing>
- DashScope: <https://www.alibabacloud.com/help/en/model-studio/billing>
