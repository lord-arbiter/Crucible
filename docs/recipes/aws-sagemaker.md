# Recipe: AWS SageMaker JumpStart

## When to use

You're already on AWS, you want a managed Qwen3-VL deployment without
managing EC2 instances or Docker, and you don't mind SageMaker's pricing
model. SageMaker JumpStart typically offers Qwen3-VL as a pre-configured
endpoint that auto-scales.

If you want full control over vLLM flags or to tune the GPU utilization,
use the [AWS EC2 recipe](./aws-ec2.md) instead.

## Cost

Verify pricing at <https://aws.amazon.com/sagemaker/pricing/>. SageMaker
real-time endpoints bill per instance-hour while running. Common GPU
instance for Qwen3-VL-8B is `ml.g5.2xlarge` (~$1.50/hr) or
`ml.g6e.2xlarge` (~$2.50/hr).

Importantly: SageMaker endpoints **bill while idle**. Stop the endpoint
when not in use.

## Setup

1. AWS Console → SageMaker → JumpStart.
2. Search "Qwen3-VL" and pick the variant you want (8B-Instruct
   recommended for cost; 32B-Instruct if your instance supports it).
3. Click **Deploy**. Pick instance type (`ml.g6e.2xlarge` is the sweet
   spot for 8B; `ml.g6e.12xlarge` for 32B).
4. Note the endpoint name and the deployment URL once it goes
   `InService` (~10 minutes from deploy).
5. SageMaker endpoints are not OpenAI-compatible by default. You have
   two options:
   - **Option A (recommended):** Deploy via JumpStart's "OpenAI-compatible"
     option if available — this exposes a standard `/v1/chat/completions`
     route directly.
   - **Option B:** Put an API Gateway in front of SageMaker that
     translates OpenAI-format requests to SageMaker's invoke format. AWS
     publishes a reference implementation; alternatively the
     [SageMaker LiteLLM proxy](https://docs.litellm.ai/docs/providers/aws_sagemaker)
     does the same job in one container.

```bash
# Once you have an OpenAI-compatible URL pointing at the endpoint:
export CRUCIBLE_VLM_ENDPOINT=https://<endpoint-id>.execute-api.<region>.amazonaws.com/v1
export CRUCIBLE_VLM_MODEL=Qwen/Qwen3-VL-8B-Instruct
export CRUCIBLE_VLM_API_KEY=<your-api-gateway-key>
```

Then follow [hosted-api-quickstart.md](./hosted-api-quickstart.md)
sections 3–5.

## Known quirks

- **Cold start** on a stopped endpoint is 5–10 minutes when AWS
  re-provisions the underlying instance. Plan accordingly.
- **`response_format=json_schema`** depends on whether JumpStart's
  serving image (typically TGI or DJL) supports it. If not, Crucible
  falls through to `json_object`.
- **Auto-scaling** can introduce latency spikes if you traffic spikes
  past the configured min instance count. Set min=1 for live demos.
- **Billing alarms.** SageMaker endpoints are easy to leave running.
  Set a CloudWatch alarm for `Invocations` going to zero for >24h.

## When to switch to EC2

If your workload is bursty (score-once, then idle for days), SageMaker
real-time endpoints waste money on idle hours. EC2 lets you stop the
instance and stop billing entirely; restart when needed.

If your workload is steady-state, SageMaker is fine and easier to
operate.
