# GPU access — decision tree for the AMD Hackathon 2026

The fastest path to an MI300X prompt today, with paid fallbacks if the
free path stalls. Submission deadline is **Sunday May 10, 12:00 PM PDT**.

---

## A. Free MI300X — ranked by speed-to-prompt

| Avenue | Latency | Cost | Notes |
|---|---|---|---|
| **AMD AI Developer Program → AMD Developer Cloud ($100 credit, ~50 hr MI300X)** | **5–10 min total**: sign-up has no approval gate; droplet provisions in **2–4 min** | Free up to $100 (~50 hr at $1.99/hr) | This is *the* path. Credits expire 30 days from account creation. |
| Lablab.ai hackathon participant credits | Same flow as above — the lablab page channels you to the AMD AI Developer Program | Free | No separate voucher portal verified; just join the program. |
| AMD Champions / Infinity Hub / ROCm community giveaways | **Days–weeks** (application + review) | Free | Useless for a 6-hour deadline. Skip. |

**Exact AMD Dev Cloud flow (verified May 2026):**

1. Go to **https://www.amd.com/en/developer/ai-dev-program.html** → click **Join** (no approval gate).
2. Inside the member portal → "Member Perks" → request **AMD Developer Cloud** credit ($100).
3. Land on **https://devcloud.amd.com** (DigitalOcean-powered), add an SSH key, click **Create GPU Droplet**, choose **1× MI300X / Ubuntu 24.04 + ROCm** image.
4. Droplet goes Creating → Active in **2–4 min**. SSH in, run `rocm-smi` to confirm.
5. `git clone https://github.com/lord-arbiter/Crucible && cd Crucible` and walk `docs/RUNBOOK.md`.

Phoronix's review and AMD's own getting-started guide both report sub-10-minute end-to-end. No queue, no approval call.

## B. If free fails: paid MI300X (instant on-demand)

1. **RunPod MI300X — $0.50/hr** (community), instant. Deploy: **https://console.runpod.io/deploy** → filter "MI300X". Lowest paid MI300X anywhere; provisioning sub-2-min from card-on-file.
2. **Hot Aisle — $1.99/hr** (1×/2×/4×/8× configs), on-demand. Deploy: **https://hotaisle.xyz** (signup → console). KYC step adds friction but reliable AMD-only host.

(Skip TensorWave — `/connect` is sales-gated, no self-serve checkout.)

## C. If MI300X is unobtainable: paid H100 80GB

The Crucible workload **fits** in 80 GB at fp16 with `--max-model-len 32768` (lose 2× context but five 4-image critics still comfortable). Drop `Qwen/Qwen3-VL-32B-Instruct` for `Qwen/Qwen3-VL-8B-Instruct` if you want extra headroom.

1. **RunPod H100 PCIe 80GB — $1.99/hr** (community) / $2.39/hr (secure). Deploy: **https://console.runpod.io/deploy** → "H100 PCIe". Provisions in <2 min.
2. **Lambda On-Demand H100 80GB SXM — ~$2.99/hr** (when stock available). Deploy: **https://cloud.lambdalabs.com/instances**. Stock is intermittent.

## D. Decision tree (deadline-aware)

```
NOW → +30 min:
  Sign up at amd.com/en/developer/ai-dev-program.html
  + claim $100 Dev Cloud credit
  + spin 1× MI300X droplet on devcloud.amd.com
  SSH in, rocm-smi, git clone Crucible, run io_smoke.

If droplet ACTIVE within 30 min → ride it to submission.
   Cost so far: $0. Credit ($100) covers ~50 hr; we need ~5.

If sign-up snags / credit not applied / droplet stuck >15 min → switch to:
  RunPod MI300X $0.50/hr  https://console.runpod.io/deploy
  $5 card top-up covers 10 hr.

If RunPod MI300X out of stock → RunPod H100 PCIe $1.99/hr (same console).
  Edit docker/Dockerfile.gpu BASE arg to vllm/vllm-openai-rocm:latest
  → wait, that's ROCm. For H100, use vllm/vllm-openai:latest (CUDA).
  Edit entrypoint.sh: drop AMD env vars (HSA_FORCE_*, VLLM_ROCM_*).
  Rest of Crucible code is GPU-agnostic.

Hard stop: be SSH'd and serving by deadline - 4h.
After that, packaging the demo / video / submission needs the time.
```

## E. Cost ceiling

If everything pays out at worst-case:

- AMD Dev Cloud (free, $100 credit) ≤ **$10** burn (5 hr at $1.99)
- RunPod MI300X $0.50/hr × 5 hr = **$2.50**
- RunPod H100 PCIe $1.99/hr × 5 hr = **$10**

Hard ceiling **<$15** even if every free path fails. Don't burn a card-of-record on Lambda SXM at $2.99/hr unless you're stuck.

## F. The moment you're SSH'd in

```bash
# 1. Confirm GPU is real
rocm-smi --showproductname        # expect MI300X line
rocm-smi --showmeminfo vram       # expect ~196,592 MB total
rocm-smi --showtemp --showpower    # idle temp <60°C, power <100W

# 2. Set up
sudo apt-get update && sudo apt-get install -y git docker.io
sudo systemctl start docker
sudo usermod -aG docker $USER     # log out and back in for this to take effect
git clone https://github.com/lord-arbiter/Crucible
cd Crucible

# 3. Walk the runbook
cat docs/RUNBOOK.md
```

Then proceed to `TESTING.md` Layer 0 (local sanity) → Layer 1 (vLLM health) → on through.

## Sources

- [AMD Developer Hackathon — lablab.ai](https://lablab.ai/ai-hackathons/amd-developer)
- [AMD AI Developer Program](https://www.amd.com/en/developer/ai-dev-program.html)
- [AMD Developer Cloud](https://www.amd.com/en/developer/resources/cloud-access/amd-developer-cloud.html)
- [AMD Dev Cloud getting-started guide](https://www.amd.com/en/developer/resources/technical-articles/2025/how-to-get-started-on-the-amd-developer-cloud-.html)
- [Phoronix: Trying Out the AMD Developer Cloud](https://www.phoronix.com/review/amd-developer-cloud)
- [MI300X Cloud Pricing — getdeploying.com](https://getdeploying.com/gpus/amd-mi300x)
- [RunPod Pricing](https://www.runpod.io/pricing)
- [Hot Aisle](https://hotaisle.xyz)
