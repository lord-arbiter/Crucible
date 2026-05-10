# Recipe: Azure VM (NCads_H100_v5 or NV-series)

## When to use

You're already on Azure (corporate AAD, existing subscription, free credits) and want a self-hosted GPU box. Azure has good NVIDIA coverage across budgets — entry-level NV-series for development, NC/ND series for production.

## Cost

Verify current pricing at <https://azure.microsoft.com/en-us/pricing/details/virtual-machines/linux/>.

Recommended SKUs (May 2026 retail in `eastus`):

| SKU | GPU | VRAM | Approx $/hr |
|---|---|---|---|
| `Standard_NV4as_v4` | 1× AMD Radeon (no NVIDIA) | 4 GB | ~$0.18 — too small, skip |
| `Standard_NC6s_v3` | 1× V100 | 16 GB | ~$3.06 |
| `Standard_NC24ads_A100_v4` | 1× A100 | 80 GB | ~$3.67 |
| `Standard_NCads_H100_v5` | 1× H100 NVL | 94 GB | ~$6.98 |
| `Standard_ND96isr_H100_v5` | 8× H100 SXM | 8×80 GB | ~$98 (overkill) |

Recommended for Crucible v0.1: **`Standard_NC24ads_A100_v4`** (1× A100 80 GB, ~$3.67/hr) — fits Qwen3-VL-32B comfortably and has the most reliable Azure stock.

## Setup

### 1. Pick a region with the SKU

Not every region carries every GPU SKU. Check availability:

```bash
az vm list-skus --location eastus --resource-type virtualMachines \
  --query "[?contains(name, 'NCads_A100')].name" -o tsv
```

`eastus`, `westus2`, `southcentralus`, `westeurope`, `northeurope`, `japaneast` typically have A100 stock. H100 is more limited.

### 2. Provision via the CLI

```bash
RESOURCE_GROUP=crucible-rg
LOCATION=eastus
VM_NAME=crucible-gpu

# 1. Resource group
az group create --name $RESOURCE_GROUP --location $LOCATION

# 2. Create the VM with a Deep Learning image (NVIDIA driver + Docker
# + nvidia-container-toolkit pre-installed).
az vm create \
  --resource-group $RESOURCE_GROUP \
  --name $VM_NAME \
  --image microsoft-dsvm:ubuntu-2204:2204:latest \
  --size Standard_NC24ads_A100_v4 \
  --admin-username azureuser \
  --ssh-key-values ~/.ssh/id_ed25519.pub \
  --os-disk-size-gb 200 \
  --public-ip-sku Standard

# 3. Open the inbound ports we need (only to your public IP)
MY_IP=$(curl -s https://api.ipify.org)
az vm open-port --resource-group $RESOURCE_GROUP --name $VM_NAME --port 8000 --priority 1010 --source-address-prefixes ${MY_IP}/32
az vm open-port --resource-group $RESOURCE_GROUP --name $VM_NAME --port 8001 --priority 1011 --source-address-prefixes ${MY_IP}/32

# 4. Get the public IP
az vm show -d --resource-group $RESOURCE_GROUP --name $VM_NAME --query publicIps -o tsv
```

The Microsoft Data Science Virtual Machine for Linux Ubuntu 22.04 image (`microsoft-dsvm:ubuntu-2204:2204:latest`) ships with NVIDIA drivers, Docker, and `nvidia-container-toolkit`. If you'd rather use a vanilla Ubuntu, see the [bare Ubuntu fallback](./cloud-gpu-vm.md#bare-ubuntu-fallback) section.

### 3. SSH in and run the container

```bash
ssh azureuser@<public-ip>
nvidia-smi   # confirm A100 visible

git clone https://github.com/lord-arbiter/Crucible
cd Crucible
docker build -f docker/Dockerfile.cuda -t crucible:cuda .

export HF_TOKEN=hf_...
docker run --rm --gpus all \
    -p 8000:8000 -p 8001:8001 \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    -e HF_TOKEN \
    -e CRUCIBLE_VLM_MODEL=Qwen/Qwen3-VL-32B-Instruct \
    -e VLLM_MAX_LEN=65536 \
    -e VLLM_GPU_UTIL=0.90 \
    crucible:cuda
```

For the smaller / cheaper path, swap `Qwen3-VL-32B-Instruct` for `Qwen3-VL-8B-Instruct` and `VLLM_MAX_LEN=32768`.

### 4. Verify and smoke test

Same as [cloud-gpu-vm.md](./cloud-gpu-vm.md) sections 6–7.

### 5. Stop the VM when done

```bash
az vm deallocate --resource-group $RESOURCE_GROUP --name $VM_NAME
# Or to delete entirely:
# az group delete --name $RESOURCE_GROUP --yes
```

`stop` keeps the VM allocated and still bills compute. `deallocate` releases the compute (only storage costs continue).

## Known quirks

- **Subscription limits.** Default Azure subscriptions cap GPU vCPU at 0 in many regions. Request a quota increase via the Azure portal → Subscriptions → Usage + quotas before provisioning, or you'll get `OperationNotAllowed`.
- **Spot pricing.** GPU SKUs are available as Spot at 60–80% discount but get evicted with 30-second notice when capacity is reclaimed. Useful for offline batch curation, not for live demos.
- **Public IP.** Default SKU is Basic; we explicitly set Standard above (better SLA, charged at $0.005/hr).
- **DSVM image** is updated weekly. If Crucible's container build fails on driver / CUDA mismatch, pick `:latest` for the freshest, or pin a known-good version like `:22.04.20260301` from the marketplace.
- **NCads_H100_v5** is the H100 NVL variant (94 GB, 80 GB usable for vLLM after KV reservation) — fine for Qwen3-VL-32B at 65k context but tight for 72B.

## Verification

```bash
nvidia-smi                              # A100 80 GB or H100 94 GB
curl -s http://localhost:8001/v1/models # served-model-name appears
python scripts/one_shot_test.py --repo lerobot/aloha_static_cups_open --critic visual
```

Pass: JSON parses, score ∈ [0, 10], total round-trip <10 seconds.

## When to use Azure SKU X vs other clouds

- **Crucible-only batch curation, willing to wait**: Spot NCads_A100_v4 at ~$1/hr is the cheapest A100 80 GB anywhere.
- **Live Space backed by always-on instance**: Standard NC24ads_A100_v4 at ~$3.67/hr is comparable to AWS p4de but with better availability.
- **You need 8× H100 for tensor-parallel 72B / 235B**: Azure ND96isr at ~$98/hr is on par with AWS p5; pick whichever has stock.
- **Other**: see the [universal cloud-gpu-vm recipe](./cloud-gpu-vm.md).
