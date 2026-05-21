# PCA Deployment — Azure Red Hat OpenShift (ARO)

This folder contains Terraform and GitOps (ArgoCD) artifacts to deploy the
**Private AI Code Assistant** on **Azure Red Hat OpenShift (ARO)** with an
NVIDIA H100 GPU node for LLM inference.

---

## Architecture Overview

```
Developer (DevSpaces / OpenCode)
  │
  │  HTTPS (cluster-internal)
  ▼
Red Hat AI Gateway (llm-d)
  │  Gateway API + HTTPRoute
  ▼
KServe Predictor Service (port 80)
  │
  ▼
vLLM v0.19.0 (Custom ServingRuntime)
  │  OpenAI-compatible API on port 8000
  ▼
Qwen/Qwen3.6-35B-A3B-FP8
  │  FP8 quantized, 35B total / 3B active MoE
  ▼
NVIDIA H100 NVL 94GB HBM3
```

All client traffic flows through the Red Hat AI Gateway — DevSpaces and OpenCode
never access the model endpoint directly. This provides a stable endpoint, TLS
termination, and a path for future EPP-based intelligent routing.

---

## Component Versions

### Platform

| Component | Version |
|-----------|---------|
| Azure Red Hat OpenShift (ARO) | 4.20.15 |
| Kubernetes | v1.33.6 |
| RHCOS | 9.6.20260217-1 (Plow) |
| CRI-O | 1.33.9 |

### Operators

| Operator | Version | Channel |
|----------|---------|---------|
| Red Hat OpenShift AI (RHOAI) | 3.3.1 | stable |
| NVIDIA GPU Operator | 26.3.1 | v26.3 |
| Node Feature Discovery (NFD) | 4.20.0 | stable |
| Red Hat DevSpaces | 3.27.1 | stable |
| DevWorkspace Operator | 0.40.1 | fast |
| Red Hat OpenShift GitOps (ArgoCD) | 1.15.4 | latest |
| Red Hat OpenShift Serverless | 1.37.1 | stable |
| Red Hat Service Mesh | 3.3.3 | stable |

### GPU / NVIDIA Stack

| Component | Version |
|-----------|---------|
| NVIDIA Kernel Driver | 550.144.03 |
| CUDA Toolkit (in container) | 12.9 |
| CUDA Compat Libs | 575.57.08 |
| GPU Hardware | NVIDIA H100 NVL 94 GB HBM3 |

### AI / ML Stack

| Component | Version | Notes |
|-----------|---------|-------|
| **vLLM** | **0.19.0 (upstream)** | Custom ServingRuntime — see [Why Upstream vLLM](#why-upstream-vllm-v0190) |
| PyTorch | 2.10.0+cu129 | Bundled with vLLM v0.19.0 |
| Transformers | 4.57.6 | Required >=5.1 for Qwen3.6 |
| Model | Qwen/Qwen3.6-35B-A3B-FP8 | 35B total / 3B active MoE, FP8, 32K ctx |
| Serving | KServe RawDeployment | Via custom ServingRuntime |
| Gateway | llm-d (Red Hat AI Gateway) | Gateway API + HTTPRoute |

### IaC / CLI Tools

| Tool | Version |
|------|---------|
| Terraform | 1.9.8 |
| Azure CLI | 2.85.0 |
| oc CLI | 4.21.5 |

---

## Why Upstream vLLM v0.19.0

RHOAI 3.3.1 bundles `registry.redhat.io/rhaiis/vllm-cuda-rhel9` based on vLLM
~0.13 with `transformers <5.x`. The Qwen3.6-35B-A3B-FP8 model uses the
`Qwen3_5MoeForConditionalGeneration` architecture class, which requires:

1. **`transformers >=5.1`** — the tokenizer and config classes for Qwen3.5-MoE
   are not present in older versions
2. **`vLLM >=0.18`** — native support for the Qwen3.5-MoE architecture,
   including DeepGEMM FP8 MoE kernels and FlashAttention v3 on H100
3. **CUDA 12.9 toolkit** — vLLM v0.19.0 ships with PyTorch 2.10 compiled against
   CUDA 12.9. The host NVIDIA driver is 550 (CUDA 12.4), so
   `VLLM_ENABLE_CUDA_COMPATIBILITY=1` bridges the gap using CUDA compat
   libraries (575.57.08)

A **custom `ServingRuntime`** (`vllm-cuda-v0190`) is registered in RHOAI to
serve the model through the standard KServe RawDeployment path, making the
model visible and manageable through the OpenShift AI dashboard.

> **Note:** This custom runtime is unsupported by Red Hat. When RHOAI ships with
> vLLM >= 0.19 and transformers >= 5.1, switch back to the bundled runtime.

---

## Prerequisites

### Tools Required

| Tool | Version | Purpose |
|------|---------|---------|
| `terraform` | >= 1.4.6 | Infrastructure provisioning |
| `az` (Azure CLI) | >= 2.50 | Azure authentication and ARO management |
| `oc` (OpenShift CLI) | >= 4.19 | Cluster interaction and GitOps bootstrap |
| `jq` | >= 1.6 | JSON processing in the GPU MachineSet script |

### Azure Permissions Required

Your Azure account needs:

- **Contributor** or **Owner** on the target subscription
- **User Access Administrator** (for role assignments created by `az aro create`)

Register the ARO resource providers if not already registered:

```bash
az provider register --namespace Microsoft.RedHatOpenShift --wait
az provider register --namespace Microsoft.Compute --wait
az provider register --namespace Microsoft.Storage --wait
az provider register --namespace Microsoft.Authorization --wait
```

### GPU Quota

Request quota for `Standard_NC40ads_H100_v5` in your target region **before**
deployment. The H100 VM requires 40 vCPUs.

```bash
az vm list-usage --location australiaeast -o table | grep -i "NC40ads"
```

### Red Hat Prerequisites

- A **Red Hat account** with an active OpenShift subscription
- **Pull secret** from [console.redhat.com/openshift/install/pull-secret](https://console.redhat.com/openshift/install/pull-secret)

---

## Cluster Specifications

| Component | Specification |
|-----------|--------------|
| Platform | Azure Red Hat OpenShift (ARO) |
| OpenShift version | 4.20.15 |
| Azure region | Australia East (`australiaeast`) |
| Master nodes | 3x `Standard_D8s_v5` |
| Worker nodes | 3x `Standard_D8s_v5` |
| GPU nodes | 1x `Standard_NC40ads_H100_v5` (NVIDIA H100 NVL 94 GB) |
| Storage class | `managed-csi` (Azure Managed Disk CSI) |

---

## Deployment Steps

### Step 1: Authenticate with Azure

```bash
az login
az account set --subscription "<your-subscription-id>"
```

### Step 2: Configure Terraform Variables

```bash
cd PCA_Deployment_ARO/terraform/
cp terraform.tfvars.example terraform.tfvars
```

Edit `terraform.tfvars`:

| Variable | Description |
|----------|-------------|
| `subscription_id` | Your Azure subscription ID |
| `pull_secret` | Red Hat pull secret (single-line JSON string) |
| `cluster_name` | Cluster name (default: `aro-pca-aue`) |
| `location` | Azure region (default: `australiaeast`) |
| `gitops_repo_url` | Your fork of the `Private_AI_Coding_Assistant` repo |

### Step 3: Deploy Infrastructure with Terraform

```bash
terraform init
terraform plan -out=aro-plan.tfplan
terraform apply aro-plan.tfplan
```

Terraform provisions: Resource Group, VNet, Subnets, ARO Cluster (~35-45 min),
GPU MachineSet, OpenShift GitOps, and ArgoCD App-of-Apps.

### Step 4: Retrieve Cluster Credentials

```bash
az aro list-credentials --name aro-pca-aue --resource-group aro-pca-aue-rg
az aro show --name aro-pca-aue --resource-group aro-pca-aue-rg --query consoleProfile.url -o tsv
oc login <API_URL> --username=kubeadmin --password=<PASSWORD>
```

### Step 5: Verify Deployment

```bash
# Check all operators
oc get csv -A | grep -v Succeeded

# Check GPU node
oc get nodes -l nvidia.com/gpu.present=true

# Check model serving
oc get inferenceservice -n ai-serving
oc get servingruntime -n ai-serving

# Check AI Gateway
oc get gateway,httproute -n ai-serving

# Check DevSpaces
oc get devworkspace -A

# Test the model via AI Gateway
GATEWAY_SVC="llm-d-gateway-data-science-gateway-class.ai-serving.svc.cluster.local"
curl -sk https://${GATEWAY_SVC}/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3.6-35B-A3B-FP8",
    "messages": [{"role": "user", "content": "Write a Python hello world"}],
    "max_tokens": 100
  }'
```

---

## GitOps Structure

ArgoCD manages the platform in five sync waves:

```
PCA_Deployment_ARO/
├── terraform/                  # Azure infrastructure
│   ├── main.tf                 # RG, VNet, subnets, ARO cluster, GPU MachineSet
│   ├── gitops-bootstrap.tf     # OpenShift GitOps operator + App-of-Apps
│   ├── variables.tf            # Input variables with defaults
│   ├── versions.tf             # Provider versions
│   ├── outputs.tf              # Credential retrieval commands
│   └── terraform.tfvars.example
├── argocd/
│   ├── 00-app-of-apps.yaml            # Root ArgoCD application
│   ├── 01-operators/                   # Wave 1: Operator subscriptions
│   │   ├── subscriptions.yaml          #   RHOAI, Service Mesh, Serverless, GPU Op, DevSpaces
│   │   ├── nvidia-cluster-policy.yaml  #   GPU Operator ClusterPolicy
│   │   ├── leader-worker-set.yaml      #   LWS controller (llm-d dependency)
│   │   ├── lws-operator-cr.yaml        #   LWS operator instance
│   │   └── cert-manager.yaml           #   cert-manager (llm-d dependency)
│   ├── 02-platform-config/             # Wave 2: Platform configuration
│   │   ├── namespaces.yaml             #   ai-serving, dev1/2/3-devspaces
│   │   ├── datasciencecluster.yaml     #   DSC with KServe Managed mode
│   │   ├── checluster.yaml             #   DevSpaces CheCluster instance
│   │   ├── hf-token-placeholder.yaml   #   HuggingFace token secret
│   │   └── rbac.yaml                   #   RoleBindings for dev users
│   ├── 03-ai-serving/                  # Wave 3: AI serving stack
│   │   ├── llminferenceservice.yaml    #   PVC + HardwareProfile + ServingRuntime +
│   │   │                               #   InferenceService (KServe RawDeployment)
│   │   ├── llm-d-gateway.yaml          #   Gateway API Gateway + HTTPRoute
│   │   ├── pvcs.yaml                   #   100Gi model cache PVC (managed-csi)
│   │   └── tls-secret-job.yaml         #   Self-signed TLS cert for gateway
│   ├── 04-devspaces/                   # Wave 4: Developer workspaces
│   │   ├── devworkspaces.yaml          #   3x DevWorkspace with OpenCode config
│   │   ├── roo-code-configmaps.yaml    #   Roo Code provider config
│   │   └── vscode-extensions-config.yaml
│   └── 05-benchmarks/                  # Wave 5: Performance benchmarks
│       └── guidellm-sweep.yaml         #   GuideLLM sweep job
└── scripts/
    ├── create-gpu-machineset.sh        # Post-cluster H100 node provisioning
    ├── deploy-full-stack.sh            # Full stack deployment script
    ├── post-terraform-fullstack.sh     # Post-terraform automation
    └── validate.sh                     # Post-deployment validation
```

---

## Key Deployment Artifacts

### HardwareProfile (`nvidia-h100-gpu`)

Registered in `redhat-ods-applications`, makes the H100 GPU visible in the
RHOAI dashboard when deploying models. Defines CPU (4-16), Memory (40-120Gi),
and GPU (1x `nvidia.com/gpu`) resource bounds.

### Custom ServingRuntime (`vllm-cuda-v0190`)

| Field | Value |
|-------|-------|
| Image | `vllm/vllm-openai:v0.19.0` |
| Entrypoint | `python3 -m vllm.entrypoints.openai.api_server` |
| Model format | vLLM |
| Protocol | REST (OpenAI-compatible) |
| CUDA compat | `VLLM_ENABLE_CUDA_COMPATIBILITY=1` + `LD_LIBRARY_PATH` |
| Cache | PVC-backed (`/model-cache`) for persistent model weights and JIT kernels |
| Probes | Startup: 60min tolerance, Readiness: 10s, Liveness: 30s |

Environment variables handle non-root container constraints:

| Variable | Value | Purpose |
|----------|-------|---------|
| `HF_HOME` | `/model-cache` | HuggingFace cache on PVC |
| `TRITON_CACHE_DIR` | `/model-cache/triton-cache` | Triton MoE kernel cache on PVC |
| `XDG_CACHE_HOME` | `/model-cache/xdg-cache` | General cache on PVC |
| `HOME` | `/tmp` | Writable home for non-root user |
| `VLLM_ENABLE_CUDA_COMPATIBILITY` | `1` | Bridge CUDA 12.4 driver ↔ 12.9 toolkit |
| `LD_LIBRARY_PATH` | `/usr/local/cuda/compat:...` | Load CUDA compat libs (575.57.08) |

### InferenceService (`qwen36-vllm`)

- **Mode**: KServe RawDeployment (no Knative/Serverless dependency)
- **Runtime**: `vllm-cuda-v0190`
- **Model args**: `--model=Qwen/Qwen3.6-35B-A3B-FP8 --tensor-parallel-size=1 --max-model-len=32768`
- **Resources**: 8-16 CPU, 80-120Gi RAM, 1x NVIDIA GPU
- **Toleration**: `nvidia.com/gpu=present:NoSchedule`

### Model Cache PVC (`model-cache`)

100Gi `managed-csi` PVC stores HuggingFace model weights (~35GB), Triton JIT
kernels, and DeepGEMM warmup artifacts. Survives pod restarts to avoid
re-downloading the model and re-compiling kernels (~15 min saved on restart).

### AI Gateway (`llm-d-gateway`)

Gateway API `Gateway` with HTTPS listener (self-signed TLS) and `HTTPRoute`
routing all `/v1` traffic to the KServe predictor service.

**Cluster-internal endpoint:**
```
https://llm-d-gateway-data-science-gateway-class.ai-serving.svc.cluster.local/v1
```

---

## DevSpaces + OpenCode

Each developer workspace runs VS Code in the browser with OpenCode pre-configured
to use the private Qwen3.6 model through the AI Gateway.

| Config | Value |
|--------|-------|
| Provider | OpenAI-compatible (vLLM) |
| Base URL | `https://llm-d-gateway-data-science-gateway-class.ai-serving.svc.cluster.local/v1` |
| Model | `Qwen/Qwen3.6-35B-A3B-FP8` |
| API Key | `EMPTY` (no auth required for cluster-internal traffic) |
| TLS | Self-signed cert (`NODE_TLS_REJECT_UNAUTHORIZED=0`) |

---

## Benchmark Results

GuideLLM sweep results for Qwen3.6-35B-A3B-FP8 on H100 NVL:

| Workload | Prompt Tokens | Output Tokens | Peak Throughput (tok/s) | Sync Latency (s) | Sync TTFT (ms) |
|----------|--------------|---------------|----------------------|-------------------|----------------|
| Code Completion | 256 | 128 | 4,512 | 0.70 | 36 |
| Code Generation | 1,024 | 512 | 12,790 | 2.78 | 83 |
| Code Review | 4,096 | 1,024 | 16,133 | 5.59 | 157 |
| File Generation | 8,192 | 2,048 | 13,976 | 11.15 | 208 |

Full results: [`testresults_h100.md`](../testresults_h100.md)

---

## Scaling GPU Nodes

```bash
# Scale to 0 (stop GPU billing)
oc scale machineset <infra_id>-gpu-h100 -n openshift-machine-api --replicas=0

# Scale back to 1
oc scale machineset <infra_id>-gpu-h100 -n openshift-machine-api --replicas=1
```

---

## Destroying the Cluster

```bash
az aro delete --name aro-pca-aue --resource-group aro-pca-aue-rg --yes
az group delete --name aro-pca-aue-rg --yes --no-wait
```

---

## Troubleshooting

**vLLM pod stuck in startup (DeepGEMM warmup):**
First launch compiles ~2,785 DeepGEMM MoE kernels via JIT (~10-15 min).
Subsequent restarts are fast when using PVC-backed cache. The startup probe
allows up to 60 minutes.

**GPU node taint preventing pod scheduling:**
The H100 node has taint `nvidia.com/gpu=present:NoSchedule`. The InferenceService
includes the matching toleration. If deploying custom pods, add the toleration.

**CUDA driver version mismatch:**
The host runs NVIDIA driver 550 (CUDA 12.4) but vLLM v0.19.0 needs CUDA 12.9.
`VLLM_ENABLE_CUDA_COMPATIBILITY=1` and the compat libs (575.57.08) bridge this gap.
If you see CUDA errors, verify `LD_LIBRARY_PATH` includes `/usr/local/cuda/compat`.

**AI Gateway returns 503:**
Check that the KServe predictor pod is Ready and the HTTPRoute points to
`qwen36-vllm-predictor` service on port 80.

**Model download slow or failing:**
The model-cache PVC persists downloads across restarts. If HuggingFace is rate-limited,
set `HF_TOKEN` in the container environment or the `hf-token` secret.
