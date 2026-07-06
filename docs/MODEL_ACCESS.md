# Model Access

The demo requires real model checkpoints. Metadata visibility on Hugging Face is
not enough; `make check-model-access` performs an actual small-file access probe.

## Required For Formal Demo

- SAM 3 image/concept segmentation: `facebook/sam3`
- SAM 3.1 multiplex video tracking: `facebook/sam3.1`
- Temporal VQA: `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`

## Optional / Hardware Dependent

- RoboPoint full checkpoint: `wentao-yuan/robopoint-v1-vicuna-v1.5-13b`
- RoboPoint LoRA: `wentao-yuan/robopoint-v1-llama-2-7b-lora`
- RoboPoint LoRA base: `meta-llama/Llama-2-7b-chat-hf`

RoboPoint 13B is not expected to fit comfortably on an 8 GB RTX 4060 without
quantization and/or offload. The LoRA path also requires access to the gated
LLaMA-2 base model.

## Current Access Result

Last checked: 2026-06-30

- `HF_TOKEN`: present for the latest check and authenticates successfully.
- `facebook/sam3`: blocked; the account is authenticated, but Hugging Face returns HTTP 403 asking to enable access to public gated repositories in the fine-grained token settings.
- `facebook/sam3.1`: blocked; the account is authenticated, but Hugging Face returns HTTP 403 asking to enable access to public gated repositories in the fine-grained token settings.
- `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`: downloaded to `outputs/hf_cache`.
- `meta-llama/Llama-2-7b-chat-hf`: blocked, gated access not granted.

The latest generated files are:

- `outputs/model_access.json`
- `outputs/model_manifest.json`

## User Action Required

1. Open the Hugging Face model pages while logged into the same account used by
   `HF_TOKEN`, then accept/request the required terms:
   - https://huggingface.co/facebook/sam3
   - https://huggingface.co/facebook/sam3.1
   - https://huggingface.co/meta-llama/Llama-2-7b-chat-hf if using RoboPoint LoRA
2. Wait until the model pages show that access has been granted.
3. Create or reuse a Hugging Face token with read access. For a fine-grained
   token, enable public gated repository access and include `facebook/sam3` and
   `facebook/sam3.1` in the token's repository permissions.
4. Export it before running the make targets:

```bash
export HF_TOKEN=hf_...
make check-model-access
make download-models
```

Do not commit `HF_TOKEN`. Do not mark SAM 3 image inference, SAM 3.1 tracking,
or RoboPoint LoRA execution as complete until checkpoint download and model load
have actually succeeded.
