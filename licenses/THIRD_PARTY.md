# Third-Party Repositories, Models, and Data

This file records sources required by the project. Formal experiment runs must capture exact revision, license, download date, container digest, and hardware in `outputs/<run_id>/`.

## Dataset

| Name | URL | Default revision | License | Notes |
| --- | --- | --- | --- | --- |
| SocialNav-SUB | https://huggingface.co/datasets/michaelmunje/SocialNav-SUB | `f750caf46e5b33e6aef8c95af6a92fb4aff1d1b1` | MIT per dataset card | Phase 1 dataset playback target. |
| SocialNavSUB code | https://github.com/michaelmunje/SocialNavSUB | not pinned yet | check repository | Reference implementation and benchmark context. |

## Planned Model Integrations

| Name | URL | Status |
| --- | --- | --- |
| SAM 3 | https://github.com/facebookresearch/sam3 | Planned Phase 3; no formal inference in Phase 1. |
| RoboPoint | https://github.com/wentaoyuan/RoboPoint | Planned Phase 4; no formal inference in Phase 1. |
| Qwen3-VL-2B-Instruct | https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct | Planned Phase 5; access and revision must be validated before use. |
| SmolVLM2-500M-Video-Instruct | https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct | Planned Phase 5; access and revision must be validated before use. |

## License Risks

- Do not mix fixture or mock outputs with formal experiment metrics.
- Do not bake model weights or datasets into container images.
- Record any gated or unavailable model access as blocked in `STATUS.md` and the run report.
