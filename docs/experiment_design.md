# Experiment Design

Formal ablations will compare:

- dataset metadata or rule baseline
- SAM 3 tracking plus geometry
- SAM 3 tracking plus VQA
- SAM 3 tracking plus RoboPoint zone plus geometry plus VQA
- oracle zone plus SAM 3 tracking plus geometry plus VQA
- manual zone plus SAM 3 tracking plus geometry plus VQA

Variables:

- time steps: 1, 3, 5, 8
- sampling interval: 0.2, 0.5, 1.0 seconds
- prediction horizon: 1.0, 2.0, 3.0 seconds
- zone source: RoboPoint, manual, dataset, oracle
- VQA model: Qwen, SmolVLM
- VQA input: raw frames, overlay frames, structured metadata combinations

No metric may be fabricated. Missing ground truth must produce unavailable metrics, not invented values.
