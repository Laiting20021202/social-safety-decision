# Docker Deployment

The dataset demo profile starts the Phase 1 browser GUI and dataset-service:

```bash
docker compose --profile dataset-demo up --build
```

Named volumes are used for caches and future outputs:

- `hf_cache`
- `dataset_cache`
- `experiment_outputs`
- `zone_configs`
- `sam3_weights`
- `robopoint_weights`
- `vqa_weights`

Datasets and model weights are not baked into images.
