# Risk Register

| Risk | Impact | Mitigation |
| --- | --- | --- |
| SocialNav-SUB format changes | Dataset adapter can break | Pin dataset revision and support local mirror tests. |
| Missing timestamps in frame sequence | Time-based playback and prediction can be misleading | Preserve `original_timestamp=null` and mark virtual timestamps in metadata. |
| SAM 3 dependency conflicts | Full stack may not run in one process | Isolate SAM 3 service and image. |
| RoboPoint dependency conflicts | Full stack may not run in one process | Isolate RoboPoint service and image. |
| Gated or unavailable VQA weights | Formal smoke cannot run | Record blocking reason; never silently replace model. |
| Mock contamination in formal metrics | Invalid research results | Mark mock/fixture runs and reject them in formal experiment validation. |
| Unsafe continue under uncertainty | Safety-critical failure | Hard rules and fail-safe policy prevent continue from insufficient information. |
| Jetson memory constraints | Runtime failure | Provide Jetson-specific images and reduced-resolution guidance. |
