# Dataset Notes

Phase 1 supports `michaelmunje/SocialNav-SUB`, a Hugging Face dataset for VQA in social robot navigation scenes. The public dataset card describes scene-understanding prompts, odometry information, 3D human tracking estimates, and multiple human labels per prompt.

The adapter treats `prompts/<scenario_id>/sample_with_bev_*.png` files as a virtual frame sequence. If source timestamps are unavailable, it uses a configured virtual interval and records that choice in frame metadata instead of pretending the source has fixed FPS.

Default revision:

`f750caf46e5b33e6aef8c95af6a92fb4aff1d1b1`
