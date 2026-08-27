# Official EdgeTAM wrapper smoke

This is an actual official-checkpoint execution, not a mocked
predictor. It validates API/runtime integration only; the arbitrary
prompts have no ground-truth masks, so accuracy remains N/A.

- Source commit: `7711e012a30a2402c4eaab637bdb00a521302c91`
- Checkpoint SHA-256: `ed2d4850b8792c239689b043c47046ec239b6e808a3d9b6ae676c803fd8780df`
- Video: `/home/david/Desktop/laiting/itri/3d_safety_decision/realtime_3d_safety_decision/third_party/Video-Depth-Anything/assets/example_videos/Tokyo-Walk_rgb.mp4`
- Input frame shape: `640x360`
- PyTorch: `2.9.0+cu128`
- Device / precision: `cuda` / `bf16`
- Model load: `1870.280 ms`
- Three bounded-window inference calls: `751.883 ms` total (`3.990 calls/s`)
- Wrapper-reported inference latency: `469.938, 219.148, 61.905 ms`
- Peak CUDA allocated: `561.634 MiB`
- Peak CUDA reserved: `592.000 MiB`
- Optional upstream CUDA extension: `unavailable`

## Calls

- frame 0: IDs=[101, 202], modes={101: 'box_points+independent_state', 202: 'box_points+independent_state'}, rebuild=initial, window=1, mask_pixels={101: 7597, 202: 20260}
- frame 1: IDs=[101, 202], modes={101: 'box_points+independent_state', 202: 'box_points+independent_state'}, rebuild=rolling_window, window=2, mask_pixels={101: 4985, 202: 20269}
- frame 2: IDs=[202], modes={202: 'box_points'}, rebuild=object_set_changed, window=2, mask_pixels={202: 20167}

Pinned upstream EdgeTAM's grouped multi-object path raised a
non-contiguous `view()` error with this PyTorch build. The
wrapper did not patch model internals: it retried each ID in a
separate official predictor state and marked each prompt mode
with `+independent_state`. ROS diagnostics expose this as WARN/
degraded. This compatibility mode is slower than a native
grouped call.

The optional upstream `_C` extension was unavailable, so
EdgeTAM skipped mask-hole post-processing and emitted its
documented warning. Core model propagation still ran, but
boundary quality may differ from a CUDA-extension build.

The first call initializes two object IDs, the second propagates
the same object set over a two-frame rolling window, and the third
removes one ID through a safe object-set rebuild. These numbers are
not ROS end-to-end FPS/latency and must not be used as a controller
deadline without a synchronized RGB-D live benchmark.
