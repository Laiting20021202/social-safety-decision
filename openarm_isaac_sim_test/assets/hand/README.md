# Hand asset

The Gazebo demo now uses the textured **LibHand 0.9 Human Hand Model** under
CC BY 3.0. Fetch and convert the upstream model locally with:

```bash
python3 scripts/fetch_libhand_asset.py
```

Generated files live in `assets/hand/libhand/` and remain untracked. The
upstream attribution and license are copied beside the mesh. Collision uses
separate hidden palm and forearm proxies.

The conversion script applies the official skeleton weights to a spread-hand pose before
export. It does not create a detector billboard or feed mesh labels to perception. The
MediaPipe/EdgeTAM path must still recognize the rendered RGB and recover metric depth.
