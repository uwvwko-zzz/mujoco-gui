# Runtime MuJoCo maps

All terrain files in this directory are robot-independent MJCF scenes. The JQG
runtime extracts their assets and world geoms, prefixes names, and combines them
with the active robot model.

`google_barkour.xml` and `barkour_assets/` are a terrain-only extraction of the
Google DeepMind Barkour v0 course from `google-deepmind/mujoco_menagerie`. The
original Apache-2.0 license is preserved in `BARKOUR_LICENSE.txt`.

`gap_jump.xml`, `hurdles.xml`, `suspended_steps.xml`, `perlin_rough.xml`, and
`dynamic_obstacles.xml` are generated locally by `generate_extra_maps.py` and
have no robot include. Re-run the generator after changing course dimensions.
The runtime rearranges the boxes in `dynamic_obstacles.xml` whenever that map is
entered again after switching to another map.
