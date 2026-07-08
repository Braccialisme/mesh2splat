# mesh2splat @ iconem — roadmap & decisions

## Current state (2026-07-08)

Branch `OfflineSplats_LargeScale`. Quadtree-leaf tiled output implemented and
verified:

- Offline converter emits quadtree **leaf** tiles on the XZ plane:
  root square = `tileSize * 2^L` covering the union bbox (origin at bbox min),
  leaves addressed `(level, x, y)` with non-negative indices,
  quadtree-y = world Z (3D Tiles 1.1 implicit-tiling convention).
- Files: `tile_L{L}_x{X}_y{Y}.ply` + `manifest.json` (v2, scheme
  `quadtree-leaves`, root bbox, leaf level, per-tile level/x/y/count/bbox).
- Verified lossless with `verify_tiles.py` (set equality against single-file
  reference): 117k gaussians and 29.1M gaussians, both TILES VALID.
- Single-file mode (tile size 0) still works — regression guard.

Division of labor: mesh2splat computes **leaves only** (lossless masters,
standard PLY). The downstream tiler (heritagewatch) builds interior LOD levels
by decimation/aggregation, the availability bitstream, `tileset.json`, and
packages each node as GLB + `KHR_gaussian_splatting`.

## CTO decisions (recorded 2026-07-08)

1. **Root region**: user-controllable square root bbox, so hierarchies from
   multiple datasets (e.g. one aerial survey + one focus area) can merge into
   a single geo scheme. Default when not provided: the extent bbox of the mesh
   being splatted (current behaviour).
2. **Octree option**: we scan very tall buildings; an option between implicit
   **quadtree (XZ)** and full **3D octree** for defining leaf tiles would be a
   game changer. Leaf tiles are then assembled into a single hierarchy by
   decimation/aggregation downstream.
3. **PLY → GLB packaging** (downstream, tiler team): use PlayCanvas
   `splat-transform`. It supports the `KHR_gaussian_splatting` glTF extension
   but **not** `KHR_gaussian_splatting_compression_spz_2`.

## Next tasks

- [ ] UI: optional user-provided root region (min_x, min_z, size) for the
      quadtree; default stays mesh-bbox-derived.
- [ ] Evaluate octree leaf mode (level, x, y, z addressing; manifest v3?).
- [ ] Phase B: SplatForge/Blender ceiling measurements
      (`phaseB-splatforge-protocol.md`), then manifest-driven frustum importer.
- [ ] Per-tile SOG via `splat-transform`; manifest georef fields with the
      heritagewatch tiler team.

## Spec context

`KHR_gaussian_splatting` is at Release Candidate (contributors: Cesium, Esri,
Niantic, NVIDIA, Huawei, Autodesk, Khronos). Splats stored as point primitives
(position/rotation/scale/opacity/SH as attributes) with graceful point-cloud
fallback. 3D Tiles + implicit tiling + GLB splat tiles is where the ecosystem
is heading.
