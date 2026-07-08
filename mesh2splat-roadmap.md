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

Division of labor — REVISED 2026-07-08 (CTO): the downstream `3dtiler/3dtiled`
does on-the-fly content conversion and exposes an OGC 3D Tiles endpoint from
arbitrary ALREADY-TILED inputs (SuperSplat SOG, XGRIDS LCC, Spark RAD chunks,
MapTiler geosplats, Potree/COPC point clouds, i3s/3MX meshes...). It does NOT
build hierarchies. **Building the LOD hierarchy is OUR job**: merge leaf
splats upward into downsampled interior nodes, up to the root, plus
`tileset.json`. Reference algorithms: Spark 2.0 Tiny-LoD (voxel-grid
bottom-up merge, training-free, fast) and Bhatt-LoD (pairwise
Bhattacharyya-similarity merge, higher quality, offline) —
https://www.worldlabs.ai/blog/spark-2.0 . SuperSplat build-lod is another
reference. Open question for CTO: node content format we should emit —
PLY per node (3dtiled converts on the fly?) or GLB KHR_gaussian_splatting
via splat-transform on our side.

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

## Offline resolution decoupling (2026-07-08, afternoon)

The live resolution combo triggers an immediate whole-model live conversion —
on a 3.4 GB model at 1:8192 that is a guaranteed freeze + driver-watchdog
crash (TDR). The offline converter used the same setting, so high-res offline
runs on big models were impossible. Fixed:

- Offline section has its own resolution dropdown (1024–8192); changing it
  never touches the live view. Live stays the viewfinder at safe resolution.
- Resolution is snapshotted at start() and used for BOTH the conversion grid
  and the PLY scale factor (scale = std / resolution) — a running export is
  immune to live UI changes (also fixes a latent mid-run corruption bug).
- Pre-run splat-count estimate in the UI: meshes x grid^2 upper bound + PLY
  GB ceiling. Actual count reported during/after the run and in the manifest
  (new `conversion_resolution` field).
- Stale-tile sweep: a tiled run now deletes leftover tile_*.ply +
  manifest.json from the target folder before writing (reruns into the same
  name used to leave orphan tiles that downstream globs would ingest).
  NOTE: sweep code untested yet (folder-with-leftovers rerun pending).
- Verified: 57.5M-splat run (full 8192 grid, quality 1.0, small mesh,
  5 leaves at level 2) TILES VALID; two independent runs produced the exact
  same set size. The old ~30M practical ceiling is gone — offline splat
  budget is disk-limited only.

## Next tasks

- [x] UI: optional user-provided root region (min_x, min_z, size) for the
      quadtree; default stays mesh-bbox-derived. Size snaps up to
      tileSize * 2^L; conversion refuses meshes outside the region;
      manifest gains `root_source: user-defined | mesh-bbox`.
- [ ] Phase L — LOD builder (`tools/build_lod_tileset.py`): read manifest v2
      + leaf PLYs, build interior quadtree nodes bottom-up (Tiny-LoD-style
      voxel merge, moment-matched gaussian aggregation, ~4x reduction per
      level), write interior tile_L{l}_x{x}_y{y}.ply + explicit tileset.json.
      MVP: PLY node content, explicit tree (implicit tiling + bitstream
      later). Upgrade path: Bhatt-LoD-style pairwise merge for quality.
- [ ] Verify the stale-tile sweep (rerun into a folder with leftovers;
      expect "Swept N stale file(s)" and a clean folder).
- [ ] Aleppo (3.4 GB) end-to-end: live at 1024, offline at 8192 tiled.
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
