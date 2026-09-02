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
build hierarchies, and our manifest+PLY format is NOT among its inputs.
**CTO decision (final): we build the full OGC 3D Tiles ourselves** — LOD
hierarchy from our leaves (merge upward to root) AND per-node GLB content
(KHR_gaussian_splatting; usually spz-compressed — spz_2 pending tooling),
plus tileset.json. Reference algorithms: Spark 2.0 Tiny-LoD (voxel-grid
bottom-up merge, training-free, fast) and Bhatt-LoD (pairwise
Bhattacharyya-similarity merge, higher quality, offline) —
https://www.worldlabs.ai/blog/spark-2.0 . SuperSplat build-lod is another
reference; splat-transform's --decimate (pairwise merging) is a candidate
quality upgrade for our interior-node downsampling.

DONE (2026-07-08 evening): `tools/build_lod_tileset.mjs` produces the full
tileset — interior nodes via voxel moment-matching (opacity x disc-area
weights; mesh splats are flat discs), explicit 3D Tiles 1.1 tileset.json
(REPLACE refinement), and with `--glb` converts every node (leaves +
interior) to KHR_gaussian_splatting GLB via @playcanvas/splat-transform
(npx, `-w`). Verified on the 117k site: 22 GLB nodes, all tileset URIs
resolve. Scale-proven same day: 56.1M splats merged to a 1.04M interior
node in ~108 s (sliced file I/O — Node caps single reads at 2 GiB, like
the GL driver); interior budget defaults to min(mean leaf, 1M); nodes
whose PLY exceeds ~1.6 GB skip GLB (32-bit container ceiling) and keep
a PLY URI with a loud warning. Visual QA done via tools/lod_viewer.html
(three.js point fallback; root node of the 117k site shows correct
footprint and colors; per-level f_dc means match). PRODUCTION RULE:
pick tile size so leaves stay under ~6M splats, then every node GLBs.
Not yet: spz_2 compression in GLB (splat-transform gap), Bhatt-quality
merging, implicit tiling, georef transform (local y-up coords for now),
real splat rendering in the QA viewer (Spark CDN import URL TBD).

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

## PBR persistence (2026-08-27)

PBR was computed per-gaussian on the GPU (converterFS.glsl writes
`pbr = vec4(metallic, roughness, 0, 1)` into `GaussianDataSSBO.pbr`) and used
by the live relighting renderer, but every disk writer dropped it on the
production/tiled path (`IncrementalPlyWriter` mirrored the 62-float standard
layout). Now optional end-to-end:

- `IncrementalPlyWriter::open(..., includePbr)` appends two float props,
  `metallicFactor` + `roughnessFactor`, AFTER the standard 62-prop block →
  64 floats / 256 B per splat. Indices 0..61 unchanged, so standard 3DGS
  viewers ignore the trailing props and `loadPlyFile` reads them back by name.
- `OfflineConverter` threads an `includePbr` flag into the single + per-tile
  writers; manifest gains `has_pbr` + `pbr_properties`, and `gaussian_format`
  becomes `3dgs-standard-ply+pbr`.
- UI: "Include PBR (roughness / metallic)" checkbox in the offline section,
  default ON (applies to single-file and tiled).
- `tools/build_lod_tileset.mjs` detects the layout from the first leaf (locks
  it, refuses a mixed set), mass-averages metallic/roughness into interior
  nodes (same opacity×disc-area weight as color/opacity). `--glb` warns loudly
  that KHR_gaussian_splatting has no material channels — PBR survives only in
  the PLY nodes.
- Python tools: `diagnose_ply.py` recognises the 64-prop PBR layout and reports
  metallic/roughness stats; `verify_tiles.py` reads stride from each header
  (62 or 64) so set-equality stays lossless with PBR.
- Verified: 62-prop regression clean (existing tiles + test.ply); synthetic
  64-prop tileset round-trips (verify TILES VALID, set-equality) and the LOD
  builder produces a 64-prop interior node with averaged metallic/roughness in
  [0,1]. STILL TODO: real GUI export smoke test (confirm a converted mesh with
  a metallicRoughness texture writes 64-prop tiles the tools accept).

## Split-on-import — path to 400M+ splats (2026-08-27)

Single welded mesh caps at ~grid² = 67M @ 8192, because grid placement is a
PER-MESH orthographic raster: converterGS sets gl_Position from each triangle's
3D position projected on its dominant plane, normalized by the per-mesh bbox
(ConversionPass sets u_bboxMin/Max from mesh.first.bbox). The UV atlas is NOT
used for placement (xatlas path dead; only original uv, for texture sampling).

Lever: "Split on import (NxN)" — `SceneManager::splitMeshesIntoGrid` buckets a
mesh's faces by XZ centroid into N×N sub-meshes (whole triangles, no cutting),
each with its own bbox → each sampled at the FULL grid → ceiling ~N²×resolution²
(3×3 ≈ 600M, enough for 400M). Sub-meshes keep the parent NAME (texture-map key)
so they share one texture upload; `loadTextures` dedups by name; setupMeshBuffers
now computes bbox per-mesh (fixed a latent running-union bug). Splat positions
are exact world coords → sub-mesh borders abut without seams (same as tiling).
UI slider 1–8 in the load section; applied at load. Composes with offline PBR +
quadtree tiling (gaussians bucket by world XZ regardless of sub-mesh).
400M @ 256 B PBR ≈ 102 GB tiled PLY; keep leaves <6M so per-node GLB stays valid.
STILL TODO: real GUI run at 3×3 to confirm the multiplication + a 400M end-to-end.

## SOG export (2026-08-27)

`tools/tiles_to_sog.mjs`: reads a tiled output folder's manifest.json and
converts each leaf PLY to a sibling `.sog` via `@playcanvas/splat-transform`
(v3.3.3, `.sog` is a direct output; GPU k-means compression). SOG is a compact
WEB-delivery format (SuperSplat / Spark / PlayCanvas), NOT a 3D Tiles content
type (that stays GLB) -- it is a parallel export of the same tiles. Flags:
`--nodes` also converts interior LOD node PLYs, `--gpu <n|cpu>`, `--dry-run`,
`--keep`, `--quiet`. Writes sog_manifest.json. PBR is DROPPED (SOG has no
material channels, like GLB) -- warned; PBR survives only in the PLY.
Proven on the 82M Aleppo run: one 27 MB / 110K-splat leaf -> 568 KB .sog
(~2% of PLY), read the 64-prop PBR PLY fine. Full-folder batch is the delivery
step. NOTE: current impl shells `npx` per tile (cold-start overhead ~2-3s x N);
if batches get large, resolve the splat-transform binary once.

## Folder-of-GLB-parts import — path to 400M input (2026-09-01)

RealityScan can't export a 400M-scale mesh as one file (GLB ~4 GB single-buffer
cap; glTF-separate .bin also uint32-capped; PLY/FBX/ABC = new heavy loaders +
huge single RAM load + PLY often lacks UV+texture). Solution: RS **"Save mesh by
parts"** → folder of small GLB parts (each geometry+UV+texture, well under the
limit), reusing RS's own spatial cut. STAGE 1 shipped: `SceneManager::
loadModelFolder(folder, splitFactor)` enumerates *.glb (sorted), parses each,
prefixes mesh names per part (textures keyed by name don't collide), appends all
into one scene, setupMeshBuffers + loadTextures once. UI button "Load folder of
GLB parts (RealityScan)" (nativeDialog::pickFolder existed); EventType::
LoadModelFolder mirrors LoadModel. Offline tiler buckets all parts into the world
quadtree. Composes with split-on-import + offline PBR.
LIMIT: Stage 1 loads ALL parts at once → all textures in VRAM together; may
exceed VRAM for hundreds of parts (true 400M). STAGE 2 (TODO): process parts
one-at-a-time into a shared tiled output (tile writers persist across parts) →
bounded VRAM, any scale.

FORMATS (2026-09-01): RS GLB-by-parts CRASHES (RS GLB exporter buggy at scale).
Added **assimp** (FetchContent v5.4.3, static, FBX+PLY+OBJ importers only) →
`SceneManager::parseMeshFileAssimp` fills utils::Mesh (pos/uv/normal/tangent +
material textures, embedded via scene->mTextures or external via stb_image, same
in-memory format as the GLB path). Routing: .glb → tinygltf, .fbx/.ply/.obj →
assimp — in both loadModel (single) and loadModelFolder (per part). getFileExt
maps fbx/obj → GLB(mesh) path; single .ply stays SPLAT (mesh PLY goes via the
folder importer, which treats .ply as mesh). Best RS export = **FBX binary,
embed textures** (self-contained like GLB, no 4 GB cap). Orientation: assimp may
bring FBX in Z-up; use the (viewer) transform / note if tiling looks vertical.

## Next tasks

- [x] Folder-of-GLB-parts import Stage 1 (2026-09-01). GUI test + Stage 2 pending.
- [ ] Stage 2: sequential per-part offline conversion into one shared tiled
      output (tile writers persist across parts) — the real 400M path.
- [x] Per-tile SOG via splat-transform (`tools/tiles_to_sog.mjs`, 2026-08-27).
- [x] Split-on-import (NxN sub-meshes) to break the single-mesh grid² ceiling
      toward 400M+ (2026-08-27). GUI run to confirm still pending.
- [x] PBR persistence: offline/tiled writer + LOD builder + tools carry
      metallicFactor/roughnessFactor (2026-08-27). GUI smoke test still pending.
- [x] UI: optional user-provided root region (min_x, min_z, size) for the
      quadtree; default stays mesh-bbox-derived. Size snaps up to
      tileSize * 2^L; conversion refuses meshes outside the region;
      manifest gains `root_source: user-defined | mesh-bbox`.
- [ ] Phase L — LOD builder (`tools/build_lod_tileset.mjs`): read manifest v2
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
