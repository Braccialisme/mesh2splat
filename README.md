# Mesh2Splat — iconem fork

A fork of [electronicarts/mesh2splat](https://github.com/electronicarts/mesh2splat)
with fixes and major extensions for using it as a **mesh → 3D Gaussian Splat
converter** on large photogrammetry models.

This fork keeps the original tool's purpose (fast, deterministic mesh→splat
conversion — no training pipeline) and adds what production needed: **offline,
disk-streamed conversion with no VRAM ceiling**, and **spatially tiled output**
ready for Blender, web viewers, and 3D Tiles pipelines.

---

## The headline: offline conversion (no VRAM ceiling)

Upstream is architected around live rendering — every gaussian stays resident in
GPU memory, capping a conversion at roughly 20–30M gaussians on a 24 GB card.
This fork adds an **offline mode** ("Convert to disk (offline)" in the File
Selector panel) that converts the mesh in triangle-range batches through one
fixed ~768 MB GPU buffer, streaming each batch straight to disk. The ceiling
becomes disk space, not VRAM.

Why batching is lossless: every gaussian is emitted by exactly one triangle's
UV-space rasterization, and triangles don't influence each other's fragments —
so N batches produce the **exact same gaussian set** as one giant pass (verified
record-for-record at 117k, 461k, and 29.1M gaussians). Batches that overflow the
buffer are always detected (the atomic counter counts past the cap) and are
split in half and retried; nothing partial or corrupt is ever written. The PLY
is written incrementally, with the vertex count patched into the header at the
end.

**Doctrine:** *live view = viewfinder* (tune density/scale with instant
feedback; trustworthy to ~20M and loud about its limits beyond), *offline =
camera* (the production export path, any size).

## Tiled output (`Tile size` > 0)

Set **Tile size** (world units) above the offline button and the converter
buckets gaussians by ground-plane XZ grid cell — cell *(i,j)* spans
`x ∈ [i·ts, (i+1)·ts)`, likewise z, boundaries on multiples of the tile size so
the grid is geo-alignable. Output is a folder:

```
<name>_tiles/
  tile_-1_0.ply      # standard 3DGS PLYs, each loadable anywhere
  tile_0_0.ply       # (SuperSplat, SplatForge, splat-transform, ...)
  ...
  manifest.json
```

`manifest.json` carries the grid convention, per-tile file / gaussian count /
actual bounding box, totals, and `crs` / `transform` placeholders for
georeferencing by a downstream pipeline (e.g. a 3D Tiles tiler). `Tile size = 0`
keeps single-file output. Verified at production scale: a 1.5 GB GLB →
29,117,452 gaussians across 15 tiles (3.7 MB to 1 GB each), zero lost, zero
duplicated, every gaussian in its cell (`verify_tiles.py`).

Note: gaussian **order** inside files is GPU-nondeterministic (atomic append);
two runs contain the same set in different orders. Use the set-comparison tools
below, never byte-diff.

---

## Mesh input: GLB, FBX, PLY, OBJ — including RealityScan "by parts"

Load a single mesh (**Select file to load**) or a whole **folder of mesh parts**
(**Load folder of mesh parts**). Supported: **`.glb`** (glTF binary, tinygltf)
and **`.fbx` / `.ply` / `.obj`** (via **assimp**, statically linked — no DLL).

This is how a 400M-scale photogrammetry model gets in when a single GLB would
blow glTF's ~4 GB single-buffer limit: export from RealityScan with **"Save mesh
by parts"** (FBX binary recommended) and point mesh2splat at the folder — one
part → its own mesh, all bucketed into the offline quadtree.

- **Folder import** loads every `.glb/.fbx/.ply/.obj` in a folder as one scene. A
  **Filename filter** loads only files whose name contains a substring — use it to
  pick ONE LOD when a folder holds several (type e.g. `LOD0`).
- **Textures.** Embedded textures are read directly; external textures load from
  the file's folder. If the material has no texture link (common with
  RealityScan), sibling `*_diffuse.*` / `*albedo*` images are matched by name.
- **UDIM.** Multi-tile diffuse (`..._diffuse.1001.jpg`, `.1002`, …) is supported:
  the mesh is split by UV tile, UVs remapped to `[0,1]`, each tile textured.
- **Split on import (N×N)** subdivides each mesh into N×N sub-meshes at load, so
  each is sampled at the full resolution grid — raises the per-mesh splat ceiling.
- A new load **replaces** the previous scene (single mesh at a time; use the
  folder importer to combine parts).

Single `.ply` still loads as a **Gaussian-splat** file (viewer); a mesh `.ply`
goes through the folder importer (it treats `.ply` as a mesh).

Caveat: parts load all at once (their textures resident together) — fine for
moderate counts; hundreds of parts with big/UDIM textures can exceed VRAM
(sequential per-part conversion is the planned fix for that scale).

---

## Fixes over upstream

- **Portable shader loading.** Shaders load relative to the exe (upstream baked
  a compile-time absolute path); CMake copies `shaders/` (and `fonts/`) next to
  the binary. Zip `bin/Release` and it runs anywhere.
- **No more corrupt/empty exports above the buffer cap** — exported count is
  clamped to real capacity.
- **64-bit buffer sizing** — allocation math no longer overflows at 2 GB.
- **Finite opacity/scale in exports.** Fully opaque meshes used to export
  `opacity = +inf` on every splat (`invSigmoid(1.0)`); many viewers reject
  inf/NaN. Exports are now sanitized (finite logits, ~±13.8).
- **Sliced GPU readbacks.** Single `glGetBufferSubData` calls beyond ~2 GB fail
  silently on common drivers; all readbacks now go in 512 MB slices with error
  checks.
- **Honest live export.** The exporter queries the buffer's *real* size
  (`GL_BUFFER_SIZE`) and `GL_MAX_SHADER_STORAGE_BLOCK_SIZE`, clamps, and warns —
  it can no longer write "ghost" (empty) records when a GPU buffer resize failed
  silently under VRAM pressure. (The live single-SSBO path is hard-capped near
  22.4M gaussians by the driver's 2 GiB shader-storage block limit; that's what
  offline mode is for.)

## Robustness / UX

- **Native Windows file dialogs** (real Explorer picker) replacing the in-app
  file browser.
- **Live VRAM estimate** (green/red) under "Max quality tweak"
  (`GL_NVX_gpu_memory_info`, NVIDIA; "n/a" elsewhere).
- **Offline progress UI**: progress bar, live gaussian/tile counts, cancel
  (deletes partial output).
- Custom UI font (VG5000, `fonts/` next to the exe; falls back to default),
  movement-speed slider, graceful shader-load failure, tunable sort cap
  (`MAX_GAUSSIANS_TO_SORT`).
- **In-app Log window** — captures `cout`/`cerr` (there is no console): import
  diagnostics (materials, textures, UDIM tiles), tile counts, warnings.
  Text is selectable and has a **Copy all** button.

---

## Building (Windows)

Requirements: Visual Studio 2022 Build Tools, CMake.

```
cmake -B build -S .
cmake --build build --config Release
```

Output: `bin/Release/Mesh2Splat.exe` with `shaders/` and `fonts/` beside it.
Re-run the first (configure) command whenever source **files are added or
removed**; a plain build suffices for edits.

**Note:** `LNK1104 ... cannot open Mesh2Splat.exe` during rebuild = the app is
still running; close it.

## Deploying

Copy `bin/Release/` only (exe + `shaders/` + `fonts/`). Runs from any local,
mapped, or UNC path.

---

## Usage

1. Load a mesh (`.glb` / `.fbx` / `.obj`) or a **folder of parts**, tune
   **Sampling density** and **Max quality tweak**
   in the live view (watch the VRAM readout — the resolution grid is
   `16 + quality × (maxRes − 16)`, and gaussian count ≈ grid² per mesh/UV atlas).
2. Pick the output folder and filename, format "PLY Standard Format".
3. Small export (≲20M): "Save splat" (live path) is fine.
   Anything serious: **"Convert to disk (offline)"**, with **Tile size** set for
   tiled output (e.g. 128–300 world units for a ~1 km site).

## Verification toolkit (numpy)

- `diagnose_ply.py f.ply` — bbox, opacity/scale ranges, inf/NaN counts.
- `compare_ply.py a b` — byte-exact comparison (same-order files only).
- `compare_ply_unordered.py a b` — **same gaussian SET**, order ignored (the
  right tool for offline-vs-live or run-vs-run comparisons).
- `diff_ply_columns.py a b` — which fields differ, and whether by a constant.
- `ghost_map.py f.ply` — locates empty ("ghost") records; diagnoses stale-buffer
  truncation patterns.
- `verify_tiles.py <tiles_dir> [reference.ply]` — manifest consistency, per-cell
  containment, and union-vs-reference set equality.

## Known limits

- Live path: ~20M gaussians (driver SSBO block limit); it now warns instead of
  corrupting. Use offline mode.
- Tiled mode caps at 256 simultaneously open tiles (Windows file handles) —
  increase the tile size if hit.
- Offline conversion writes standard 3DGS PLY only (PBR/compressed formats:
  live exporter).

---

*Fork maintained by iconem. Upstream: Electronic Arts Inc. — see
`README_upstream.md`, `LICENSE.txt`, `NOTICE.txt`.*