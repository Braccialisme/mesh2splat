# Mesh2Splat — iconem fork

A fork of [electronicarts/mesh2splat](https://github.com/electronicarts/mesh2splat)
with fixes and quality-of-life changes for using it as a **mesh → 3D Gaussian Splat
converter** on large photogrammetry models.

This fork keeps the original tool's purpose (fast, deterministic mesh→splat
conversion — no training pipeline) and makes it portable, more robust, and usable
on bigger models.

---

## What's different from the upstream repo

### Fixes
- **Portable shader loading.** Upstream resolved shader files from a compile-time
  absolute path (`__FILE__`), so the built `.exe` only ran on the machine it was
  compiled on. Now shaders are loaded relative to the executable, and a CMake
  post-build step copies the `shaders/` folder next to the `.exe`. The build is
  self-contained: zip `bin/Release` (exe + `shaders/`) and it runs anywhere.
- **Export no longer produces corrupt/empty `.ply` files above the buffer cap.**
  The gaussian count reported to the exporter is now clamped to the buffer's real
  capacity, so an over-cap conversion produces a valid (capped) file instead of a
  file full of zeros.
- **64-bit buffer sizing.** GPU buffer allocation used a 32-bit size calculation
  that overflowed past ~2 GB, silently allocating too-small buffers and corrupting
  large exports. The size math is now 64-bit, so the real limit is VRAM, not a
  32-bit integer.

### Robustness / UX
- **Live VRAM estimate** shown in the Properties panel under "Max quality tweak":
  displays estimated GPU memory need vs. free VRAM, in green (OK) or red (may
  corrupt). NVIDIA-only (uses `GL_NVX_gpu_memory_info`); shows "n/a" on other GPUs.
- **Graceful shader-load failure.** A missing shader file now shows a message box
  naming the problem instead of crashing silently.
- **Adjustable camera navigation speed.** A "Movement speed" slider (logarithmic,
  with Reset) was added to the Properties panel, for flying around large scenes.
- **Raised gaussian cap** (`MAX_GAUSSIANS_TO_SORT`) — tunable in
  `src/renderer/renderPasses/RenderPass.hpp`.

---

## Building (Windows)

Requirements: Visual Studio 2022 Build Tools, CMake.

```
cmake -B build -S .
cmake --build build --config Release
```

Output: `bin/Release/Mesh2Splat.exe` with a `shaders/` folder next to it.

**Note:** if you get `LNK1104 ... cannot open Mesh2Splat.exe` during a rebuild,
the app is still running — close it and rebuild.

## Deploying (e.g. to a NAS / for a colleague)

Copy only `bin/Release/` (the `.exe` + the `shaders/` folder). Nothing else is
needed at runtime — no `src/`, no source tree, no specific drive letter. It runs
from any local, mapped, or UNC path.

---

## Usage notes

- **Format:** export as "PLY Standard Format" for compatibility with SuperSplat,
  Blender importers, etc.
- **Density / quality:** "Sampling density" and "Max quality tweak" (resolution)
  together control how many gaussians are generated. Watch the live VRAM readout.

## Known limits

- **VRAM-bound output.** The tool is built around live rendering: all gaussians
  stay resident in GPU memory. On a 24 GB card the practical ceiling is roughly
  20–30M gaussians per conversion pass; pushing higher risks corruption (watch the
  VRAM readout).
- To exceed a single VRAM-load, convert the mesh in spatial regions and merge the
  resulting `.ply` files offline.
- Truly large (offline, disk-streamed) conversion would require decoupling
  conversion from live rendering — a larger architectural change, not yet done.

---

## Diagnosing a bad `.ply`

`diagnose_ply.py` (numpy) reports a file's bounding box, scale/opacity ranges, and
inf/NaN counts — useful to confirm an export is valid before loading it elsewhere.

```
python diagnose_ply.py yourfile.ply
```

A valid file has real (non-zero, varied) positions. All-zero positions or a
collapsed bounding box means the export was corrupt (see the export fixes above).

---

*Fork maintained by iconem. Upstream: Electronic Arts Inc.*