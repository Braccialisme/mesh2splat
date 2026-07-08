#!/usr/bin/env python3
"""
build_lod_tileset.py -- Phase L: quadtree LOD builder for mesh2splat tiles.

Reads a mesh2splat manifest v2 (scheme "quadtree-leaves") plus its leaf
PLY tiles, builds the interior quadtree levels bottom-up by merging and
downsampling splats (Tiny-LoD-style voxel aggregation, training-free),
and writes:

  - interior node PLYs  tile_L{l}_x{x}_y{y}.ply  for l = leafLevel-1 .. 0
  - an explicit 3D Tiles tileset.json (refine REPLACE, box bounding
    volumes, per-level geometric error)

The leaves are never modified -- they stay the lossless masters.

Merging math (moment matching): each splat is an anisotropic gaussian
with mass m = opacity * sx*sy*sz. A group of splats in one voxel becomes
a single gaussian with the group's mass-weighted mean center, combined
covariance  E[Cov_i + (p_i-mu)(p_i-mu)^T],  mass-weighted color/SH, and
mass-preserving opacity (clamped). Downsampling targets a roughly
constant splat budget per node, so each level has ~1/4 the splats of the
one below it.

Usage:
  python build_lod_tileset.py <tiles_dir> [--budget N] [--dry-run]

  --budget N   target splats per interior node (default: mean leaf count)
  --dry-run    report the plan, write nothing
"""

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

# Standard 3DGS PLY property order (62 float32 per record), as written by
# mesh2splat's IncrementalPlyWriter.
PROPS = (
    ["x", "y", "z", "nx", "ny", "nz"]
    + [f"f_dc_{i}" for i in range(3)]
    + [f"f_rest_{i}" for i in range(45)]
    + ["opacity"]
    + [f"scale_{i}" for i in range(3)]
    + [f"rot_{i}" for i in range(4)]
)
N_PROPS = len(PROPS)  # 62


# ----------------------------------------------------------------- PLY I/O

def read_ply(path):
    """Read a standard 3DGS PLY into an (N, 62) float32 array."""
    with open(path, "rb") as f:
        header = b""
        while not header.endswith(b"end_header\n"):
            chunk = f.readline()
            if not chunk:
                raise ValueError(f"{path}: truncated header")
            header += chunk
        lines = header.decode("ascii", "replace").splitlines()
        count = None
        names = []
        for ln in lines:
            if ln.startswith("element vertex"):
                count = int(ln.split()[-1])
            elif ln.startswith("property"):
                names.append(ln.split()[-1])
        if count is None:
            raise ValueError(f"{path}: no element vertex")
        if names != PROPS:
            raise ValueError(f"{path}: unexpected property layout ({len(names)} props)")
        data = np.fromfile(f, dtype="<f4", count=count * N_PROPS)
    if data.size != count * N_PROPS:
        raise ValueError(f"{path}: body shorter than header count")
    return data.reshape(count, N_PROPS)


def write_ply(path, arr):
    """Write an (N, 62) float32 array as a standard 3DGS PLY."""
    header = ["ply", "format binary_little_endian 1.0",
              f"element vertex {arr.shape[0]}"]
    header += [f"property float {p}" for p in PROPS]
    header.append("end_header")
    with open(path, "wb") as f:
        f.write(("\n".join(header) + "\n").encode("ascii"))
        arr.astype("<f4", copy=False).tofile(f)


# ------------------------------------------------------------- splat math

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def inv_sigmoid(y):
    y = np.clip(y, 1e-6, 1.0 - 1e-6)
    return np.log(y / (1.0 - y))


def quats_to_matrices(q):
    """(N,4) unnormalized (w,x,y,z) quaternions -> (N,3,3) rotation matrices."""
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    m = np.empty((q.shape[0], 3, 3), dtype=np.float64)
    m[:, 0, 0] = 1 - 2 * (y * y + z * z)
    m[:, 0, 1] = 2 * (x * y - w * z)
    m[:, 0, 2] = 2 * (x * z + w * y)
    m[:, 1, 0] = 2 * (x * y + w * z)
    m[:, 1, 1] = 1 - 2 * (x * x + z * z)
    m[:, 1, 2] = 2 * (y * z - w * x)
    m[:, 2, 0] = 2 * (x * z - w * y)
    m[:, 2, 1] = 2 * (y * z + w * x)
    m[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return m


def matrices_to_quats(m):
    """(N,3,3) rotation matrices -> (N,4) (w,x,y,z) quaternions, vectorized."""
    n = m.shape[0]
    q = np.empty((n, 4), dtype=np.float64)
    t = np.einsum("nii->n", m)  # trace
    # Four standard branches, selected per row by the largest diagonal term.
    c0 = t
    c1 = m[:, 0, 0]
    c2 = m[:, 1, 1]
    c3 = m[:, 2, 2]
    case = np.where(
        c0 > np.maximum(c1, np.maximum(c2, c3)) - 1e-9, 0,
        np.where(c1 >= np.maximum(c2, c3), 1,
                 np.where(c2 >= c3, 2, 3)))

    s = np.sqrt(np.maximum(t + 1.0, 1e-12)) * 2.0
    sel = case == 0
    q[sel, 0] = 0.25 * s[sel]
    q[sel, 1] = (m[sel, 2, 1] - m[sel, 1, 2]) / s[sel]
    q[sel, 2] = (m[sel, 0, 2] - m[sel, 2, 0]) / s[sel]
    q[sel, 3] = (m[sel, 1, 0] - m[sel, 0, 1]) / s[sel]

    s = np.sqrt(np.maximum(1.0 + m[:, 0, 0] - m[:, 1, 1] - m[:, 2, 2], 1e-12)) * 2.0
    sel = case == 1
    q[sel, 0] = (m[sel, 2, 1] - m[sel, 1, 2]) / s[sel]
    q[sel, 1] = 0.25 * s[sel]
    q[sel, 2] = (m[sel, 0, 1] + m[sel, 1, 0]) / s[sel]
    q[sel, 3] = (m[sel, 0, 2] + m[sel, 2, 0]) / s[sel]

    s = np.sqrt(np.maximum(1.0 - m[:, 0, 0] + m[:, 1, 1] - m[:, 2, 2], 1e-12)) * 2.0
    sel = case == 2
    q[sel, 0] = (m[sel, 0, 2] - m[sel, 2, 0]) / s[sel]
    q[sel, 1] = (m[sel, 0, 1] + m[sel, 1, 0]) / s[sel]
    q[sel, 2] = 0.25 * s[sel]
    q[sel, 3] = (m[sel, 1, 2] + m[sel, 2, 1]) / s[sel]

    s = np.sqrt(np.maximum(1.0 - m[:, 0, 0] - m[:, 1, 1] + m[:, 2, 2], 1e-12)) * 2.0
    sel = case == 3
    q[sel, 0] = (m[sel, 1, 0] - m[sel, 0, 1]) / s[sel]
    q[sel, 1] = (m[sel, 0, 2] + m[sel, 2, 0]) / s[sel]
    q[sel, 2] = (m[sel, 1, 2] + m[sel, 2, 1]) / s[sel]
    q[sel, 3] = 0.25 * s[sel]
    return q


def merge_groups(arr, group_ids, n_groups):
    """Moment-match each group of splats into one splat. Returns (G, 62)."""
    pos = arr[:, 0:3].astype(np.float64)
    nrm = arr[:, 3:6].astype(np.float64)
    sh = arr[:, 6:54].astype(np.float64)          # f_dc(3) + f_rest(45)
    alpha = sigmoid(arr[:, 54].astype(np.float64))
    sigma = np.exp(arr[:, 55:58].astype(np.float64))
    quat = arr[:, 58:62].astype(np.float64)

    # Weight by opacity x disc area (two largest axes): mesh2splat splats
    # are near-flat surface discs, so a 3D-volume weight would be ~zero
    # for every splat and collapse the weighting.
    s_sorted = np.sort(sigma, axis=1)          # ascending: [thin, mid, major]
    mass = alpha * s_sorted[:, 1] * s_sorted[:, 2]
    mass = np.maximum(mass, 1e-30)

    w_sum = np.bincount(group_ids, weights=mass, minlength=n_groups)

    def wmean(v):  # mass-weighted mean of per-splat columns
        out = np.empty((n_groups, v.shape[1]), dtype=np.float64)
        for c in range(v.shape[1]):
            out[:, c] = np.bincount(group_ids, weights=mass * v[:, c],
                                    minlength=n_groups)
        return out / w_sum[:, None]

    mu = wmean(pos)

    # Per-splat covariance R diag(sigma^2) R^T, plus center spread.
    rot = quats_to_matrices(quat)
    cov = np.einsum("nij,nj,nkj->nik", rot, sigma ** 2, rot)
    d = pos - mu[group_ids]
    cov = cov + d[:, :, None] * d[:, None, :]

    cov_sum = np.zeros((n_groups, 3, 3), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            cov_sum[:, i, j] = np.bincount(group_ids, weights=mass * cov[:, i, j],
                                           minlength=n_groups)
    cov_m = cov_sum / w_sum[:, None, None]

    # Eigendecompose -> new scales (sqrt of eigenvalues) + rotation.
    eigval, eigvec = np.linalg.eigh(cov_m)
    eigval = np.maximum(eigval, 1e-16)
    new_sigma = np.sqrt(eigval)
    # Right-handed frames only (quaternion needs det = +1).
    flip = np.linalg.det(eigvec) < 0
    eigvec[flip, :, 2] *= -1.0
    new_quat = matrices_to_quats(eigvec)

    # Opacity: mass-weighted mean of the children (mass-preserving
    # opacity = w_sum / volume breaks down for flat splats).
    alpha_mean = np.bincount(group_ids, weights=mass * alpha,
                             minlength=n_groups) / w_sum
    new_alpha = np.clip(alpha_mean, 1e-4, 1.0 - 1e-6)

    out = np.empty((n_groups, N_PROPS), dtype=np.float32)
    out[:, 0:3] = mu
    out[:, 3:6] = wmean(nrm)
    out[:, 6:54] = wmean(sh)
    out[:, 54] = inv_sigmoid(new_alpha)
    out[:, 55:58] = np.log(np.maximum(new_sigma, 1e-12))
    out[:, 58:62] = new_quat
    return out


def downsample(arr, target):
    """Voxel-merge an (N, 62) splat array down to ~target splats.

    Bisection on voxel size: occupied-voxel count decreases monotonically
    as voxels grow. Groups smaller than 2 pass through untouched.
    """
    n = arr.shape[0]
    if n <= target:
        return arr

    pos = arr[:, 0:3].astype(np.float64)
    lo, hi = pos.min(axis=0), pos.max(axis=0)
    extent = float(np.max(hi - lo))
    if extent <= 0:
        return arr[:1]

    # Bisect voxel size until occupied-cell count is within 15% of target.
    v_lo, v_hi = extent / 4096.0, extent
    best = None
    for _ in range(24):
        v = math.sqrt(v_lo * v_hi)  # geometric midpoint
        keys = np.floor((pos - lo) / v).astype(np.int64)
        flat = (keys[:, 0] << 42) | (keys[:, 1] << 21) | keys[:, 2]
        uniq, inv = np.unique(flat, return_inverse=True)
        cnt = uniq.size
        if best is None or abs(cnt - target) < abs(best[0] - target):
            best = (cnt, v, inv, uniq.size)
        if cnt > target * 1.15:
            v_lo = v          # too many cells -> larger voxels
        elif cnt < target * 0.85:
            v_hi = v          # too few cells -> smaller voxels
        else:
            break
    _, v, inv, n_groups = best
    return merge_groups(arr, inv, n_groups), v


# ------------------------------------------------------------- tree build

def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = [a for a in sys.argv[1:] if a.startswith("--")]
    if not args:
        print(__doc__)
        sys.exit(1)
    tiles_dir = Path(args[0])
    dry_run = "--dry-run" in flags
    budget = None
    for fl in flags:
        if fl.startswith("--budget"):
            budget = int(fl.split("=")[1] if "=" in fl else flags[flags.index(fl) + 1])

    manifest = json.loads((tiles_dir / "manifest.json").read_text())
    if manifest.get("scheme") != "quadtree-leaves":
        sys.exit("manifest is not scheme=quadtree-leaves (need manifest v2)")

    leaf_level = manifest["leaf_level"]
    leaf_size = manifest["leaf_size"]
    root = manifest["root"]
    leaves = {(t["x"], t["y"]): t for t in manifest["tiles"]}

    if budget is None:
        budget = int(np.mean([t["count"] for t in manifest["tiles"]]))
    print(f"leaf level {leaf_level}, {len(leaves)} leaves, "
          f"budget {budget:,} splats/interior node")

    # levels[l] = {(x, y): {"file": name, "count": int, "bbox": (min, max)}}
    levels = {leaf_level: {}}
    for (x, y), t in leaves.items():
        levels[leaf_level][(x, y)] = {
            "file": t["file"], "count": t["count"],
            "bbox": (np.array(t["bbox_min"]), np.array(t["bbox_max"])),
        }

    for l in range(leaf_level - 1, -1, -1):
        levels[l] = {}
        below = levels[l + 1]
        parents = sorted({(x // 2, y // 2) for (x, y) in below})
        for (px, py) in parents:
            children = [(x, y) for (x, y) in below
                        if x // 2 == px and y // 2 == py]
            total = sum(below[c]["count"] for c in children)
            fname = f"tile_L{l}_x{px}_y{py}.ply"
            if dry_run:
                est = min(total, budget)
                print(f"  L{l} ({px},{py}): {len(children)} children, "
                      f"{total:,} -> ~{est:,}   {fname}")
                bmins = np.min([below[c]["bbox"][0] for c in children], axis=0)
                bmaxs = np.max([below[c]["bbox"][1] for c in children], axis=0)
                levels[l][(px, py)] = {"file": fname, "count": est,
                                       "bbox": (bmins, bmaxs)}
                continue

            t0 = time.time()
            arr = np.concatenate([read_ply(tiles_dir / below[c]["file"])
                                  for c in children])
            res = downsample(arr, budget)
            merged, voxel = res if isinstance(res, tuple) else (res, 0.0)
            write_ply(tiles_dir / fname, merged)
            bmins = merged[:, 0:3].min(axis=0)
            bmaxs = merged[:, 0:3].max(axis=0)
            levels[l][(px, py)] = {"file": fname, "count": merged.shape[0],
                                   "bbox": (bmins, bmaxs)}
            print(f"  L{l} ({px},{py}): {total:,} -> {merged.shape[0]:,} "
                  f"(voxel {voxel:.3f}) in {time.time() - t0:.1f}s")

    if dry_run:
        print("dry run -- nothing written")
        return

    # ----- explicit tileset.json (3D Tiles 1.1, refine REPLACE) -----
    def node(l, x, y):
        e = levels[l][(x, y)]
        bmin, bmax = e["bbox"]
        center = (np.asarray(bmin) + np.asarray(bmax)) / 2.0
        half = np.maximum((np.asarray(bmax) - np.asarray(bmin)) / 2.0, 1e-3)
        box = [float(center[0]), float(center[1]), float(center[2]),
               float(half[0]), 0, 0, 0, float(half[1]), 0, 0, 0, float(half[2])]
        # Geometric error ~ merge resolution of this level; leaves are exact.
        g_err = 0.0 if l == leaf_level else leaf_size * (2 ** (leaf_level - l)) / 32.0
        n = {"boundingVolume": {"box": box},
             "geometricError": g_err,
             "refine": "REPLACE",
             "content": {"uri": e["file"]}}
        kids = [node(l + 1, cx, cy)
                for (cx, cy) in (levels.get(l + 1) or {})
                if cx // 2 == x and cy // 2 == y]
        if kids:
            n["children"] = kids
        return n

    root_node = node(0, 0, 0)
    tileset = {
        "asset": {"version": "1.1",
                  "generator": "mesh2splat build_lod_tileset.py"},
        "geometricError": root_node["geometricError"] * 2.0,
        "root": root_node,
        "extras": {
            "note": "Node content is standard 3DGS PLY in the source local "
                    "frame (y-up, no georeferencing); downstream tiler "
                    "converts content format and applies the manifest "
                    "transform when it is filled in.",
            "source_manifest": "manifest.json",
        },
    }
    (tiles_dir / "tileset.json").write_text(json.dumps(tileset, indent=1))

    total_interior = sum(e["count"] for l in range(leaf_level)
                         for e in levels[l].values())
    print(f"tileset.json written; {total_interior:,} interior splats across "
          f"{sum(len(levels[l]) for l in range(leaf_level))} nodes")


if __name__ == "__main__":
    main()
