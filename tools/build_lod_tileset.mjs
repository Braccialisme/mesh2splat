#!/usr/bin/env node
/*
 * build_lod_tileset.mjs -- Phase L: quadtree LOD builder for mesh2splat tiles.
 *
 * Reads a mesh2splat manifest v2 (scheme "quadtree-leaves") plus its leaf
 * PLY tiles, builds the interior quadtree levels bottom-up by merging and
 * downsampling splats (Tiny-LoD-style voxel aggregation, training-free),
 * and writes:
 *
 *   - interior node PLYs  tile_L{l}_x{x}_y{y}.ply  for l = leafLevel-1 .. 0
 *   - an explicit 3D Tiles tileset.json (refine REPLACE, box bounding
 *     volumes, per-level geometric error)
 *   - with --glb: every node (leaves + interior) converted to GLB with
 *     KHR_gaussian_splatting via @playcanvas/splat-transform (npx), and
 *     tileset content URIs pointing at the .glb files
 *
 * The leaves are never modified -- they stay the lossless masters.
 *
 * Merging math (moment matching): each splat is weighted by
 * opacity x disc area (product of its two largest axes -- mesh-derived
 * splats are near-flat discs, a 3D-volume weight would collapse to zero).
 * A group of splats in one voxel becomes a single gaussian with the
 * group's weighted mean center, combined covariance
 * E[Cov_i + (p_i-mu)(p_i-mu)^T], weighted color/SH, and weighted-mean
 * opacity. Downsampling targets a roughly constant splat budget per node.
 *
 * Usage:
 *   node build_lod_tileset.mjs <tiles_dir> [--budget N] [--glb] [--dry-run]
 */

import { readFileSync, writeFileSync, existsSync, statSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { join } from "node:path";

// Standard 3DGS PLY property order (62 float32 per record), as written by
// mesh2splat's IncrementalPlyWriter.
const PROPS = [
  "x", "y", "z", "nx", "ny", "nz",
  ...Array.from({ length: 3 }, (_, i) => `f_dc_${i}`),
  ...Array.from({ length: 45 }, (_, i) => `f_rest_${i}`),
  "opacity",
  ...Array.from({ length: 3 }, (_, i) => `scale_${i}`),
  ...Array.from({ length: 4 }, (_, i) => `rot_${i}`),
];
const NP = PROPS.length; // 62

// ------------------------------------------------------------- PLY I/O

function readPly(path) {
  const buf = readFileSync(path);
  const headEnd = buf.indexOf("end_header\n") + "end_header\n".length;
  if (headEnd <= 0) throw new Error(`${path}: no end_header`);
  const head = buf.subarray(0, headEnd).toString("ascii");
  let count = null;
  const names = [];
  for (const ln of head.split("\n")) {
    if (ln.startsWith("element vertex")) count = parseInt(ln.split(/\s+/)[2]);
    else if (ln.startsWith("property")) names.push(ln.trim().split(/\s+/)[2]);
  }
  if (count === null) throw new Error(`${path}: no element vertex`);
  if (names.length !== NP || names.some((n, i) => n !== PROPS[i]))
    throw new Error(`${path}: unexpected property layout (${names.length} props)`);
  const bytes = buf.subarray(headEnd);
  if (bytes.length < count * NP * 4) throw new Error(`${path}: body too short`);
  // Copy to an aligned buffer so we can view it as Float32Array.
  const aligned = new Uint8Array(count * NP * 4);
  aligned.set(bytes.subarray(0, count * NP * 4));
  return { data: new Float32Array(aligned.buffer), n: count };
}

function writePly(path, data, n) {
  const head =
    ["ply", "format binary_little_endian 1.0", `element vertex ${n}`,
     ...PROPS.map((p) => `property float ${p}`), "end_header", ""].join("\n");
  const body = Buffer.from(data.buffer, data.byteOffset, n * NP * 4);
  writeFileSync(path, Buffer.concat([Buffer.from(head, "ascii"), body]));
}

// ------------------------------------------------------------- splat math

const sigmoid = (x) => 1 / (1 + Math.exp(-x));
const invSigmoid = (y) => {
  const c = Math.min(Math.max(y, 1e-6), 1 - 1e-6);
  return Math.log(c / (1 - c));
};

// Cyclic Jacobi eigensolver for a symmetric 3x3 matrix given as
// [xx, xy, xz, yy, yz, zz]. Returns { val: [3], vec: 3x3 column-major
// eigenvectors (vec[3*c + r]) }, eigenvalues ascending.
function eigh3(m) {
  let a = [
    [m[0], m[1], m[2]],
    [m[1], m[3], m[4]],
    [m[2], m[4], m[5]],
  ];
  let v = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  for (let sweep = 0; sweep < 12; sweep++) {
    let off = Math.abs(a[0][1]) + Math.abs(a[0][2]) + Math.abs(a[1][2]);
    if (off < 1e-15) break;
    for (let p = 0; p < 2; p++) {
      for (let q = p + 1; q < 3; q++) {
        if (Math.abs(a[p][q]) < 1e-18) continue;
        const theta = (a[q][q] - a[p][p]) / (2 * a[p][q]);
        const t = Math.sign(theta) / (Math.abs(theta) + Math.sqrt(theta * theta + 1)) || 1 / (theta + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;
        for (let k = 0; k < 3; k++) {
          const akp = a[k][p], akq = a[k][q];
          a[k][p] = c * akp - s * akq;
          a[k][q] = s * akp + c * akq;
        }
        for (let k = 0; k < 3; k++) {
          const apk = a[p][k], aqk = a[q][k];
          a[p][k] = c * apk - s * aqk;
          a[q][k] = s * apk + c * aqk;
        }
        for (let k = 0; k < 3; k++) {
          const vkp = v[k][p], vkq = v[k][q];
          v[k][p] = c * vkp - s * vkq;
          v[k][q] = s * vkp + c * vkq;
        }
      }
    }
  }
  const order = [0, 1, 2].sort((i, j) => a[i][i] - a[j][j]);
  const val = order.map((i) => a[i][i]);
  const vec = new Float64Array(9);
  for (let c = 0; c < 3; c++)
    for (let r = 0; r < 3; r++) vec[3 * c + r] = v[r][order[c]];
  return { val, vec };
}

// Rotation matrix (column-major 3x3) -> quaternion (w, x, y, z).
function matToQuat(m) {
  const r = (row, col) => m[3 * col + row];
  const t = r(0, 0) + r(1, 1) + r(2, 2);
  let w, x, y, z;
  if (t > 0) {
    const s = Math.sqrt(t + 1) * 2;
    w = 0.25 * s;
    x = (r(2, 1) - r(1, 2)) / s;
    y = (r(0, 2) - r(2, 0)) / s;
    z = (r(1, 0) - r(0, 1)) / s;
  } else if (r(0, 0) >= r(1, 1) && r(0, 0) >= r(2, 2)) {
    const s = Math.sqrt(Math.max(1 + r(0, 0) - r(1, 1) - r(2, 2), 1e-12)) * 2;
    w = (r(2, 1) - r(1, 2)) / s;
    x = 0.25 * s;
    y = (r(0, 1) + r(1, 0)) / s;
    z = (r(0, 2) + r(2, 0)) / s;
  } else if (r(1, 1) >= r(2, 2)) {
    const s = Math.sqrt(Math.max(1 - r(0, 0) + r(1, 1) - r(2, 2), 1e-12)) * 2;
    w = (r(0, 2) - r(2, 0)) / s;
    x = (r(0, 1) + r(1, 0)) / s;
    y = 0.25 * s;
    z = (r(1, 2) + r(2, 1)) / s;
  } else {
    const s = Math.sqrt(Math.max(1 - r(0, 0) - r(1, 1) + r(2, 2), 1e-12)) * 2;
    w = (r(1, 0) - r(0, 1)) / s;
    x = (r(0, 2) + r(2, 0)) / s;
    y = (r(1, 2) + r(2, 1)) / s;
    z = 0.25 * s;
  }
  return [w, x, y, z];
}

// Merge each group of splats into one moment-matched splat. groupId maps
// each input splat to [0, G). Returns { data, n: G }.
function mergeGroups(data, n, groupId, G) {
  const mass = new Float64Array(n);
  const rotM = new Float64Array(n * 9); // per-splat rotation matrix
  for (let i = 0; i < n; i++) {
    const o = i * NP;
    const alpha = sigmoid(data[o + 54]);
    const s0 = Math.exp(data[o + 55]);
    const s1 = Math.exp(data[o + 56]);
    const s2 = Math.exp(data[o + 57]);
    // opacity x disc area: product of the two largest axes.
    const mx = Math.max(s0, s1, s2);
    const mn = Math.min(s0, s1, s2);
    const mid = s0 + s1 + s2 - mx - mn;
    mass[i] = Math.max(alpha * mx * mid, 1e-30);

    // Normalized quaternion (w,x,y,z) -> rotation matrix.
    let qw = data[o + 58], qx = data[o + 59], qy = data[o + 60], qz = data[o + 61];
    const qn = Math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz) || 1;
    qw /= qn; qx /= qn; qy /= qn; qz /= qn;
    const r = rotM.subarray(i * 9, i * 9 + 9); // column-major
    r[0] = 1 - 2 * (qy * qy + qz * qz); r[3] = 2 * (qx * qy - qw * qz); r[6] = 2 * (qx * qz + qw * qy);
    r[1] = 2 * (qx * qy + qw * qz);     r[4] = 1 - 2 * (qx * qx + qz * qz); r[7] = 2 * (qy * qz - qw * qx);
    r[2] = 2 * (qx * qz - qw * qy);     r[5] = 2 * (qy * qz + qw * qx);     r[8] = 1 - 2 * (qx * qx + qy * qy);
  }

  // Pass 1: weighted sums for mean position, normals, SH, opacity.
  const wSum = new Float64Array(G);
  const mu = new Float64Array(G * 3);
  const nrmSum = new Float64Array(G * 3);
  const shSum = new Float64Array(G * 48);
  const alphaSum = new Float64Array(G);
  for (let i = 0; i < n; i++) {
    const g = groupId[i];
    const o = i * NP;
    const m = mass[i];
    wSum[g] += m;
    mu[g * 3] += m * data[o];
    mu[g * 3 + 1] += m * data[o + 1];
    mu[g * 3 + 2] += m * data[o + 2];
    nrmSum[g * 3] += m * data[o + 3];
    nrmSum[g * 3 + 1] += m * data[o + 4];
    nrmSum[g * 3 + 2] += m * data[o + 5];
    for (let c = 0; c < 48; c++) shSum[g * 48 + c] += m * data[o + 6 + c];
    alphaSum[g] += m * sigmoid(data[o + 54]);
  }
  for (let g = 0; g < G; g++) {
    mu[g * 3] /= wSum[g];
    mu[g * 3 + 1] /= wSum[g];
    mu[g * 3 + 2] /= wSum[g];
  }

  // Pass 2: combined covariance  E[R diag(s^2) R^T + d d^T].
  const cov = new Float64Array(G * 6); // xx xy xz yy yz zz
  for (let i = 0; i < n; i++) {
    const g = groupId[i];
    const o = i * NP;
    const m = mass[i];
    const s0 = Math.exp(data[o + 55]) ** 2;
    const s1 = Math.exp(data[o + 56]) ** 2;
    const s2 = Math.exp(data[o + 57]) ** 2;
    const r = rotM.subarray(i * 9, i * 9 + 9);
    // C = R diag(s) R^T with R column-major (columns are rotated axes).
    const cxx = s0 * r[0] * r[0] + s1 * r[3] * r[3] + s2 * r[6] * r[6];
    const cxy = s0 * r[0] * r[1] + s1 * r[3] * r[4] + s2 * r[6] * r[7];
    const cxz = s0 * r[0] * r[2] + s1 * r[3] * r[5] + s2 * r[6] * r[8];
    const cyy = s0 * r[1] * r[1] + s1 * r[4] * r[4] + s2 * r[7] * r[7];
    const cyz = s0 * r[1] * r[2] + s1 * r[4] * r[5] + s2 * r[7] * r[8];
    const czz = s0 * r[2] * r[2] + s1 * r[5] * r[5] + s2 * r[8] * r[8];
    const dx = data[o] - mu[g * 3];
    const dy = data[o + 1] - mu[g * 3 + 1];
    const dz = data[o + 2] - mu[g * 3 + 2];
    cov[g * 6] += m * (cxx + dx * dx);
    cov[g * 6 + 1] += m * (cxy + dx * dy);
    cov[g * 6 + 2] += m * (cxz + dx * dz);
    cov[g * 6 + 3] += m * (cyy + dy * dy);
    cov[g * 6 + 4] += m * (cyz + dy * dz);
    cov[g * 6 + 5] += m * (czz + dz * dz);
  }

  const out = new Float32Array(G * NP);
  const cm = new Float64Array(6);
  for (let g = 0; g < G; g++) {
    const o = g * NP;
    const w = wSum[g];
    for (let k = 0; k < 6; k++) cm[k] = cov[g * 6 + k] / w;
    const { val, vec } = eigh3(cm);
    // Right-handed frame (det +1) for a valid quaternion.
    const det =
      vec[0] * (vec[4] * vec[8] - vec[5] * vec[7]) -
      vec[3] * (vec[1] * vec[8] - vec[2] * vec[7]) +
      vec[6] * (vec[1] * vec[5] - vec[2] * vec[4]);
    if (det < 0) { vec[6] *= -1; vec[7] *= -1; vec[8] *= -1; }
    const q = matToQuat(vec);

    out[o] = mu[g * 3];
    out[o + 1] = mu[g * 3 + 1];
    out[o + 2] = mu[g * 3 + 2];
    out[o + 3] = nrmSum[g * 3] / w;
    out[o + 4] = nrmSum[g * 3 + 1] / w;
    out[o + 5] = nrmSum[g * 3 + 2] / w;
    for (let c = 0; c < 48; c++) out[o + 6 + c] = shSum[g * 48 + c] / w;
    out[o + 54] = invSigmoid(alphaSum[g] / w);
    out[o + 55] = Math.log(Math.max(Math.sqrt(Math.max(val[0], 1e-16)), 1e-12));
    out[o + 56] = Math.log(Math.max(Math.sqrt(Math.max(val[1], 1e-16)), 1e-12));
    out[o + 57] = Math.log(Math.max(Math.sqrt(Math.max(val[2], 1e-16)), 1e-12));
    out[o + 58] = q[0];
    out[o + 59] = q[1];
    out[o + 60] = q[2];
    out[o + 61] = q[3];
  }
  return { data: out, n: G };
}

// Voxel-merge an (n, 62) splat set down to ~target splats. Bisection on
// voxel size: occupied-voxel count decreases monotonically as voxels grow.
function downsample(data, n, target) {
  if (n <= target) return { data, n, voxel: 0 };

  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  for (let i = 0; i < n; i++) {
    const o = i * NP;
    if (data[o] < minX) minX = data[o];
    if (data[o] > maxX) maxX = data[o];
    if (data[o + 1] < minY) minY = data[o + 1];
    if (data[o + 1] > maxY) maxY = data[o + 1];
    if (data[o + 2] < minZ) minZ = data[o + 2];
    if (data[o + 2] > maxZ) maxZ = data[o + 2];
  }
  const extent = Math.max(maxX - minX, maxY - minY, maxZ - minZ);
  if (extent <= 0) return { data: data.slice(0, NP), n: 1, voxel: 0 };

  // Grid indices stay < 4096 per axis, so a packed 36-bit key fits safely
  // in a double.
  const countCells = (v) => {
    const seen = new Set();
    for (let i = 0; i < n; i++) {
      const o = i * NP;
      const kx = Math.min(Math.floor((data[o] - minX) / v), 4095);
      const ky = Math.min(Math.floor((data[o + 1] - minY) / v), 4095);
      const kz = Math.min(Math.floor((data[o + 2] - minZ) / v), 4095);
      seen.add((kx * 4096 + ky) * 4096 + kz);
    }
    return seen.size;
  };

  let vLo = extent / 4096, vHi = extent;
  let best = null;
  for (let it = 0; it < 18; it++) {
    const v = Math.sqrt(vLo * vHi);
    const cnt = countCells(v);
    if (!best || Math.abs(cnt - target) < Math.abs(best.cnt - target))
      best = { cnt, v };
    if (cnt > target * 1.15) vLo = v;
    else if (cnt < target * 0.85) vHi = v;
    else break;
  }

  const v = best.v;
  const cellToGroup = new Map();
  const groupId = new Uint32Array(n);
  let G = 0;
  for (let i = 0; i < n; i++) {
    const o = i * NP;
    const kx = Math.min(Math.floor((data[o] - minX) / v), 4095);
    const ky = Math.min(Math.floor((data[o + 1] - minY) / v), 4095);
    const kz = Math.min(Math.floor((data[o + 2] - minZ) / v), 4095);
    const key = (kx * 4096 + ky) * 4096 + kz;
    let g = cellToGroup.get(key);
    if (g === undefined) { g = G++; cellToGroup.set(key, g); }
    groupId[i] = g;
  }
  const merged = mergeGroups(data, n, groupId, G);
  return { ...merged, voxel: v };
}

// ------------------------------------------------------------- tree build

function bboxOf(data, n) {
  const mn = [Infinity, Infinity, Infinity];
  const mx = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < n; i++) {
    const o = i * NP;
    for (let a = 0; a < 3; a++) {
      if (data[o + a] < mn[a]) mn[a] = data[o + a];
      if (data[o + a] > mx[a]) mx[a] = data[o + a];
    }
  }
  return { mn, mx };
}

function main() {
  const argv = process.argv.slice(2);
  const args = argv.filter((a) => !a.startsWith("--"));
  const flags = argv.filter((a) => a.startsWith("--"));
  if (!args.length) {
    console.log("usage: node build_lod_tileset.mjs <tiles_dir> [--budget N] [--glb] [--dry-run]");
    process.exit(1);
  }
  const tilesDir = args[0];
  const dryRun = flags.includes("--dry-run");
  const toGlb = flags.includes("--glb");
  let budget = null;
  for (const fl of flags)
    if (fl.startsWith("--budget=")) budget = parseInt(fl.split("=")[1]);
  const bIdx = argv.indexOf("--budget");
  if (bIdx >= 0 && argv[bIdx + 1]) budget = parseInt(argv[bIdx + 1]);

  const manifest = JSON.parse(readFileSync(join(tilesDir, "manifest.json"), "utf8"));
  if (manifest.scheme !== "quadtree-leaves")
    throw new Error("manifest is not scheme=quadtree-leaves (need manifest v2)");

  const leafLevel = manifest.leaf_level;
  const leafSize = manifest.leaf_size;
  if (budget === null)
    budget = Math.round(manifest.tiles.reduce((s, t) => s + t.count, 0) / manifest.tiles.length);
  console.log(`leaf level ${leafLevel}, ${manifest.tiles.length} leaves, ` +
              `budget ${budget.toLocaleString()} splats/interior node`);

  // levels[l] = Map "x,y" -> { file, count, bbox }
  const levels = new Map();
  levels.set(leafLevel, new Map());
  for (const t of manifest.tiles)
    levels.get(leafLevel).set(`${t.x},${t.y}`, {
      x: t.x, y: t.y, file: t.file, count: t.count,
      bbox: { mn: t.bbox_min, mx: t.bbox_max },
    });

  for (let l = leafLevel - 1; l >= 0; l--) {
    levels.set(l, new Map());
    const below = levels.get(l + 1);
    const parents = new Map();
    for (const e of below.values()) {
      const key = `${e.x >> 1},${e.y >> 1}`;
      if (!parents.has(key)) parents.set(key, []);
      parents.get(key).push(e);
    }
    for (const [key, children] of [...parents.entries()].sort()) {
      const [px, py] = key.split(",").map(Number);
      const total = children.reduce((s, c) => s + c.count, 0);
      const fname = `tile_L${l}_x${px}_y${py}.ply`;
      if (dryRun) {
        const est = Math.min(total, budget);
        console.log(`  L${l} (${px},${py}): ${children.length} children, ` +
                    `${total.toLocaleString()} -> ~${est.toLocaleString()}   ${fname}`);
        levels.get(l).set(key, {
          x: px, y: py, file: fname, count: est,
          bbox: {
            mn: [0, 1, 2].map((a) => Math.min(...children.map((c) => c.bbox.mn[a]))),
            mx: [0, 1, 2].map((a) => Math.max(...children.map((c) => c.bbox.mx[a]))),
          },
        });
        continue;
      }

      const t0 = Date.now();
      let total_n = 0;
      const parts = children.map((c) => {
        const p = readPly(join(tilesDir, c.file));
        total_n += p.n;
        return p;
      });
      const all = new Float32Array(total_n * NP);
      let off = 0;
      for (const p of parts) { all.set(p.data.subarray(0, p.n * NP), off); off += p.n * NP; }

      const res = downsample(all, total_n, budget);
      writePly(join(tilesDir, fname), res.data, res.n);
      levels.get(l).set(key, {
        x: px, y: py, file: fname, count: res.n, bbox: bboxOf(res.data, res.n),
      });
      console.log(`  L${l} (${px},${py}): ${total.toLocaleString()} -> ` +
                  `${res.n.toLocaleString()} (voxel ${res.voxel.toFixed(3)}) ` +
                  `in ${((Date.now() - t0) / 1000).toFixed(1)}s`);
    }
  }

  if (dryRun) { console.log("dry run -- nothing written"); return; }

  // ----- optional PLY -> GLB (KHR_gaussian_splatting) per node -----
  if (toGlb) {
    const all = [...levels.values()].flatMap((m) => [...m.values()]);
    console.log(`converting ${all.length} nodes to GLB (splat-transform)...`);
    for (const e of all) {
      const ply = join(tilesDir, e.file);
      const glb = ply.replace(/\.ply$/, ".glb");
      const t0 = Date.now();
      const r = spawnSync("npx", ["--yes", "@playcanvas/splat-transform", "-w", ply, glb],
                          { shell: process.platform === "win32", encoding: "utf8" });
      if (r.status !== 0 || !existsSync(glb))
        throw new Error(`splat-transform failed on ${e.file}:\n${(r.stderr || "").slice(-2000)}`);
      e.content = e.file.replace(/\.ply$/, ".glb");
      console.log(`  ${e.content} (${(statSync(glb).size / 1e6).toFixed(1)} MB) ` +
                  `in ${((Date.now() - t0) / 1000).toFixed(1)}s`);
    }
  } else {
    for (const m of levels.values())
      for (const e of m.values()) e.content = e.file;
  }

  // ----- explicit tileset.json (3D Tiles 1.1, refine REPLACE) -----
  const node = (l, x, y) => {
    const e = levels.get(l).get(`${x},${y}`);
    const c = [0, 1, 2].map((a) => (e.bbox.mn[a] + e.bbox.mx[a]) / 2);
    const h = [0, 1, 2].map((a) => Math.max((e.bbox.mx[a] - e.bbox.mn[a]) / 2, 1e-3));
    const gErr = l === leafLevel ? 0 : (leafSize * 2 ** (leafLevel - l)) / 32;
    const out = {
      boundingVolume: { box: [c[0], c[1], c[2], h[0], 0, 0, 0, h[1], 0, 0, 0, h[2]] },
      geometricError: gErr,
      refine: "REPLACE",
      content: { uri: e.content },
    };
    const kids = [];
    const nxt = levels.get(l + 1);
    if (nxt)
      for (const k of nxt.values())
        if ((k.x >> 1) === x && (k.y >> 1) === y) kids.push(node(l + 1, k.x, k.y));
    if (kids.length) out.children = kids;
    return out;
  };

  const rootNode = node(0, 0, 0);
  const tileset = {
    asset: { version: "1.1", generator: "mesh2splat build_lod_tileset.mjs" },
    geometricError: rootNode.geometricError * 2,
    root: rootNode,
    extras: {
      note: "Node content is KHR_gaussian_splatting GLB (or 3DGS PLY " +
            "without --glb) in the source local frame (y-up, no " +
            "georeferencing); the manifest transform applies when filled in.",
      source_manifest: "manifest.json",
    },
  };
  writeFileSync(join(tilesDir, "tileset.json"), JSON.stringify(tileset, null, 1));

  let interior = 0, nodes = 0;
  for (let l = 0; l < leafLevel; l++)
    for (const e of levels.get(l).values()) { interior += e.count; nodes++; }
  console.log(`tileset.json written; ${interior.toLocaleString()} interior ` +
              `splats across ${nodes} nodes`);
}

main();
