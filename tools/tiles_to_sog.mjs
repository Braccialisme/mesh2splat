/**
 * tiles_to_sog.mjs -- convert a mesh2splat tiled output folder to per-tile SOG.
 *
 * SOG (@playcanvas/splat-transform .sog) is a compact, self-contained,
 * GPU-compressed splat format for WEB viewers (SuperSplat, Spark, PlayCanvas).
 * It is NOT a 3D Tiles content type -- that is GLB (see build_lod_tileset.mjs
 * --glb). So SOG is a parallel, web-delivery export of the same tiles.
 *
 * Reads <tiles_dir>/manifest.json and converts each leaf PLY to a sibling .sog
 * via `npx @playcanvas/splat-transform`. With --nodes it also converts any
 * interior LOD node PLYs produced by build_lod_tileset.mjs (every tile_L*.ply
 * in the folder). Writes sog_manifest.json listing the outputs.
 *
 * PBR NOTE: SOG has no material channels -- metallicFactor / roughnessFactor
 * are dropped (same limitation as GLB). PBR survives only in the PLY tiles.
 *
 * Usage:
 *   node tiles_to_sog.mjs <tiles_dir> [--nodes] [--gpu <n|cpu>]
 *                         [--dry-run] [--keep] [--quiet]
 *   --nodes    also convert interior LOD node PLYs (not just manifest leaves)
 *   --gpu      pass a GPU adapter index, or 'cpu' to disable GPU compression
 *   --dry-run  list what would be converted, convert nothing
 *   --keep     skip a tile whose .sog already exists (default: overwrite)
 *   --quiet    less output
 */

import { readFileSync, writeFileSync, existsSync, statSync, readdirSync } from "node:fs";
import { join, basename } from "node:path";
import { spawnSync } from "node:child_process";

function parseArgs(argv) {
  const a = { dir: null, nodes: false, gpu: null, dryRun: false, keep: false, quiet: false };
  for (let i = 0; i < argv.length; i++) {
    const t = argv[i];
    if (t === "--nodes") a.nodes = true;
    else if (t === "--gpu") a.gpu = argv[++i];
    else if (t === "--dry-run") a.dryRun = true;
    else if (t === "--keep") a.keep = true;
    else if (t === "--quiet") a.quiet = true;
    else if (!t.startsWith("--") && a.dir === null) a.dir = t;
  }
  return a;
}

function main() {
  const args = parseArgs(process.argv.slice(2));
  if (!args.dir) {
    console.log("usage: node tiles_to_sog.mjs <tiles_dir> [--nodes] [--gpu <n|cpu>] [--dry-run] [--keep] [--quiet]");
    process.exit(2);
  }

  const manPath = join(args.dir, "manifest.json");
  if (!existsSync(manPath)) {
    console.error(`no manifest.json in ${args.dir}`);
    process.exit(1);
  }
  const man = JSON.parse(readFileSync(manPath, "utf8"));

  if (man.has_pbr)
    console.log(
      "NOTE: these tiles carry PBR (metallicFactor/roughnessFactor). SOG has no " +
      "material channels -- PBR will be DROPPED in the .sog outputs; it stays in the PLY.");

  // Work list: manifest leaves, plus (optionally) every other tile_*.ply
  // in the folder (interior LOD nodes written by build_lod_tileset.mjs).
  const files = new Set(man.tiles.map((t) => t.file));
  if (args.nodes) {
    for (const f of readdirSync(args.dir))
      if (f.startsWith("tile_") && f.endsWith(".ply")) files.add(f);
  }
  const list = [...files].sort();
  console.log(`${list.length} PLY tile(s) -> SOG in ${args.dir}` + (args.dryRun ? "  (dry run)" : ""));

  const outputs = [];
  let plyBytes = 0, sogBytes = 0, done = 0, skipped = 0, failed = 0;

  for (const file of list) {
    const ply = join(args.dir, file);
    if (!existsSync(ply)) { console.log(`  MISSING ${file} -- skipping`); continue; }
    const sog = ply.replace(/\.ply$/, ".sog");
    const sogName = basename(sog);

    if (args.keep && existsSync(sog)) {
      skipped++;
      if (!args.quiet) console.log(`  keep ${sogName} (already exists)`);
      outputs.push({ ply: file, sog: sogName });
      continue;
    }

    if (args.dryRun) { outputs.push({ ply: file, sog: sogName }); continue; }

    const cmd = ["--yes", "@playcanvas/splat-transform", "-w"];
    if (args.gpu !== null) cmd.push("--gpu", args.gpu);
    cmd.push(ply, sog);

    const r = spawnSync("npx", cmd, { shell: process.platform === "win32", encoding: "utf8" });
    if (r.status !== 0 || !existsSync(sog)) {
      failed++;
      console.error(`  FAIL ${file}:\n${(r.stderr || "").slice(-1500)}`);
      continue;
    }

    const pb = statSync(ply).size, sb = statSync(sog).size;
    plyBytes += pb; sogBytes += sb; done++;
    outputs.push({ ply: file, sog: sogName, ply_bytes: pb, sog_bytes: sb });
    if (!args.quiet)
      console.log(`  ${sogName}  ${(sb / 1e6).toFixed(1)} MB  (${(100 * sb / pb).toFixed(0)}% of PLY)`);
  }

  if (!args.dryRun) {
    const sogMan = {
      source_manifest: "manifest.json",
      scheme: man.scheme,
      has_pbr: false,
      pbr_note: man.has_pbr ? "PBR present in source PLY tiles was dropped in SOG" : "no PBR in source",
      leaf_level: man.leaf_level,
      leaf_size: man.leaf_size,
      root: man.root,
      count: outputs.length,
      tiles: outputs,
    };
    writeFileSync(join(args.dir, "sog_manifest.json"), JSON.stringify(sogMan, null, 2));

    const ratio = plyBytes > 0 ? (100 * sogBytes / plyBytes).toFixed(1) : "n/a";
    console.log(
      `\ndone: ${done} converted` + (skipped ? `, ${skipped} kept` : "") +
      (failed ? `, ${failed} FAILED` : "") +
      `  |  ${(sogBytes / 1e6).toFixed(1)} MB SOG vs ${(plyBytes / 1e6).toFixed(1)} MB PLY (${ratio}%)` +
      `  |  sog_manifest.json written`);
    if (failed) process.exit(1);
  } else {
    for (const o of outputs) console.log(`  would write ${o.sog}`);
  }
}

main();
