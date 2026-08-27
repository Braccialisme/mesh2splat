/**
 * tiles_to_sog.mjs -- convert a mesh2splat tiled output folder to per-tile SOG.
 *
 * SOG (@playcanvas/splat-transform .sog) is a compact, self-contained,
 * GPU-compressed splat format for WEB viewers (SuperSplat, Spark, PlayCanvas).
 * It is NOT a 3D Tiles content type -- that is GLB (see build_lod_tileset.mjs
 * --glb). So SOG is a parallel, web-delivery export of the same tiles.
 *
 * Reads <tiles_dir>/manifest.json and converts each leaf PLY to a sibling .sog
 * via `npx @playcanvas/splat-transform`. Conversions run concurrently (a bounded
 * pool; see --jobs). With --nodes it also converts any interior LOD node PLYs
 * produced by build_lod_tileset.mjs. Writes sog_manifest.json.
 *
 * PBR NOTE: SOG has no material channels -- metallicFactor / roughnessFactor
 * are dropped (same limitation as GLB). PBR survives only in the PLY tiles.
 *
 * Usage:
 *   node tiles_to_sog.mjs <tiles_dir> [--jobs N] [--nodes] [--gpu <n|cpu>]
 *                         [--dry-run] [--keep] [--quiet]
 *   --jobs N   max concurrent conversions (default 4; use 1 for serial).
 *              Each SOG job uses some GPU memory -- lower this if you OOM.
 *   --nodes    also convert interior LOD node PLYs (not just manifest leaves)
 *   --gpu      pass a GPU adapter index, or 'cpu' to disable GPU compression
 *   --dry-run  list what would be converted, convert nothing
 *   --keep     skip a tile whose .sog already exists (default: overwrite)
 *   --quiet    less output
 */

import { readFileSync, writeFileSync, existsSync, statSync, readdirSync } from "node:fs";
import { join, basename } from "node:path";
import { spawn } from "node:child_process";

// npx via a shell: Node >=22 refuses to spawn .cmd without shell:true on
// Windows (EINVAL), and shell:true wants npx (not npx.cmd). Paths are quoted
// below since shell:true concatenates args.
const WIN = process.platform === "win32";

function parseArgs(argv) {
  const a = { dir: null, jobs: 4, nodes: false, gpu: null, dryRun: false, keep: false, quiet: false };
  for (let i = 0; i < argv.length; i++) {
    const t = argv[i];
    if (t === "--jobs") a.jobs = Math.max(1, parseInt(argv[++i], 10) || 1);
    else if (t === "--nodes") a.nodes = true;
    else if (t === "--gpu") a.gpu = argv[++i];
    else if (t === "--dry-run") a.dryRun = true;
    else if (t === "--keep") a.keep = true;
    else if (t === "--quiet") a.quiet = true;
    else if (!t.startsWith("--") && a.dir === null) a.dir = t;
  }
  return a;
}

// Convert one PLY -> SOG. Resolves to a result record; never rejects.
function convertOne(args, ply, sog) {
  return new Promise((resolve) => {
    const q = (p) => `"${p}"`; // quote for shell:true (paths may contain spaces)
    const cmd = ["npx", "--yes", "@playcanvas/splat-transform", "-w"];
    if (args.gpu !== null) cmd.push("--gpu", args.gpu);
    cmd.push(q(ply), q(sog));

    const child = spawn(cmd.join(" "), { shell: true });
    let stderr = "";
    child.stderr.on("data", (d) => { stderr += d; });
    child.on("error", (e) => resolve({ ok: false, err: e.message }));
    child.on("close", (code) => {
      if (code !== 0 || !existsSync(sog)) resolve({ ok: false, err: stderr.slice(-1500) });
      else resolve({ ok: true });
    });
  });
}

// Run tasks with a bounded concurrency pool. onDone(result, task) after each.
async function runPool(tasks, jobs, worker) {
  let next = 0;
  const runners = Array.from({ length: Math.min(jobs, tasks.length) }, async () => {
    while (next < tasks.length) {
      const i = next++;
      await worker(tasks[i], i);
    }
  });
  await Promise.all(runners);
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (!args.dir) {
    console.log("usage: node tiles_to_sog.mjs <tiles_dir> [--jobs N] [--nodes] [--gpu <n|cpu>] [--dry-run] [--keep] [--quiet]");
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
  // (interior LOD nodes written by build_lod_tileset.mjs).
  const files = new Set(man.tiles.map((t) => t.file));
  if (args.nodes)
    for (const f of readdirSync(args.dir))
      if (f.startsWith("tile_") && f.endsWith(".ply")) files.add(f);

  const list = [...files].sort();

  // Pre-resolve each task, applying keep/missing/dry-run filters.
  const outputs = [];       // records for sog_manifest (all, incl. kept)
  const toConvert = [];     // {file, ply, sog} actually needing a run
  let skipped = 0;
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
    toConvert.push({ file, ply, sog, sogName });
  }

  console.log(`${list.length} tile(s), ${toConvert.length} to convert` +
              (skipped ? `, ${skipped} kept` : "") +
              (args.dryRun ? "  (dry run)" : `  |  jobs=${args.jobs}`));

  if (args.dryRun) {
    for (const t of toConvert) console.log(`  would write ${t.sogName}`);
    return;
  }

  let plyBytes = 0, sogBytes = 0, done = 0, failed = 0, finished = 0;
  const total = toConvert.length;

  await runPool(toConvert, args.jobs, async (t) => {
    const r = await convertOne(args, t.ply, t.sog);
    finished++;
    const tag = `[${finished}/${total}]`;
    if (!r.ok) {
      failed++;
      console.error(`  ${tag} FAIL ${t.file}:\n${r.err}`);
      return;
    }
    const pb = statSync(t.ply).size, sb = statSync(t.sog).size;
    plyBytes += pb; sogBytes += sb; done++;
    outputs.push({ ply: t.file, sog: t.sogName, ply_bytes: pb, sog_bytes: sb });
    if (!args.quiet)
      console.log(`  ${tag} ${t.sogName}  ${(sb / 1e6).toFixed(1)} MB  (${(100 * sb / pb).toFixed(0)}% of PLY)`);
  });

  // Stable order in the manifest regardless of completion order.
  outputs.sort((a, b) => a.sog.localeCompare(b.sog));

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
}

main();
