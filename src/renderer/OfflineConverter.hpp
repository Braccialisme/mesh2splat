///////////////////////////////////////////////////////////////////////////////
//         Mesh2Splat: fast mesh to 3D gaussian splat conversion             //
//        Copyright (c) 2025 Electronic Arts Inc. All rights reserved.       //
///////////////////////////////////////////////////////////////////////////////
//
// OfflineConverter
// ----------------
// Converts the loaded mesh to gaussian-splat PLY output in triangle-range
// BATCHES, streaming each batch's gaussians from GPU straight to disk, so
// the whole result never resides in VRAM. Output size is limited by disk.
//
// Why triangle ranges are safe (design note, section 3): every gaussian is
// emitted by exactly one triangle's UV rasterization; triangles do not
// influence each other's fragments. Converting ranges yields the exact same
// gaussian SET as one giant pass -- no seams, no duplicates. (Order within
// files differs run-to-run: atomic-counter append follows GPU scheduling.)
//
// PHASE T -- TILED OUTPUT: when tileSize > 0, each batch's gaussians are
// bucketed on the CPU by ground-plane (XZ) grid cell, cell (i,j) covering
// x in [i*tileSize, (i+1)*tileSize), z likewise (indices may be negative;
// boundaries sit on multiples of tileSize, so the grid is geo-alignable).
// One IncrementalPlyWriter per non-empty tile, created lazily, all streaming
// simultaneously into  <output>_tiles/tile_<i>_<j>.ply,  plus a
// manifest.json (grid scheme, per-tile bbox/count/file, georef placeholder)
// written at the end. tileSize == 0 keeps the original single-file behavior.
//
// Driving model: start() once, then step() once per frame from the mediator;
// each step converts one work range. Ranges that overflow the batch buffer
// (the counter keeps counting past the cap, so overflow is always detected)
// are split in half and re-queued -- nothing partial is ever written.

#pragma once

#include <deque>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <cstdint>

#include "renderPasses/RenderContext.hpp"
#include "parsers/IncrementalPlyWriter.hpp"

class OfflineConverter
{
public:
    // Batch buffer budget in gaussians. 8M * 96 B = ~768 MB of VRAM,
    // reused for every batch. Lower it temporarily (e.g. 200000) to force
    // multi-batch runs / overflow splitting when testing.
    static constexpr unsigned int kDefaultBatchCapacity = 8'000'000u;

    // Safety cap on simultaneously open tile files (Windows CRT handle
    // limits). Exceeding it fails the run with advice to raise tileSize.
    static constexpr size_t kMaxOpenTiles = 256;

    // Begins an offline conversion of the currently loaded mesh(es).
    // tileSize: ground-plane tile edge in world units; 0 = single file.
    // Returns false (with status set) if preconditions fail.
    bool start(RenderContext& ctx,
               const std::string& outputPath,
               float tileSize = 0.0f,
               unsigned int batchCapacity = kDefaultBatchCapacity);

    // Runs ONE batch. Call once per frame while isRunning().
    // Returns true while more work remains; false when finished or failed.
    bool step(RenderContext& ctx);

    // Cancels the run: deletes partial output file(s), frees GPU buffer.
    void cancel();

    bool               isRunning()    const { return running; }
    float              progress01()   const;
    uint64_t           writtenCount() const { return totalWritten; }
    const std::string& statusText()   const { return status; }

private:
    struct WorkRange {
        size_t  meshIndex;
        GLint   firstVertex;   // multiple of 3
        GLsizei vertexCount;   // multiple of 3
    };

    struct TileState {
        parsers::IncrementalPlyWriter writer;
        std::string filename;      // relative, e.g. "tile_-2_5.ply"
        uint64_t    count = 0;
        glm::vec3   bboxMin{  1e30f,  1e30f,  1e30f };
        glm::vec3   bboxMax{ -1e30f, -1e30f, -1e30f };
    };

    // Returns the writer for tile (i,j), creating it lazily; nullptr on failure.
    TileState* tileFor(int i, int j);

    bool finishAndWriteManifest();
    void fail(const std::string& why);
    void cleanupGpu();
    void abortAllWriters();

    std::deque<WorkRange>                    work;
    std::vector<utils::GaussianDataSSBO>     readback;      // reused across batches
    std::vector<utils::GaussianDataSSBO>     tileBucket;    // reused per tile per batch

    // Single-file mode (tileSize == 0)
    parsers::IncrementalPlyWriter            singleWriter;

    // Tiled mode
    std::map<std::pair<int, int>, TileState> tiles;
    std::string  tilesDir;         // absolute folder for tiles + manifest
    float        tileSize          = 0.0f;
    bool         tiled             = false;

    std::string  outputPathStored;
    std::string  sourceName;
    float        scaleMultiplierStored = 1.0f;

    GLuint       offlineGaussianBuffer = 0;
    unsigned int batchCapacity         = 0;
    uint64_t     totalVertices         = 0;
    uint64_t     processedVertices     = 0;
    uint64_t     totalWritten          = 0;
    int          batchesDone           = 0;
    bool         running               = false;
    std::string  status;
};