///////////////////////////////////////////////////////////////////////////////
//         Mesh2Splat: fast mesh to 3D gaussian splat conversion             //
//        Copyright (c) 2025 Electronic Arts Inc. All rights reserved.       //
///////////////////////////////////////////////////////////////////////////////
//
// OfflineConverter
// ----------------
// Converts the loaded mesh to a gaussian-splat PLY in triangle-range BATCHES,
// streaming each batch's gaussians from GPU straight to disk, so the whole
// result never resides in VRAM. Output size is limited by disk, not by the
// 24 GB live-view ceiling.
//
// Why triangle ranges are safe (design note, section 3): every gaussian is
// emitted by exactly one triangle's UV rasterization; triangles do not
// influence each other's fragments. Converting ranges [0..k), [k..n) yields
// the exact same gaussian SET as one giant pass -- no seams, no duplicates.
// (The ORDER within the file differs run-to-run: parallel fragments append
// via an atomic counter in GPU scheduling order. Viewers don't care.)
//
// Driving model: start() once, then step() once per frame from the mediator
// (keeps the UI responsive); each step converts one work range. If a range
// overflows the batch buffer (the atomic counter keeps counting past the cap,
// so overflow is always detected), the range is split in half and re-queued --
// nothing partial is ever written.

#pragma once

#include <deque>
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

    // Begins an offline conversion of the currently loaded mesh(es).
    // Returns false (with status set) if preconditions fail.
    bool start(RenderContext& ctx,
               const std::string& outputPath,
               unsigned int batchCapacity = kDefaultBatchCapacity);

    // Runs ONE batch. Call once per frame while isRunning().
    // Returns true while more work remains; false when finished or failed.
    bool step(RenderContext& ctx);

    // Cancels the run: deletes the partial output file, frees GPU buffer.
    void cancel();

    bool               isRunning()    const { return running; }
    float              progress01()   const;
    uint64_t           writtenCount() const { return writer.writtenCount(); }
    const std::string& statusText()   const { return status; }

private:
    struct WorkRange {
        size_t  meshIndex;
        GLint   firstVertex;   // multiple of 3
        GLsizei vertexCount;   // multiple of 3
    };

    void fail(const std::string& why);
    void cleanupGpu();

    std::deque<WorkRange>                    work;
    parsers::IncrementalPlyWriter            writer;
    std::vector<utils::GaussianDataSSBO>     readback;   // reused across batches

    GLuint       offlineGaussianBuffer = 0;
    unsigned int batchCapacity         = 0;
    uint64_t     totalVertices         = 0;
    uint64_t     processedVertices     = 0;
    int          batchesDone           = 0;
    bool         running               = false;
    std::string  status;
};