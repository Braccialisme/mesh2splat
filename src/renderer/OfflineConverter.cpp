///////////////////////////////////////////////////////////////////////////////
//         Mesh2Splat: fast mesh to 3D gaussian splat conversion             //
//        Copyright (c) 2025 Electronic Arts Inc. All rights reserved.       //
///////////////////////////////////////////////////////////////////////////////

#include "OfflineConverter.hpp"
#include "utils/params.hpp"

#include <iostream>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <filesystem>

namespace
{
    std::string withCommas(uint64_t v)
    {
        std::string s = std::to_string(v);
        for (int i = static_cast<int>(s.size()) - 3; i > 0; i -= 3)
            s.insert(static_cast<size_t>(i), ",");
        return s;
    }
}

bool OfflineConverter::start(RenderContext& ctx,
                             const std::string& outputPath,
                             float requestedTileSize,
                             unsigned int requestedBatchCapacity)
{
    if (running) return false;

    if (ctx.dataMeshAndGlMesh.empty()) { status = "No mesh loaded.";   return false; }
    if (outputPath.empty())            { status = "No output path.";   return false; }

    batchCapacity     = std::max(requestedBatchCapacity, 1000u);
    tileSize          = requestedTileSize;
    tiled             = (tileSize > 0.0f);
    totalVertices     = 0;
    processedVertices = 0;
    totalWritten      = 0;
    batchesDone       = 0;
    work.clear();
    tiles.clear();
    outputPathStored  = outputPath;

    for (size_t i = 0; i < ctx.dataMeshAndGlMesh.size(); ++i)
    {
        GLsizei vertexCount = static_cast<GLsizei>(ctx.dataMeshAndGlMesh[i].second.vertexCount);
        if (vertexCount <= 0) continue;
        work.push_back({ i, 0, vertexCount });
        totalVertices += static_cast<uint64_t>(vertexCount);
    }
    if (work.empty()) { status = "Mesh has no triangles."; return false; }

    // Same scale semantics as the live exporter.
    scaleMultiplierStored = ctx.gaussianStd / static_cast<float>(ctx.resolutionTarget);

    if (tiled)
    {
        // <output-without-extension>_tiles/  next to the requested output.
        std::filesystem::path p(outputPath);
        sourceName = p.stem().string();
        std::filesystem::path dir = p.parent_path() / (p.stem().string() + "_tiles");
        std::error_code ec;
        std::filesystem::create_directories(dir, ec);
        if (ec) { status = "Could not create tile folder: " + dir.string(); return false; }
        tilesDir = dir.string();
        // Tile writers are created lazily in tileFor() as gaussians arrive.
    }
    else
    {
        sourceName = std::filesystem::path(outputPath).stem().string();
        if (!singleWriter.open(outputPath, scaleMultiplierStored)) {
            status = "Could not open output file: " + outputPath;
            return false;
        }
    }

    glGenBuffers(1, &offlineGaussianBuffer);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, offlineGaussianBuffer);
    GLsizeiptr bufferSize = static_cast<GLsizeiptr>(batchCapacity) * sizeof(utils::GaussianDataSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, bufferSize, nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    running = true;
    status  = "Starting...";
    std::cout << "[Offline] Conversion started -> "
              << (tiled ? tilesDir + "  (tile size " + std::to_string(tileSize) + ")" : outputPath)
              << "  (batch capacity " << withCommas(batchCapacity) << " gaussians, "
              << withCommas(bufferSize / (1024 * 1024)) << " MB VRAM)" << std::endl;
    return true;
}

OfflineConverter::TileState* OfflineConverter::tileFor(int i, int j)
{
    auto key = std::make_pair(i, j);
    auto it  = tiles.find(key);
    if (it != tiles.end()) return &it->second;

    if (tiles.size() >= kMaxOpenTiles) {
        fail("More than " + std::to_string(kMaxOpenTiles) +
             " tiles -- tile size is too small for this model. Increase it.");
        return nullptr;
    }

    TileState& t = tiles[key];   // constructed in place (writer is not movable)
    t.filename = "tile_" + std::to_string(i) + "_" + std::to_string(j) + ".ply";
    if (!t.writer.open(tilesDir + "/" + t.filename, scaleMultiplierStored)) {
        fail("Could not open tile file: " + t.filename);
        return nullptr;
    }
    return &t;
}

bool OfflineConverter::step(RenderContext& ctx)
{
    if (!running) return false;

    if (work.empty()) {
        finishAndWriteManifest(); // sets final status; run ends either way
        return false;
    }

    WorkRange range = work.front();
    work.pop_front();

    auto& mesh = ctx.dataMeshAndGlMesh[range.meshIndex];

    GLuint program = ctx.shaderRegistry.getProgramID(glUtils::ShaderProgramTypes::ConverterProgram);
    glUseProgram(program);

    glUtils::resetAtomicCounter(ctx.atomicCounterBufferConversionPass);
    glUtils::setUniform1i(program, "u_maxGaussians", static_cast<int>(batchCapacity));

    GLuint framebuffer;
    GLuint drawBuffers = glUtils::setupFrameBuffer(framebuffer, ctx.resolutionTarget, ctx.resolutionTarget);
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, offlineGaussianBuffer);
    glBindBufferBase(GL_ATOMIC_COUNTER_BUFFER, 1, ctx.atomicCounterBufferConversionPass);

    glViewport(0, 0, static_cast<int>(ctx.resolutionTarget), static_cast<int>(ctx.resolutionTarget));
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glDisable(GL_CULL_FACE);

    glUtils::setUniform1i(program, "hasAlbedoMap", 0);
    glUtils::setUniform1i(program, "hasNormalMap", 0);
    glUtils::setUniform1i(program, "hasMetallicRoughnessMap", 0);

    if (ctx.meshToTextureData.find(mesh.first.name) != ctx.meshToTextureData.end())
    {
        auto& textureMap = ctx.meshToTextureData.at(mesh.first.name);

        if (textureMap.find(BASE_COLOR_TEXTURE) != textureMap.end()) {
            glUtils::setTexture2D(program, "albedoTexture", textureMap.at(BASE_COLOR_TEXTURE).glTextureID, 0);
            glUtils::setUniform1i(program, "hasAlbedoMap", 1);
        }
        if (textureMap.find(NORMAL_TEXTURE) != textureMap.end()) {
            glUtils::setTexture2D(program, "normalTexture", textureMap.at(NORMAL_TEXTURE).glTextureID, 1);
            glUtils::setUniform1i(program, "hasNormalMap", 1);
        }
        if (textureMap.find(METALLIC_ROUGHNESS_TEXTURE) != textureMap.end()) {
            glUtils::setTexture2D(program, "metallicRoughnessTexture", textureMap.at(METALLIC_ROUGHNESS_TEXTURE).glTextureID, 2);
            glUtils::setUniform1i(program, "hasMetallicRoughnessMap", 1);
        }
    }

    glUtils::setUniform4f(program, "u_materialFactor", mesh.first.material.baseColorFactor);
    glUtils::setUniform3f(program, "u_bboxMin", mesh.first.bbox.min);
    glUtils::setUniform3f(program, "u_bboxMax", mesh.first.bbox.max);

    glBindVertexArray(mesh.second.vao);
    glDrawArrays(GL_TRIANGLES, range.firstVertex, range.vertexCount);

    glFinish();

    uint32_t numGs = 0;
    glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, ctx.atomicCounterBufferConversionPass);
    glGetBufferSubData(GL_ATOMIC_COUNTER_BUFFER, 0, sizeof(uint32_t), &numGs);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteRenderbuffers(1, &drawBuffers);
    glDeleteFramebuffers(1, &framebuffer);

    if (numGs > batchCapacity)
    {
        if (range.vertexCount <= 3) {
            fail("A single triangle produced " + withCommas(numGs) +
                 " gaussians, exceeding the batch capacity of " + withCommas(batchCapacity) +
                 ". Increase OfflineConverter::kDefaultBatchCapacity or lower the resolution target.");
            return false;
        }
        GLsizei halfTriangles = (range.vertexCount / 3) / 2;
        GLsizei half          = std::max<GLsizei>(halfTriangles, 1) * 3;
        work.push_front({ range.meshIndex, range.firstVertex + half, range.vertexCount - half });
        work.push_front({ range.meshIndex, range.firstVertex,        half });
        status = "Batch too dense (" + withCommas(numGs) + " gaussians) -- splitting and retrying...";
        std::cout << "[Offline] " << status << std::endl;
        return true;
    }

    if (numGs > 0)
    {
        readback.resize(numGs);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, offlineGaussianBuffer);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        {
            const GLsizeiptr sliceBytes = 512ll * 1024ll * 1024ll;
            const GLsizeiptr totalBytes =
                static_cast<GLsizeiptr>(numGs) * static_cast<GLsizeiptr>(sizeof(utils::GaussianDataSSBO));
            char*      dst    = reinterpret_cast<char*>(readback.data());
            GLsizeiptr offset = 0;
            while (offset < totalBytes)
            {
                GLsizeiptr chunk = (totalBytes - offset < sliceBytes) ? (totalBytes - offset) : sliceBytes;
                glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, offset, chunk, dst + offset);
                offset += chunk;
            }
        }
        GLenum err = glGetError();
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
        if (err != GL_NO_ERROR) {
            fail("GPU readback failed (GL error) -- nothing corrupt was written.");
            return false;
        }

        if (!tiled)
        {
            if (!singleWriter.appendBatch(readback)) {
                fail("Write to disk failed (disk full?). Partial file removed.");
                return false;
            }
            totalWritten += numGs;
        }
        else
        {
            // Bucket this batch by ground-plane grid cell, then stream each
            // bucket to its tile writer. Bucketing is CPU-side and cheap:
            // one pass to group, one append per touched tile.
            std::map<std::pair<int, int>, std::vector<uint32_t>> groups;
            const float invTs = 1.0f / tileSize;
            for (uint32_t g = 0; g < numGs; ++g)
            {
                const auto& pos = readback[g].position;
                int i = 0, j = 0;
                if (std::isfinite(pos.x) && std::isfinite(pos.z)) {
                    i = static_cast<int>(std::floor(pos.x * invTs));
                    j = static_cast<int>(std::floor(pos.z * invTs));
                }
                groups[{i, j}].push_back(g);
            }

            for (auto& [key, indices] : groups)
            {
                TileState* tile = tileFor(key.first, key.second);
                if (!tile) return false; // fail() already called

                tileBucket.clear();
                tileBucket.reserve(indices.size());
                for (uint32_t idx : indices)
                {
                    const auto& gaus = readback[idx];
                    tileBucket.push_back(gaus);
                    tile->bboxMin = glm::min(tile->bboxMin, glm::vec3(gaus.position));
                    tile->bboxMax = glm::max(tile->bboxMax, glm::vec3(gaus.position));
                }
                if (!tile->writer.appendBatch(tileBucket)) {
                    fail("Write to tile " + tile->filename + " failed (disk full?).");
                    return false;
                }
                tile->count  += indices.size();
                totalWritten += indices.size();
            }
        }
    }

    processedVertices += static_cast<uint64_t>(range.vertexCount);
    ++batchesDone;

    std::ostringstream ss;
    ss << "Batch " << batchesDone << " done -- " << withCommas(totalWritten) << " gaussians";
    if (tiled) ss << " across " << tiles.size() << " tiles";
    ss << " (" << (totalWritten * sizeof(float) * 62) / (1024 * 1024) << " MB)";
    status = ss.str();

    return true;
}

bool OfflineConverter::finishAndWriteManifest()
{
    bool ok = true;

    if (!tiled)
    {
        ok = singleWriter.finalize();
        if (ok) status = "Done: " + withCommas(totalWritten) + " gaussians written.";
        else    status = "FAILED while finalizing the output file (disk full?).";
    }
    else
    {
        for (auto& [key, tile] : tiles)
            if (!tile.writer.finalize()) { ok = false; break; }

        if (ok)
        {
            std::ofstream m(tilesDir + "/manifest.json", std::ios::trunc);
            m << std::setprecision(9);
            m << "{\n";
            m << "  \"version\": 1,\n";
            m << "  \"generator\": \"mesh2splat OfflineConverter (tiled)\",\n";
            m << "  \"source\": \"" << sourceName << "\",\n";
            m << "  \"gaussian_format\": \"3dgs-standard-ply\",\n";
            m << "  \"tile_size\": " << tileSize << ",\n";
            m << "  \"grid_plane\": \"xz\",\n";
            m << "  \"grid_convention\": \"tile (i,j) spans x in [i*tile_size,(i+1)*tile_size), z likewise; y unbounded\",\n";
            m << "  \"total_gaussians\": " << totalWritten << ",\n";
            m << "  \"crs\": null,\n";
            m << "  \"transform\": null,\n";
            m << "  \"transform_note\": \"4x4 row-major local-to-georeferenced transform; filled by downstream pipeline\",\n";
            m << "  \"tiles\": [\n";
            size_t n = 0;
            for (auto& [key, tile] : tiles)
            {
                m << "    { \"i\": " << key.first << ", \"j\": " << key.second
                  << ", \"file\": \"" << tile.filename << "\""
                  << ", \"count\": " << tile.count
                  << ", \"bbox_min\": [" << tile.bboxMin.x << ", " << tile.bboxMin.y << ", " << tile.bboxMin.z << "]"
                  << ", \"bbox_max\": [" << tile.bboxMax.x << ", " << tile.bboxMax.y << ", " << tile.bboxMax.z << "] }"
                  << (++n < tiles.size() ? "," : "") << "\n";
            }
            m << "  ]\n}\n";
            ok = m.good();
            m.close();

            status = "Done: " + withCommas(totalWritten) + " gaussians across "
                   + std::to_string(tiles.size()) + " tiles + manifest.json";
        }
        else
        {
            status = "FAILED while finalizing tile files (disk full?).";
        }
    }

    if (ok) std::cout << "[Offline] " << status << std::endl;
    else    std::cerr << "[Offline] " << status << std::endl;

    cleanupGpu();
    running = false;
    return ok;
}

void OfflineConverter::abortAllWriters()
{
    if (!tiled) { singleWriter.abort(); return; }
    for (auto& [key, tile] : tiles) tile.writer.abort(); // closes + deletes each tile file
    std::error_code ec;
    std::filesystem::remove(std::filesystem::path(tilesDir) / "manifest.json", ec);
    std::filesystem::remove(tilesDir, ec); // removed only if empty
}

void OfflineConverter::cancel()
{
    if (!running) return;
    abortAllWriters();
    cleanupGpu();
    work.clear();
    tiles.clear();
    running = false;
    status  = "Cancelled -- partial output deleted.";
    std::cout << "[Offline] " << status << std::endl;
}

float OfflineConverter::progress01() const
{
    if (totalVertices == 0) return 0.0f;
    return static_cast<float>(static_cast<double>(processedVertices) /
                              static_cast<double>(totalVertices));
}

void OfflineConverter::fail(const std::string& why)
{
    status = "FAILED: " + why;
    std::cerr << "[Offline] " << status << std::endl;
    abortAllWriters();
    cleanupGpu();
    work.clear();
    tiles.clear();
    running = false;
}

void OfflineConverter::cleanupGpu()
{
    if (offlineGaussianBuffer != 0) {
        glDeleteBuffers(1, &offlineGaussianBuffer);
        offlineGaussianBuffer = 0;
    }
    readback.clear();
    readback.shrink_to_fit();
    tileBucket.clear();
    tileBucket.shrink_to_fit();
}