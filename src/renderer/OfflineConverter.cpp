///////////////////////////////////////////////////////////////////////////////
//         Mesh2Splat: fast mesh to 3D gaussian splat conversion             //
//        Copyright (c) 2025 Electronic Arts Inc. All rights reserved.       //
///////////////////////////////////////////////////////////////////////////////

#include "OfflineConverter.hpp"
#include "utils/params.hpp"

#include <iostream>
#include <sstream>
#include <algorithm>

namespace
{
    // Format big counts with thousands separators for the status line.
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
                             unsigned int requestedBatchCapacity)
{
    if (running) return false;

    if (ctx.dataMeshAndGlMesh.empty()) {
        status = "No mesh loaded.";
        return false;
    }
    if (outputPath.empty()) {
        status = "No output path.";
        return false;
    }

    batchCapacity     = std::max(requestedBatchCapacity, 1000u);
    totalVertices     = 0;
    processedVertices = 0;
    batchesDone       = 0;
    work.clear();

    // One initial work range per mesh: the whole mesh. Ranges that overflow
    // the batch buffer get split adaptively in step().
    for (size_t i = 0; i < ctx.dataMeshAndGlMesh.size(); ++i)
    {
        GLsizei vertexCount = static_cast<GLsizei>(ctx.dataMeshAndGlMesh[i].second.vertexCount);
        if (vertexCount <= 0) continue;
        work.push_back({ i, 0, vertexCount });
        totalVertices += static_cast<uint64_t>(vertexCount);
    }
    if (work.empty()) {
        status = "Mesh has no triangles.";
        return false;
    }

    // Same scale semantics as the live exporter (SceneManager::exportPly).
    float scaleMultiplier = ctx.gaussianStd / static_cast<float>(ctx.resolutionTarget);

    if (!writer.open(outputPath, scaleMultiplier)) {
        status = "Could not open output file: " + outputPath;
        return false;
    }

    // One fixed-size gaussian output buffer, reused for every batch.
    glGenBuffers(1, &offlineGaussianBuffer);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, offlineGaussianBuffer);
    GLsizeiptr bufferSize = static_cast<GLsizeiptr>(batchCapacity) * sizeof(utils::GaussianDataSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, bufferSize, nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    running = true;
    status  = "Starting...";
    std::cout << "[Offline] Conversion started -> " << outputPath
              << "  (batch capacity " << withCommas(batchCapacity) << " gaussians, "
              << withCommas(bufferSize / (1024 * 1024)) << " MB VRAM)" << std::endl;
    return true;
}

bool OfflineConverter::step(RenderContext& ctx)
{
    if (!running) return false;

    if (work.empty()) {
        // All ranges done: patch the header with the final count and close.
        if (writer.finalize()) {
            status = "Done: " + withCommas(writer.writtenCount()) + " gaussians written.";
            std::cout << "[Offline] " << status << std::endl;
        } else {
            status = "FAILED while finalizing the output file (disk full?).";
            std::cerr << "[Offline] " << status << std::endl;
        }
        cleanupGpu();
        running = false;
        return false;
    }

    WorkRange range = work.front();
    work.pop_front();

    auto& mesh = ctx.dataMeshAndGlMesh[range.meshIndex];

    // --- One conversion pass over this triangle range. Mirrors
    // ConversionPass::execute/conversion, but draws only the range and
    // appends into our own fixed-size buffer.
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

    // Per-mesh material uniforms/textures (same as ConversionPass::conversion).
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

    // How many gaussians did this range WANT to produce? (The counter keeps
    // counting past the cap; excess fragments were discarded, so overflow is
    // always detectable and nothing corrupt can be written.)
    uint32_t numGs = 0;
    glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, ctx.atomicCounterBufferConversionPass);
    glGetBufferSubData(GL_ATOMIC_COUNTER_BUFFER, 0, sizeof(uint32_t), &numGs);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteRenderbuffers(1, &drawBuffers);
    glDeleteFramebuffers(1, &framebuffer);

    if (numGs > batchCapacity)
    {
        // Overflow: this range's output doesn't fit. Split it in half
        // (triangle-aligned) and re-queue both halves at the front so
        // processing order is preserved. Written file is untouched.
        if (range.vertexCount <= 3) {
            fail("A single triangle produced " + withCommas(numGs) +
                 " gaussians, exceeding the batch capacity of " + withCommas(batchCapacity) +
                 ". Increase OfflineConverter::kDefaultBatchCapacity or lower the resolution target.");
            return false;
        }
        GLsizei halfTriangles = (range.vertexCount / 3) / 2;
        GLsizei half          = std::max<GLsizei>(halfTriangles, 1) * 3;

        WorkRange a{ range.meshIndex, range.firstVertex,        half };
        WorkRange b{ range.meshIndex, range.firstVertex + half, range.vertexCount - half };
        work.push_front(b);
        work.push_front(a);

        status = "Batch too dense (" + withCommas(numGs) + " gaussians) -- splitting and retrying...";
        std::cout << "[Offline] " << status << std::endl;
        return true;
    }

    // Read back only what was produced and stream it to disk.
    if (numGs > 0)
    {
        readback.resize(numGs);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, offlineGaussianBuffer);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
                           static_cast<GLsizeiptr>(numGs) * sizeof(utils::GaussianDataSSBO),
                           readback.data());
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

        if (!writer.appendBatch(readback)) {
            fail("Write to disk failed (disk full?). Partial file removed.");
            return false;
        }
    }

    processedVertices += static_cast<uint64_t>(range.vertexCount);
    ++batchesDone;

    std::ostringstream ss;
    ss << "Batch " << batchesDone << " done -- "
       << withCommas(writer.writtenCount()) << " gaussians written ("
       << (writer.writtenCount() * sizeof(float) * 62) / (1024 * 1024) << " MB)";
    status = ss.str();

    return true;
}

void OfflineConverter::cancel()
{
    if (!running) return;
    writer.abort();  // closes and deletes the partial file
    cleanupGpu();
    work.clear();
    running = false;
    status  = "Cancelled -- partial file deleted.";
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
    writer.abort();
    cleanupGpu();
    work.clear();
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
}