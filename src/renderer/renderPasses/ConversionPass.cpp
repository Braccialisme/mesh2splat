///////////////////////////////////////////////////////////////////////////////
//         Mesh2Splat: fast mesh to 3D gaussian splat conversion             //
//        Copyright (c) 2025 Electronic Arts Inc. All rights reserved.       //
///////////////////////////////////////////////////////////////////////////////

#pragma once
#include "ConversionPass.hpp"

void ConversionPass::execute(RenderContext &renderContext)
{
#ifdef  _DEBUG
    glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, PassesDebugIDs::CONVERSION_PASS, -1, "CONVERSION_VS_GS_PS_PASS");
#endif 

    glUseProgram(renderContext.shaderRegistry.getProgramID(glUtils::ShaderProgramTypes::ConverterProgram));

    renderContext.numberOfGaussians = 0;
    glUtils::resetAtomicCounter(renderContext.atomicCounterBufferConversionPass);

    // Scale buffer capacity by mesh count so multi-mesh GLBs have room for all gaussians
    unsigned int meshCount = static_cast<unsigned int>(std::max(size_t(1), renderContext.dataMeshAndGlMesh.size()));
    unsigned int maxGaussians = renderContext.resolutionTarget * renderContext.resolutionTarget * 6 * meshCount;
    // We clamp to MAX_GAUSSIANS_TO_SORT since downstream buffers (sort, prepass) are fixed at that size
    maxGaussians = std::min(maxGaussians, static_cast<unsigned int>(MAX_GAUSSIANS_TO_SORT));
    GLsizeiptr bufferSize = static_cast<GLsizeiptr>(maxGaussians) * sizeof(glm::vec4) * 6;
    GLint currentSize;
    
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, renderContext.gaussianBuffer);    
    glGetBufferParameteriv(GL_SHADER_STORAGE_BUFFER, GL_BUFFER_SIZE, &currentSize);
    
    if (currentSize != bufferSize) {
        glBufferData(GL_SHADER_STORAGE_BUFFER, bufferSize, nullptr, GL_DYNAMIC_DRAW);
    }

    GLuint converterProgramIDSetup = renderContext.shaderRegistry.getProgramID(glUtils::ShaderProgramTypes::ConverterProgram);
    glUtils::setUniform1i(converterProgramIDSetup, "u_maxGaussians", static_cast<int>(maxGaussians));

    GLuint framebuffer;
    GLuint drawBuffers = glUtils::setupFrameBuffer(framebuffer, renderContext.resolutionTarget, renderContext.resolutionTarget);
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
    
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, renderContext.gaussianBuffer);    
    glBindBufferBase(GL_ATOMIC_COUNTER_BUFFER, 1, renderContext.atomicCounterBufferConversionPass);
    
    glViewport(0, 0, int(renderContext.resolutionTarget), int(renderContext.resolutionTarget));
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glDisable(GL_CULL_FACE);

    for (auto& mesh : renderContext.dataMeshAndGlMesh) {
        conversion(renderContext, mesh, framebuffer);
    }

    glFinish();

    glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, renderContext.atomicCounterBufferConversionPass);
    uint32_t numGs;
    glGetBufferSubData(GL_ATOMIC_COUNTER_BUFFER, 0, sizeof(uint32_t), &numGs);
    renderContext.numberOfGaussians = std::min(numGs, maxGaussians);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    glDeleteRenderbuffers(1, &drawBuffers); 
    glDeleteFramebuffers(1, &framebuffer);

#ifdef  _DEBUG
    glPopDebugGroup();
#endif 
}

void ConversionPass::conversion(
        RenderContext& renderContext, std::pair<utils::Mesh, utils::GLMesh>& mesh, GLuint dummyFramebuffer
    ) 
{

    GLuint converterProgramID = renderContext.shaderRegistry.getProgramID(glUtils::ShaderProgramTypes::ConverterProgram);

    glUtils::setUniform1i(converterProgramID, "hasAlbedoMap", 0);
    glUtils::setUniform1i(converterProgramID, "hasNormalMap", 0);
    glUtils::setUniform1i(converterProgramID, "hasMetallicRoughnessMap", 0);

    if (renderContext.meshToTextureData.find(mesh.first.name) != renderContext.meshToTextureData.end())
    {
        auto& textureMap = renderContext.meshToTextureData.at(mesh.first.name);

        if (textureMap.find(BASE_COLOR_TEXTURE) != textureMap.end())
        {
            glUtils::setTexture2D(converterProgramID, "albedoTexture", textureMap.at(BASE_COLOR_TEXTURE).glTextureID, 0);
            glUtils::setUniform1i(converterProgramID, "hasAlbedoMap", 1);
        }
        if (textureMap.find(NORMAL_TEXTURE) != textureMap.end())
        {
            glUtils::setTexture2D(converterProgramID, "normalTexture", textureMap.at(NORMAL_TEXTURE).glTextureID,         1);
            glUtils::setUniform1i(converterProgramID, "hasNormalMap", 1);
        }
        if (textureMap.find(METALLIC_ROUGHNESS_TEXTURE) != textureMap.end())
        {
            glUtils::setTexture2D(converterProgramID, "metallicRoughnessTexture", textureMap.at(METALLIC_ROUGHNESS_TEXTURE).glTextureID,     2);
            glUtils::setUniform1i(converterProgramID, "hasMetallicRoughnessMap", 1);
        }
    }

    glUtils::setUniform4f(converterProgramID,      "u_materialFactor", mesh.first.material.baseColorFactor);
    glUtils::setUniform3f(converterProgramID,      "u_bboxMin", mesh.first.bbox.min);
    glUtils::setUniform3f(converterProgramID,      "u_bboxMax", mesh.first.bbox.max);

    glBindVertexArray(mesh.second.vao);

    glDrawArrays(GL_TRIANGLES, 0, mesh.second.vertexCount); 
 
}