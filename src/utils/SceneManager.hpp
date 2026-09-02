///////////////////////////////////////////////////////////////////////////////
//         Mesh2Splat: fast mesh to 3D gaussian splat conversion             //
//        Copyright (c) 2025 Electronic Arts Inc. All rights reserved.       //
///////////////////////////////////////////////////////////////////////////////

#pragma once
#include "utils.hpp"
#include "renderer/renderPasses/RenderContext.hpp"
#include "normalizedUvUnwrapping.hpp"
#include "parsers/parsers.hpp"
#include "renderer/RenderPasses.hpp"

class SceneManager {
public:
    SceneManager(RenderContext& context);
    ~SceneManager();

    bool loadModel(const std::string& filePath, const std::string& parentFolder, int splitFactor = 1);
    // Loads every .glb in a folder (RealityScan "save mesh by parts" output) as
    // ONE combined scene, appending all parts' meshes. Each part keeps its own
    // material/texture (names are prefixed per part so textures don't collide).
    // The offline tiler then buckets all parts into the world quadtree.
    bool loadModelFolder(const std::string& folderPath, int splitFactor = 1,
                         const std::string& nameFilter = "");
    // FBX / OBJ mesh import via assimp (RealityScan parts that aren't GLB/PLY).
    bool parseMeshFileAssimp(const std::string& path, std::vector<utils::Mesh>& meshes);
    // Binary mesh PLY via happly directly (NO assimp) -- lets a 180M-face mesh
    // load without assimp's transient memory blow-up. Same UDIM / diffuse-in-
    // folder texturing as the assimp path.
    bool parseMeshPly(const std::string& path, std::vector<utils::Mesh>& meshes);
    bool loadPly(const std::string& filePath);
    void exportPly(const std::string outputFile, unsigned int exportFormat);

    void updateMeshes();
    void cleanup();

private:
    RenderContext& renderContext;

    bool parseGltfFile(const std::string& filePath, const std::string& parentFolder, std::vector<utils::Mesh>& meshes);
    void parseGltfMaterial(const tinygltf::Model& model, int materialIndex, std::string base_folder, utils::MaterialGltf& materialGltf);
    void parseGltfTextureInfo(const tinygltf::Model& model, const tinygltf::Parameter& textureParameter, std::string base_folder, std::string name, utils::TextureInfo& info);
    void generateNormalizedUvCoordinates(std::vector<utils::Mesh>& meshes);
    void loadTextures(const std::vector<utils::Mesh>& meshes);
    void setupMeshBuffers(std::vector<utils::Mesh>& meshes);
    template <typename T>
    const T* getBufferData(const tinygltf::Model& model, int accessorIndex);
};
