///////////////////////////////////////////////////////////////////////////////
//         Mesh2Splat: fast mesh to 3D gaussian splat conversion             //
//        Copyright (c) 2025 Electronic Arts Inc. All rights reserved.       //
///////////////////////////////////////////////////////////////////////////////

#include "SceneManager.hpp"
#include <iostream>
#include <cstring>
#include <functional>
#include <filesystem>
#include <algorithm>
#include <map>
#include <cmath>
#include <cctype>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <assimp/material.h>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

SceneManager::SceneManager(RenderContext& context) : renderContext(context)
{
}

SceneManager::~SceneManager() {
    cleanup();
}


// Splits each mesh into an n x n grid of sub-meshes by triangle centroid on
// the XZ plane (whole triangles, no cutting). Each sub-mesh keeps the parent's
// material AND name -- name is the texture-map key (ConversionPass looks up by
// name), so sub-meshes share the parent's single texture upload. The payoff:
// the converter projects EACH mesh's own bbox into the full resolution grid
// (converterGS: gl_Position = bbox-normalized position), so N^2 sub-meshes
// yield up to N^2 x resolution^2 gaussians -- the way past the single-mesh
// ~grid^2 ceiling (67M at 8192) toward hundreds of millions. Splat positions
// stay exact world coords, so sub-mesh borders abut without seams.
static std::vector<utils::Mesh> splitMeshesIntoGrid(const std::vector<utils::Mesh>& meshes, int n)
{
    if (n <= 1) return meshes;

    std::vector<utils::Mesh> out;
    for (const auto& src : meshes)
    {
        if (src.faces.empty()) { out.push_back(src); continue; }

        float minX = FLT_MAX, minZ = FLT_MAX, maxX = -FLT_MAX, maxZ = -FLT_MAX;
        for (const auto& f : src.faces)
            for (int i = 0; i < 3; ++i) {
                minX = std::min(minX, f.pos[i].x); maxX = std::max(maxX, f.pos[i].x);
                minZ = std::min(minZ, f.pos[i].z); maxZ = std::max(maxZ, f.pos[i].z);
            }
        const float spanX = std::max(maxX - minX, 1e-6f);
        const float spanZ = std::max(maxZ - minZ, 1e-6f);

        std::vector<utils::Mesh> cells;
        cells.reserve(n * n);
        for (int c = 0; c < n * n; ++c) {
            utils::Mesh m(src.name);          // SAME name -> shared texture
            m.material = src.material;
            cells.push_back(std::move(m));
        }

        for (const auto& f : src.faces) {
            const float cx = (f.pos[0].x + f.pos[1].x + f.pos[2].x) / 3.0f;
            const float cz = (f.pos[0].z + f.pos[1].z + f.pos[2].z) / 3.0f;
            int ix = static_cast<int>((cx - minX) / spanX * n);
            int iz = static_cast<int>((cz - minZ) / spanZ * n);
            ix = std::min(std::max(ix, 0), n - 1);
            iz = std::min(std::max(iz, 0), n - 1);
            cells[ix * n + iz].faces.push_back(f);
        }

        for (auto& c : cells)
            if (!c.faces.empty()) out.push_back(std::move(c));
    }

    std::cout << "[Split] " << meshes.size() << " mesh(es) -> " << out.size()
              << " sub-mesh(es) (" << n << "x" << n << " grid per mesh)" << std::endl;
    return out;
}

// Fills a TextureInfo (in-memory pixels) from an assimp material texture slot,
// handling both embedded textures (path "*N" -> scene->mTextures) and external
// files (loaded from baseDir). Same in-memory format the GLB path produces.
static void loadAssimpTexture(const aiScene* scene, const aiMaterial* mat, aiTextureType type,
                              const std::string& baseDir, utils::TextureInfo& out)
{
    aiString texPath;
    if (mat->GetTexture(type, 0, &texPath) != AI_SUCCESS) return;
    std::string p = texPath.C_Str();
    std::cout << "  [tex] type " << (int)type << " path=\"" << p << "\""
              << (p.size() && p[0]=='*' ? " (embedded)" : " (external)") << std::endl;
    int w = 0, h = 0, c = 0;
    unsigned char* data = nullptr;

    if (!p.empty() && p[0] == '*') {                       // embedded texture
        int idx = std::atoi(p.c_str() + 1);
        if (idx >= 0 && idx < static_cast<int>(scene->mNumTextures)) {
            const aiTexture* tex = scene->mTextures[idx];
            if (tex->mHeight == 0) {                        // compressed (png/jpg bytes)
                data = stbi_load_from_memory(reinterpret_cast<const unsigned char*>(tex->pcData),
                                             static_cast<int>(tex->mWidth), &w, &h, &c, 0);
            } else {                                        // raw aiTexel (BGRA)
                w = tex->mWidth; h = tex->mHeight; c = 4;
                out.texture.resize(static_cast<size_t>(w) * h * 4);
                for (size_t i = 0; i < static_cast<size_t>(w) * h; ++i) {
                    const aiTexel& t = tex->pcData[i];
                    out.texture[i*4+0] = t.r; out.texture[i*4+1] = t.g;
                    out.texture[i*4+2] = t.b; out.texture[i*4+3] = t.a;
                }
                out.width = w; out.height = h; out.channels = 4; out.path = "embedded";
                return;
            }
        }
    } else {                                               // external file
        std::string full = p;
        if (!baseDir.empty() && !std::filesystem::path(p).is_absolute())
            full = baseDir + "/" + p;
        data = stbi_load(full.c_str(), &w, &h, &c, 0);
        if (!data) {                                       // RS sometimes writes absolute/old paths: retry basename in baseDir
            std::string bn = std::filesystem::path(p).filename().string();
            data = stbi_load((baseDir + "/" + bn).c_str(), &w, &h, &c, 0);
        }
    }

    if (data) {
        out.texture.assign(data, data + static_cast<size_t>(w) * h * c);
        out.width = w; out.height = h; out.channels = c; out.path = "assimp";
        stbi_image_free(data);
        std::cout << "        loaded " << w << "x" << h << " (" << c << " ch)" << std::endl;
    } else if (out.texture.empty()) {
        std::cout << "        FAILED to load texture (missing file / unreadable)" << std::endl;
    }
}

// Loads an external image file into a TextureInfo (in-memory pixels).
static bool loadTextureFile(const std::string& fullPath, utils::TextureInfo& out)
{
    int w = 0, h = 0, c = 0;
    unsigned char* data = stbi_load(fullPath.c_str(), &w, &h, &c, 0);
    if (!data) return false;
    out.texture.assign(data, data + static_cast<size_t>(w) * h * c);
    out.width = w; out.height = h; out.channels = c; out.path = fullPath;
    stbi_image_free(data);
    return true;
}

// UDIM: build the per-tile filename from the pattern assimp reported. Handles a
// literal "<UDIM>" token, or a 4-digit UDIM number (1001-1100) between dots.
static std::string udimFilename(const std::string& pattern, int udim)
{
    std::string s = pattern;
    // RealityScan "_u<col>_v<row>_" naming (1-based tiles). Decode udim back to
    // tiles: udim = 1001 + uTile + 10*vTile -> column = uTile+1, row = vTile+1.
    if (s.find("<U>") != std::string::npos || s.find("<V>") != std::string::npos) {
        const int uTile = (udim - 1001) % 10;
        const int vTile = (udim - 1001) / 10;
        size_t p;
        while ((p = s.find("<U>")) != std::string::npos) s.replace(p, 3, std::to_string(uTile + 1));
        while ((p = s.find("<V>")) != std::string::npos) s.replace(p, 3, std::to_string(vTile + 1));
        return s;
    }
    size_t pos = s.find("<UDIM>");
    if (pos != std::string::npos) { s.replace(pos, 6, std::to_string(udim)); return s; }
    // replace ".NNNN." where NNNN starts with 1 (UDIM tiles are 1001+)
    for (size_t i = 1; i + 5 < s.size(); ++i) {
        if (s[i-1] == '.' && s[i] == '1' && s[i+4] == '.' &&
            std::isdigit((unsigned char)s[i+1]) && std::isdigit((unsigned char)s[i+2]) && std::isdigit((unsigned char)s[i+3])) {
            s.replace(i, 4, std::to_string(udim));
            return s;
        }
    }
    return s;
}

// Turn a "_u<N>_v<M>_" RealityScan UDIM marker into "_u<U>_v<M>..._v<V>" tokens
// so udimFilename can substitute per tile. Returns true if it rewrote s.
static bool tokenizeUvUdim(std::string& s)
{
    for (size_t i = 0; i + 4 < s.size(); ++i) {
        if (s[i] == '_' && (s[i+1]=='u'||s[i+1]=='U') && std::isdigit((unsigned char)s[i+2])) {
            size_t j = i + 2; while (j < s.size() && std::isdigit((unsigned char)s[j])) ++j;
            if (j + 2 < s.size() && s[j]=='_' && (s[j+1]=='v'||s[j+1]=='V') && std::isdigit((unsigned char)s[j+2])) {
                size_t k = j + 2; while (k < s.size() && std::isdigit((unsigned char)s[k])) ++k;
                // replace the v-digits first (later in string) then the u-digits
                s.replace(j + 2, k - (j + 2), "<V>");
                s.replace(i + 2, j - (i + 2), "<U>");
                return true;
            }
        }
    }
    return false;
}

// RealityScan often exports the mesh material WITHOUT a texture link -- the
// diffuse images just sit next to the file (e.g. "Name_diffuse.1001.jpg"). When
// the material carries no diffuse, find those files by name and return a
// filename pattern (with "<UDIM>" when tiled) to use as the diffuse source.
static std::string findDiffuseInFolder(const std::string& baseDir, const std::string& fileStem, bool& isUdimOut)
{
    namespace fs = std::filesystem;
    isUdimOut = false;
    auto tolow = [](std::string s){ std::transform(s.begin(), s.end(), s.begin(),
                    [](unsigned char c){ return (char)std::tolower(c); }); return s; };

    // Strip a trailing LOD marker so "Name_LOD4" matches "Name_diffuse.*".
    std::string stem = fileStem, low = tolow(fileStem);
    size_t lp = low.rfind("_lod");
    if (lp == std::string::npos) lp = low.rfind(".lod");
    if (lp != std::string::npos) stem = stem.substr(0, lp);
    std::string steml = tolow(stem);

    auto isImg = [&](const std::string& e){ std::string x = tolow(e);
        return x==".jpg"||x==".jpeg"||x==".png"||x==".tif"||x==".tiff"||x==".bmp"; };
    auto isDiffuse = [&](const std::string& n){ std::string x = tolow(n);
        return x.find("diffuse")!=std::string::npos || x.find("albedo")!=std::string::npos ||
               x.find("basecolor")!=std::string::npos || x.find("base_color")!=std::string::npos; };

    std::error_code ec;
    std::vector<std::string> diffuse, stemMatch;
    for (const auto& e : fs::directory_iterator(baseDir, ec)) {
        if (!e.is_regular_file() || !isImg(e.path().extension().string())) continue;
        std::string fn = e.path().filename().string();
        if (isDiffuse(fn)) diffuse.push_back(fn);
        else if (!steml.empty() && tolow(fn).rfind(steml, 0) == 0) stemMatch.push_back(fn);
    }
    std::vector<std::string>& cands = !diffuse.empty() ? diffuse : stemMatch;
    if (cands.empty()) return "";
    std::sort(cands.begin(), cands.end());

    // Turn a UDIM token in the first candidate into a substitutable pattern:
    // ".NNNN." (Mari/standard) -> "<UDIM>", or "_uN_vM_" (RealityScan) -> tokens.
    std::string pat = cands.front();
    bool tokenized = false;
    for (size_t i = 1; i + 5 < pat.size(); ++i) {
        if (pat[i-1]=='.' && pat[i]=='1' && pat[i+4]=='.' &&
            std::isdigit((unsigned char)pat[i+1]) && std::isdigit((unsigned char)pat[i+2]) && std::isdigit((unsigned char)pat[i+3])) {
            pat.replace(i, 4, "<UDIM>"); isUdimOut = true; tokenized = true; break;
        }
    }
    if (!tokenized && cands.size() > 1 && tokenizeUvUdim(pat)) isUdimOut = true;
    std::cout << "  [diffuse-in-folder] using \"" << pat << "\""
              << (isUdimOut ? " (UDIM)" : "") << " from " << cands.size() << " candidate file(s)" << std::endl;
    return pat;
}

// Loads a mesh file (FBX / PLY / OBJ) via assimp into utils::Mesh(es), matching
// the format parseGltfFile produces (per-triangle Face with pos/uv/normal/tangent
// + material textures). This is how RealityScan "by parts" (FBX/PLY) gets in.
// UDIM diffuse textures (multiple 1001/1002/... tiles) are split: faces are
// grouped by UV tile, UVs remapped to [0,1], and each group gets its tile image.
bool SceneManager::parseMeshFileAssimp(const std::string& path, std::vector<utils::Mesh>& meshes)
{
    Assimp::Importer importer;
    const aiScene* scene = importer.ReadFile(path,
        aiProcess_Triangulate | aiProcess_GenSmoothNormals | aiProcess_CalcTangentSpace |
        aiProcess_JoinIdenticalVertices | aiProcess_FlipUVs | aiProcess_PreTransformVertices);
    if (!scene || (scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE) || !scene->mRootNode) {
        std::cerr << "assimp load failed (" << path << "): " << importer.GetErrorString() << std::endl;
        return false;
    }
    const std::string baseDir = std::filesystem::path(path).parent_path().string();
    std::cout << "[assimp] " << std::filesystem::path(path).filename().string() << ": "
              << scene->mNumMeshes << " mesh(es), " << scene->mNumMaterials << " material(s), "
              << scene->mNumTextures << " embedded texture(s)"
              << (scene->mNumTextures == 0 ? "  (textures external or none)" : "") << std::endl;

    // Fallback diffuse pattern from sibling files, used when a material has no
    // texture link (common with RealityScan FBX/PLY exports).
    bool folderIsUdim = false;
    const std::string folderDiffusePattern =
        findDiffuseInFolder(baseDir, std::filesystem::path(path).stem().string(), folderIsUdim);

    for (unsigned mi = 0; mi < scene->mNumMeshes; ++mi) {
        const aiMesh* am = scene->mMeshes[mi];
        if (!am->HasPositions() || am->mNumFaces == 0) continue;

        const std::string baseName = am->mName.length ? std::string(am->mName.C_Str())
                                                      : ("mesh_" + std::to_string(mi));

        // --- material (shared across this mesh's UDIM tiles) ---
        utils::MaterialGltf material;          // normal/metallic + factor; base color set per tile
        std::string diffusePattern;            // external diffuse path pattern (may contain <UDIM>)
        bool diffuseEmbedded = false;
        const aiMaterial* mat = (am->mMaterialIndex < scene->mNumMaterials)
                              ? scene->mMaterials[am->mMaterialIndex] : nullptr;
        if (mat) {
            aiString ap;
            if (mat->GetTexture(aiTextureType_BASE_COLOR, 0, &ap) == AI_SUCCESS) diffusePattern = ap.C_Str();
            else if (mat->GetTexture(aiTextureType_DIFFUSE, 0, &ap) == AI_SUCCESS) diffusePattern = ap.C_Str();
            diffuseEmbedded = (!diffusePattern.empty() && diffusePattern[0] == '*');
            loadAssimpTexture(scene, mat, aiTextureType_NORMALS,  baseDir, material.normalTexture);
            loadAssimpTexture(scene, mat, aiTextureType_METALNESS, baseDir, material.metallicRoughnessTexture);
            aiColor4D col;
            if (aiGetMaterialColor(mat, AI_MATKEY_COLOR_DIFFUSE, &col) == AI_SUCCESS)
                material.baseColorFactor = glm::vec4(col.r, col.g, col.b, col.a);
        }
        // No texture in the material? Use the sibling files found in the folder.
        if (diffusePattern.empty() && !diffuseEmbedded && !folderDiffusePattern.empty())
            diffusePattern = folderDiffusePattern;

        const bool hasUV = am->HasTextureCoords(0);
        const bool hasN  = am->HasNormals();
        const bool hasT  = am->HasTangentsAndBitangents();

        // --- detect UDIM (multiple tiles): <UDIM> token, or UVs beyond [0,1) ---
        bool isUdim = false;
        if (hasUV && !diffusePattern.empty() && !diffuseEmbedded) {
            if (diffusePattern.find("<UDIM>") != std::string::npos) isUdim = true;
            else for (unsigned i = 0; i < am->mNumVertices && !isUdim; ++i)
                if (am->mTextureCoords[0][i].x >= 1.0f || am->mTextureCoords[0][i].y >= 1.0f) isUdim = true;
        }

        // --- group faces by UDIM tile (single group when not UDIM), remap UVs ---
        std::map<int, std::vector<utils::Face>> tileFaces;  // key = UDIM number
        for (unsigned fi = 0; fi < am->mNumFaces; ++fi) {
            const aiFace& f = am->mFaces[fi];
            if (f.mNumIndices != 3) continue;
            int tileU = 0, tileV = 0;
            if (isUdim) {
                const aiVector3D& t0 = am->mTextureCoords[0][f.mIndices[0]];
                tileU = std::max(0, (int)std::floor(t0.x));
                tileV = std::max(0, (int)std::floor(t0.y));
            }
            const int udim = 1001 + tileU + 10 * tileV;
            utils::Face face{};
            for (int k = 0; k < 3; ++k) {
                unsigned idx = f.mIndices[k];
                const aiVector3D& v = am->mVertices[idx];
                face.pos[k] = glm::vec3(v.x, v.y, v.z);
                if (hasUV) { const aiVector3D& t = am->mTextureCoords[0][idx];
                             face.uv[k] = glm::vec2(t.x - tileU, t.y - tileV); }   // -> [0,1]
                else       face.uv[k] = glm::vec2(0.0f);
                if (hasN)  { const aiVector3D& n = am->mNormals[idx]; face.normal[k] = glm::vec3(n.x, n.y, n.z); }
                else       face.normal[k] = glm::vec3(0.0f, 1.0f, 0.0f);
                if (hasT)  { const aiVector3D& t = am->mTangents[idx]; face.tangent[k] = glm::vec4(t.x, t.y, t.z, 1.0f); }
                else       face.tangent[k] = glm::vec4(1.0f, 0.0f, 0.0f, 1.0f);
            }
            tileFaces[udim].push_back(face);
        }

        // --- one sub-mesh per tile, each with its own base-color texture ---
        for (auto& kv : tileFaces) {
            const int udim = kv.first;
            utils::Mesh mesh(isUdim ? (baseName + "_udim" + std::to_string(udim)) : baseName);
            mesh.material = material;

            if (diffuseEmbedded && mat) {
                loadAssimpTexture(scene, mat, aiTextureType_BASE_COLOR, baseDir, mesh.material.baseColorTexture);
                if (mesh.material.baseColorTexture.texture.empty())
                    loadAssimpTexture(scene, mat, aiTextureType_DIFFUSE, baseDir, mesh.material.baseColorTexture);
            } else if (!diffusePattern.empty()) {
                std::string fn   = isUdim ? udimFilename(diffusePattern, udim) : diffusePattern;
                std::string full = std::filesystem::path(fn).is_absolute() ? fn : (baseDir + "/" + fn);
                if (!loadTextureFile(full, mesh.material.baseColorTexture))
                    loadTextureFile(baseDir + "/" + std::filesystem::path(fn).filename().string(),
                                    mesh.material.baseColorTexture);
            }

            if (mesh.material.baseColorTexture.texture.empty())
                std::cout << "  [warn] '" << mesh.name << "': no base-color texture -> WHITE "
                          << (diffusePattern.empty() ? "(no diffuse in material)"
                                                     : ("(tried " + (isUdim ? udimFilename(diffusePattern, udim) : diffusePattern) + ")"))
                          << std::endl;
            else if (isUdim)
                std::cout << "  [udim] tile " << udim << ": " << kv.second.size() << " faces, tex "
                          << mesh.material.baseColorTexture.width << "x" << mesh.material.baseColorTexture.height << std::endl;

            mesh.faces = std::move(kv.second);
            if (!mesh.faces.empty()) meshes.push_back(std::move(mesh));
        }
    }
    return true;
}

// Binary mesh PLY via happly (no assimp). Mirrors parseMeshFileAssimp's UDIM /
// diffuse-in-folder texturing, but sources geometry from happly's compact
// arrays -- so it fits a 180M-face mesh in RAM that assimp cannot import.
bool SceneManager::parseMeshPly(const std::string& path, std::vector<utils::Mesh>& meshes)
{
    parsers::MeshPlyRaw raw;
    if (!parsers::readMeshPlyGeometry(path, raw)) {
        std::cerr << "[ply-mesh] no usable geometry in " << path << std::endl;
        return false;
    }
    const size_t nV = raw.x.size();
    const size_t nF = raw.faces.size();

    // Recenter: GNSS / geo meshes carry huge global coordinates (e.g. ~1e7) that
    // wreck float32 precision in the shader and make every splat viewer unable to
    // render the output (splats sit millions of units from the origin). Subtract
    // a floored origin so positions sit near 0. Logged so the offset can be added
    // back for georeferencing. Only applied when coords are actually large.
    double oxD = raw.x[0], oyD = raw.y[0], ozD = raw.z[0];
    for (size_t i = 1; i < nV; ++i) {
        oxD = std::min(oxD, (double)raw.x[i]);
        oyD = std::min(oyD, (double)raw.y[i]);
        ozD = std::min(ozD, (double)raw.z[i]);
    }
    const bool recenter = (std::fabs(oxD) > 1e4 || std::fabs(oyD) > 1e4 || std::fabs(ozD) > 1e4);
    const float subX = recenter ? (float)std::floor(oxD) : 0.0f;
    const float subY = recenter ? (float)std::floor(oyD) : 0.0f;
    const float subZ = recenter ? (float)std::floor(ozD) : 0.0f;
    if (recenter)
        std::cout << "[ply-mesh] recentering by origin (" << (double)subX << ", " << (double)subY
                  << ", " << (double)subZ << ") -- add back to georeference." << std::endl;

    const std::string baseDir  = std::filesystem::path(path).parent_path().string();
    const std::string baseName = std::filesystem::path(path).stem().string();

    // Diffuse pattern from sibling files (RealityScan PLY has no texture link).
    bool folderIsUdim = false;
    std::string diffusePattern = findDiffuseInFolder(baseDir, baseName, folderIsUdim);

    // UDIM if the pattern is tiled, or any UV lands outside [0,1).
    bool isUdim = false;
    if (raw.hasUV && !diffusePattern.empty()) {
        if (diffusePattern.find("<UDIM>") != std::string::npos) isUdim = true;
        else for (size_t i = 0; i < nV && !isUdim; ++i)
            if (raw.s[i] >= 1.0f || raw.t[i] >= 1.0f) isUdim = true;
    }

    // Group faces by UDIM tile (single group when not UDIM), remap UV to [0,1].
    std::map<int, std::vector<utils::Face>> tileFaces;
    for (size_t fi = 0; fi < nF; ++fi) {
        const std::vector<int>& f = raw.faces[fi];
        if (f.size() != 3) continue;
        if (f[0] < 0 || f[1] < 0 || f[2] < 0 ||
            (size_t)f[0] >= nV || (size_t)f[1] >= nV || (size_t)f[2] >= nV) continue;

        int tileU = 0, tileV = 0;
        if (isUdim && raw.hasUV) {
            tileU = std::max(0, (int)std::floor(raw.s[f[0]]));
            tileV = std::max(0, (int)std::floor(raw.t[f[0]]));
        }
        const int udim = 1001 + tileU + 10 * tileV;

        // Mesh PLY carries no normals -> flat face normal (converter recomputes
        // its own splat orientation anyway; this only feeds the output normal).
        const glm::vec3 p0(raw.x[f[0]], raw.y[f[0]], raw.z[f[0]]);
        const glm::vec3 p1(raw.x[f[1]], raw.y[f[1]], raw.z[f[1]]);
        const glm::vec3 p2(raw.x[f[2]], raw.y[f[2]], raw.z[f[2]]);
        glm::vec3 fn = glm::cross(p1 - p0, p2 - p0);
        float fl = glm::length(fn);
        fn = (fl > 0.0f) ? fn / fl : glm::vec3(0.0f, 1.0f, 0.0f);

        utils::Face face{};
        for (int k = 0; k < 3; ++k) {
            const int idx = f[k];
            face.pos[k]    = glm::vec3(raw.x[idx] - subX, raw.y[idx] - subY, raw.z[idx] - subZ);
            face.uv[k]     = raw.hasUV ? glm::vec2(raw.s[idx] - tileU, raw.t[idx] - tileV)
                                       : glm::vec2(0.0f);
            face.normal[k] = fn;
            face.tangent[k]= glm::vec4(1.0f, 0.0f, 0.0f, 1.0f);
            if (raw.hasColor) face.color[k] = glm::vec3(raw.r[idx], raw.g[idx], raw.b[idx]);
        }
        tileFaces[udim].push_back(face);
    }
    // Free the big happly arrays before we hold the face-soup + textures.
    raw.faces.clear(); raw.faces.shrink_to_fit();

    for (auto& kv : tileFaces) {
        const int udim = kv.first;
        utils::Mesh mesh(isUdim ? (baseName + "_udim" + std::to_string(udim)) : baseName);

        if (!diffusePattern.empty()) {
            std::string fn   = isUdim ? udimFilename(diffusePattern, udim) : diffusePattern;
            std::string full = std::filesystem::path(fn).is_absolute() ? fn : (baseDir + "/" + fn);
            if (!loadTextureFile(full, mesh.material.baseColorTexture))
                loadTextureFile(baseDir + "/" + std::filesystem::path(fn).filename().string(),
                                mesh.material.baseColorTexture);
        }

        if (mesh.material.baseColorTexture.texture.empty())
            std::cout << "  [warn] '" << mesh.name << "': no base-color texture -> WHITE "
                      << (diffusePattern.empty() ? "(no diffuse found in folder)"
                                                 : ("(tried " + (isUdim ? udimFilename(diffusePattern, udim) : diffusePattern) + ")"))
                      << std::endl;
        else if (isUdim)
            std::cout << "  [udim] tile " << udim << ": " << kv.second.size() << " faces, tex "
                      << mesh.material.baseColorTexture.width << "x" << mesh.material.baseColorTexture.height << std::endl;

        mesh.faces = std::move(kv.second);
        if (!mesh.faces.empty()) meshes.push_back(std::move(mesh));
    }
    return !meshes.empty();
}

// Route a mesh file to the right parser: GLB via tinygltf, PLY via happly
// (mesh), FBX/OBJ via assimp.
static bool lowerExtIs(const std::string& path, const char* ext) {
    std::string e = std::filesystem::path(path).extension().string();
    std::transform(e.begin(), e.end(), e.begin(), [](unsigned char c){ return (char)std::tolower(c); });
    return e == ext;
}

bool SceneManager::loadModel(const std::string& filePath, const std::string& parentFolder, int splitFactor) {
    std::vector<utils::Mesh> meshes;
    const bool ok = lowerExtIs(filePath, ".glb") ? parseGltfFile(filePath, parentFolder, meshes)
                  : lowerExtIs(filePath, ".ply") ? parseMeshPly(filePath, meshes)
                                                 : parseMeshFileAssimp(filePath, meshes);
    if (!ok) {
        std::cerr << "Failed to parse mesh file: " << filePath << std::endl;
        return false;
    }

    if (splitFactor > 1)
        meshes = splitMeshesIntoGrid(meshes, splitFactor);

    //generateNormalizedUvCoordinates(meshes);
    setupMeshBuffers(meshes);
    loadTextures(meshes);
    glUtils::generateTextures(renderContext.meshToTextureData);

    return true;
}

bool SceneManager::loadModelFolder(const std::string& folderPath, int splitFactor,
                                   const std::string& nameFilter) {
    namespace fs = std::filesystem;
    std::vector<utils::Mesh> allMeshes;
    int parts = 0;

    std::error_code ec;
    if (!fs::is_directory(folderPath, ec)) {
        std::cerr << "loadModelFolder: not a folder: " << folderPath << std::endl;
        return false;
    }

    // Deterministic order so tiling/addresses are reproducible across runs.
    // Accept any supported mesh part format (RealityScan "by parts").
    auto isMeshPart = [](const fs::path& p) {
        std::string e = p.extension().string();
        std::transform(e.begin(), e.end(), e.begin(), [](unsigned char c){ return (char)std::tolower(c); });
        return e == ".glb" || e == ".ply" || e == ".fbx" || e == ".obj";
    };
    std::vector<fs::path> partFiles;
    for (const auto& entry : fs::directory_iterator(folderPath, ec)) {
        if (!entry.is_regular_file() || !isMeshPart(entry.path())) continue;
        // Filename filter: pick one LOD when a folder holds several per part.
        if (!nameFilter.empty() &&
            entry.path().filename().string().find(nameFilter) == std::string::npos) continue;
        partFiles.push_back(entry.path());
    }
    std::sort(partFiles.begin(), partFiles.end());
    if (!nameFilter.empty())
        std::cout << "[Folder] filter \"" << nameFilter << "\" -> " << partFiles.size()
                  << " file(s) match" << std::endl;

    const std::string parentFolder = folderPath + "/";
    for (const auto& file : partFiles) {
        std::vector<utils::Mesh> partMeshes;
        const bool ok = lowerExtIs(file.string(), ".glb") ? parseGltfFile(file.string(), parentFolder, partMeshes)
                      : lowerExtIs(file.string(), ".ply") ? parseMeshPly(file.string(), partMeshes)
                                                          : parseMeshFileAssimp(file.string(), partMeshes);
        if (!ok) {
            std::cerr << "  skipped (parse failed): " << file.filename().string() << std::endl;
            continue;
        }
        // Prefix each part's mesh names so textures (keyed by mesh name) from
        // different parts never collide in loadTextures' dedup.
        const std::string prefix = file.stem().string() + "::";
        for (auto& m : partMeshes) m.name = prefix + m.name;
        for (auto& m : partMeshes) allMeshes.push_back(std::move(m));
        ++parts;
    }

    if (allMeshes.empty()) {
        std::cerr << "loadModelFolder: no loadable .glb parts in " << folderPath << std::endl;
        return false;
    }

    if (splitFactor > 1)
        allMeshes = splitMeshesIntoGrid(allMeshes, splitFactor);

    std::cout << "[Folder] loaded " << parts << " GLB part(s) -> " << allMeshes.size()
              << " mesh(es) total from " << folderPath << std::endl;

    setupMeshBuffers(allMeshes);
    loadTextures(allMeshes);
    glUtils::generateTextures(renderContext.meshToTextureData);
    return true;
}

bool SceneManager::loadPly(const std::string& filePath) {
    try {
        parsers::loadPlyFile(filePath, renderContext.readGaussians, renderContext.plyHasPbr);
        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error loading PLY file: " << filePath << std::endl;
        return false;
    }
    return false;
}

template <typename T>
const T* SceneManager::getBufferData(const tinygltf::Model& model, int accessorIndex) {

    const auto& accessor    = model.accessors[accessorIndex];
    const auto& bufferView  = model.bufferViews[accessor.bufferView];
    const auto& buffer      = model.buffers[bufferView.buffer];
    
    //TODO: This may not be too safe
    const T* dataPtr = reinterpret_cast<const T*>(&buffer.data[bufferView.byteOffset + accessor.byteOffset]);
    
    return dataPtr;
}

void SceneManager::parseGltfTextureInfo(const tinygltf::Model& model, const tinygltf::Parameter& textureParameter, std::string base_folder, std::string name, utils::TextureInfo& info) {

    auto it = textureParameter.json_double_value.find("index");

    if (it != textureParameter.json_double_value.end()) {
        int textureIndex = static_cast<int>(it->second);
        if (textureIndex >= 0 && textureIndex < model.textures.size()) {
            const tinygltf::Texture& texture = model.textures[textureIndex];
            if (texture.source >= 0 && texture.source < model.images.size()) {
                const tinygltf::Image& image = model.images[texture.source];

                std::string fileExtension = image.mimeType.substr(image.mimeType.find_last_of('/') + 1);
                info.path = base_folder + image.name + "." + fileExtension;

                info.width      = image.width;
                info.height     = image.height;

                info.texture.resize(image.image.size());
                std::memcpy(info.texture.data(), image.image.data(), image.image.size());
                
                info.path       = name;
                info.channels   = image.component;
            }

            // Handling texCoord index if present
            auto texCoordIt = textureParameter.json_double_value.find("texCoord");
            if (texCoordIt != textureParameter.json_double_value.end()) {
                info.texCoordIndex = static_cast<int>(texCoordIt->second);
            }
            else {
                info.texCoordIndex = 0; // Default texture coordinate set
            }
        }
    }
}

void SceneManager::parseGltfMaterial(const tinygltf::Model& model, int materialIndex, std::string base_folder, utils::MaterialGltf& materialGltf) {

    if (materialIndex < 0 || materialIndex >= model.materials.size()) {
        return;
    }

    const tinygltf::Material& material = model.materials[materialIndex];

    materialGltf.name = material.name;

    // Base Color Factor
    auto colorIt = material.values.find("baseColorFactor");
    if (colorIt != material.values.end()) {
        materialGltf.baseColorFactor = glm::vec4(
            static_cast<float>(colorIt->second.ColorFactor()[0]),
            static_cast<float>(colorIt->second.ColorFactor()[1]),
            static_cast<float>(colorIt->second.ColorFactor()[2]),
            static_cast<float>(colorIt->second.ColorFactor()[3])
        );
    }

    //R=ambient occlusion G=roughness B=metallic for AO_Roughness_Metallic
    // Base Color Texture
    auto baseColorTexIt = material.values.find("baseColorTexture");
    if (baseColorTexIt != material.values.end()) {
         parseGltfTextureInfo(model, baseColorTexIt->second, base_folder, "baseColorTexture", materialGltf.baseColorTexture);
    }
    else {
        materialGltf.baseColorTexture.path = EMPTY_TEXTURE;
    }

    // Normal Texture
    auto normalTexIt = material.additionalValues.find("normalTexture");
    if (normalTexIt != material.additionalValues.end()) {
         parseGltfTextureInfo(model, normalTexIt->second, base_folder, "normalTexture", materialGltf.normalTexture);

        auto scaleIt = normalTexIt->second.json_double_value.find("scale");
        if (scaleIt != normalTexIt->second.json_double_value.end()) {
            materialGltf.normalScale = static_cast<float>(scaleIt->second);
        }
        else {
            materialGltf.normalScale = 1.0f; // Default scale if not specified
        }
    }
    else {
        materialGltf.normalTexture.path = EMPTY_TEXTURE;
    }

    // Metallic-Roughness Texture
    auto metalRoughTexIt = material.values.find("metallicRoughnessTexture");
    if (metalRoughTexIt != material.values.end()) {
         parseGltfTextureInfo(model, metalRoughTexIt->second, base_folder, "metallicRoughnessTexture", materialGltf.metallicRoughnessTexture);
    }

    // Occlusion Texture
    auto occlusionTexIt = material.additionalValues.find("occlusionTexture");
    if (occlusionTexIt != material.additionalValues.end()) {
        parseGltfTextureInfo(model, occlusionTexIt->second, base_folder, "occlusionTexture", materialGltf.occlusionTexture);

        auto scaleIt = occlusionTexIt->second.json_double_value.find("strength");
        if (scaleIt != occlusionTexIt->second.json_double_value.end()) {
            materialGltf.occlusionStrength = static_cast<float>(scaleIt->second);
        }
        else {
            materialGltf.occlusionStrength = 1.0f; // Default scale if not specified
        }
    }
    else {
        materialGltf.occlusionTexture.path = EMPTY_TEXTURE;
    }

    // Emissive Texture
    auto emissiveTexIt = material.additionalValues.find("emissiveTexture");
    if (emissiveTexIt != material.additionalValues.end()) {
        parseGltfTextureInfo(model, emissiveTexIt->second, base_folder, "emissiveTexture", materialGltf.emissiveTexture);
    }
    else {
        materialGltf.emissiveTexture.path = EMPTY_TEXTURE;
    }


    // Emissive Factor
    auto emissiveFactorIt = material.values.find("emissiveFactor");
    if (emissiveFactorIt != material.values.end()) {
        materialGltf.emissiveFactor = glm::vec3(
            static_cast<float>(emissiveFactorIt->second.number_array[0]),
            static_cast<float>(emissiveFactorIt->second.number_array[1]),
            static_cast<float>(emissiveFactorIt->second.number_array[2])
        );
    }

    // Metallic and Roughness Factors
    materialGltf.metallicFactor = material.pbrMetallicRoughness.metallicFactor;
    materialGltf.roughnessFactor = material.pbrMetallicRoughness.roughnessFactor;
}

bool SceneManager::parseGltfFile(const std::string& filePath, const std::string& parentFolder, std::vector<utils::Mesh>& meshes) {
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err;
    std::string warn;

    
    bool ret = loader.LoadBinaryFromFile(&model, &err, &warn, filePath);
    if (!ret) {
        std::cerr << "Failed to load glTF: " << err << std::endl;
        return false;
    }
    

    if (!warn.empty()) {
        std::cout << "glTF parse warning: " << warn << std::endl;
    }

    struct MeshInstance {
        int meshIndex;
        glm::mat4 transform;
    };
    std::vector<MeshInstance> meshInstances;

    std::function<void(int, const glm::mat4&)> traverseNode = [&](int nodeIndex, const glm::mat4& parentTransform) {
        if (nodeIndex < 0 || nodeIndex >= static_cast<int>(model.nodes.size())) return;
        const tinygltf::Node& node = model.nodes[nodeIndex];

        // Build local transform
        glm::mat4 localTransform(1.0f);

        if (node.matrix.size() == 16) {
            // Column-major matrix provided directly
            for (int c = 0; c < 4; c++)
                for (int r = 0; r < 4; r++)
                    localTransform[c][r] = static_cast<float>(node.matrix[c * 4 + r]);
        } else {
            // TRS
            glm::mat4 T(1.0f), R(1.0f), S(1.0f);
            if (node.translation.size() == 3) {
                T = glm::translate(glm::mat4(1.0f), glm::vec3(
                    static_cast<float>(node.translation[0]),
                    static_cast<float>(node.translation[1]),
                    static_cast<float>(node.translation[2])));
            }
            if (node.rotation.size() == 4) {
                glm::quat q(
                    static_cast<float>(node.rotation[3]),  // w
                    static_cast<float>(node.rotation[0]),  // x
                    static_cast<float>(node.rotation[1]),  // y
                    static_cast<float>(node.rotation[2])); // z
                R = glm::mat4_cast(q);
            }
            if (node.scale.size() == 3) {
                S = glm::scale(glm::mat4(1.0f), glm::vec3(
                    static_cast<float>(node.scale[0]),
                    static_cast<float>(node.scale[1]),
                    static_cast<float>(node.scale[2])));
            }
            localTransform = T * R * S;
        }

        glm::mat4 worldTransform = parentTransform * localTransform;

        if (node.mesh >= 0 && node.mesh < static_cast<int>(model.meshes.size())) {
            meshInstances.push_back({ node.mesh, worldTransform });
        }

        for (int child : node.children) {
            traverseNode(child, worldTransform);
        }
    };

    // Traverse the scene graph to collect mesh instances with their world transforms
    if (!model.scenes.empty()) {
        int sceneIndex = model.defaultScene >= 0 ? model.defaultScene : 0;
        const tinygltf::Scene& scene = model.scenes[sceneIndex];
        for (int rootNode : scene.nodes) {
            traverseNode(rootNode, glm::mat4(1.0f));
        }
    }

    // Fallback: if no mesh instances were found via the scene graph we add all meshes with identity transform
    if (meshInstances.empty()) {
        for (int i = 0; i < static_cast<int>(model.meshes.size()); i++) {
            meshInstances.push_back({ i, glm::mat4(1.0f) });
        }
    }

    //remember that "when a 3D model is created as GLTF it is already triangulated"
    int meshCounter = 0;

    for (const auto& instance : meshInstances) {
        const tinygltf::Mesh& mesh = model.meshes[instance.meshIndex];
        glm::mat4 worldTransform = instance.transform;
        glm::mat3 normalMatrix = glm::transpose(glm::inverse(glm::mat3(worldTransform)));

        for (const auto& primitive : mesh.primitives) {
            // Skip non-triangle primitives
            if (primitive.mode != TINYGLTF_MODE_TRIANGLES && primitive.mode != -1) {
                std::cout << "Skipping non-triangle primitive (mode=" << primitive.mode << ") in mesh: " << mesh.name << std::endl;
                continue;
            }

            // Must have POSITION attribute
            if (primitive.attributes.find("POSITION") == primitive.attributes.end()) {
                std::cerr << "Warning: Primitive in mesh '" << mesh.name << "' has no POSITION attribute, skipping." << std::endl;
                continue;
            }

            std::string baseName = mesh.name.empty() ? "mesh" : mesh.name;
            utils::Mesh myMesh(baseName + "_" + std::to_string(meshCounter));
            ++meshCounter;

            // --- Build index list ---
            std::vector<uint32_t> indices;

            if (primitive.indices >= 0) {
                const tinygltf::Accessor& indicesAccessor = model.accessors[primitive.indices];
                const tinygltf::BufferView& bufferView = model.bufferViews[indicesAccessor.bufferView];
                const tinygltf::Buffer& buffer = model.buffers[bufferView.buffer];
                const unsigned char* indexData = buffer.data.data() + bufferView.byteOffset + indicesAccessor.byteOffset;
                indices.resize(indicesAccessor.count);

                if (indicesAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
                    const uint16_t* buf = reinterpret_cast<const uint16_t*>(indexData);
                    for (size_t i = 0; i < indicesAccessor.count; i++) {
                        indices[i] = buf[i];
                    }
                }
                else if (indicesAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT) {
                    const uint32_t* buf = reinterpret_cast<const uint32_t*>(indexData);
                    for (size_t i = 0; i < indicesAccessor.count; i++) {
                        indices[i] = buf[i];
                    }
                }
                else if (indicesAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE) {
                    const uint8_t* buf = reinterpret_cast<const uint8_t*>(indexData);
                    for (size_t i = 0; i < indicesAccessor.count; i++) {
                        indices[i] = buf[i];
                    }
                }
                else {
                    std::cerr << "Warning: Unsupported index component type " << indicesAccessor.componentType << " in mesh '" << mesh.name << "', skipping primitive." << std::endl;
                    continue;
                }
            } else {
                // Non-indexed geometry: generate sequential indices
                const tinygltf::Accessor& posAccessor = model.accessors[primitive.attributes.at("POSITION")];
                indices.resize(posAccessor.count);
                for (uint32_t i = 0; i < static_cast<uint32_t>(posAccessor.count); i++) {
                    indices[i] = i;
                }
            }

            if (indices.size() < 3 || indices.size() % 3 != 0) {
                std::cerr << "Warning: Invalid index count (" << indices.size() << ") in mesh '" << mesh.name << "', skipping primitive." << std::endl;
                continue;
            }

            // Extract vertex data
            auto vertices = getBufferData<glm::vec3>(model, primitive.attributes.at("POSITION"));
            
            const glm::vec3* normals = nullptr;
            bool hasNormals = false;
            if (primitive.attributes.find("NORMAL") != primitive.attributes.end())
            {
                normals = getBufferData<glm::vec3>(model, primitive.attributes.at("NORMAL"));
                hasNormals = true;
            }

            const glm::vec2* uvs = nullptr;
            bool hasUvs = false;
            if (primitive.attributes.find("TEXCOORD_0") != primitive.attributes.end())
            {
                uvs = getBufferData<glm::vec2>(model, primitive.attributes.at("TEXCOORD_0"));
                hasUvs = true;
            }

            const glm::vec4* tangents = nullptr;
            bool hasTangents = false;
            if (primitive.attributes.find("TANGENT") != primitive.attributes.end())
            {
                tangents = getBufferData<glm::vec4>(model, primitive.attributes.at("TANGENT"));
                hasTangents = true;
            }
            
            parseGltfMaterial(model, primitive.material, parentFolder, myMesh.material);

            myMesh.faces.resize(indices.size() / 3);
            utils::Face* dst = myMesh.faces.data();
            
            for (size_t i = 0, count = indices.size(); i < count; i += 3, ++dst) {             

                uint32_t idx[3] = { indices[i], indices[i + 1], indices[i + 2] };

                for (int e = 0; e < 3; e++)
                {
                    // Apply world transform to position
                    glm::vec4 worldPos = worldTransform * glm::vec4(vertices[idx[e]], 1.0f);
                    dst->pos[e] = glm::vec3(worldPos);

                    if (hasUvs) dst->uv[e] = uvs[idx[e]];

                    if (hasNormals) {
                        // Transform normal by the normal matrix
                        dst->normal[e] = glm::normalize(normalMatrix * normals[idx[e]]);
                    }
                }

                // If no normals in the file, compute a flat face normal from the (transformed) positions
                if (!hasNormals) {
                    glm::vec3 faceNormal = glm::normalize(glm::cross(
                        dst->pos[1] - dst->pos[0],
                        dst->pos[2] - dst->pos[0]));
                    dst->normal[0] = faceNormal;
                    dst->normal[1] = faceNormal;
                    dst->normal[2] = faceNormal;
                }

                if (hasTangents)
                {
                    for (int e = 0; e < 3; e++) {
                        glm::vec3 tVec = glm::normalize(glm::mat3(worldTransform) * glm::vec3(tangents[idx[e]]));
                        dst->tangent[e] = glm::vec4(tVec, tangents[idx[e]].w);
                    }
                } else {
                    //TODO: Should use Mikktspace algorithm http://www.mikktspace.com/
                    //but tbh just reimport it in Blender and export the tangents (there is a checkbox on the right of the exporter window under "data->mesh->Tangents")
                    glm::vec3 dp1 = dst->pos[1] - dst->pos[0];
                    glm::vec3 dp2 = dst->pos[2] - dst->pos[0];
                    glm::vec2 duv1 = dst->uv[1] - dst->uv[0];
                    glm::vec2 duv2 = dst->uv[2] - dst->uv[0];

                    float det = duv1.x * duv2.y - duv1.y * duv2.x;
                    if (fabs(det) < 1e-8f)
                        det = 1.0f; // divide-by-zero

                    float invDet = 1.0f / det;

                    glm::vec3 tangent = (dp1 * duv2.y - dp2 * duv1.y) * invDet;
                    glm::vec3 bitangent = (dp2 * duv1.x - dp1 * duv2.x) * invDet;

                    tangent = glm::normalize(tangent);
                    bitangent = glm::normalize(bitangent);

                    glm::vec3 normal = glm::normalize(glm::cross(dp1, dp2));

                    float handedness = (glm::dot(glm::cross(normal, tangent), bitangent) < 0.0f) ? -1.0f : 1.0f;

                    glm::vec4 finalTangent = glm::vec4(tangent, handedness);

                    // set to all three verts (still naive)
                    dst->tangent[0] = finalTangent;
                    dst->tangent[1] = finalTangent;
                    dst->tangent[2] = finalTangent;
                }


            }
            meshes.push_back(myMesh);
        }
    }
    return true;
}

// Generate Normalized UV Coordinates
void SceneManager::generateNormalizedUvCoordinates(std::vector<utils::Mesh>& meshes)
{
    uvUnwrapping::generateNormalizedUvCoordinatesPerMesh(renderContext.normalizedUvSpaceWidth, renderContext.normalizedUvSpaceHeight, meshes);
}

// Setup Mesh Buffers
void SceneManager::setupMeshBuffers(std::vector<utils::Mesh>& meshes)
{
    // Free the PREVIOUS model's GPU objects before dropping their handles --
    // otherwise every reload leaks its VAO/VBO (and textures). Fatal for the
    // sequential folder path, where hundreds of ~10 GB parts load in turn and
    // would exhaust VRAM within a few parts and crash.
    for (auto& pair : renderContext.dataMeshAndGlMesh) {
        if (pair.second.vao) glDeleteVertexArrays(1, &pair.second.vao);
        if (pair.second.vbo) glDeleteBuffers(1, &pair.second.vbo);
    }
    for (auto& meshTex : renderContext.meshToTextureData)
        for (auto& tex : meshTex.second)
            if (tex.second.glTextureID) { GLuint id = tex.second.glTextureID; glDeleteTextures(1, &id); }

    renderContext.dataMeshAndGlMesh.clear();
    renderContext.dataMeshAndGlMesh.reserve(meshes.size());
    // Fresh load: drop the previous model's texture keys and surface tally so a
    // reload (and the shared-name texture dedup in loadTextures) starts clean.
    renderContext.meshToTextureData.clear();
    renderContext.totalSurfaceArea = 0;

    float totalSurface = 0;

    for (auto& mesh : meshes) {
        utils::GLMesh glMesh;
        std::vector<float> vertices;
        float meshSurface = 0;
        // Per-mesh bbox: the converter projects THIS mesh's bbox into the grid
        // (converterGS u_bboxMin/Max), so it must be the mesh's own extent, not
        // a running union across meshes. Essential for split-on-import.
        glm::vec3 minBB(FLT_MAX);
        glm::vec3 maxBB(-FLT_MAX);
        for (const auto& face : mesh.faces) {
            for (int i = 0; i < 3; ++i) { // Assuming each face is a triangle (and it must be as we are only reading .gltf/.glb files)
                // Position
                vertices.push_back(face.pos[i].x);
                vertices.push_back(face.pos[i].y);
                vertices.push_back(face.pos[i].z);

                // Normal
                vertices.push_back(face.normal[i].x);
                vertices.push_back(face.normal[i].y);
                vertices.push_back(face.normal[i].z);

                // Tangent
                vertices.push_back(face.tangent[i].x);
                vertices.push_back(face.tangent[i].y);
                vertices.push_back(face.tangent[i].z);
                vertices.push_back(face.tangent[i].w);

                // UV
                vertices.push_back(face.uv[i].x);
                vertices.push_back(face.uv[i].y);

                // NORMALIZED UV
                vertices.push_back(face.normalizedUvs[i].x);
                vertices.push_back(face.normalizedUvs[i].y);

                // Scale
                vertices.push_back(face.scale.x);
                vertices.push_back(face.scale.y);
                vertices.push_back(face.scale.z);

                // Vertex colour (white by default; real values for vertex-coloured PLY)
                vertices.push_back(face.color[i].x);
                vertices.push_back(face.color[i].y);
                vertices.push_back(face.color[i].z);

                minBB.x = std::min(minBB.x, face.pos[i].x);
                minBB.y = std::min(minBB.y, face.pos[i].y);
                minBB.z = std::min(minBB.z, face.pos[i].z);

                maxBB.x = std::max(maxBB.x, face.pos[i].x);
                maxBB.y = std::max(maxBB.y, face.pos[i].y);
                maxBB.z = std::max(maxBB.z, face.pos[i].z);

            }
            
            mesh.surfaceArea += utils::triangleArea(face.pos[0], face.pos[1], face.pos[2]);
            
        }
        mesh.bbox = utils::BBox(minBB, maxBB);

        renderContext.totalSurfaceArea += mesh.surfaceArea;

        
        unsigned int floatsPerVertex = 20;
        glMesh.vertexCount = vertices.size() / floatsPerVertex; // Number of vertices

        // 3 position, 3 normal, 4 tangent, 2 UV, 2 NORMALIZED UVs, 3 scale, 3 colour = 20
        size_t vertexStride = floatsPerVertex * sizeof(float);

        // Generate and bind VAO
        glGenVertexArrays(1, &glMesh.vao);
        glBindVertexArray(glMesh.vao);

        // Generate and bind VBO
        glGenBuffers(1, &glMesh.vbo);
        glBindBuffer(GL_ARRAY_BUFFER, glMesh.vbo);
        glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);

        // Vertex attribute pointers
        // Position attribute
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, vertexStride, (void*)0);
        glEnableVertexAttribArray(0);
        // Normal attribute
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, vertexStride, (void*)(3 * sizeof(float)));
        glEnableVertexAttribArray(1);
        // Tangent attribute
        glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, vertexStride, (void*)(6 * sizeof(float)));
        glEnableVertexAttribArray(2);
        // UV attribute
        glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, vertexStride, (void*)(10 * sizeof(float)));
        glEnableVertexAttribArray(3);
        // Normalized UV attribute
        glVertexAttribPointer(4, 2, GL_FLOAT, GL_FALSE, vertexStride, (void*)(12 * sizeof(float)));
        glEnableVertexAttribArray(4);
        // Scale attribute
        glVertexAttribPointer(5, 3, GL_FLOAT, GL_FALSE, vertexStride, (void*)(14 * sizeof(float)));
        glEnableVertexAttribArray(5);
        // Vertex colour attribute
        glVertexAttribPointer(6, 3, GL_FLOAT, GL_FALSE, vertexStride, (void*)(17 * sizeof(float)));
        glEnableVertexAttribArray(6);

        //Should use array indices for per face data such as rotation and scale or directly compute it in the shader, should actually do it in a compute shader and be done

        // Unbind VAO
        glBindVertexArray(0);

        // Add to list of GLMeshes
        renderContext.dataMeshAndGlMesh.push_back(std::make_pair(mesh, glMesh));
    }

}

void SceneManager::loadTextures(const std::vector<utils::Mesh>& meshes)
{
    for (auto& mesh : meshes)
    {
        // Sub-meshes from split-on-import share the parent's name (the texture
        // key). Load each unique name once -- otherwise N^2 sub-meshes reload
        // the same source image N^2 times.
        if (renderContext.meshToTextureData.find(mesh.name) != renderContext.meshToTextureData.end())
            continue;

        std::map<std::string, utils::TextureDataGl> textureMapForThisMesh;
        //BASECOLOR ALBEDO TEXTURE LOAD
        if (mesh.material.baseColorTexture.path != EMPTY_TEXTURE)
        {
            utils::TextureDataGl tdgl(mesh.material.baseColorTexture);
            textureMapForThisMesh.insert_or_assign(BASE_COLOR_TEXTURE, tdgl);
        }
        else {
            renderContext.material.baseColorTexture.width = MAX_RESOLUTION_TARGET;
            renderContext.material.baseColorTexture.height = MAX_RESOLUTION_TARGET;
            textureMapForThisMesh.erase(BASE_COLOR_TEXTURE);
        }

        //METALLIC-ROUGHNESS TEXTURE LOAD
        if (mesh.material.metallicRoughnessTexture.path != EMPTY_TEXTURE)
        {
            utils::TextureDataGl tdgl(mesh.material.metallicRoughnessTexture);
            textureMapForThisMesh.insert_or_assign(METALLIC_ROUGHNESS_TEXTURE, tdgl);
        }
        else {
            renderContext.material.metallicRoughnessTexture.width = MAX_RESOLUTION_TARGET;
            renderContext.material.metallicRoughnessTexture.height = MAX_RESOLUTION_TARGET;
            textureMapForThisMesh.erase(METALLIC_ROUGHNESS_TEXTURE);
        }

        //NORMAL TEXTURE LOAD
        if (mesh.material.normalTexture.path != EMPTY_TEXTURE)
        {
            
            utils::TextureDataGl tdgl(mesh.material.normalTexture);
            textureMapForThisMesh.insert_or_assign(NORMAL_TEXTURE, tdgl);
        }
        else {
            renderContext.material.normalTexture.width = MAX_RESOLUTION_TARGET;
            renderContext.material.normalTexture.height = MAX_RESOLUTION_TARGET;
            textureMapForThisMesh.erase(NORMAL_TEXTURE);

        }

        //OCCLUSION TEXTURE LOAD
        if (mesh.material.occlusionTexture.path != EMPTY_TEXTURE)
        {
            utils::TextureDataGl tdgl(mesh.material.occlusionTexture);
            textureMapForThisMesh.insert_or_assign(AO_TEXTURE, tdgl);
        }
        else {
            renderContext.material.occlusionTexture.width = MAX_RESOLUTION_TARGET;
            renderContext.material.occlusionTexture.height = MAX_RESOLUTION_TARGET;
            textureMapForThisMesh.erase(AO_TEXTURE);
        }

        //EMISSIVE TEXTURE LOAD
        if (mesh.material.emissiveTexture.path != EMPTY_TEXTURE)
        {
            utils::TextureDataGl tdgl(mesh.material.emissiveTexture);
            textureMapForThisMesh.insert_or_assign(EMISSIVE_TEXTURE, tdgl);
        }
        else {
            renderContext.material.emissiveTexture.width = MAX_RESOLUTION_TARGET;
            renderContext.material.emissiveTexture.height = MAX_RESOLUTION_TARGET;
            textureMapForThisMesh.erase(EMISSIVE_TEXTURE);
        }

        renderContext.meshToTextureData.insert_or_assign(mesh.name, textureMapForThisMesh);

    }
    
}

void SceneManager::exportPly(const std::string outputFile, unsigned int exportFormat)
{
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, renderContext.gaussianBuffer);

    while (glGetError() != GL_NO_ERROR) {} // clear stale errors first

    // --- Honesty check: how big is the buffer REALLY? A glBufferData that
    // failed under VRAM pressure leaves the previous, smaller buffer behind
    // while the code believes the resize succeeded; the shader then silently
    // drops every write past the real end and the export reads ghosts.
    // Query the driver instead of trusting our own bookkeeping.
    GLint64 realBufferBytes = 0;
    glGetBufferParameteri64v(GL_SHADER_STORAGE_BUFFER, GL_BUFFER_SIZE, &realBufferBytes);
    GLint64 maxBlockBytes = 0;
    glGetInteger64v(GL_MAX_SHADER_STORAGE_BLOCK_SIZE, &maxBlockBytes);

    const unsigned long long requested    = renderContext.numberOfGaussians;
    const unsigned long long realCapacity =
        static_cast<unsigned long long>(realBufferBytes) / sizeof(utils::GaussianDataSSBO);
    const unsigned long long exportCount  = (requested < realCapacity) ? requested : realCapacity;

    std::cout << "[Export] gaussians requested: " << requested
              << " | real buffer: " << realBufferBytes << " B (capacity " << realCapacity
              << ") | GL_MAX_SHADER_STORAGE_BLOCK_SIZE: " << maxBlockBytes << " B" << std::endl;

    if (exportCount < requested) {
        std::cerr << "[Export] WARNING: buffer really holds only " << exportCount << " of "
                  << requested << " gaussians -- a GPU buffer resize failed silently "
                  << "(VRAM pressure?). Exporting the valid part only. "
                  << "Use offline conversion for exports of this size." << std::endl;
    }
    if (maxBlockBytes > 0 && realBufferBytes > maxBlockBytes) {
        std::cerr << "[Export] WARNING: buffer (" << realBufferBytes
                  << " B) exceeds GL_MAX_SHADER_STORAGE_BLOCK_SIZE (" << maxBlockBytes
                  << " B) -- shader writes past that limit are undefined on this driver. "
                  << "Use offline conversion." << std::endl;
    }

    std::vector<utils::GaussianDataSSBO> cpuData(exportCount);

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    // Sliced readback (single calls beyond ~2 GB are unreliable on many
    // drivers), with per-slice error reporting.
    {
        const GLsizeiptr sliceBytes = 512ll * 1024ll * 1024ll; // 512 MB per read
        const GLsizeiptr totalBytes =
            static_cast<GLsizeiptr>(exportCount) *
            static_cast<GLsizeiptr>(sizeof(utils::GaussianDataSSBO));

        char*      dst    = reinterpret_cast<char*>(cpuData.data());
        GLsizeiptr offset = 0;
        int        slice  = 0;
        while (offset < totalBytes)
        {
            GLsizeiptr chunk = (totalBytes - offset < sliceBytes) ? (totalBytes - offset) : sliceBytes;
            glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, offset, chunk, dst + offset);
            GLenum err = glGetError();
            if (err != GL_NO_ERROR) {
                std::cerr << "[Export] readback slice " << slice << " (offset " << offset
                          << ", " << chunk << " B) FAILED, GL error 0x"
                          << std::hex << err << std::dec << std::endl;
            }
            offset += chunk;
            ++slice;
        }
    }
    
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    
    float scaleMultiplier = renderContext.gaussianStd / static_cast<float>(renderContext.resolutionTarget);
    auto format           = exportFormat;  

    std::thread(
        [=, data = std::move(cpuData)]() mutable 
        {
            parsers::savePlyVector(outputFile, data, format, scaleMultiplier);
        }
    ).detach();
    
}


void SceneManager::updateMeshes()
{
}

void SceneManager::cleanup()
{
}