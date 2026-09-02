///////////////////////////////////////////////////////////////////////////////
//         Mesh2Splat: fast mesh to 3D gaussian splat conversion             //
//        Copyright (c) 2025 Electronic Arts Inc. All rights reserved.       //
///////////////////////////////////////////////////////////////////////////////

#pragma once
#include "tiny_gltf.h"
#include "stb_image.h"   
#include "stb_image_resize.h"
#include "stb_image_write.h"
#include "happly.h"
#include "utils/utils.hpp"

namespace parsers
{
	utils::TextureDataGl loadImageAndBpp(std::string texturePath, int& textureWidth, int& textureHeight);

	void writePbrPLY(const std::string& filename, std::vector<utils::GaussianDataSSBO>& gaussians, float scaleMultiplier);

	void writeBinaryPlyStandardFormat(const std::string& filename, const std::vector<utils::GaussianDataSSBO>& gaussians, float scaleMultiplier);

	void loadPlyFile(std::string plyFileLocation, std::vector<utils::GaussianDataSSBO>& gaussians, bool& hasPbr);

	void savePlyVector(std::string outputFileLocation, std::vector<utils::GaussianDataSSBO> gaussians_3D_list, unsigned int format, float scaleMultiplier);

	unsigned char* combineMetallicRoughness(const char* path1, const char* path2, int& width, int& height, int& channels);

	bool extractImageNames(const std::string& combinedName, std::string& path, std::string& name1, std::string& name2);

	// --- Mesh PLY (RealityScan-style) -------------------------------------
	// A binary mesh PLY (positions + optional UV + optional vertex color +
	// triangle faces), as opposed to a Gaussian-splat PLY (f_dc_*/scale_*/rot_*).
	// Read via happly directly -- no assimp -- so a 180M-face mesh loads without
	// assimp's transient memory blow-up (it only has to fit the compact arrays).
	struct MeshPlyRaw {
		std::vector<float> x, y, z;          // positions (required)
		std::vector<float> s, t;             // UV (empty if absent)
		std::vector<float> r, g, b;          // vertex color 0..1 (empty if absent)
		std::vector<std::vector<int>> faces; // triangle vertex indices
		bool hasUV = false;
		bool hasColor = false;
	};

	// Returns false (and logs) if the file has no vertex/face elements or fails.
	bool readMeshPlyGeometry(const std::string& path, MeshPlyRaw& out);

	// Header sniff: true if the PLY looks like a Gaussian-splat PLY (has f_dc_0).
	// Reads only the ascii header, so it's instant even on multi-GB files.
	bool plyIsGaussianSplat(const std::string& path);
}


