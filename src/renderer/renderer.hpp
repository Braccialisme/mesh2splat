///////////////////////////////////////////////////////////////////////////////
//         Mesh2Splat: fast mesh to 3D gaussian splat conversion             //
//        Copyright (c) 2025 Electronic Arts Inc. All rights reserved.       //
///////////////////////////////////////////////////////////////////////////////

#pragma once
#include <filesystem>
#include <algorithm>
#include <iostream>
#include "utils/utils.hpp"
#include "IoHandler.hpp"
#include "imGuiUi/ImGuiUi.hpp"
#include "parsers/parsers.hpp"
#include "utils/glUtils.hpp"
#include "RadixSort.hpp"
#include "renderPasses/RenderContext.hpp"
#include "RenderPasses.hpp"
#include "utils/SceneManager.hpp"
#include "utils/ShaderRegistry.hpp"
#include "OfflineConverter.hpp"

class Renderer {
public:
	Renderer(GLFWwindow* window, Camera& cameraInstance);

	~Renderer();

	void initialize();
	void renderFrame();        
	void clearingPrePass(glm::vec4 clearColor); 
	void updateTransformations();

	//TODO: For now not using this, will implement a render-pass based structure and change how the render-loop is implemented
	bool updateShadersIfNeeded(bool forceReload = false);
	unsigned int getVisibleGaussianCount();
	unsigned int getTotalGaussianCount();
	void setLastShaderCheckTime(double lastShaderCheckedTime);
	double getLastShaderCheckTime();
	RenderContext* getRenderContext();
	void enableRenderPass(std::string renderPassName);
	void setViewportResolutionForConversion(int resolutionTarget);
	void setFormatType(unsigned int format);

	void setStdDevFromImGui(float stdDev);
	void resetRendererViewportResolution();
	SceneManager& getSceneManager();
	double getTotalGpuFrameTimeMs() const;
	void updateGaussianBuffer();
	void gaussianBufferFromSize(unsigned int size);
	void setRenderMode(ImGuiUI::VisualizationOption renderMode);
	void resetModelMatrices();
	void createGBuffer();
	void deleteGBuffer();
	void createDepthTexture();
	void deleteDepthTexture();
	void setDepthTestEnabled(bool depthTest);
	void setLightingEnabled(bool isEnabled);
	void setLightIntensity(float lightIntensity);
	void setLightColor(glm::vec3 lightColor);
	bool hasWindowSizeChanged();
	bool isWindowMinimized();

	void createMeshGBuffer();
	void deleteMeshGBuffer();
	void setSplitScreenEnabled(bool enabled);
	void setSplitScreenPosition(float position);

	// --- Offline (chunked-to-disk) conversion. Driven one batch per frame
	// by the mediator; see OfflineConverter for the details.
	bool startOfflineConversion(const std::string& outputPath, float tileSize,
	                            const OfflineConverter::RootRegion& rootRegion = {},
	                            int offlineResolution = 0,
	                            bool includePbr = false) {
		return offlineConverter.start(renderContext, outputPath, tileSize, rootRegion, offlineResolution,
		                              OfflineConverter::kDefaultBatchCapacity, includePbr);
	}
	size_t getMeshCount() const { return renderContext.dataMeshAndGlMesh.size(); }
	void stepOfflineConversion()          { offlineConverter.step(renderContext); }
	void cancelOfflineConversion()        { offlineConverter.cancel(); }
	bool isOfflineConversionRunning() const     { return offlineConverter.isRunning(); }
	float getOfflineProgress() const            { return offlineConverter.progress01(); }
	unsigned long long getOfflineWritten() const { return offlineConverter.writtenCount(); }
	const std::string& getOfflineStatus() const { return offlineConverter.statusText(); }

	// --- Sequential folder conversion: convert a folder of mesh parts (from the
	// out-of-core splitter) into ONE shared tiled output, loading/converting/
	// freeing one part at a time so a mesh too big to hold whole still LODs.
	// Requires a custom root region (shared quadtree across all parts).
	bool startSequentialFolderConversion(const std::string& folder, const std::string& outputPath,
	                                     float tileSize, const OfflineConverter::RootRegion& rootRegion,
	                                     int offlineResolution, bool includePbr) {
		namespace fs = std::filesystem;
		seqParts.clear(); seqIndex = 0; seqParentFolder = folder + "/";
		std::error_code ec;
		if (!fs::is_directory(folder, ec)) { seqStatus = "Not a folder: " + folder; return false; }
		for (const auto& e : fs::directory_iterator(folder, ec)) {
			if (!e.is_regular_file()) continue;
			std::string ext = e.path().extension().string();
			std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c){ return (char)std::tolower(c); });
			if (ext == ".ply") seqParts.push_back(e.path().string());
		}
		std::sort(seqParts.begin(), seqParts.end());
		if (seqParts.empty()) { seqStatus = "No .ply parts in " + folder; return false; }
		if (!offlineConverter.beginMultiPart(renderContext, outputPath, tileSize, rootRegion,
		                                     offlineResolution, OfflineConverter::kDefaultBatchCapacity, includePbr))
			{ seqStatus = offlineConverter.statusText(); return false; }
		if (!sceneManager->loadModel(seqParts[0], seqParentFolder, 1)) {
			seqStatus = "Failed to load part: " + seqParts[0]; offlineConverter.cancel(); return false;
		}
		offlineConverter.queuePart(renderContext);
		seqRunning = true;
		seqStatus = "Converting part 1/" + std::to_string(seqParts.size());
		std::cout << "[Offline] Sequential folder: " << seqParts.size() << " parts from " << folder << std::endl;
		return true;
	}
	void stepSequentialFolderConversion() {
		if (!seqRunning) return;
		if (offlineConverter.step(renderContext)) return;   // batch done, part still has work
		if (!offlineConverter.isRunning()) { seqRunning = false; seqStatus = offlineConverter.statusText(); return; } // failed
		++seqIndex;                                          // current part drained -> next
		if (seqIndex >= seqParts.size()) {
			offlineConverter.finishMultiPart();
			seqRunning = false;
			seqStatus = offlineConverter.statusText();
			return;
		}
		if (sceneManager->loadModel(seqParts[seqIndex], seqParentFolder, 1))
			offlineConverter.queuePart(renderContext);
		else
			std::cerr << "[Offline] skipping unloadable part: " << seqParts[seqIndex] << std::endl;
		seqStatus = "Converting part " + std::to_string(seqIndex + 1) + "/" + std::to_string(seqParts.size());
	}
	void cancelSequentialFolderConversion()  { offlineConverter.cancel(); seqRunning = false; seqStatus = "Cancelled."; }
	bool  isSequentialRunning() const        { return seqRunning; }
	float getSequentialProgress() const      { return seqParts.empty() ? 0.0f : (float)seqIndex / (float)seqParts.size(); }
	const std::string& getSequentialStatus() const { return seqStatus; }




private:
	std::map<std::string, std::unique_ptr<IRenderPass>> renderPasses;
	std::vector<std::string> renderPassesOrder;

	GLFWwindow* rendererGlfwWindow;

	std::unique_ptr<SceneManager> sceneManager;
	RenderContext renderContext;

	OfflineConverter offlineConverter;

	// Sequential folder (multi-part) conversion state
	std::vector<std::string> seqParts;
	size_t                   seqIndex = 0;
	std::string              seqParentFolder;
	std::string              seqStatus;
	bool                     seqRunning = false;

	double lastShaderCheckTime;

	double gpuFrameTimeMs;

	Camera& camera;

};