///////////////////////////////////////////////////////////////////////////////
//         Mesh2Splat: fast mesh to 3D gaussian splat conversion             //
//        Copyright (c) 2025 Electronic Arts Inc. All rights reserved.       //
///////////////////////////////////////////////////////////////////////////////

#pragma once
#include <string>
#include <optional> 
#include <filesystem>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <glm/glm.hpp>
#include "utils/utils.hpp"
#include "Imguizmo.hpp"
#include "ImGuiFileDialog.h"



class ImGuiUI {
public:
    ImGuiUI(float defaultGaussianStd, float defaultMesh2SPlatQuality);
    ~ImGuiUI();

    void preframe();
    void initialize(GLFWwindow* window);
    void renderUI();
    void displayGaussianCounts(unsigned int gaussianCount, unsigned int visibleGaussianCount);
    void postframe();

    bool shouldRunConversion() const;
    bool shouldLoadNewMesh() const;
    //Batch processing stuff
    bool hasBatchWork() const;                        // any Queued left?
    bool isBatchRunning() const;                      // UI is in "running" state

                          // flag cancel; new items not dispatched

    bool shouldBatchLoadNewMeshes() const ;
    bool shouldSavePly() const;
    bool wasMeshLoaded() const;
    bool shouldLoadPly() const;
    bool isLightingEnabled() const;
    bool isLightSelected() const;
    bool wasPlyLoaded() const;

    std::string getMeshFilePath() const;
    std::string getMeshFilePathParentFolder() const;
    std::string getMeshFullFilePathDestination() const;
    std::string getPlyFilePathParentFolder() const;
    std::string getPlyFilePath() const;


    float getGaussianStd() const;
    int getResolutionTarget() const;
    unsigned int getFormatOption() const;

    glm::vec4 getSceneBackgroundColor() const;

    void setLoadNewMesh(bool shouldLoadNewMesh);
    void setBatchLoadNewMeshes(bool shouldBatchLoadNewMeshes);
    void setMeshLoaded(bool loaded);
    void setRunConversion(bool shouldRunConversionFlag);
    void setShouldSavePly(bool shouldSavePly);
    void setFrameMetrics(double gpuFrameTime);
    void setLoadNewPly(bool loadedPly);
    void setPlyLoaded(bool loadedPly);

    void renderGizmoUi(glm::mat4& glmViewMat, glm::mat4& glmProjMat, glm::mat4& glmModelMat);
    void renderFileSelectorWindow();
    void renderPropertiesWindow();
    void renderGpuFrametime();
    void renderLightingSettings();
    float getLightIntensity() const;
    glm::vec3 getLightColor() const;
    
    void setEnableDepthTest(bool depthTest);
    bool getIsDepthTestEnabled() const;

    bool isSplitScreenEnabled() const;
    float getSplitScreenPosition() const;

    float getMovementSpeed() const { return movementSpeed; }

    // --- Offline (chunked-to-disk) conversion UI plumbing
    float getOfflineTileSize() const          { return offlineTileSize; }
    // Offline conversion resolution: same quality slider, but its own max
    // resolution -- so cranking it never triggers a live reconversion.
    int getOfflineResolutionTarget() const
    {
        return static_cast<int>(minRes + quality * (resolutionOptions[offlineResolutionIndex] - minRes));
    }
    void setLoadedMeshCount(size_t n)         { loadedMeshCount = n; }
    bool  getOfflineUseCustomRoot() const     { return offlineUseCustomRoot; }
    float getOfflineRootMinX() const          { return offlineRootOrigin[0]; }
    float getOfflineRootMinZ() const          { return offlineRootOrigin[1]; }
    float getOfflineRootSize() const          { return offlineRootSize; }
    bool shouldStartOfflineConversion() const { return offlineConvertRequested; }
    void clearOfflineConvertRequest()         { offlineConvertRequested = false; }
    bool wantsOfflineCancel() const           { return offlineCancelRequested; }
    void clearOfflineCancelRequest()          { offlineCancelRequested = false; }
    void setOfflineState(bool running, float progress01, unsigned long long written, const std::string& statusText)
    {
        offlineRunning  = running;
        offlineProgress = progress01;
        offlineWritten  = written;
        offlineStatus   = statusText;
    }


    enum class VisualizationOption
    {
        ALBEDO = 0,
        DEPTH = 1,
        NORMAL = 2,
        GEOMETRY = 3,
        OVERDRAW = 4,
        PBR = 5,
        FINAL = 6
    };
    


    ImGuiUI::VisualizationOption selectedRenderMode() const ;

    //Batch functions
    struct BatchItem {
        std::string path;
        std::string parent;
        utils::ModelFileExtension ext;
        enum class Status { Queued, Processing, Done, Failed } status = Status::Queued;
        std::string outPath; // filled by UI using your current output naming rules
        std::string error;   // filled if failed
    };

    BatchItem* popNextBatchItem();      // get next Queued -> set to Processing
    void markBatchItemDone(const std::string& path);  // Processing -> Done
    void markBatchItemFailed(const std::string& path, const std::string& err);
    void cancelBatch();     
    const std::vector<ImGuiUI::BatchItem>& getBatchItems() const;

private:
    int resolutionIndex = 0;
    const int resolutionOptions[4] = { 1024, 2048, 4096, 8192 };
    const char* resolutionLabels[4] = { "1024", "2048", "4096", "8192" };
    
    int formatIndex = 0;
    const unsigned int formatOptions[3] = { 0, 1, 2 };
    const char* formatLabels[3] = { "PLY Standard Format", "PLY PBR", "PLY Compressed PBR"};

    int renderIndex = 0;
    int prevRenderIndexBeforeLighting = 0;
    const ImGuiUI::VisualizationOption renderOptions[7] = {
        ImGuiUI::VisualizationOption::FINAL,
        ImGuiUI::VisualizationOption::ALBEDO,
        ImGuiUI::VisualizationOption::DEPTH,
        ImGuiUI::VisualizationOption::NORMAL,
        ImGuiUI::VisualizationOption::GEOMETRY,
        ImGuiUI::VisualizationOption::OVERDRAW,
        ImGuiUI::VisualizationOption::PBR
    };
    const char* renderLabels[7] = { "Final (Shaded)", "Albedo", "Depth", "Normals", "Geometry", "Overdraw", "PBR (metallic-roughness)"};

    float gaussian_std;
    float lightIntensity = 0;
    float quality;
    bool runConversionFlag = false;
    bool loadNewMesh = false;
    bool batchLoadNewMeshes = false; 
    bool loadNewPly = false;

    //Rendering flags
    bool hasRenderModeChanged = false;

    bool hasPlyBeenLoaded = false;
    bool hasMeshBeenLoaded = false;

    bool lightSelected = false;
    bool lightingEnabled = false;

    bool savePly = false;

    bool enableDepthTest = false;

    bool splitScreenEnabled = false;
    float splitScreenPosition = 0.5f;

    float movementSpeed = 2.0f; // camera navigation speed, exposed via Properties slider

    // Offline (chunked-to-disk) conversion state
    int   offlineResolutionIndex = 0;      // index into resolutionOptions; independent of live maxRes
    size_t loadedMeshCount       = 0;      // for the pre-run splat-count upper bound
    float offlineTileSize        = 0.0f;   // 0 = single file; >0 = XZ tile edge in world units
    bool  offlineUseCustomRoot   = false;  // shared site convention: user-provided quadtree root
    float offlineRootOrigin[2]   = { 0.0f, 0.0f };   // root min X / min Z (world units)
    float offlineRootSize        = 0.0f;   // requested root edge; snapped up to tileSize * 2^L
    bool offlineConvertRequested = false;
    bool offlineCancelRequested  = false;
    bool offlineRunning          = false;
    float offlineProgress        = 0.0f;
    unsigned long long offlineWritten = 0;
    std::string offlineStatus;

    std::string meshFilePath;
    std::string meshParentFolder;

    std::string plyFilePath;
    std::string plyParentFolder;

    utils::ModelFileExtension currentModelFormat = utils::ModelFileExtension::NONE;

    std::string destinationFilePathFolder = "";
    char outputFilename[256] = "DefaultFilename.ply";


    const float minStd = 0.1f;
    const float maxStd = 2.0f;
    const float minLightIntensity = 0.0;
    const float maxLightIntensity = 1000.0;
    int maxRes = 1024; //TBH not sure what best value is here
    int minRes = 16;

    //Gpu timing data
    std::vector<float> frameTimeHistory = {0.0f};
    static constexpr size_t MAX_FRAME_HISTORY = 100;
    double gpuFrameTime = 0;
    float maxPlotTimeMs = 100.0f; 
    float targetFrameTimeThreshold = 16.6f; 

    glm::vec4 sceneBackgroundColor = { 0,0,0,1 };
    glm::vec3 lightColor = { 1,1,1 };

    //Batching data
    std::vector<BatchItem> batchItems;
    bool batchIncludeSubfolders = false;
    bool batchRunning = false;
    bool batchCancelRequested = false;
    float batchProgress01 = 0.0f;
    int batchSelectedRow = -1;

    void renderBatchWindow();                  // NEW panel
    void enqueueFolder(const std::string& dir);
    
    static bool isSupportedMesh(const std::filesystem::directory_entry& e)
    {
        if (!e.is_regular_file()) return false;
        auto ext = e.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        return (ext == ".glb" || ext == ".ply");
    };

    static utils::ModelFileExtension extFromPath(const std::string& p)
    {
        auto ext = utils::getFileExtension(p);
        if (ext == utils::ModelFileExtension::GLB) return utils::ModelFileExtension::GLB;
        if (ext == utils::ModelFileExtension::PLY) return utils::ModelFileExtension::PLY;
        return utils::ModelFileExtension::NONE;
    };

};