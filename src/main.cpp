///////////////////////////////////////////////////////////////////////////////
//         Mesh2Splat: fast mesh to 3D gaussian splat conversion             //
//        Copyright (c) 2025 Electronic Arts Inc. All rights reserved.       //
///////////////////////////////////////////////////////////////////////////////

#include "utils/normalizedUvUnwrapping.hpp"
#include "renderer/renderer.hpp"
#include "glewGlfwHandlers/glewGlfwHandler.hpp"
#include "renderer/guiRendererConcreteMediator.hpp"
#include "utils/Logger.hpp"
#include <cstring>
#include <cstdlib>
#include <iostream>

// ---- CLI arg helpers -------------------------------------------------------
static const char* argVal(int argc, char** argv, const char* flag, const char* def = nullptr) {
    for (int i = 1; i < argc - 1; ++i) if (std::strcmp(argv[i], flag) == 0) return argv[i + 1];
    return def;
}
static bool argFlag(int argc, char** argv, const char* flag) {
    for (int i = 1; i < argc; ++i) if (std::strcmp(argv[i], flag) == 0) return true;
    return false;
}

// Headless folder-of-parts conversion. Returns exit code, or -999 if not CLI.
static int runCli(int argc, char** argv) {
    const char* folder = argVal(argc, argv, "--convert-folder");
    if (!folder) return -999;   // not CLI -> fall through to the interactive app

    const char* out = argVal(argc, argv, "--out");
    if (!out) { std::cerr << "[CLI] --out <output.ply> is required\n"; return 2; }
    const float tile  = std::atof(argVal(argc, argv, "--tile", "20"));
    const int   grid  = std::atoi(argVal(argc, argv, "--grid", "288"));   // final grid resolution R
    const float rootX = std::atof(argVal(argc, argv, "--root-x", "0"));
    const float rootZ = std::atof(argVal(argc, argv, "--root-z", "0"));
    const float rootS = std::atof(argVal(argc, argv, "--root-size", "0"));
    const float gstd  = std::atof(argVal(argc, argv, "--std", "0.65"));
    const bool  pbr   = argFlag(argc, argv, "--pbr");

    std::cout << "[CLI] headless convert: folder=" << folder << " out=" << out
              << " tile=" << tile << " grid=" << grid
              << " root=(" << rootX << "," << rootZ << ") size=" << rootS
              << " std=" << gstd << " pbr=" << (pbr ? 1 : 0) << std::endl;

    GlewGlfwHandler handler(glm::ivec2(1080, 720), "Mesh2SplatCLI", /*visible=*/false);
    if (handler.init() == -1) { std::cerr << "[CLI] GL init failed\n"; return 3; }
    Camera camera(glm::vec3(0.0f, 0.0f, 5.0f), glm::vec3(0.0f, 1.0f, 0.0f), -90.0f, 0.0f);
    Renderer renderer(handler.getWindow(), camera);
    renderer.initialize();
    renderer.setStdDevFromImGui(gstd);   // sets renderContext.gaussianStd (scaleMult = std/grid)

    // Flush any benign GL errors accumulated during setup, so the converter's
    // own glGetError check isn't tripped by them, then report anything left.
    { GLenum e; int flushed = 0; while ((e = glGetError()) != GL_NO_ERROR) { ++flushed;
        std::cerr << "[CLI] pre-convert GL error 0x" << std::hex << e << std::dec << " flushed\n"; }
      if (flushed == 0) std::cout << "[CLI] GL state clean before convert\n"; }

    OfflineConverter::RootRegion root;
    root.enabled = true; root.minX = rootX; root.minZ = rootZ; root.size = rootS;

    if (!renderer.startSequentialFolderConversion(folder, out, tile, root, grid, pbr)) {
        std::cerr << "[CLI] failed to start: " << renderer.getSequentialStatus() << std::endl;
        return 4;
    }
    while (renderer.isSequentialRunning()) {
        renderer.stepSequentialFolderConversion();
        glfwPollEvents();
    }
    std::cout << "[CLI] finished: " << renderer.getSequentialStatus() << std::endl;
    glfwTerminate();
    return 0;
}

int main(int argc, char** argv) {
    utils::initLog();

    int cli = runCli(argc, argv);
    if (cli != -999) return cli;   // ran (or failed) headless; done.
   // capture cout/cerr into the in-app Log window
    GlewGlfwHandler glewGlfwHandler(glm::ivec2(1080, 720), "Mesh2Splat");
    
    Camera camera(
        glm::vec3(0.0f, 0.0f, 5.0f), 
        glm::vec3(0.0f, 1.0f, 0.0f), 
        -90.0f, 
        0.0f
    );  

    IoHandler ioHandler(glewGlfwHandler.getWindow(), camera);
    if(glewGlfwHandler.init() == -1) return -1;

    ioHandler.setupCallbacks();

    ImGuiUI ImGuiUI(0.65f, 0.5f); //TODO: give a meaning to these params
    ImGuiUI.initialize(glewGlfwHandler.getWindow());

    Renderer renderer(glewGlfwHandler.getWindow(), camera);
    renderer.initialize();
    GuiRendererConcreteMediator guiRendererMediator(renderer, ImGuiUI);

    float deltaTime = 0.0f; 
    float lastFrame = 0.0f;

    while (!glfwWindowShouldClose(glewGlfwHandler.getWindow())) {
        
        float currentFrame = static_cast<float>(glfwGetTime());
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        glfwPollEvents();
        
        camera.SetMovementSpeed(ImGuiUI.getMovementSpeed());
        ioHandler.processInput(deltaTime);

        renderer.clearingPrePass(ImGuiUI.getSceneBackgroundColor());

        ImGuiUI.preframe();
        ImGuiUI.renderUI();
        
        guiRendererMediator.update();

        renderer.renderFrame();

        ImGuiUI.displayGaussianCounts(renderer.getTotalGaussianCount(), renderer.getVisibleGaussianCount());
        ImGuiUI.postframe();

        glfwSwapBuffers(glewGlfwHandler.getWindow());
    }

    glfwTerminate();

    return 0;
}

