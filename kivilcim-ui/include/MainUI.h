#pragma once
#include <GLFW/glfw3.h>

#include "imgui.h"
#include "w_RightPanel.h"
#include "w_TopPanel.h"
#include "w_BackgroundPanel.h"
#include "w_LeftPanel.h"

// ==========================================
// YENİ: CUDA KÖPRÜLERİ VE SHADER'LAR
// ==========================================
#include "Workspace.h"
#include "Shaders/LiquidShader.cuh"

class CudaDynamicTexture;

enum class AppMode {
    START_SCREEN,
    WORKSPACE
};

class MainUI {
private:
    TopPanel topPanel;
    LeftPanel leftPanel;
    RightPanel rightPanel;
    BackgroundPanel backgroundPanel;

    AppMode currentMode = AppMode::START_SCREEN;
    Workspace workspaceUI;

    GLFWwindow* windowHandle;
    unsigned int logoTextureId;
    ImFont* logFont;

    // YENİ: Dinamik Likit Cam Tuvalimiz
    CudaDynamicTexture* liquidCanvas;

public:
    MainUI(GLFWwindow* window);
    ~MainUI();

    void newFrame();
    void renderPanels();
    void renderDrawData();
};