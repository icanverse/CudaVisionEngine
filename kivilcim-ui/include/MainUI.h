#pragma once
#include <GLFW/glfw3.h>

#include "imgui.h"
#include "w_RightPanel.h"
#include "w_TopPanel.h"
#include "w_BackgroundPanel.h"
#include "w_LeftPanel.h"
#include "Workspace.h"
#include "Shaders/LiquidShader.cuh"

// YENİ: Motorun ve arayüzün tek gerçek veri kaynağı
#include "Data/WorkspaceStateData.h"

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

    Kdata::WorkspaceStateData appState;

    Kdata::PreferenceData userPrefs;
    Kivilcim::UI::PreferencesPanel preferencesPanel;

    GLFWwindow* windowHandle;
    unsigned int logoTextureId;
    ImFont* logFont;

    CudaDynamicTexture* liquidCanvas;

public:
    MainUI(GLFWwindow* window);
    ~MainUI();

    void newFrame();
    void renderPanels();
    void renderDrawData();
};