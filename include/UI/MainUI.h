#pragma once
#include <GLFW/glfw3.h>

#include "imgui.h"
#include "Workspace.h"
#include "w_RightPanel.h"
#include "w_TopPanel.h"
#include "w_BackgroundPanel.h"
#include "w_LeftPanel.h"

// YENİ: Motorun Çalışma Modları (State Machine)
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

    // Mod Yöneticisi ve Çalışma Alanı Nesnesi
    AppMode currentMode = AppMode::START_SCREEN;
    Workspace workspaceUI;

    GLFWwindow* windowHandle;
    unsigned int logoTextureId; // Logonun VRAM'deki adresi
    ImFont* logFont;

public:
    MainUI(GLFWwindow* window);
    ~MainUI();

    void newFrame();
    void renderPanels();
    void renderDrawData();
};