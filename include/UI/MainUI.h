#pragma once
#include <GLFW/glfw3.h>
#include "w_RightPanel.h"
#include "w_TopPanel.h"
#include "w_BackgroundPanel.h"

class MainUI {
private:
    TopPanel topPanel;
    RightPanel rightPanel;
    BackgroundPanel backgroundPanel;
    GLFWwindow* windowHandle;
    unsigned int logoTextureId; // Logonun VRAM'deki adresi
public:
    MainUI(GLFWwindow* window);
    ~MainUI();

    void newFrame();
    void renderPanels();
    void renderDrawData();
};
