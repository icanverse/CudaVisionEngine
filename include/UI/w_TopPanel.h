#pragma once
#include <GLFW/glfw3.h>

class TopPanel {
private:
    bool isDragging = false;
    int dragOffsetX = 0;
    int dragOffsetY = 0;

public:
    void render(GLFWwindow* window, float displayWidth, unsigned int logoTextureId);
};