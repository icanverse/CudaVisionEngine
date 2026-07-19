#include "UI/w_TopPanel.h"
#include "imgui.h"

#ifdef _WIN32
#include <windows.h>
#endif

#include <GLFW/glfw3.h>

// DUZELTME: "static" eklendi. Onceden bu global bir degiskendi ve baska bir
// .cpp dosyasinda ayni isimde global bir sey tanimlanirsa ODR (One Definition
// Rule) ihlali / link hatasi riski vardi. Artik bu degisken sadece bu
// translation unit'e ozel.
static float panelHeight = 60.0f;

void TopPanel::render(GLFWwindow* window, float displayWidth, unsigned int logoTextureId) {
    ImGuiViewport* viewport = ImGui::GetMainViewport();

    // Paneli ana pencereye hapseder
    ImGui::SetNextWindowViewport(viewport->ID);
    ImGui::SetNextWindowPos(viewport->Pos, ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(viewport->Size.x, panelHeight), ImGuiCond_Always);

    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.05f, 0.05f, 0.06f, 1.0f));

    ImGuiWindowFlags flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                             ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar |
                             ImGuiWindowFlags_NoScrollWithMouse | ImGuiWindowFlags_NoDocking;

    ImGui::Begin("TopPanel", nullptr, flags);

    ImVec2 minPos = ImGui::GetWindowPos();
    ImVec2 maxPos = ImVec2(minPos.x + ImGui::GetWindowWidth(), minPos.y + ImGui::GetWindowHeight());

    ImU32 colorTopLeft  = IM_COL32(55, 30, 10, 255);
    ImU32 colorTopRight = IM_COL32(55, 30, 10, 255);
    ImU32 colorBotLeft  = IM_COL32(0, 0, 0, 255);
    ImU32 colorBotRight = IM_COL32(0, 0, 0, 255);

    ImDrawList* drawList = ImGui::GetWindowDrawList();
    drawList->AddRectFilledMultiColor(minPos, maxPos, colorTopLeft, colorTopRight, colorBotRight, colorBotLeft);

    if (logoTextureId != 0) {
        float padding = 16.0f;
        float logoHeight = panelHeight - padding;
        float logoWidth = logoHeight;

        glBindTexture(GL_TEXTURE_2D, logoTextureId);
        int texWidth = 0, texHeight = 0;
        glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, &texWidth);
        glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT, &texHeight);

        if (texHeight > 0) {
            logoWidth = logoHeight * ((float)texWidth / (float)texHeight);
        }

        ImGui::SetCursorPos(ImVec2(15.0f, padding * 0.5f));
        ImGui::Image((void*)(intptr_t)logoTextureId, ImVec2(logoWidth, logoHeight), ImVec2(0.0f, 1.0f), ImVec2(1.0f, 0.0f));
        ImGui::SameLine();
    }

    ImGui::SetCursorPosY(0.0f);

    float buttonWidth = panelHeight;
    float buttonHeight = panelHeight;

    float dragAreaWidth = displayWidth - ImGui::GetCursorPosX() - (buttonWidth * 2.0f);
    if (dragAreaWidth < 10.0f) dragAreaWidth = 10.0f;

    ImGui::InvisibleButton("DragArea", ImVec2(dragAreaWidth, panelHeight));

    if (ImGui::IsItemHovered() && ImGui::IsMouseClicked(0)) {
        isDragging = true;
        double mX, mY;
        glfwGetCursorPos(window, &mX, &mY);
        dragOffsetX = (int)mX;
        dragOffsetY = (int)mY;
    }

    if (isDragging && ImGui::IsMouseDown(0)) {
#ifdef _WIN32
        POINT p;
        GetCursorPos(&p);
        glfwSetWindowPos(window, p.x - dragOffsetX, p.y - dragOffsetY);
#endif
    } else {
        isDragging = false;
    }

    ImGui::SameLine(0, 0);
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 0.0f);

    ImGui::SetCursorPosY(0.0f);
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.0f, 0.0f, 0.0f, 0.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(1.0f, 1.0f, 1.0f, 0.1f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(1.0f, 1.0f, 1.0f, 0.2f));

    if (ImGui::Button("-", ImVec2(buttonWidth, buttonHeight))) glfwIconifyWindow(window);

    ImGui::PopStyleColor(3);
    ImGui::SameLine(0, 0);

    ImGui::SetCursorPosY(0.0f);
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.0f, 0.0f, 0.0f, 0.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.9f, 0.1f, 0.15f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.7f, 0.1f, 0.1f, 1.0f));

    if (ImGui::Button("X", ImVec2(buttonWidth, buttonHeight))) glfwSetWindowShouldClose(window, 1);

    ImGui::PopStyleColor(3);
    ImGui::PopStyleVar();

    ImGui::End();
    ImGui::PopStyleColor();
    ImGui::PopStyleVar();
}

float TopPanel::getPanelHeight() {
    return panelHeight;
}