#include "../../include/UI/Workspace.h"
#include "UI/Layers/WorkspaceToolboxes.h"
#include "UI/WorkspaceTopPanel.h"
#include "imgui.h"

#include <GLFW/glfw3.h>

#include <iostream>
#include <string>

namespace {
// Workspace.h'in mevcut kokunu degistirmemek icin yeni paneller burada tutulur.
// Uygulamada tek editor Workspace'i oldugu varsayimiyla durumlari korunur.
WorkspaceToolboxes additionalToolboxes;
WorkspaceTopPanel workspaceTopPanel;

void synchronizeWorkspaceViewport(
    GLFWwindow* workspaceWindow,
    ImGuiViewport* workspaceViewport
) {
    if (workspaceWindow == nullptr || workspaceViewport == nullptr) return;

    // Ozel baslik cubugu GLFW penceresini dogrudan tasiyor. ImGui'nin mantiksal
    // viewport konumu da ayni anda guncellenmezse goruntu ile hitbox ayrisir.
    int windowScreenX = 0;
    int windowScreenY = 0;
    int windowWidth = 0;
    int windowHeight = 0;
    glfwGetWindowPos(workspaceWindow, &windowScreenX, &windowScreenY);
    glfwGetWindowSize(workspaceWindow, &windowWidth, &windowHeight);

    const ImVec2 actualWindowPosition(
        static_cast<float>(windowScreenX),
        static_cast<float>(windowScreenY)
    );

    workspaceViewport->Pos = actualWindowPosition;
    workspaceViewport->Size = ImVec2(
        static_cast<float>(windowWidth),
        static_cast<float>(windowHeight)
    );
    ImGui::SetWindowPos(actualWindowPosition, ImGuiCond_Always);
    ImGui::SetWindowSize(workspaceViewport->Size, ImGuiCond_Always);
}
}

Workspace::Workspace() : activeProject(nullptr) {
}

void Workspace::loadProject(Kivilcim::ProjectData* project) {
    activeProject = project;
    workspaceTopPanel.setLayers({
        {0, activeProject->name, true}
    });
    std::cout << "[Workspace] Tuval yuklendi. Proje: " << activeProject->name << std::endl;
}

void Workspace::render(float displayWidth, float displayHeight) {
    if (!activeProject) return;

    ImGuiWindowClass windowClass;
    windowClass.ViewportFlagsOverrideSet =
        ImGuiViewportFlags_NoAutoMerge |
        ImGuiViewportFlags_NoDecoration;
    ImGui::SetNextWindowClass(&windowClass);

    ImGui::SetNextWindowPos(ImVec2(100, 100), ImGuiCond_Appearing);
    ImGui::SetNextWindowSize(ImVec2(displayWidth * 0.9f, displayHeight * 0.9f), ImGuiCond_Appearing);

    const ImGuiWindowFlags workspaceFlags =
        ImGuiWindowFlags_NoTitleBar |
        ImGuiWindowFlags_NoCollapse;

    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.08f, 0.08f, 0.09f, 1.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 8.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));

    const std::string windowTitle =
        "Kivilcim Editor - " + activeProject->name + "###Workspace_Main";

    bool closeWorkspace = false;

    if (ImGui::Begin(windowTitle.c_str(), nullptr, workspaceFlags)) {
        ImGuiViewport* workspaceViewport = ImGui::GetWindowViewport();
        GLFWwindow* workspaceWindow = workspaceViewport != nullptr
            ? static_cast<GLFWwindow*>(workspaceViewport->PlatformHandle)
            : nullptr;

        synchronizeWorkspaceViewport(workspaceWindow, workspaceViewport);

        const float workspaceWidth = ImGui::GetWindowSize().x;
        closeWorkspace = workspaceTopPanel.render(workspaceWindow, workspaceWidth, 0);

        const float contentTop = workspaceTopPanel.getPanelHeight();
        ImGui::SetCursorPos(ImVec2(0.0f, contentTop));
        ImGui::BeginChild(
            "Workspace_Content",
            ImVec2(0.0f, 0.0f),
            false,
            ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse
        );

        const float editorWidth = ImGui::GetWindowSize().x;
        const float editorHeight = ImGui::GetWindowSize().y;

        quickToolbar.render(editorWidth, editorHeight);
        topToolbox.render(editorWidth, editorHeight);
        additionalToolboxes.render(editorWidth, editorHeight);

        ImGui::EndChild();
    }

    ImGui::End();
    ImGui::PopStyleVar(2);
    ImGui::PopStyleColor();

    if (closeWorkspace && onClose) {
        onClose();
    }
}