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

    // Ozel baslik cubugu GLFW penceresini dogrudan tasiyor. Yalnizca mantiksal
    // konumu esitlemek hitbox kaymasini engeller. Boyutu burada zorlamiyoruz;
    // yeniden boyutlandirmayi ImGui/GLFW platform backend'i yonetir.
    int windowScreenX = 0;
    int windowScreenY = 0;
    glfwGetWindowPos(workspaceWindow, &windowScreenX, &windowScreenY);

    const ImVec2 actualWindowPosition(
        static_cast<float>(windowScreenX),
        static_cast<float>(windowScreenY)
    );

    workspaceViewport->Pos = actualWindowPosition;
    ImGui::SetWindowPos(actualWindowPosition, ImGuiCond_Always);
}
}

Workspace::Workspace() : activeProject(nullptr) {
}

void Workspace::loadProject(Kivilcim::ProjectData* project) {
    activeProject = project;
    if (activeProject == nullptr) return;

    // Once Canvas GPU/display texture'i hazirlanir. Katman thumbnail'i da
    // ayni texture kimligiyle ilk setLayers cagrisi icinde verilir.
    additionalToolboxes.canvas().setProject(activeProject);

    workspaceTopPanel.setLayers({
        {activeProject->id, activeProject->name, true}
    });
    additionalToolboxes.layers().setLayers({
        {
            activeProject->id,
            activeProject->name,
            additionalToolboxes.canvas().getDisplayTextureId(),
            true,
            false
        }
    });

    std::cout << "[Workspace] Tuval yuklendi. Proje: "
              << activeProject->name << std::endl;
}

void Workspace::render(float displayWidth, float displayHeight) {
    if (!activeProject) return;

    ImGuiWindowClass windowClass;
    windowClass.ViewportFlagsOverrideSet =
        ImGuiViewportFlags_NoAutoMerge |
        ImGuiViewportFlags_NoDecoration;

    // Ayrik Workspace framebuffer'inin her kareden once temizlenmesini saglar.
    // Bu bayrak temizlenmezse resize sirasinda eski toolbox kareleri birikir.
    windowClass.ViewportFlagsOverrideClear =
        ImGuiViewportFlags_NoRendererClear;

    ImGui::SetNextWindowClass(&windowClass);

    ImGui::SetNextWindowPos(ImVec2(100.0f, 100.0f), ImGuiCond_Appearing);
    ImGui::SetNextWindowSize(
        ImVec2(displayWidth * 0.9f, displayHeight * 0.9f),
        ImGuiCond_Appearing
    );

    const ImGuiWindowFlags workspaceFlags =
        ImGuiWindowFlags_NoTitleBar |
        ImGuiWindowFlags_NoCollapse;

    ImGui::PushStyleColor(
        ImGuiCol_WindowBg,
        ImVec4(0.0f, 0.0f, 0.0f, 1.0f)
    );
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

        // Aktif GPU goruntusunu dogrudan ProjectData uzerinden Canvas'a aktar.
        // d_imageData hazir degilse CanvasPanel textureID thumbnail'ina doner.
        additionalToolboxes.canvas().setProject(activeProject);

        // Asimetrik, yumusak Workspace gradyani.
        const ImVec2 windowMin = ImGui::GetWindowPos();
        const ImVec2 windowMax(
            windowMin.x + ImGui::GetWindowSize().x,
            windowMin.y + ImGui::GetWindowSize().y
        );

        const ImU32 colorTopLeft = IM_COL32(0, 0, 0, 255);
        const ImU32 colorTopRight = IM_COL32(5, 6, 10, 255);
        const ImU32 colorBottomRight = IM_COL32(0, 0, 0, 255);
        const ImU32 colorBottomLeft = IM_COL32(18, 10, 5, 255);

        ImGui::GetWindowDrawList()->AddRectFilledMultiColor(
            windowMin,
            windowMax,
            colorTopLeft,
            colorTopRight,
            colorBottomRight,
            colorBottomLeft
        );

        const float workspaceWidth = ImGui::GetWindowSize().x;
        closeWorkspace = workspaceTopPanel.render(
            workspaceWindow,
            workspaceWidth,
            0
        );

        const float contentTop = workspaceTopPanel.getPanelHeight();
        ImGui::SetCursorPos(ImVec2(0.0f, contentTop));

        // Child arka plani seffaf; ana Workspace gradyani gorunmeye devam eder.
        ImGui::PushStyleColor(
            ImGuiCol_ChildBg,
            ImVec4(0.0f, 0.0f, 0.0f, 0.0f)
        );
        ImGui::BeginChild(
            "Workspace_Content",
            ImVec2(0.0f, 0.0f),
            false,
            ImGuiWindowFlags_NoScrollbar |
            ImGuiWindowFlags_NoScrollWithMouse
        );

        const float editorWidth = ImGui::GetWindowSize().x;
        const float editorHeight = ImGui::GetWindowSize().y;

        quickToolbar.render(editorWidth, editorHeight);
        topToolbox.render(editorWidth, editorHeight);
        additionalToolboxes.render(editorWidth, editorHeight);

        ImGui::EndChild();
        ImGui::PopStyleColor();
    }

    ImGui::End();
    ImGui::PopStyleVar(2);
    ImGui::PopStyleColor();

    if (closeWorkspace && onClose) {
        onClose();
    }
}