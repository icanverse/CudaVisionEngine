#include "../Workspace.h"

#include "Layers/WorkspaceToolboxes.h"
#include "Tools/IsoDepthEditor.h"
#include "WorkspaceTopPanel.h"

#include "imgui.h"

#include <GLFW/glfw3.h>

#include <iostream>
#include <string>

namespace Kivilcim {
    struct ProjectData;
}

namespace {

// Workspace.h dosyasının mevcut yapısını değiştirmemek için
// ek paneller burada tutuluyor.
WorkspaceToolboxes additionalToolboxes;
WorkspaceTopPanel workspaceTopPanel;
Kivilcim::Tools::IsoDepthEditor isoEditor;

void synchronizeWorkspaceViewport(
    GLFWwindow* workspaceWindow,
    ImGuiViewport* workspaceViewport
) {
    if (workspaceWindow == nullptr || workspaceViewport == nullptr) {
        return;
    }

    int windowScreenX = 0;
    int windowScreenY = 0;

    glfwGetWindowPos(
        workspaceWindow,
        &windowScreenX,
        &windowScreenY
    );

    const ImVec2 actualWindowPosition(
        static_cast<float>(windowScreenX),
        static_cast<float>(windowScreenY)
    );

    workspaceViewport->Pos = actualWindowPosition;

    ImGui::SetWindowPos(
        actualWindowPosition,
        ImGuiCond_Always
    );
}

} // namespace

Workspace::Workspace()
    : activeProject(nullptr) {
}

void Workspace::loadProject(Kivilcim::ProjectData* project) {
    activeProject = project;

    if (activeProject == nullptr) {
        return;
    }

    // Canvas GPU/display texture'ını hazırla.
    additionalToolboxes.canvas().setProject(activeProject);

    workspaceTopPanel.setLayers({
        {
            activeProject->id,
            activeProject->name,
            true
        }
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

    std::cout
        << "[Workspace] Tuval yuklendi. Proje: "
        << activeProject->name
        << std::endl;
}

void Workspace::render(
    float displayWidth,
    float displayHeight
) {
    if (activeProject == nullptr) {
        return;
    }

    ImGuiWindowClass windowClass;

    windowClass.ViewportFlagsOverrideSet =
        ImGuiViewportFlags_NoAutoMerge |
        ImGuiViewportFlags_NoDecoration;

    // Ayrık Workspace framebuffer'ının her karede temizlenmesini sağlar.
    windowClass.ViewportFlagsOverrideClear =
        ImGuiViewportFlags_NoRendererClear;

    ImGui::SetNextWindowClass(&windowClass);

    ImGui::SetNextWindowPos(
        ImVec2(100.0f, 100.0f),
        ImGuiCond_Appearing
    );

    ImGui::SetNextWindowSize(
        ImVec2(
            displayWidth * 0.9f,
            displayHeight * 0.9f
        ),
        ImGuiCond_Appearing
    );

    const ImGuiWindowFlags workspaceFlags =
        ImGuiWindowFlags_NoTitleBar |
        ImGuiWindowFlags_NoCollapse;

    ImGui::PushStyleColor(
        ImGuiCol_WindowBg,
        ImVec4(0.0f, 0.0f, 0.0f, 1.0f)
    );

    ImGui::PushStyleVar(
        ImGuiStyleVar_WindowRounding,
        8.0f
    );

    ImGui::PushStyleVar(
        ImGuiStyleVar_WindowPadding,
        ImVec2(0.0f, 0.0f)
    );

    const std::string windowTitle =
        "Kivilcim Editor - " +
        activeProject->name +
        "###Workspace_Main";

    bool closeWorkspace = false;

    if (ImGui::Begin(
        windowTitle.c_str(),
        nullptr,
        workspaceFlags
    )) {
        ImGuiViewport* workspaceViewport =
            ImGui::GetWindowViewport();

        GLFWwindow* workspaceWindow =
            workspaceViewport != nullptr
                ? static_cast<GLFWwindow*>(
                    workspaceViewport->PlatformHandle
                )
                : nullptr;

        synchronizeWorkspaceViewport(
            workspaceWindow,
            workspaceViewport
        );

        // Aktif proje görüntüsünü Canvas'a aktar.
        additionalToolboxes.canvas().setProject(activeProject);

        // Workspace arka plan gradyanı.
        const ImVec2 windowMin =
            ImGui::GetWindowPos();

        const ImVec2 windowMax(
            windowMin.x + ImGui::GetWindowSize().x,
            windowMin.y + ImGui::GetWindowSize().y
        );

        const ImU32 colorTopLeft =
            IM_COL32(0, 0, 0, 255);

        const ImU32 colorTopRight =
            IM_COL32(5, 6, 10, 255);

        const ImU32 colorBottomRight =
            IM_COL32(0, 0, 0, 255);

        const ImU32 colorBottomLeft =
            IM_COL32(18, 10, 5, 255);

        ImGui::GetWindowDrawList()->AddRectFilledMultiColor(
            windowMin,
            windowMax,
            colorTopLeft,
            colorTopRight,
            colorBottomRight,
            colorBottomLeft
        );

        const float workspaceWidth =
            ImGui::GetWindowSize().x;

        closeWorkspace = workspaceTopPanel.render(
            workspaceWindow,
            workspaceWidth,
            0
        );

        // Üst panelde seçilen menü eylemini yakala.
        const WorkspaceMenuAction menuAction =
            workspaceTopPanel.consumeLastAction();

        switch (menuAction) {
            case WorkspaceMenuAction::
                CREATE_PHYSICAL_DEPTH_LAYER:

                isoEditor.isOpen = true;
                break;

            case WorkspaceMenuAction::
                CREATE_VIRTUAL_LIGHT:

                // Sanal ışık editörü eklendiğinde
                // burada açılacak.
                break;

            default:
                break;
        }

        const float contentTop =
            workspaceTopPanel.getPanelHeight();

        ImGui::SetCursorPos(
            ImVec2(0.0f, contentTop)
        );

        // Child arka planını şeffaf bırak.
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

        const float editorWidth =
            ImGui::GetWindowSize().x;

        const float editorHeight =
            ImGui::GetWindowSize().y;

        quickToolbar.render(
            editorWidth,
            editorHeight
        );

        topToolbox.render(
            editorWidth,
            editorHeight
        );

        additionalToolboxes.render(
            editorWidth,
            editorHeight
        );

        ImGui::EndChild();
        ImGui::PopStyleColor();
    }

    ImGui::End();

    ImGui::PopStyleVar(2);
    ImGui::PopStyleColor();

    // Workspace kapatılıyorsa bağlı editörleri de kapat.
    if (closeWorkspace) {
        isoEditor.isOpen = false;

        if (onClose) {
            onClose();
        }

        return;
    }

    // IsoDepthEditor yalnızca bir kez ve ana Workspace
    // penceresinden sonra render edilir.
    if (isoEditor.isOpen) {
        isoEditor.render(
            activeProject->textureID,
            static_cast<float>(
                activeProject->projectWidth
            ),
            static_cast<float>(
                activeProject->projectHeight
            )
        );
    }
}