#include "Workspace.h"

#include "imgui.h"
#include <GLFW/glfw3.h>

#include <iostream>
#include <string>

namespace {

    void synchronizeWorkspaceViewport(GLFWwindow* workspaceWindow, ImGuiViewport* workspaceViewport) {
        if (workspaceWindow == nullptr || workspaceViewport == nullptr) return;

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

} // namespace

Workspace::Workspace() {
}

void Workspace::render(Kdata::WorkspaceStateData* state, float displayWidth, float displayHeight) {
    // Güvenlik kontrolü: State yoksa veya geçerli bir proje başlatılmadıysa çizme
    if (!state || (state->project.id == 0 && state->project.name.empty())) {
        return;
    }

    ImGuiWindowClass windowClass;
    windowClass.ViewportFlagsOverrideSet = ImGuiViewportFlags_NoAutoMerge | ImGuiViewportFlags_NoDecoration;
    windowClass.ViewportFlagsOverrideClear = ImGuiViewportFlags_NoRendererClear;

    ImGui::SetNextWindowClass(&windowClass);
    ImGui::SetNextWindowPos(ImVec2(100.0f, 100.0f), ImGuiCond_Appearing);
    ImGui::SetNextWindowSize(ImVec2(displayWidth * 1.24f, displayHeight * 1.24f), ImGuiCond_Appearing);

    const ImGuiWindowFlags workspaceFlags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse;

    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.0f, 0.0f, 0.0f, 1.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));

    // Başlığı State üzerinden çekiyoruz
    const std::string windowTitle = "Kivilcim Editor - " + state->project.name + "###Workspace_Main";

    bool closeWorkspace = false;

    if (ImGui::Begin(windowTitle.c_str(), nullptr, workspaceFlags)) {
        ImGuiViewport* workspaceViewport = ImGui::GetWindowViewport();
        GLFWwindow* workspaceWindow = workspaceViewport != nullptr
            ? static_cast<GLFWwindow*>(workspaceViewport->PlatformHandle) : nullptr;

        synchronizeWorkspaceViewport(workspaceWindow, workspaceViewport);

        const ImVec2 windowMin = ImGui::GetWindowPos();
        const ImVec2 windowMax(windowMin.x + ImGui::GetWindowSize().x, windowMin.y + ImGui::GetWindowSize().y);

        const ImU32 colorTopLeft = IM_COL32(22, 22, 25, 255);
        const ImU32 colorTopRight = IM_COL32(22, 22, 25, 255);
        const ImU32 colorBottomRight = IM_COL32(10, 10, 12, 255);
        const ImU32 colorBottomLeft = IM_COL32(10, 10, 12, 255);

        ImGui::GetWindowDrawList()->AddRectFilledMultiColor(
            windowMin, windowMax, colorTopLeft, colorTopRight, colorBottomRight, colorBottomLeft
        );

        const float workspaceWidth = ImGui::GetWindowSize().x;

        // YENİ: Top panel artık state parametresi alıyor
        closeWorkspace = workspaceTopPanel.render(state, workspaceWindow, workspaceWidth, 0);

        // NOT: Eski consumeLastAction() ve switch-case bloğu tamamen silindi.
        // Eylemler zaten alt panellerin içinde doğrudan State'e yazılıyor.

        // getPanelHeight statik bir fonksiyon olduğu için sınıf üzerinden doğrudan çağırıyoruz
        const float contentTop = WorkspaceTopPanel::getPanelHeight();

        ImGui::SetCursorPos(ImVec2(0.0f, contentTop));
        ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.0f, 0.0f, 0.0f, 0.0f));

        ImGui::BeginChild(
            "Workspace_Content", ImVec2(0.0f, 0.0f), false,
            ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse
        );

        const float editorWidth = ImGui::GetWindowSize().x;
        const float editorHeight = ImGui::GetWindowSize().y;

        // YENİ: Alt panellerin TAMAMI artık ilk parametre olarak 'state' alıyor.
        quickInspectorToolbox.render(state, editorWidth, editorHeight);
        quickToolbar.render(state, editorWidth, editorHeight);
        topToolbox.render(state, editorWidth, editorHeight);
        additionalToolboxes.render(state, editorWidth, editorHeight);

        ImGui::EndChild();
        ImGui::PopStyleColor();
    }

    ImGui::End();
    ImGui::PopStyleVar(2);
    ImGui::PopStyleColor();

    if (closeWorkspace) {
        // Workspace kapanırken editörleri de kapat
        state->editors.activeEditor = Kdata::ExclusiveEditor::NONE;
        if (onClose) onClose();
        return;
    }

    // YENİ MİMARİ: Editör kontrolü state üzerinden yapılıyor
    if (state->editors.isEditorActive() && state->editors.activeEditor == Kdata::ExclusiveEditor::ISO_DEPTH) {
        isoEditor.render(
            state->project.textureID,
            static_cast<float>(state->project.projectWidth),
            static_cast<float>(state->project.projectHeight)
        );
    }
}