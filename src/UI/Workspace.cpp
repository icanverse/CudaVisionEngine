#include "../../include/UI/Workspace.h"
#include "UI/Layers/WorkspaceToolboxes.h"
#include "imgui.h"

#include <iostream>
#include <string>

namespace {
// Workspace.h'in mevcut kokunu degistirmemek icin yeni paneller burada tutulur.
// Uygulamada tek editor Workspace'i oldugu varsayimiyla durumlari korunur.
WorkspaceToolboxes additionalToolboxes;
}

Workspace::Workspace() : activeProject(nullptr) {
}

void Workspace::loadProject(Kivilcim::ProjectData* project) {
    activeProject = project;
    std::cout << "[Workspace] Tuval yuklendi. Proje: " << activeProject->name << std::endl;
}

void Workspace::render(float displayWidth, float displayHeight) {
    if (!activeProject) return;

    ImGuiWindowClass windowClass;
    windowClass.ViewportFlagsOverrideSet = ImGuiViewportFlags_NoAutoMerge;
    ImGui::SetNextWindowClass(&windowClass);

    ImGui::SetNextWindowPos(ImVec2(100, 100), ImGuiCond_Appearing);
    ImGui::SetNextWindowSize(ImVec2(displayWidth * 0.9f, displayHeight * 0.9f), ImGuiCond_Appearing);

    const ImGuiWindowFlags workspaceFlags = ImGuiWindowFlags_NoCollapse;

    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.08f, 0.08f, 0.09f, 1.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 8.0f);

    const std::string windowTitle =
        "Kivilcim Editor - " + activeProject->name + "###Workspace_Main";

    if (ImGui::Begin(windowTitle.c_str(), nullptr, workspaceFlags)) {
        if (ImGui::Button("<- Hub'a Don", ImVec2(150, 35))) {
            if (onClose) onClose();
        }

        ImGui::SameLine();
        ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 8.0f);
        ImGui::TextColored(
            ImVec4(0.5f, 0.5f, 0.5f, 1.0f),
            "Aktif Proje: %s",
            activeProject->name.c_str()
        );

        const float editorWidth = ImGui::GetWindowSize().x;
        const float editorHeight = ImGui::GetWindowSize().y;

        // Var olan iki arac kutusunun akisi aynen korunuyor.
        quickToolbar.render(editorWidth, editorHeight);
        topToolbox.render(editorWidth, editorHeight);

        // Yeni paneller mevcut Workspace penceresinin icinde ciziliyor.
        additionalToolboxes.render(editorWidth, editorHeight);
    }

    ImGui::End();
    ImGui::PopStyleVar();
    ImGui::PopStyleColor();
}
