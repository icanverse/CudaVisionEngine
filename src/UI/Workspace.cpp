#include "../../include/UI/Workspace.h"
#include "imgui.h"
#include <iostream>
#include <string>

Workspace::Workspace() : activeProject(nullptr) {
}

void Workspace::loadProject(Kivilcim::ProjectData* project) {
    activeProject = project;
    std::cout << "[Workspace] Tuval yuklendi. Proje: " << activeProject->name << std::endl;
}

void Workspace::render(float displayWidth, float displayHeight) {
    if (!activeProject) return;

    // ==========================================
    // YENİ: UNITY EDITOR GİBİ BAĞIMSIZ PENCERE ZORLAMASI
    // ==========================================
    ImGuiWindowClass window_class;
    window_class.ViewportFlagsOverrideSet = ImGuiViewportFlags_NoAutoMerge; // Asla Hub'a (Ana Pencereye) yapışma!
    ImGui::SetNextWindowClass(&window_class);

    // İlk açıldığında Hub'ın boyutlarında açılır, sonra özgürdür.
    ImGui::SetNextWindowPos(ImVec2(100, 100), ImGuiCond_Appearing);
    ImGui::SetNextWindowSize(ImVec2(displayWidth * 0.9f, displayHeight * 0.9f), ImGuiCond_Appearing);

    ImGuiWindowFlags workspaceFlags = ImGuiWindowFlags_NoCollapse;

    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.08f, 0.08f, 0.09f, 1.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 8.0f);

    std::string windowTitle = "Kivilcim Editor - " + activeProject->name + "###Workspace_Main";

    // PENCEREYİ BAŞLAT
    if (ImGui::Begin(windowTitle.c_str(), nullptr, workspaceFlags)) {

        if (ImGui::Button("<- Hub'a Don", ImVec2(150, 35))) {
            if (onClose) onClose();
        }

        ImGui::SameLine();
        ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 8.0f);
        ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "Aktif Proje: %s", activeProject->name.c_str());

        // ==========================================
        // YENİ: ARAÇ KUTULARINI EDITOR'ÜN İÇİNE GÖMÜYORUZ
        // ==========================================
        // Artık ekranın değil, Workspace penceresinin kendi genişlik ve yüksekliğini alıyoruz
        float editorWidth = ImGui::GetWindowSize().x;
        float editorHeight = ImGui::GetWindowSize().y;

        // Toolbar'ları dışarıda değil, BURADA (Begin ve End arasında) çağırıyoruz!
        quickToolbar.render(editorWidth, editorHeight);
        topToolbox.render(editorWidth, editorHeight);
    }

    ImGui::End();

    ImGui::PopStyleVar();
    ImGui::PopStyleColor();
}