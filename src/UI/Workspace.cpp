#include "../../include/UI/Workspace.h"
#include "imgui.h"
#include <iostream>

Workspace::Workspace() : activeProject(nullptr) {
    // Constructor artık tertemiz
}

void Workspace::loadProject(Kivilcim::ProjectData* project) {
    activeProject = project;
    std::cout << "[Workspace] Tuval yuklendi. Proje: " << activeProject->name << std::endl;
}

void Workspace::render(float displayWidth, float displayHeight) {
    if (!activeProject) return;

    // ==========================================
    // 1. ANA ÇALIŞMA ALANI (FULL SCREEN TUVAL)
    // ==========================================
    ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(displayWidth, displayHeight), ImGuiCond_Always);

    // Pencereyi arka plana sabitle ve başlık çubuğunu gizle
    ImGuiWindowFlags workspaceFlags = ImGuiWindowFlags_NoCollapse |
                                      ImGuiWindowFlags_NoResize |
                                      ImGuiWindowFlags_NoMove |
                                      ImGuiWindowFlags_NoTitleBar |
                                      ImGuiWindowFlags_NoBringToFrontOnFocus;

    // Göz yormayan, endüstriyel koyu gri bir çalışma alanı arka planı
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.08f, 0.08f, 0.09f, 1.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);

    if (ImGui::Begin("Workspace_Main", nullptr, workspaceFlags)) {

        // --- ÜST BİLGİ VE NAVİGASYON ---
        if (ImGui::Button("<- Ana Ekrana Don", ImVec2(150, 35))) {
            if (onClose) onClose();
        }

        ImGui::SameLine();
        ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 8.0f); // Yazıyı butona ortalamak için
        ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "Aktif Proje: %s", activeProject->name.c_str());

        // TUVAL (CANVAS) ÇİZİMLERİ İLERİDE BURAYA GELECEK...
        // (Şu an bilerek boş bıraktık)
    }
    ImGui::End();

    ImGui::PopStyleVar();
    ImGui::PopStyleColor();

    // ==========================================
    // 2. ARAÇ KUTUSU (QUICK TOOLBAR)
    // ==========================================
    // Ayrı bir ImGui penceresi olarak tuvalin üzerinde yüzecek
    quickToolbar.render();
}