#include "../../include/UI/Workspace.h"
#include "imgui.h"
#include <iostream>

Workspace::Workspace() : activeProject(nullptr) {
    bgAlpha = 0.8f;

    // --- CUDA SHADER BAŞLANGIÇ DEĞERLERİ ---
    waveFrequency = 40.0f;
    waveSpeed = 3.0f;
    waveAmplitude = 0.05f;

    // Kıvılcım Turuncusu
    liquidColor[0] = 0.85f; liquidColor[1] = 0.45f; liquidColor[2] = 0.00f;
    // Koyu Cam Grisi
    shaderBgColor[0] = 0.05f; shaderBgColor[1] = 0.05f; shaderBgColor[2] = 0.06f;
    liquidAlpha = 1.0f;
}

void Workspace::loadProject(Kivilcim::ProjectData* project) {
    activeProject = project;
    std::cout << "[Workspace] Deneysel Laboratuvar moduna gecildi. Proje: " << activeProject->name << std::endl;
}

void Workspace::render(float displayWidth, float displayHeight) {
    if (!activeProject) return;

    ImVec2 windowSize(850.0f, 650.0f); // Arayüzü biraz büyüttük
    ImGui::SetNextWindowPos(
        ImVec2((displayWidth - windowSize.x) * 0.5f, (displayHeight - windowSize.y) * 0.5f),
        ImGuiCond_FirstUseEver
    );
    ImGui::SetNextWindowSize(windowSize, ImGuiCond_FirstUseEver);

    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.08f, 0.08f, 0.09f, bgAlpha));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 8.0f);

    ImGuiWindowFlags flags = ImGuiWindowFlags_NoCollapse;

    if (ImGui::Begin((activeProject->name + " - Deneysel Laboratuvar").c_str(), nullptr, flags)) {

        // ==========================================
        // YENİ: PANELİN KENDİSİNİ LİKİT CAM YAPMAK
        // ==========================================
        if (activeProject->textureID > 0) {
            ImDrawList* drawList = ImGui::GetWindowDrawList();
            ImVec2 winPos = ImGui::GetWindowPos();   // Panelin ekrandaki X, Y konumu
            ImVec2 winSize = ImGui::GetWindowSize(); // Panelin tam Genişlik ve Yüksekliği

            // Düşük seviyeli çizim API'si ile dokuyu panelin tam sınırlarına gerdir
            drawList->AddImage(
                (ImTextureID)(intptr_t)activeProject->textureID,
                winPos,
                ImVec2(winPos.x + winSize.x, winPos.y + winSize.y)
            );
        }

        // --- KONTROL PANELİ ---
        if (ImGui::Button("<- Ana Ekrana Don", ImVec2(150, 35))) {
            if (onClose) onClose();
        }

        ImGui::SameLine(0, 30.0f);
        ImGui::SetNextItemWidth(300.0f);
        ImGui::SliderFloat("Pencere Seffafligi", &bgAlpha, 0.0f, 1.0f, "Alpha: %.2f");
        ImGui::SameLine(0, 20.0f);
        ImGui::SliderFloat("Likit Seffafligi", &liquidAlpha, 0.0f, 1.0f, "%.2f");

        ImGui::Separator();
        ImGui::Dummy(ImVec2(0.0f, 10.0f));

        // ==========================================
        // YENİ: DİNAMİK SHADER KONTROLLERİ
        // ==========================================
        ImGui::TextColored(ImVec4(0.85f, 0.45f, 0.0f, 1.0f), "[ LİKİT CAM PARAMETRELERİ ]");

        ImGui::PushItemWidth(200.0f);
        ImGui::SliderFloat("Dalga Frekansi", &waveFrequency, 5.0f, 100.0f, "%.1f");
        ImGui::SameLine(0, 20.0f);
        ImGui::SliderFloat("Dalga Hizi", &waveSpeed, 0.0f, 15.0f, "%.1f");
        ImGui::SameLine(0, 20.0f);
        ImGui::SliderFloat("Genlik (Siddet)", &waveAmplitude, 0.01f, 0.30f, "%.3f");
        ImGui::PopItemWidth();

        ImGui::Dummy(ImVec2(0.0f, 5.0f));

        ImGuiColorEditFlags colorFlags = ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_NoLabel;
        ImGui::Text("Likit Rengi:"); ImGui::SameLine();
        ImGui::ColorEdit3("##LiqCol", liquidColor, colorFlags);

        ImGui::SameLine(0, 30.0f);

        ImGui::Text("Arka Plan:"); ImGui::SameLine();
        ImGui::ColorEdit3("##BgCol", shaderBgColor, colorFlags);

        ImGui::Dummy(ImVec2(0.0f, 15.0f));
        ImGui::Separator();
        ImGui::Dummy(ImVec2(0.0f, 10.0f));

    }
    ImGui::End();

    ImGui::PopStyleVar();
    ImGui::PopStyleColor();
}