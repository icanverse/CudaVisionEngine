#include "../../include/UI/w_LeftPanel.h"
#include <iostream>
#include <algorithm>

#include "imgui.h"
#include "UI/w_TopPanel.h"

// --- YENİ: KAYIT PARSER'I VE GÖRSEL YÜKLEYİCİLER ---
#include <stb_image.h>
#include <stb_image_resize.h>
#include <GLFW/glfw3.h>

#include "io/UI/KvlcmProjectParser.h"

// Linker çakışmasını önlemek için 'static' tanımlandı
static unsigned int CreateSolidColorTexture_Local(float r, float g, float b) {
    GLuint textureID;
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);
    unsigned char data[4] = { (unsigned char)(r * 255.0f), (unsigned char)(g * 255.0f), (unsigned char)(b * 255.0f), 255 };
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
    return textureID;
}

static unsigned int LoadThumbnailTexture_Local(const std::string& path, int targetW, int targetH, int& outOrigW, int& outOrigH) {
    int w, h, channels;
    stbi_set_flip_vertically_on_load(true);
    unsigned char* data = stbi_load(path.c_str(), &w, &h, &channels, 4);
    if (!data) return 0;

    outOrigW = w; outOrigH = h;
    unsigned char* resizedData = (unsigned char*)malloc(targetW * targetH * 4);
    stbir_resize_uint8(data, w, h, 0, resizedData, targetW, targetH, 0, 4);

    GLuint textureID;
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, targetW, targetH, 0, GL_RGBA, GL_UNSIGNED_BYTE, resizedData);

    free(resizedData);
    stbi_image_free(data);
    return textureID;
}

// --- RENDER DÖNGÜSÜ ---
void LeftPanel::render(float displayWidth, float displayHeight) {
    float topPanelHeight = TopPanel::getPanelHeight();
    float realScreenHeight = ImGui::GetIO().DisplaySize.y;

    float panelWidth = 840.0f;
    float xPos = 15.0f;
    float yPos = 50.0f + topPanelHeight * 0.3f;
    float panelHeight = realScreenHeight - yPos - 15.0f;

    if (panelHeight < 100.0f) panelHeight = 100.0f;

    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.02f, 0.02f, 0.03f, 0.6f));

    ImGui::SetNextWindowSize(ImVec2(panelWidth, panelHeight), ImGuiCond_Always);
    ImGui::SetNextWindowPos(ImVec2(xPos, yPos), ImGuiCond_Always);

    ImGuiWindowFlags leftPanel_flags = ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoTitleBar;

    ImGui::Begin("Hadi Başlayalım!", nullptr, leftPanel_flags);
    ImGui::SetWindowFontScale(1.8f);
    ImGui::TextColored(ImVec4(1.0f, 1.0f, 1.0f, 1.0f), "Hadi Baslayalim!");
    ImGui::SetWindowFontScale(1.0f);
    ImGui::Separator();
    ImGui::Dummy(ImVec2(0.0f, 15.0f));

    float windowVisibleX2 = ImGui::GetWindowPos().x + ImGui::GetWindowContentRegionMax().x;
    ImGuiStyle& style = ImGui::GetStyle();

    float tileWidth = 256.0f;
    float tileHeight = 144.0f;

    for (size_t i = 0; i < projectStack.size(); ++i) {
        ImGui::PushID((int)i);
        ImGui::BeginGroup();

        ImVec2 startPos = ImGui::GetCursorPos();

        if (projectStack[i].textureID > 0) {
            float origW = (float)projectStack[i].size.x;
            float origH = (float)projectStack[i].size.y;

            if (origW <= 0.0f) origW = tileWidth;
            if (origH <= 0.0f) origH = tileHeight;

            float scale = std::min(tileWidth / origW, tileHeight / origH);
            float renderW = origW * scale;
            float renderH = origH * scale;
            float offsetX = (tileWidth - renderW) * 0.5f;
            float offsetY = (tileHeight - renderH) * 0.5f;

            ImGui::SetCursorPos(ImVec2(startPos.x + offsetX, startPos.y + offsetY));

            if (ImGui::ImageButton(projectStack[i].name.c_str(), (ImTextureID)(intptr_t)projectStack[i].textureID, ImVec2(renderW, renderH), ImVec2(0, 1), ImVec2(1, 0))) {
                std::cout << "[UI] Proje secildi: " << projectStack[i].name << std::endl;
                projectStack[i].isSelected = true;
            }

            // --- ÇİFT TIKLAMA KONTROLÜ (GÖRSEL İÇİN) ---
            if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
                std::cout << "[UI] Projeye CIFT TIKLANDI: " << projectStack[i].name << std::endl;
                if (onProjectDoubleClicked) onProjectDoubleClicked(projectStack[i].id);
            }
        } else {
            if (ImGui::Button("Gorsel\nYok", ImVec2(tileWidth, tileHeight))) {
                std::cout << "[UI] Proje secildi: " << projectStack[i].name << std::endl;
            }
            // ---  ÇİFT TIKLAMA KONTROLÜ (NORMAL BUTON İÇİN) ---
            if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
                std::cout << "[UI] Projeye CIFT TIKLANDI: " << projectStack[i].name << std::endl;
                if (onProjectDoubleClicked) onProjectDoubleClicked(projectStack[i].id);
            }
        }

        ImGui::SetCursorPos(ImVec2(startPos.x, startPos.y + tileHeight + 5.0f));

        float textWidth = ImGui::CalcTextSize(projectStack[i].name.c_str()).x;
        float textIndent = (tileWidth - textWidth) * 0.5f;
        if (textIndent > 0.0f) ImGui::SetCursorPosX(ImGui::GetCursorPosX() + textIndent);
        ImGui::TextColored(ImVec4(0.8f, 0.8f, 0.8f, 1.0f), "%s", projectStack[i].name.c_str());

        ImGui::EndGroup();

        float lastGroupX2 = startPos.x + tileWidth;
        float nextGroupX2 = lastGroupX2 + style.ItemSpacing.x + tileWidth;
        if (i + 1 < projectStack.size() && nextGroupX2 < windowVisibleX2) {
            ImGui::SameLine();
        } else {
            ImGui::Dummy(ImVec2(0.0f, 20.0f));
        }

        ImGui::PopID();
    }

    ImGui::End();
    ImGui::PopStyleColor();
    ImGui::PopStyleVar(2);
}

// --- YENİ PROJE EKLEME ---
void LeftPanel::addProjectToStack(Kivilcim::ProjectData newProject) {
    if (newProject.id == 0) {
        newProject.id = projectCounter++;
    } else if (newProject.id >= projectCounter) {
        projectCounter = newProject.id + 1; // ID sayacını disktan gelen veriye göre senkronize et
    }

    if (newProject.name == "İsimsiz-1" || newProject.name.empty()) {
        newProject.name = "İsimsiz Proje " + std::to_string(newProject.id);
    }

    projectStack.insert(projectStack.begin(), newProject);
    std::cout << "[Kivilcim UI] Proje eklendi: " << newProject.name << std::endl;
}

// --- ÇALIŞMA ALANINI DİSKTEN YÜKLE ---
// --- ÇALIŞMA ALANINI DİSKTEN YÜKLE ---
// --- ÇALIŞMA ALANINI DİSKTEN YÜKLE ---
void LeftPanel::loadWorkspace() {
    // YENİ: Okumaya başlamadan önce mevcut listeyi tamamen sıfırla (X2 Kopyalanma BUG'ını sonsuza dek çözer)
    projectStack.clear();

    // Yolu doğrudan masaüstüne verdik
    std::vector<Kivilcim::ProjectData> savedProjects = Kivilcim::KvlcmProjectParser::load("C:/Users/Can/Desktop/sirca_workspace.kvlcm_proj");

    for (auto it = savedProjects.rbegin(); it != savedProjects.rend(); ++it) {
        Kivilcim::ProjectData& p = *it;
        if (!p.imagePath.empty()) {
            int origW = 0, origH = 0;
            p.textureID = LoadThumbnailTexture_Local(p.imagePath, 256, 144, origW, origH);
            if (p.textureID > 0) p.size = {origW, origH};
        } else {
            p.textureID = CreateSolidColorTexture_Local(p.bgColor[0], p.bgColor[1], p.bgColor[2]);
        }
        this->addProjectToStack(p);
    }
}
// --- ÇALIŞMA ALANINI DİSKE KAYDET ---
void LeftPanel::saveWorkspace() {
    // Yolu doğrudan masaüstüne verdik
    Kivilcim::KvlcmProjectParser::save("C:/Users/Can/Desktop/sirca_workspace.kvlcm_proj", projectStack);
}