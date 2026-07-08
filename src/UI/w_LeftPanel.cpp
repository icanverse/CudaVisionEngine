#include "../../include/UI/w_LeftPanel.h"
#include <iostream>

#include "imgui.h"
#include "UI/w_TopPanel.h"

// Görsel işleme ve VRAM aktarımı için
#include <stb_image.h>
#include <stb_image_resize.h>
#include <GLFW/glfw3.h>

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

    // ==========================================
    // PROJE IZGARASI (16:9)
    // ==========================================
    float windowVisibleX2 = ImGui::GetWindowPos().x + ImGui::GetWindowContentRegionMax().x;
    ImGuiStyle& style = ImGui::GetStyle();

    float tileWidth = 256.0f;
    float tileHeight = 144.0f;

    for (size_t i = 0; i < projectStack.size(); ++i) {
        ImGui::PushID(i);
        ImGui::BeginGroup();

        // 1. Görsel Alanı (Texture ID 0'dan büyükse başarılı yüklenmiştir)
        if (projectStack[i].textureID > 0) {
            // UV koordinatlarını ekledik: ImVec2(0, 1) ve ImVec2(1, 0) görseli dikeyde aynalar
            if (ImGui::ImageButton(projectStack[i].name.c_str(), (ImTextureID)(intptr_t)projectStack[i].textureID, ImVec2(tileWidth, tileHeight), ImVec2(0, 1), ImVec2(1, 0))) {
                std::cout << "[UI] Proje secildi: " << projectStack[i].name << std::endl;
                projectStack[i].isSelected = true;
            }
        } else {
            // Görsel yüklenemediyse normal buton
            if (ImGui::Button("Gorsel\nYok", ImVec2(tileWidth, tileHeight))) {
                std::cout << "[UI] Proje secildi: " << projectStack[i].name << std::endl;
            }
        }

        // 2. Altındaki İsim
        float textWidth = ImGui::CalcTextSize(projectStack[i].name.c_str()).x;
        float textIndent = (tileWidth - textWidth) * 0.5f;
        if (textIndent > 0.0f) ImGui::SetCursorPosX(ImGui::GetCursorPosX() + textIndent);
        ImGui::TextColored(ImVec4(0.8f, 0.8f, 0.8f, 1.0f), "%s", projectStack[i].name.c_str());

        ImGui::EndGroup();

        // 3. Grid Mantığı
        float lastGroupX2 = ImGui::GetItemRectMax().x;
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

// --- YARDIMCI FONKSİYON: THUMBNAIL OLUŞTURUCU ---
unsigned int LoadThumbnailTexture(const std::string& path, int targetW, int targetH) {
    int w, h, channels;
    stbi_set_flip_vertically_on_load(true);
    unsigned char* data = stbi_load(path.c_str(), &w, &h, &channels, 4); // 4 Kanal (RGBA) zorunlu

    if (!data) {
        std::cerr << "[Kivilcim UI] HATA: Gorsel okunamadi -> " << path << std::endl;
        return 0; // Hata durumunda 0 (Geçersiz Texture ID) döner
    }

    // Küçültülmüş resim için RAM'de yer ayır
    unsigned char* resizedData = (unsigned char*)malloc(targetW * targetH * 4);
    stbir_resize_uint8(data, w, h, 0, resizedData, targetW, targetH, 0, 4);

    // OpenGL (VRAM) üzerinde Doku (Texture) oluştur
    GLuint textureID;
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);

    // Filtreleme (Küçültülen resimler için Linear en iyisidir)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // Veriyi GPU'ya yolla
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, targetW, targetH, 0, GL_RGBA, GL_UNSIGNED_BYTE, resizedData);

    // RAM'i temizle (Pikseller artık GPU'da)
    free(resizedData);
    stbi_image_free(data);

    return textureID;
}

void LeftPanel::addProjectToStack(const std::string& photoPath) {
    // 1. Yeni projeyi oluştur (Constructor otomatik olarak diğer değişkenleri sıfırlar)
    std::string projName = "Isimsiz " + std::to_string(projectCounter++);
    Kivilcim::ProjectData newProject(projectCounter, projName, photoPath);

    // 2. Thumbnail'i oluştur ve VRAM'deki ID'sini kaydet
    newProject.textureID = LoadThumbnailTexture(photoPath, 256, 144);

    // İleride orijinal görsel boyutlarını (metadata) almak istersen stbi_info kullanabilirsin
    // stbi_info(photoPath.c_str(), &newProject.size.x, &newProject.size.y, &newProject.channels);

    // 3. Stack'e ekle
    projectStack.insert(projectStack.begin(), newProject);
    std::cout << "[Kivilcim UI] Yeni proje olusturuldu ve VRAM'e aktarildi: " << newProject.name << std::endl;
}