#include "imgui.h"
#include <iostream>
#include <thread>

// CUDA ve Shader Header'ları
#include "UI/w_RightPanel.h"

#include "../io/TextureUtility/CudaDynamicTexture.cuh"
#include "../Compute/Shaders/LiquidShader.cuh"

// Diğer arayüz bağımlılıkların (kendi yollarına göre düzenle gerekirse)
// #include "w_TopPanel.h"
// #include "stb_image.h"
// #include "stb_image_resize.h"
// #include "UI_Utils.h" // openFileDialog, DrawSpinner vb. varsa

RightPanel::RightPanel() {
    memset(projectNameBuf, 0, sizeof(projectNameBuf));
    docWidth = 1920;
    docHeight = 1080;
    dimMetric = 0; 
    orientation = 1; 
    resolution = 72; 
    resMetric = 0; 
    bgContentMode = 0; 
    bgColor[0] = 1.0f; bgColor[1] = 1.0f; bgColor[2] = 1.0f;
    selectedImagePath = "";
    keepOriginalSize = true; 
    
    // İşletim sistemi erişim izni hatalarını (Access Violation) önlemek için güvenli varsayılan yol
    projectSavePath = "C:\\Users\\Can\\Documents\\KivilcimProjects";

    // Shader için VRAM dokusunu başlat
    shaderPreviewTexture = new CudaDynamicTexture(512, 288);
    flowTime = 0.0f;
}

RightPanel::~RightPanel() {
    if (shaderPreviewTexture) {
        delete shaderPreviewTexture;
        shaderPreviewTexture = nullptr;
    }
}

void RightPanel::render(float displayWidth, float displayHeight) {
    float panelWidth = 450.0f;
    
    // TopPanel yüksekliğini kendi projenin yapısına göre al
    // float topPanelHeight = TopPanel::getPanelHeight();
    float topPanelHeight = 50.0f; // Geçici sabit değer (Kendi kodunla değiştirebilirsin)

    float realScreenHeight = ImGui::GetIO().DisplaySize.y;
    float realScreenWidth = ImGui::GetIO().DisplaySize.x;

    float xPos = realScreenWidth - panelWidth - 15.0f;
    float yPos = 50.0f + topPanelHeight * 0.3f;
    float panelHeight = realScreenHeight - yPos - 15.0f;

    if (panelHeight < 100.0f) panelHeight = 100.0f;

    // ==========================================
    // TEMA VE STİL AYARLARI (TURUNCU KONSEPT & CAM EFEKTİ)
    // ==========================================
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 1.2f);

    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.0f, 0.0f, 0.0f, 0.0f));
    ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0.85f, 0.45f, 0.0f, 0.3f));
    ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.12f, 0.12f, 0.13f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, ImVec4(0.18f, 0.18f, 0.19f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_FrameBgActive, ImVec4(0.85f, 0.45f, 0.0f, 0.6f));
    ImGui::PushStyleColor(ImGuiCol_PopupBg, ImVec4(0.08f, 0.08f, 0.09f, 0.98f));
    ImGui::PushStyleColor(ImGuiCol_Header, ImVec4(0.85f, 0.45f, 0.0f, 0.5f));
    ImGui::PushStyleColor(ImGuiCol_HeaderHovered, ImVec4(0.85f, 0.45f, 0.0f, 0.7f));
    ImGui::PushStyleColor(ImGuiCol_HeaderActive, ImVec4(0.85f, 0.45f, 0.0f, 0.9f));
    ImGui::PushStyleColor(ImGuiCol_CheckMark, ImVec4(1.0f, 0.6f, 0.0f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_TextSelectedBg, ImVec4(0.85f, 0.45f, 0.0f, 0.4f));

    ImGui::SetNextWindowSize(ImVec2(panelWidth, panelHeight), ImGuiCond_Always);
    ImGui::SetNextWindowPos(ImVec2(xPos, yPos), ImGuiCond_Always);

    ImGuiWindowFlags rightPanel_flags = ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoTitleBar;

    ImGui::Begin("Proje Hazirlik", nullptr, rightPanel_flags);
    ImDrawList* drawList = ImGui::GetWindowDrawList();

    // ==========================================
    // CANLI LİKİT AKIŞI ÖNİZLEMESİ (SHADER)
    // ==========================================
    flowTime += ImGui::GetIO().DeltaTime;
    
    cudaSurfaceObject_t surface = shaderPreviewTexture->map();
    
    Kivilcim::Shaders::launchLiquidFlowShader(
        surface,
        shaderPreviewTexture->getWidth(),
        shaderPreviewTexture->getHeight(),
        flowTime,
        1.0f,    // Flow Speed
        2.0f,   // Frequency
        make_float3(0.85f, 0.45f, 0.0f) // Kıvılcım Turuncusu
    );
    
    shaderPreviewTexture->unmap();

    // Shader görselini panel genişliğine uyarlayıp çiz
    ImVec2 winPos = ImGui::GetWindowPos();   // Panelin ekrandaki X, Y konumu
    ImVec2 winSize = ImGui::GetWindowSize(); // Panelin tam Genişlik ve Yüksekliği

    // Shader'ı panelin arka planına tam boyutlu (stretch) olarak çizdiriyoruz
    drawList->AddImage(
        (ImTextureID)(intptr_t)shaderPreviewTexture->getTextureID(),
        winPos,
        ImVec2(winPos.x + winSize.x, winPos.y + winSize.y)
    );

    // Proje Detayları yazısından önce biraz boşluk bırakalım
    ImGui::Dummy(ImVec2(0.0f, 5.0f));

    // ==========================================
    // 1. PROJE ŞABLONU (ÜST KISIM)
    // ==========================================
    ImGui::Dummy(ImVec2(0.0f, 5.0f));
    ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.0f, 1.0f), "PROJE DETAYLARI");
    ImGui::Dummy(ImVec2(0.0f, 5.0f));

    // Proje Adı
    ImGui::Text("Proje Adı:");
    ImGui::PushItemWidth(panelWidth * 0.9f);
    ImGui::InputTextWithHint("##ProjAdi", "İsimsiz-1", projectNameBuf, IM_ARRAYSIZE(projectNameBuf));
    ImGui::PopItemWidth();
    ImGui::Dummy(ImVec2(0.0f, 10.0f));

    // Orijinal Boyutları Koru Checkbox'ı
    ImGui::Checkbox("Orijinal Çözünürlüğü Koru", &keepOriginalSize);
    ImGui::Dummy(ImVec2(0.0f, 5.0f));

    if (keepOriginalSize) ImGui::BeginDisabled();

    // Genişlik ve Yükseklik YAN YANA
    float colWidth = 130.0f;
    const char* dimMetrics[] = { "Piksel", "İnç", "Santimetre" };

    ImGui::Text("Genişlik:");
    ImGui::SameLine(colWidth + 25.0f);
    ImGui::Text("Yükseklik:");

    ImGui::PushItemWidth(colWidth);
    ImGui::InputInt("##Width", &docWidth, 0, 0);
    ImGui::PopItemWidth();

    ImGui::SameLine(colWidth + 25.0f);
    ImGui::PushItemWidth(colWidth);
    ImGui::InputInt("##Height", &docHeight, 0, 0);
    ImGui::PopItemWidth();

    ImGui::SameLine();
    ImGui::PushItemWidth(panelWidth - (colWidth * 2) - 45.0f);
    ImGui::Combo("##DimMetric", &dimMetric, dimMetrics, IM_ARRAYSIZE(dimMetrics));
    ImGui::PopItemWidth();

    ImGui::Dummy(ImVec2(0.0f, 10.0f));

    // Oryantasyon
    ImGui::Text("Oryantasyon:");
    ImGui::SameLine(100.0f);

    if (orientation == 0) ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.85f, 0.45f, 0.0f, 1.0f));
    else ImGui::PushStyleColor(ImGuiCol_Button, ImGui::GetStyle().Colors[ImGuiCol_Button]);
    if (ImGui::Button(" | ", ImVec2(35, 25))) { orientation = 0; }
    ImGui::PopStyleColor();

    ImGui::SameLine();

    if (orientation == 1) ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.85f, 0.45f, 0.0f, 1.0f));
    else ImGui::PushStyleColor(ImGuiCol_Button, ImGui::GetStyle().Colors[ImGuiCol_Button]);
    if (ImGui::Button(" - ", ImVec2(35, 25))) { orientation = 1; }
    ImGui::PopStyleColor();

    ImGui::Dummy(ImVec2(0.0f, 10.0f));

    // Çözünürlük (Resolution)
    const char* resMetrics[] = { "Piksel/İnç", "Piksel/cm" };
    ImGui::Text("Çözünürlük:");
    ImGui::PushItemWidth(colWidth);
    ImGui::InputInt("##Res", &resolution, 0, 0);
    ImGui::PopItemWidth();

    ImGui::SameLine(colWidth + 25.0f);
    ImGui::PushItemWidth(colWidth);
    ImGui::Combo("##ResMetric", &resMetric, resMetrics, IM_ARRAYSIZE(resMetrics));
    ImGui::PopItemWidth();

    if (keepOriginalSize) ImGui::EndDisabled();

    ImGui::Dummy(ImVec2(0.0f, 10.0f));

    // Arka Plan İçeriği (Background Contents)
    ImGui::Text("Arka Plan İçeriği:");
    const char* bgContents[] = { "Beyaz", "Siyah", "Şeffaf", "Özel Renk" };

    ImGui::PushItemWidth(colWidth * 1.5f);
    if (ImGui::Combo("##BgContent", &bgContentMode, bgContents, IM_ARRAYSIZE(bgContents))) {
        if (bgContentMode == 0) { bgColor[0] = 1.0f; bgColor[1] = 1.0f; bgColor[2] = 1.0f; }
        if (bgContentMode == 1) { bgColor[0] = 0.0f; bgColor[1] = 0.0f; bgColor[2] = 0.0f; }
    }
    ImGui::PopItemWidth();

    ImGui::SameLine();

    ImGuiColorEditFlags colorFlags = ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_NoLabel | ImGuiColorEditFlags_AlphaPreview;
    if (ImGui::ColorEdit3("##ColorBox", bgColor, colorFlags)) {
        bgContentMode = 3;
    }

    ImGui::Dummy(ImVec2(0.0f, 15.0f));
    ImGui::Separator();
    ImGui::Dummy(ImVec2(0.0f, 10.0f));

    // ==========================================
    // 2. GÖRSEL YÜKLEME ALANI
    // ==========================================
    ImVec2 dropZoneSize = ImVec2(ImGui::GetContentRegionAvail().x * 0.9f, 120.0f);
    float dropZoneIndent = (ImGui::GetContentRegionAvail().x - dropZoneSize.x) * 0.5f;
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + dropZoneIndent);

    ImVec2 cursorScreenPos = ImGui::GetCursorScreenPos();

    if (!isProcessingImage) {
        ImGui::InvisibleButton("DropZone", dropZoneSize);
        if (ImGui::IsItemClicked()) {
            // openFileDialog fonksiyonunu kendi util dosyana göre yapılandır
            // std::string tempPath = openFileDialog();
            // if (!tempPath.empty()) { selectedImagePath = tempPath; }
        }
    } else {
        ImGui::Dummy(dropZoneSize); 
    }

    bool isHovered = ImGui::IsItemHovered() && !isProcessingImage;
    ImU32 bgColorZone = isHovered ? IM_COL32(50, 50, 60, 200) : IM_COL32(30, 30, 35, 180);
    ImU32 borderColor = isHovered ? IM_COL32(255, 165, 0, 255) : IM_COL32(100, 100, 110, 150);

    drawList->AddRectFilled(cursorScreenPos, ImVec2(cursorScreenPos.x + dropZoneSize.x, cursorScreenPos.y + dropZoneSize.y), bgColorZone, 8.0f);
    drawList->AddRect(cursorScreenPos, ImVec2(cursorScreenPos.x + dropZoneSize.x, cursorScreenPos.y + dropZoneSize.y), borderColor, 8.0f, 0, isHovered ? 2.0f : 1.0f);

    ImVec2 centerPos = ImVec2(cursorScreenPos.x + dropZoneSize.x * 0.5f, cursorScreenPos.y + dropZoneSize.y * 0.5f);

    if (isProcessingImage) {
        // DrawSpinner(drawList, centerPos, 20.0f, 4.0f, IM_COL32(255, 165, 0, 255));
        std::string loadingText = "Gorsel Isleniyor...";
        ImVec2 textSize = ImGui::CalcTextSize(loadingText.c_str());
        drawList->AddText(ImVec2(centerPos.x - textSize.x * 0.5f, centerPos.y + 25.0f), IM_COL32(255, 165, 0, 255), loadingText.c_str());
    } else {
        std::string dropText = selectedImagePath.empty() ? "Gorsel Secin (Istege Bagli)" : "Gorsel Hazir!\nDegistirmek icin tikla.";
        ImVec2 textSize = ImGui::CalcTextSize(dropText.c_str());
        ImVec2 textPos = ImVec2(centerPos.x - textSize.x * 0.5f, centerPos.y - textSize.y * 0.5f);
        ImU32 textColor = selectedImagePath.empty() ? (isHovered ? IM_COL32(255, 204, 102, 255) : IM_COL32(180, 180, 180, 255)) : IM_COL32(120, 255, 120, 255);
        drawList->AddText(textPos, textColor, dropText.c_str());
    }

    // ==========================================
    // 2.5 GELİŞMİŞ SEÇENEKLER 
    // ==========================================
    ImGui::Dummy(ImVec2(0.0f, 15.0f));

    ImGui::PushStyleColor(ImGuiCol_Header, ImVec4(0.12f, 0.12f, 0.13f, 0.6f));
    ImGui::PushStyleColor(ImGuiCol_HeaderHovered, ImVec4(0.18f, 0.18f, 0.19f, 0.8f));
    ImGui::PushStyleColor(ImGuiCol_HeaderActive, ImVec4(0.85f, 0.45f, 0.0f, 0.6f));

    if (ImGui::CollapsingHeader("Gelismiş Secenekler")) {
        ImGui::Dummy(ImVec2(0.0f, 5.0f));
        ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Proje Kayit Konumu:");

        char pathBuf[512];
        strncpy(pathBuf, projectSavePath.c_str(), sizeof(pathBuf));
        pathBuf[sizeof(pathBuf) - 1] = '\0'; 

        ImGui::PushItemWidth(panelWidth * 0.65f); 
        if (ImGui::InputText("##FolderPath", pathBuf, sizeof(pathBuf))) {
            projectSavePath = pathBuf;
        }
        ImGui::PopItemWidth();

        ImGui::SameLine();

        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.2f, 0.22f, 1.0f));
        if (ImGui::Button("Gozat...", ImVec2(panelWidth * 0.2f, 0))) {
            // std::string selectedFolder = openFolderDialog();
            // if (!selectedFolder.empty()) { projectSavePath = selectedFolder; }
        }
        ImGui::PopStyleColor(); 
        ImGui::Dummy(ImVec2(0.0f, 10.0f));
    }
    ImGui::PopStyleColor(3); 

    // ==========================================
    // --- THREAD'DEN GELEN VERİYİ YAKALAMA ---
    // ==========================================
    if (isImageReadyForGPU) {
        GLuint textureID = 0;
        if (rawResizedData) {
            glGenTextures(1, &textureID);
            glBindTexture(GL_TEXTURE_2D, textureID);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 256, 144, 0, GL_RGBA, GL_UNSIGNED_BYTE, rawResizedData);
            free(rawResizedData);
            rawResizedData = nullptr;
        }

        std::string finalName = std::string(projectNameBuf);
        if (finalName.empty()) { finalName = "İsimsiz Proje"; }

        Kivilcim::ProjectData newData(0, finalName, selectedImagePath);
        newData.textureID = textureID;
        newData.size = {loadedOrigW, loadedOrigH};
        newData.projectWidth = docWidth;
        newData.projectHeight = docHeight;
        newData.bgColor[0] = bgColor[0]; newData.bgColor[1] = bgColor[1]; newData.bgColor[2] = bgColor[2];
        newData.keepOriginalSize = keepOriginalSize;
        newData.dimMetric = dimMetric;
        newData.orientation = orientation;
        newData.resolution = resolution;
        newData.kvlcmDir = projectSavePath;

        if (onProjectCreated) onProjectCreated(newData);

        selectedImagePath = "";
        memset(projectNameBuf, 0, sizeof(projectNameBuf));
        isImageReadyForGPU = false;
        isProcessingImage = false; 
    }

    // ==========================================
    // 3. OLUŞTUR BUTONU
    // ==========================================
    float buttonHeight = 45.0f;
    float currentY = ImGui::GetCursorPosY();
    float spaceLeft = panelHeight - currentY - buttonHeight - 35.0f;

    if (spaceLeft > 0) {
        ImGui::Dummy(ImVec2(0.0f, spaceLeft));
    }

    bool isDisabled = isProcessingImage;
    if (isDisabled) ImGui::BeginDisabled(); 

    ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 1.5f);

    // 2. Renkleri ayarla
    ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0.85f, 0.45f, 0.0f, 1.0f)); // ÇERÇEVE RENGİ (Kıvılcım Turuncusu)
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.85f, 0.45f, 0.0f, 0.0f)); // İÇİ SAYDAM (Senin kodun)
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(1.0f, 0.55f, 0.0f, 0.8f)); // Üzerine gelince hafif dolsun
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.65f, 0.35f, 0.0f, 1.0f)); // Tıklayınca tam dolsun

    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + dropZoneIndent);

    if (ImGui::Button("PROJEYI OLUSTUR", ImVec2(dropZoneSize.x, buttonHeight))) {
        if (!selectedImagePath.empty()) {
            isProcessingImage = true;
            std::string pathCopy = selectedImagePath;

            std::thread([this, pathCopy]() {
                // stb_image kütüphanelerini çağıracağın alan
                /*
                int w, h, channels;
                stbi_set_flip_vertically_on_load(true);
                unsigned char* data = stbi_load(pathCopy.c_str(), &w, &h, &channels, 4);
                if (data) {
                    this->loadedOrigW = w;
                    this->loadedOrigH = h;
                    this->rawResizedData = (unsigned char*)malloc(256 * 144 * 4);
                    stbir_resize_uint8(data, w, h, 0, this->rawResizedData, 256, 144, 0, 4);
                    stbi_image_free(data);
                }
                */
                this->isImageReadyForGPU = true;
            }).detach();

        } else {
            std::string finalName = std::string(projectNameBuf);
            if (finalName.empty()) { finalName = "İsimsiz Proje"; }

            Kivilcim::ProjectData newData(0, finalName, "");
            // newData.textureID = CreateSolidColorTexture(bgColor[0], bgColor[1], bgColor[2]);
            newData.size = {docWidth, docHeight};
            newData.projectWidth = docWidth;
            newData.projectHeight = docHeight;
            newData.bgColor[0] = bgColor[0]; newData.bgColor[1] = bgColor[1]; newData.bgColor[2] = bgColor[2];
            newData.keepOriginalSize = keepOriginalSize;
            newData.dimMetric = dimMetric;
            newData.orientation = orientation;
            newData.resolution = resolution;
            newData.kvlcmDir = projectSavePath;

            if (onProjectCreated) {
                onProjectCreated(newData);
            }
            memset(projectNameBuf, 0, sizeof(projectNameBuf));
        }
    }

    ImGui::PopStyleColor(4);
    ImGui::PopStyleVar(1);
    if (isDisabled) ImGui::EndDisabled();

    ImGui::End();

    ImGui::PopStyleColor(11);
    ImGui::PopStyleVar(2);
}

