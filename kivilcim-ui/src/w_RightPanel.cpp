#include "w_RightPanel.h"
#include "imgui.h"
#include <iostream>
#include <cstring>
#include <cstdlib>

#ifdef _WIN32
#include <windows.h>
#include <commdlg.h>
#include <shlobj.h>
#endif

#include "TextureUtility/CudaDynamicTexture.cuh"
#include "TextureUtility/TextureUtility.h"
#include "../../kivilcim-graphics/include/Shaders/LiquidShader.cuh"

// İsimsiz namespace: Sadece bu dosyaya özel güvenli C++ diyalog fonksiyonları
namespace {
#ifdef _WIN32
    std::string openFileDialog() {
        char filename[MAX_PATH];
        OPENFILENAMEA ofn;
        ZeroMemory(&filename, sizeof(filename));
        ZeroMemory(&ofn, sizeof(ofn));
        ofn.lStructSize = sizeof(ofn);
        ofn.hwndOwner = NULL;
        ofn.lpstrFilter = "Gorsel Dosyalari\0*.png;*.jpg;*.jpeg;*.bmp\0Tum Dosyalar\0*.*\0";
        ofn.lpstrFile = filename;
        ofn.nMaxFile = MAX_PATH;
        ofn.lpstrTitle = "Projeye Gorsel Ekle";
        ofn.Flags = OFN_DONTADDTORECENT | OFN_FILEMUSTEXIST | OFN_NOCHANGEDIR;

        if (GetOpenFileNameA(&ofn)) {
            return std::string(filename);
        }
        return "";
    }

    std::string openFolderDialog() {
        char path[MAX_PATH];
        BROWSEINFOA bi = { 0 };
        bi.lpszTitle = "Proje Kayit Klasorunu Sec";
        bi.ulFlags = BIF_RETURNONLYFSDIRS | BIF_NEWDIALOGSTYLE;

        LPITEMIDLIST pidl = SHBrowseForFolderA(&bi);
        if (pidl != nullptr) {
            std::string result;
            if (SHGetPathFromIDListA(pidl, path)) {
                result = std::string(path);
            }
            CoTaskMemFree(pidl);
            return result;
        }
        return "";
    }
#else
    std::string openFileDialog() { return ""; }
    std::string openFolderDialog() { return ""; }
#endif
}

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

    projectSavePath = "C:\\Users\\Can\\Documents\\KivilcimProjects";
    shaderPreviewTexture = new CudaDynamicTexture(512, 288);
    flowTime = 0.0f;
}

RightPanel::~RightPanel() {
    // THREAD GÜVENLİĞİ: Arka planda okuma yaparken pencere kapanırsa bekle!
    if (workerThread.joinable()) {
        workerThread.join();
    }

    if (shaderPreviewTexture) {
        delete shaderPreviewTexture;
        shaderPreviewTexture = nullptr;
    }

    if (rawResizedData) {
        std::free(rawResizedData);
        rawResizedData = nullptr;
    }
}

void RightPanel::startImageProcessing(const std::string& path) {
    if (workerThread.joinable()) {
        workerThread.join(); // Varsa önceki thread'i temizle
    }

    isProcessingImage = true;
    std::string pathCopy = path;

    // Bağımsız ve güvenli iş parçacığı (Thread) oluşturuluyor
    workerThread = std::thread([this, pathCopy]() {
        int originalWidth = 0;
        int originalHeight = 0;

        unsigned char* resizedPixels = TextureUtility::LoadResizedPixels(
            pathCopy, 256, 144, originalWidth, originalHeight
        );

        if (!resizedPixels) {
            this->isProcessingImage = false;
            return;
        }

        this->loadedOrigW = originalWidth;
        this->loadedOrigH = originalHeight;
        this->rawResizedData = resizedPixels;

        // Atomik bayrak: Ana thread (ImGui) artık dokuyu GPU'ya atabilir
        this->isImageReadyForGPU = true;
    });
}

void RightPanel::render(float displayWidth, float displayHeight) {
    ImGuiViewport* viewport = ImGui::GetMainViewport();
    float panelWidth = 450.0f;
    float topPanelHeight = 50.0f;

    float realScreenHeight = ImGui::GetIO().DisplaySize.y;
    float xPos = viewport->Pos.x + viewport->Size.x - panelWidth - 15.0f;
    float yPos = viewport->Pos.y + 50.0f + topPanelHeight * 0.3f;
    float panelHeight = realScreenHeight - yPos - 15.0f;

    if (panelHeight < 100.0f) panelHeight = 100.0f;

    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 1.2f);

    // ... [STIL AYARLARIN AYNI KALACAK] ...
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

    ImGui::SetNextWindowViewport(viewport->ID);
    ImGui::SetNextWindowPos(ImVec2(xPos, yPos), ImGuiCond_Always);

    ImGuiWindowFlags rightPanel_flags = ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse |
                                        ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoTitleBar |
                                        ImGuiWindowFlags_NoDocking;

    ImGui::Begin("Proje Hazirlik", nullptr, rightPanel_flags);
    ImDrawList* drawList = ImGui::GetWindowDrawList();

    // --- SHADER ÇİZİMİ ---
    flowTime += ImGui::GetIO().DeltaTime;
    cudaSurfaceObject_t surface = shaderPreviewTexture->map();
    Kivilcim::Shaders::launchLiquidFlowShader(
        surface, shaderPreviewTexture->getWidth(), shaderPreviewTexture->getHeight(),
        flowTime, 1.0f, 2.0f, make_float3(0.85f, 0.45f, 0.0f)
    );
    shaderPreviewTexture->unmap();

    ImVec2 winPos = ImGui::GetWindowPos();
    ImVec2 winSize = ImGui::GetWindowSize();
    drawList->AddImage(
        (ImTextureID)(intptr_t)shaderPreviewTexture->getTextureID(),
        winPos, ImVec2(winPos.x + winSize.x, winPos.y + winSize.y)
    );

    // ... [INPUT KISIMLARI AYNI KALACAK] ...
    ImGui::Dummy(ImVec2(0.0f, 10.0f));
    ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.0f, 1.0f), "PROJE DETAYLARI");
    ImGui::Dummy(ImVec2(0.0f, 5.0f));

    ImGui::Text("Proje Adi:");
    ImGui::PushItemWidth(panelWidth * 0.9f);
    ImGui::InputTextWithHint("##ProjAdi", "İsimsiz-1", projectNameBuf, IM_ARRAYSIZE(projectNameBuf));
    ImGui::PopItemWidth();
    ImGui::Dummy(ImVec2(0.0f, 10.0f));

    ImGui::Checkbox("Orijinal Cozunurlugu Koru", &keepOriginalSize);
    ImGui::Dummy(ImVec2(0.0f, 5.0f));

    if (keepOriginalSize) ImGui::BeginDisabled();

    float colWidth = 130.0f;
    const char* dimMetrics[] = { "Piksel", "Inc", "Santimetre" };

    ImGui::Text("Genislik:"); ImGui::SameLine(colWidth + 25.0f); ImGui::Text("Yukseklik:");
    ImGui::PushItemWidth(colWidth); ImGui::InputInt("##Width", &docWidth, 0, 0); ImGui::PopItemWidth();
    ImGui::SameLine(colWidth + 25.0f);
    ImGui::PushItemWidth(colWidth); ImGui::InputInt("##Height", &docHeight, 0, 0); ImGui::PopItemWidth();
    ImGui::SameLine();
    ImGui::PushItemWidth(panelWidth - (colWidth * 2) - 45.0f);
    ImGui::Combo("##DimMetric", &dimMetric, dimMetrics, IM_ARRAYSIZE(dimMetrics));
    ImGui::PopItemWidth();

    ImGui::Dummy(ImVec2(0.0f, 10.0f));
    ImGui::Text("Oryantasyon:"); ImGui::SameLine(100.0f);

    if (orientation == 0) ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.85f, 0.45f, 0.0f, 1.0f));
    else ImGui::PushStyleColor(ImGuiCol_Button, ImGui::GetStyle().Colors[ImGuiCol_Button]);
    if (ImGui::Button(" | ", ImVec2(35, 25))) { orientation = 0; } ImGui::PopStyleColor();
    ImGui::SameLine();
    if (orientation == 1) ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.85f, 0.45f, 0.0f, 1.0f));
    else ImGui::PushStyleColor(ImGuiCol_Button, ImGui::GetStyle().Colors[ImGuiCol_Button]);
    if (ImGui::Button(" - ", ImVec2(35, 25))) { orientation = 1; } ImGui::PopStyleColor();

    ImGui::Dummy(ImVec2(0.0f, 10.0f));
    const char* resMetrics[] = { "Piksel/Inc", "Piksel/cm" };
    ImGui::Text("Cozunurluk:");
    ImGui::PushItemWidth(colWidth); ImGui::InputInt("##Res", &resolution, 0, 0); ImGui::PopItemWidth();
    ImGui::SameLine(colWidth + 25.0f);
    ImGui::PushItemWidth(colWidth); ImGui::Combo("##ResMetric", &resMetric, resMetrics, IM_ARRAYSIZE(resMetrics)); ImGui::PopItemWidth();

    if (keepOriginalSize) ImGui::EndDisabled();

    ImGui::Dummy(ImVec2(0.0f, 10.0f));
    ImGui::Text("Arka Plan Icerigi:");
    const char* bgContents[] = { "Beyaz", "Siyah", "Seffaf", "Ozel Renk" };
    ImGui::PushItemWidth(colWidth * 1.5f);
    if (ImGui::Combo("##BgContent", &bgContentMode, bgContents, IM_ARRAYSIZE(bgContents))) {
        if (bgContentMode == 0) { bgColor[0] = 1.0f; bgColor[1] = 1.0f; bgColor[2] = 1.0f; }
        if (bgContentMode == 1) { bgColor[0] = 0.0f; bgColor[1] = 0.0f; bgColor[2] = 0.0f; }
    }
    ImGui::PopItemWidth();
    ImGui::SameLine();

    ImGuiColorEditFlags colorFlags = ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_NoLabel | ImGuiColorEditFlags_AlphaPreview;
    if (ImGui::ColorEdit3("##ColorBox", bgColor, colorFlags)) bgContentMode = 3;

    ImGui::Dummy(ImVec2(0.0f, 15.0f)); ImGui::Separator(); ImGui::Dummy(ImVec2(0.0f, 10.0f));

    // --- DOSYA SEÇİM ALANI ---
    ImVec2 dropZoneSize = ImVec2(ImGui::GetContentRegionAvail().x * 0.9f, 120.0f);
    float dropZoneIndent = (ImGui::GetContentRegionAvail().x - dropZoneSize.x) * 0.5f;
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + dropZoneIndent);
    ImVec2 cursorScreenPos = ImGui::GetCursorScreenPos();

    if (!isProcessingImage.load()) {
        ImGui::InvisibleButton("DropZone", dropZoneSize);
        if (ImGui::IsItemClicked()) {
            std::string tempPath = openFileDialog();
            if (!tempPath.empty()) selectedImagePath = tempPath;
        }
    } else {
        ImGui::Dummy(dropZoneSize);
    }

    bool isHovered = ImGui::IsItemHovered() && !isProcessingImage.load();
    ImU32 bgColorZone = isHovered ? IM_COL32(50, 50, 60, 200) : IM_COL32(30, 30, 35, 180);
    ImU32 borderColor = isHovered ? IM_COL32(255, 165, 0, 255) : IM_COL32(100, 100, 110, 150);

    drawList->AddRectFilled(cursorScreenPos, ImVec2(cursorScreenPos.x + dropZoneSize.x, cursorScreenPos.y + dropZoneSize.y), bgColorZone, 8.0f);
    drawList->AddRect(cursorScreenPos, ImVec2(cursorScreenPos.x + dropZoneSize.x, cursorScreenPos.y + dropZoneSize.y), borderColor, 8.0f, 0, isHovered ? 2.0f : 1.0f);

    ImVec2 centerPos = ImVec2(cursorScreenPos.x + dropZoneSize.x * 0.5f, cursorScreenPos.y + dropZoneSize.y * 0.5f);

    if (isProcessingImage.load()) {
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

    ImGui::Dummy(ImVec2(0.0f, 15.0f));

    // --- KLASÖR SEÇİMİ ---
    ImGui::PushStyleColor(ImGuiCol_Header, ImVec4(0.12f, 0.12f, 0.13f, 0.6f));
    ImGui::PushStyleColor(ImGuiCol_HeaderHovered, ImVec4(0.18f, 0.18f, 0.19f, 0.8f));
    ImGui::PushStyleColor(ImGuiCol_HeaderActive, ImVec4(0.85f, 0.45f, 0.0f, 0.6f));

    if (ImGui::CollapsingHeader("Gelismis Secenekler")) {
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
            std::string chosenFolder = openFolderDialog();
            if (!chosenFolder.empty()) projectSavePath = chosenFolder;
        }
        ImGui::PopStyleColor();
        ImGui::Dummy(ImVec2(0.0f, 10.0f));
    }
    ImGui::PopStyleColor(3);

    // --- THREAD SONUCUNU YAKALAMA (ANA DÖNGÜDE GPU'YA YAZILIR) ---
    if (isImageReadyForGPU.load()) {
        unsigned int textureID = 0;
        if (rawResizedData) {
            textureID = TextureUtility::CreateTextureFromPixels(rawResizedData, 256, 144);
            std::free(rawResizedData);
            rawResizedData = nullptr;
        }

        std::string finalName = std::string(projectNameBuf);
        if (finalName.empty()) finalName = "İsimsiz Proje";

        Kdata::ProjectData newData(0, finalName, selectedImagePath);
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

        // Callback tetikleniyor! (MainUI bu sinyali alıp Workspace'e atacak)
        if (onProjectCreated) onProjectCreated(newData);

        selectedImagePath = "";
        memset(projectNameBuf, 0, sizeof(projectNameBuf));

        isImageReadyForGPU = false;
        isProcessingImage = false;
    }

    // --- PROJE OLUŞTUR BUTONU ---
    float buttonHeight = 45.0f;
    float currentY = ImGui::GetCursorPosY();
    float targetY = ImGui::GetWindowHeight() - buttonHeight - 15.0f + ImGui::GetScrollY();
    if (currentY < targetY) ImGui::SetCursorPosY(targetY);

    bool isDisabled = isProcessingImage.load();
    if (isDisabled) ImGui::BeginDisabled();

    ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 1.5f);
    ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0.85f, 0.45f, 0.0f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.85f, 0.45f, 0.0f, 0.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(1.0f, 0.55f, 0.0f, 0.8f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.65f, 0.35f, 0.0f, 1.0f));
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + dropZoneIndent);

    if (ImGui::Button("PROJEYI OLUSTUR", ImVec2(dropZoneSize.x, buttonHeight))) {
        if (!selectedImagePath.empty()) {
            startImageProcessing(selectedImagePath); // Güvenli Thread'i başlat
        } else {
            // Görsel yoksa anında oluştur
            std::string finalName = std::string(projectNameBuf);
            if (finalName.empty()) finalName = "İsimsiz Proje";

            Kdata::ProjectData newData(0, finalName, "");
            newData.size = {docWidth, docHeight};
            newData.projectWidth = docWidth;
            newData.projectHeight = docHeight;
            newData.bgColor[0] = bgColor[0]; newData.bgColor[1] = bgColor[1]; newData.bgColor[2] = bgColor[2];
            newData.keepOriginalSize = keepOriginalSize;
            newData.dimMetric = dimMetric;
            newData.orientation = orientation;
            newData.resolution = resolution;
            newData.kvlcmDir = projectSavePath;

            if (onProjectCreated) onProjectCreated(newData);
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