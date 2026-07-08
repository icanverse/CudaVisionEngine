#include "UI/Workspace.h"
#include "imgui.h"
#include <iostream>

Workspace::Workspace() : activeProject(nullptr) {}

void Workspace::loadProject(Kivilcim::ProjectData* project) {
    activeProject = project;
    std::cout << "[Workspace] Proje yuklendi: " << activeProject->name << std::endl;
}

void Workspace::render(float displayWidth, float displayHeight) {
    if (!activeProject) return; // Proje yoksa hiçbir şey çizme

    // 1. Üst Menü Çubuğu (File, Edit, Layer...)
    renderTopMenu();
    
    // Üst menünün yüksekliğini alıyoruz ki diğer panelleri onun altına hizalayalım
    float menuHeight = ImGui::GetFrameHeight(); 

    // 2. Sol Araç Kutusu (Toolbox - Fırçalar, Silgiler)
    renderToolbox(displayHeight, menuHeight);

    // 3. Sağ Paneller (Katmanlar, Özellikler)
    renderRightPanels(displayWidth, displayHeight, menuHeight);

    // 4. Orta Tuval (Canvas - Asıl görüntünün renderlanacağı yer)
    renderCanvas(displayWidth, displayHeight, menuHeight);
}

void Workspace::renderTopMenu() {
    if (ImGui::BeginMainMenuBar()) {
        if (ImGui::BeginMenu("Dosya")) {
            if (ImGui::MenuItem("Kaydet", "Ctrl+S")) { /* Kaydetme mantığı */ }
            if (ImGui::MenuItem("Farkli Kaydet...")) { }
            ImGui::Separator();
            if (ImGui::MenuItem("Ana Ekrana Don")) {
                if (onClose) onClose(); // MainUI'ye çıkış sinyali gönder
            }
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Duzenle")) {
            if (ImGui::MenuItem("Geri Al", "Ctrl+Z")) {}
            if (ImGui::MenuItem("Ileri Al", "Ctrl+Y")) {}
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Filtreler")) {
            if (ImGui::MenuItem("Likit Cam Efekti")) {}
            if (ImGui::MenuItem("Bulaniklik (Blur)")) {}
            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();
    }
}

void Workspace::renderToolbox(float displayHeight, float menuHeight) {
    ImGui::SetNextWindowPos(ImVec2(0.0f, menuHeight), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(toolboxWidth, displayHeight - menuHeight), ImGuiCond_Always);
    
    ImGuiWindowFlags flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse;
    
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.12f, 0.12f, 0.13f, 1.0f));
    
    ImGui::Begin("Toolbox", nullptr, flags);
    
    // Araç Butonları (Şimdilik Metin/Harf, ileride Icon Font eklenebilir)
    ImVec2 btnSize(toolboxWidth - 16.0f, 40.0f);
    
    if (ImGui::Button("TASI", btnSize)) selectedTool = 0;
    if (ImGui::Button("SECİM", btnSize)) selectedTool = 1;
    if (ImGui::Button("FIRCA", btnSize)) selectedTool = 2;
    if (ImGui::Button("SİLGI", btnSize)) selectedTool = 3;
    
    ImGui::End();
    
    ImGui::PopStyleColor();
    ImGui::PopStyleVar();
}

void Workspace::renderRightPanels(float displayWidth, float displayHeight, float menuHeight) {
    ImGui::SetNextWindowPos(ImVec2(displayWidth - rightPanelWidth, menuHeight), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(rightPanelWidth, displayHeight - menuHeight), ImGuiCond_Always);
    
    ImGuiWindowFlags flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse;
    
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.12f, 0.12f, 0.13f, 1.0f));
    
    ImGui::Begin("RightPanels", nullptr, flags);
    
    // --- ÖZELLİKLER BÖLÜMÜ ---
    ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.0f, 1.0f), "PROJE OZELLIKLERI");
    ImGui::Separator();
    ImGui::Text("Ad: %s", activeProject->name.c_str());
    ImGui::Text("Boyut: %dx%d px", activeProject->projectWidth, activeProject->projectHeight);
    
    ImGui::Dummy(ImVec2(0.0f, 20.0f));
    
    // --- KATMANLAR (LAYERS) BÖLÜMÜ ---
    ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.0f, 1.0f), "KATMANLAR (LAYERS)");
    ImGui::Separator();
    
    // Temsili Katman Listesi (Photoshop tarzı)
    ImGui::Selectable("Katman 2 (Dinamik Efekt)");
    ImGui::Selectable("Katman 1 (Gorsel)", true); // Seçili katman
    ImGui::Selectable("Arka Plan");

    ImGui::End();
    
    ImGui::PopStyleColor();
    ImGui::PopStyleVar();
}

void Workspace::renderCanvas(float displayWidth, float displayHeight, float menuHeight) {
    // Canvas, sol araç kutusu ile sağ panelin arasında kalan devasa boşluktur.
    float canvasX = toolboxWidth;
    float canvasWidth = displayWidth - toolboxWidth - rightPanelWidth;
    float canvasHeight = displayHeight - menuHeight;

    ImGui::SetNextWindowPos(ImVec2(canvasX, menuHeight), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(canvasWidth, canvasHeight), ImGuiCond_Always);
    
    ImGuiWindowFlags flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse;
    
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.05f, 0.05f, 0.06f, 1.0f)); // Koyu arka plan
    
    ImGui::Begin("CanvasArea", nullptr, flags);
    
    // Ekranda projenin görselini ortalayarak çizdiriyoruz
    if (activeProject->textureID > 0) {
        ImVec2 availSize = ImGui::GetContentRegionAvail();
        
        // Görselin orijinal en/boy oranını koruyarak ekrana sığdırmak için ufak bir matematik
        float scale = std::min(availSize.x / activeProject->size.x, availSize.y / activeProject->size.y);
        float drawW = activeProject->size.x * scale * 0.9f; // %10 boşluk payı (Padding) bırak
        float drawH = activeProject->size.y * scale * 0.9f;
        
        float offsetX = (availSize.x - drawW) * 0.5f;
        float offsetY = (availSize.y - drawH) * 0.5f;
        
        ImGui::SetCursorPos(ImVec2(ImGui::GetCursorPosX() + offsetX, ImGui::GetCursorPosY() + offsetY));
        
        // Asıl Şov: Projenin VRAM'deki piksellerini devasa bir şekilde ekrana bas!
        ImGui::Image((ImTextureID)(intptr_t)activeProject->textureID, ImVec2(drawW, drawH), ImVec2(0, 1), ImVec2(1, 0));
    }

    ImGui::End();
    
    ImGui::PopStyleColor();
    ImGui::PopStyleVar();
}