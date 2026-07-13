#include "UI/Layers/QuickToolbar.h"
#include "imgui.h"


QuickToolbar::QuickToolbar() {
    currentTool = ActiveTool::NONE;

    // YENİ: Tanımsız icon değişkenleri yerine "nullptr" kullanıyoruz.
    // İleride buraya kendi ikonlarının TextureID'lerini gireceksin.
    // availableTools.push_back({ id, name, tooltip, iconTexture, defination });

    // availableTools.push_back({ id, name, tooltip, iconTexture, defination });

    availableTools.push_back({
        ActiveTool::REGION_SELECT,
        "Secim",
        "Alani secmek icin surukle", // tooltip buraya geldi
        nullptr,                     // iconTexture
        "Istege bagli tanim 1"       // defination (yeni eklediğin değişken)
    });

    availableTools.push_back({
        ActiveTool::REGION_CROP,
        "Kirp",
        "Secili alani kes",
        nullptr,
        "Istege bagli tanim 2"
    });

    availableTools.push_back({
        ActiveTool::PAN,
        "Kaydir",
        "Tuvali hareket ettir",
        nullptr,
        "Istege bagli tanim 3"
    });

}

void QuickToolbar::render(float displayWidth, float displayHeight) {
    // 1. KONUM VE BOYUT AYARLARI
    float toolbarWidth = 80.0f;
    float toolbarHeight = 20.0f + (55.0f * availableTools.size());

    float xPos = 15.0f; // Sol kenara daya
    float yPos = (displayHeight - toolbarHeight) * 0.5f; // Ekranın dikeyde tam ortasına hizala

    ImGui::SetNextWindowPos(ImVec2(xPos, yPos), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(toolbarWidth, toolbarHeight), ImGuiCond_Always);

    // ==========================================
    // YUVARLAK CAM VE KAPSÜL BUTON TASARIMI
    // ==========================================

    // Yarı saydam koyu füme arka plan ve turuncu sınır çizgisi
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.08f, 0.08f, 0.09f, 0.75f));
    ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0.85f, 0.45f, 0.0f, 0.4f));

    // Panel dış hatları
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 15.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 1.5f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(10.0f, 10.0f));

    // YENİ: İç elemanların (butonların) köşelerini yuvarlat
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 12.0f);

    // 2. BAYRAKLAR
    ImGuiWindowFlags toolbox_flags = ImGuiWindowFlags_NoResize |
                                     ImGuiWindowFlags_NoMove |
                                     ImGuiWindowFlags_NoScrollbar |
                                     ImGuiWindowFlags_NoCollapse |
                                     ImGuiWindowFlags_NoTitleBar;

    // 3. PENCEREYİ BAŞLAT
    ImGui::Begin("Arac Kutusu", nullptr, toolbox_flags);

    // Tüm butonları tek bir for döngüsü ile çiz
    for (const auto& tool : availableTools) {

        // Seçili araca göre renk belirle
        if (currentTool == tool.id) {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.85f, 0.45f, 0.0f, 1.0f)); // Aktif turuncu
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(1.0f, 0.55f, 0.0f, 1.0f));
        } else {
            // Pasif butonları tamamen şeffaf yap
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.0f, 0.0f, 0.0f, 0.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.2f, 0.2f, 0.2f, 0.5f));
        }

        // Buton çizimi
        if (ImGui::Button(tool.name.c_str(), ImVec2(60.0f, 50.0f))) {
            currentTool = tool.id;
        }

        ImGui::PopStyleColor(2);

        // Butonlar arası boşluk
        ImGui::Dummy(ImVec2(0.0f, 5.0f));

        // Fareyle üzerine gelince açıklama göster
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("%s", tool.tooltip.c_str());
        }
    }

    // Push ettiğimiz 2 Renk ve ARTIK 4 Değişkeni temizliyoruz
    ImGui::PopStyleColor(2);
    ImGui::PopStyleVar(4);

    ImGui::End();
}

QuickToolbar::~QuickToolbar() {
    // std::vector<ToolUI_Info> availableTools; zaten otomatik temizlenir.
    // İleride iconTexture (void*) için bellek temizliği gerekirse buraya yazacağız.
}