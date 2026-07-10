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

void QuickToolbar::render() {
    ImGui::Begin("Arac Kutusu");

    // Tüm butonları tek bir for döngüsü ile çiz
    for (const auto& tool : availableTools) {

        // Seçili araca göre renk belirle (Aktifse Turuncu, Pasifse Koyu Gri)
        if (currentTool == tool.id) {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.85f, 0.45f, 0.0f, 1.0f));
        } else {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.2f, 0.2f, 1.0f));
        }

        // İsimli buton çizimi (İleride ImGui::ImageButton'a çevrilebilir)
        if (ImGui::Button(tool.name.c_str(), ImVec2(50, 50))) {
            currentTool = tool.id;
        }

        ImGui::PopStyleColor();

        // Fareyle üzerine gelince açıklama göster
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("%s", tool.tooltip.c_str());
        }
    }

    ImGui::End();
}

QuickToolbar::~QuickToolbar() {
    // std::vector<ToolUI_Info> availableTools; zaten otomatik temizlenir.
    // İleride iconTexture (void*) için bellek temizliği gerekirse buraya yazacağız.
}