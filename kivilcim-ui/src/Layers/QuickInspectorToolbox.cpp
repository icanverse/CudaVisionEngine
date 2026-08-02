#include "Layers/QuickInspectorToolbox.h"
#include "imgui.h"

QuickInspectorToolbox::QuickInspectorToolbox() {}
QuickInspectorToolbox::~QuickInspectorToolbox() {}

void QuickInspectorToolbox::render(Kdata::WorkspaceStateData* state, float displayWidth, float displayHeight) {
    if (!state) return;

    const float panelWidth = getPanelWidth();
    const float panelHeight = getPanelHeight();
    
    const float panelTop = 60.0f;
    const float padding = 8.0f;

    // Layer paneli ile tamamen aynı X (Sağ) hizasına oturtuyoruz
    ImGui::SetCursorPos(ImVec2(displayWidth - panelWidth - 15.0f, panelTop));
    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.08f, 0.08f, 0.09f, 0.98f));
    ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0.15f, 0.15f, 0.16f, 1.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 2.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_ChildBorderSize, 1.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(padding, padding));

    ImGui::BeginChild("ContextPanel", ImVec2(panelWidth, panelHeight), true, ImGuiWindowFlags_NoScrollbar);

    ImGui::TextColored(ImVec4(0.85f, 0.85f, 0.85f, 1.0f), "ARAC OZELLIKLERI");
    ImGui::Separator();
    ImGui::Dummy(ImVec2(0.0f, 5.0f));

    // ==========================================
    // DİNAMİK İÇERİK YÖNETİMİ
    // ==========================================
    int currentToolId = static_cast<int>(state->tools.lastFiredAction);

    // ÇÖZÜM: avail değişkenini buraya, if bloğunun dışına alıyoruz!
    ImVec2 avail = ImGui::GetContentRegionAvail();

    if (currentToolId == 101) { // ÖRNEK: Kdata::InstantAction::FREE_ROI
        ImGui::TextColored(ImVec4(1.0f, 0.7f, 0.2f, 1.0f), "Serbest Secim Araci (ROI)");
        ImGui::Dummy(ImVec2(0.0f, 10.0f));

        ImGui::Text("Hassasiyet Büyüteci:");

        // Büyüteç için sahte bir kare
        ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.02f, 0.02f, 0.02f, 1.0f));
        ImGui::BeginChild("MagnifierScreen", ImVec2(avail.x, 140.0f), true);
        ImGui::SetCursorPos(ImVec2(avail.x / 2 - 40, 60.0f)); // Yazıyı ortala
        ImGui::TextDisabled("[ 4x Büyüteç ]");
        ImGui::EndChild();
        ImGui::PopStyleColor();
    }
    else if (currentToolId == 102) { // ÖRNEK: Kdata::InstantAction::ADJUSTMENT
        ImGui::TextColored(ImVec4(1.0f, 0.7f, 0.2f, 1.0f), "Renk ve Isik Ayarlari");
        ImGui::Dummy(ImVec2(0.0f, 15.0f));

        static float temp = 0.0f, contrast = 1.0f, brightness = 0.0f;

        // Artık avail.x sorunsuz bir şekilde okunabilir
        ImGui::PushItemWidth(avail.x - 10.0f);
        ImGui::SliderFloat("##Temp", &temp, -100.0f, 100.0f, "Sicaklik: %.1f");
        ImGui::Dummy(ImVec2(0.0f, 5.0f));
        ImGui::SliderFloat("##Contrast", &contrast, 0.0f, 2.0f, "Kontrast: %.2f");
        ImGui::Dummy(ImVec2(0.0f, 5.0f));
        ImGui::SliderFloat("##Brightness", &brightness, -1.0f, 1.0f, "Parlaklik: %.2f");
        ImGui::PopItemWidth();
    }
    else {
        ImGui::Dummy(ImVec2(0.0f, 20.0f));
        ImGui::TextDisabled("Aktif araca ait bir ayar\nbulunmamaktadir.");
    }

    ImGui::EndChild();
    ImGui::PopStyleVar(3);
    ImGui::PopStyleColor(2);
}