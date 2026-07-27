#include "Layers/QuickTopRightToolbox.h"
#include "Layers/ToolboxIconButton.h"
#include "imgui.h"
#include <algorithm> // std::clamp için gerekli

QuickTopRightToolbox::QuickTopRightToolbox()
    : lastAction(TopAction::NONE) {
    // push_back yerine modern başlatma listesi (Initializer List)
    availableTools = {
        {TopAction::UNDO, "Geri Al", "Son islemi geri alir", Icon::Undo, "Islem gecmisinde bir adim geriye gider"},
        {TopAction::REDO, "Yinele", "Geri alinan islemi yineler", Icon::Redo, "Islem gecmisinde bir adim ileri gider"},
        {TopAction::TURN_LEFT, "Sola Dondur", "90 Derece Sola Dondur", Icon::Turn_Left, "Tuvali saat yonunun tersine cevirir"},
        {TopAction::TURN_RIGHT, "Saga Dondur", "90 Derece Saga Dondur", Icon::Turn_Right, "Tuvali saat yonunde cevirir"},
        {TopAction::ZOOM_IN, "Yakinlastir", "Tuvali Yakinlastir (+)", Icon::Zoom_In, "Calisma alanina yaklasir"},
        {TopAction::ZOOM_OUT, "Uzaklastir", "Tuvali Uzaklastir (-)", Icon::Zoom_Out, "Calisma alanindan uzaklasir"},
        {TopAction::MIRROR_HORIZONTAL, "Yatay Aynala", "Yatay Eksende Aynala", Icon::Mirror_Horizontal, "Goruntuyu yatay eksende cevirir"},
        {TopAction::MIRROR_VERTICAL, "Dikey Aynala", "Dikey Eksende Aynala", Icon::Mirror_Vertical, "Goruntuyu dikey eksende cevirir"}
    };
}

void QuickTopRightToolbox::render(float displayWidth, float displayHeight) {
    // TopCenter ile aynı dinamik ölçeklendirme mantığı
    const float iconSide = std::clamp(displayHeight / 32.0f, 22.0f, 34.0f);
    const float padding = 8.0f;
    const float spacing = 4.0f;
    const float buttonWidth = iconSide + ImGui::GetStyle().FramePadding.x * 2.0f;
    const float toolbarWidth = padding * 2.0f +
        static_cast<float>(availableTools.size()) * buttonWidth +
        static_cast<float>(availableTools.size() - 1) * spacing;
    const float toolbarHeight = padding * 2.0f + iconSide + ImGui::GetStyle().FramePadding.y * 2.0f;

    // Konumlandırma: Sağ üstte, Y ekseni 15.0f olarak ayarlandı
    ImGui::SetCursorPos(ImVec2(displayWidth - toolbarWidth - 15.0f, 15.0f));

    // TopCenter ile uyumlu arka plan ve çerçeve renkleri
    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.08f, 0.08f, 0.09f, 0.78f));
    ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0.85f, 0.45f, 0.0f, 0.4f));
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 15.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_ChildBorderSize, 1.5f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(padding, padding));
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 12.0f);

    ImGui::BeginChild("Sag Ust Arac Kutusu", ImVec2(toolbarWidth, toolbarHeight), true,
                      ToolboxUI::FloatingPanelFlags());

    for (std::size_t i = 0; i < availableTools.size(); ++i) {
        const auto& tool = availableTools[i];

        // ==========================================
        // İŞTE İSTEDİĞİN TURUNCU HOVER EFEKTİ
        // ==========================================
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.0f, 0.0f, 0.0f, 0.0f)); // Normal durumda şeffaf
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(1.0f, 0.55f, 0.0f, 0.7f)); // Turuncu Hover
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.65f, 0.35f, 0.0f, 1.0f)); // Tıklanma anı (Koyu Turuncu)

        if (ToolboxUI::IconButton(tool.name.c_str(), tool.icon, ImVec2(iconSide, iconSide))) {
            lastAction = tool.id;
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", tool.tooltip.c_str());

        ImGui::PopStyleColor(3);
        if (i + 1 < availableTools.size()) ImGui::SameLine(0.0f, spacing);
    }

    ImGui::EndChild();
    ImGui::PopStyleVar(4);
    ImGui::PopStyleColor(2);
}

QuickTopRightToolbox::~QuickTopRightToolbox() = default;