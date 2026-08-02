#include "Layers/QuickLayersToolbox.h"
#include "imgui.h"

#include <algorithm>
#include <cstdint>

#include "Layers/ToolboxIconButton.h"

QuickLayersToolbox::QuickLayersToolbox() {
    availableTools = {
        {Kdata::InstantAction::LAYER_DOWN,  "AltaTasi##Layers",   Icon::Layers_toDown, "Katmani bir alta tasir"},
        // OPEN_LAYERS (Sadece arayüz odağı için, InstantAction'a map edilmedi)
        {Kdata::InstantAction::LAYER_ADD,   "KatmanEkle##Layers", Icon::Layers_Add,    "Yeni katman ekler"},
        {Kdata::InstantAction::LAYER_LOCK,  "Kilitle##Layers",    Icon::Lock,          "Secili katmani kilitler"},
        // UNION_LAYERS (Gelecekte eklenecek)
        {Kdata::InstantAction::LAYER_VIS,   "Gorunurluk##Layers", Icon::Visible,       "Secili katmanin gorunurlugunu degistirir"}
    };
}

QuickLayersToolbox::~QuickLayersToolbox() = default;

void QuickLayersToolbox::render(Kdata::WorkspaceStateData* state, float displayWidth, float displayHeight) {
    if (!state) return;

    const float panelWidth = getPanelWidth();
    const float panelTop = getPanelTop() + 270.0f;
    const float requestedHeight = displayHeight - panelTop - 15.0f;
    const float panelHeight = requestedHeight > 220.0f ? requestedHeight : 220.0f;
    const float iconSide = std::clamp(displayHeight / 34.0f, 24.0f, 31.0f);
    const float padding = 8.0f;
    const float spacing = 6.0f;

    ImGui::SetCursorPos(ImVec2(displayWidth - panelWidth - 15.0f, panelTop));

    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.08f, 0.08f, 0.09f, 0.98f));
    ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0.15f, 0.15f, 0.16f, 1.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 2.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_ChildBorderSize, 1.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(padding, padding));
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 2.0f);

    ImGui::BeginChild(
        "LayersPanel",
        ImVec2(panelWidth, panelHeight),
        true,
        ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse
    );

    renderToolbar(state, iconSide, spacing);
    ImGui::Dummy(ImVec2(0.0f, 2.0f));
    ImGui::PushStyleColor(ImGuiCol_Separator, ImVec4(0.18f, 0.18f, 0.20f, 1.0f));
    ImGui::Separator();
    ImGui::PopStyleColor();
    ImGui::Dummy(ImVec2(0.0f, 2.0f));

    ImGui::TextColored(ImVec4(0.85f, 0.85f, 0.85f, 1.0f), "KATMANLAR");
    ImGui::SameLine();
    ImGui::TextDisabled("%d", static_cast<int>(state->layers.layers.size()));
    ImGui::Dummy(ImVec2(0.0f, 2.0f));

    renderLayerList(state);

    ImGui::EndChild();
    ImGui::PopStyleVar(4);
    ImGui::PopStyleColor(2);
}

void QuickLayersToolbox::renderToolbar(Kdata::WorkspaceStateData* state, float iconSide, float spacing) {
    const float buttonWidth = iconSide + ImGui::GetStyle().FramePadding.x * 2.0f;
    const float totalWidth =
        buttonWidth * static_cast<float>(availableTools.size()) +
        spacing * static_cast<float>(availableTools.size() - 1);
    const float offsetX = (ImGui::GetContentRegionAvail().x - totalWidth) * 0.5f;
    if (offsetX > 0.0f) ImGui::SetCursorPosX(ImGui::GetCursorPosX() + offsetX);

    for (std::size_t i = 0; i < availableTools.size(); ++i) {
        const Tool& tool = availableTools[i];
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.0f, 0.0f, 0.0f, 0.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.25f, 0.25f, 0.27f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.35f, 0.35f, 0.38f, 1.0f));

        if (ToolboxUI::IconButton(tool.name.c_str(), tool.icon, ImVec2(iconSide, iconSide))) {
            // YENİ: Araç çubuğuna tıklandığında motorun merkezine komut gönder
            state->tools.lastFiredAction = tool.actionId;

            // Anlık UI güncellemeleri (Opsiyonel, asıl işi motor da yapabilir)
            if (state->layers.activeLayerIndex >= 0 && state->layers.activeLayerIndex < state->layers.layers.size()) {
                auto& activeLayer = state->layers.layers[state->layers.activeLayerIndex];

                if (tool.actionId == Kdata::InstantAction::LAYER_LOCK) {
                    activeLayer.isLocked = !activeLayer.isLocked;
                } else if (tool.actionId == Kdata::InstantAction::LAYER_VIS) {
                    activeLayer.isVisible = !activeLayer.isVisible;
                }
            }
        }

        if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", tool.tooltip.c_str());
        ImGui::PopStyleColor(3);

        if (i + 1 < availableTools.size()) ImGui::SameLine(0.0f, spacing);
    }
}

void QuickLayersToolbox::renderLayerList(Kdata::WorkspaceStateData* state) {
    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.05f, 0.05f, 0.06f, 0.95f));
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 2.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(4.0f, 4.0f));

    ImGui::BeginChild(
        "LayersScrollableList",
        ImVec2(0.0f, 0.0f),
        true,
        ImGuiWindowFlags_AlwaysVerticalScrollbar
    );

    if (state->layers.layers.empty()) {
        ImGui::Dummy(ImVec2(0.0f, 12.0f));
        ImGui::TextDisabled("Henuz katman yok");
    } else {
        // En üstteki katman en üstte görünsün diye ters döngü (opsiyonel ama önerilir)
        for (auto it = state->layers.layers.rbegin(); it != state->layers.layers.rend(); ++it) {
            renderLayerRow(state, *it);
        }
    }

    ImGui::EndChild();
    ImGui::PopStyleVar(2);
    ImGui::PopStyleColor();
}

void QuickLayersToolbox::renderLayerRow(Kdata::WorkspaceStateData* state, Kdata::Layer& layer) {
    ImGui::PushID(layer.id);

    const bool selected = (state->layers.activeLayerIndex != -1) &&
                          (state->layers.layers[state->layers.activeLayerIndex].id == layer.id);

    ImGui::PushStyleColor(
        ImGuiCol_ChildBg,
        selected ? ImVec4(0.35f, 0.10f, 0.05f, 1.0f) : ImVec4(0.90f, 0.10f, 0.11f, 0.90f)
    );
    ImGui::PushStyleColor(
        ImGuiCol_Border,
        selected ? ImVec4(0.90f, 0.50f, 0.10f, 0.90f) : ImVec4(0.0f, 0.0f, 0.0f, 0.0f)
    );

    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 2.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_ChildBorderSize, selected ? 1.0f : 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));

    const float rowHeight = 36.0f;
    ImGui::BeginChild("LayerRow", ImVec2(0.0f, rowHeight), true,
                      ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);

    const float rowWidth = ImGui::GetWindowSize().x;

    float checkboxSize = ImGui::GetFrameHeight();
    ImGui::SetCursorPos(ImVec2(8.0f, (rowHeight - checkboxSize) * 0.5f));
    if (ImGui::Checkbox("##Visible", &layer.isVisible)) {
        // Görünürlük değişti
    }

    const float maxThumbSize = 28.0f;
    ImVec2 actualThumbSize(maxThumbSize, maxThumbSize);

    float thumbX = 34.0f + (maxThumbSize - actualThumbSize.x) * 0.5f;
    float thumbY = (rowHeight - actualThumbSize.y) * 0.5f;
    ImGui::SetCursorPos(ImVec2(thumbX, thumbY));

    // Thumbnail dokusu artık doğrudan Kdata::Layer objesinden geliyor
    if (layer.thumbnailTextureID != 0) {
        ImGui::Image(
            (ImTextureID)(intptr_t)layer.thumbnailTextureID,
            actualThumbSize,
            ImVec2(0.0f, 1.0f),
            ImVec2(1.0f, 0.0f)
        );
    } else {
        ImGui::Dummy(actualThumbSize);
        const ImVec2 rowScreenPos = ImGui::GetWindowPos();
        const ImVec2 thumbMin(rowScreenPos.x + thumbX, rowScreenPos.y + thumbY);
        const ImVec2 thumbMax(thumbMin.x + actualThumbSize.x, thumbMin.y + actualThumbSize.y);

        ImDrawList* drawList = ImGui::GetWindowDrawList();
        drawList->AddRectFilled(thumbMin, thumbMax, IM_COL32(40, 40, 45, 255), 0.0f);
        drawList->AddRect(thumbMin, thumbMax, IM_COL32(70, 70, 75, 220), 0.0f);
    }

    float textHeight = ImGui::GetTextLineHeight();
    ImGui::SetCursorPos(ImVec2(72.0f, (rowHeight - textHeight) * 0.5f));

    const float nameWidth = rowWidth - 110.0f;

    ImGui::SetWindowFontScale(1.05f);
    ImGui::SetNextItemAllowOverlap();
    if (ImGui::Selectable(
            (layer.name + "##LayerName").c_str(),
            selected,
            ImGuiSelectableFlags_None,
            ImVec2(nameWidth > 40.0f ? nameWidth : 40.0f, textHeight)
        )) {
        // Katman seçildiğinde State üzerindeki aktif katman indeksini bul ve güncelle
        for (int i = 0; i < state->layers.layers.size(); ++i) {
            if (state->layers.layers[i].id == layer.id) {
                state->layers.activeLayerIndex = i;
                break;
            }
        }
    }
    ImGui::SetWindowFontScale(1.0f);

    float lockBtnSize = 20.0f;
    ImGui::SetCursorPos(ImVec2(rowWidth - 30.0f, (rowHeight - lockBtnSize) * 0.5f));
    ImGui::PushStyleColor(
        ImGuiCol_Button,
        layer.isLocked ? ImVec4(0.6f, 0.6f, 0.6f, 0.45f) : ImVec4(0.0f, 0.0f, 0.0f, 0.0f)
    );
    if (ToolboxUI::IconButton("Lock##LayerRow", Icon::Lock, ImVec2(lockBtnSize, lockBtnSize))) {
        layer.isLocked = !layer.isLocked;
    }
    ImGui::PopStyleColor();

    ImGui::EndChild();
    ImGui::PopStyleVar(3);
    ImGui::PopStyleColor(2);
    ImGui::Dummy(ImVec2(0.0f, 3.0f));
    ImGui::PopID();
}