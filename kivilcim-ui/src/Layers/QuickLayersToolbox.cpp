#include "Layers/QuickLayersToolbox.h"

#include "Layers/ToolboxIconButton.h"
#include "imgui.h"

#include <algorithm>
#include <cstdint>

QuickLayersToolbox::QuickLayersToolbox()
    : lastAction(LayerToolAction::NONE) {
    availableTools = {
        {LayerToolAction::MOVE_DOWN,      "AltaTasi##Layers",   Icon::Layers_toDown, "Katmani bir alta tasir"},
        {LayerToolAction::OPEN_LAYERS,    "Katmanlar##Layers",  Icon::Layers,        "Katman listesini odaklar"},
        {LayerToolAction::ADD_LAYER,      "KatmanEkle##Layers", Icon::Layers_Add,    "Yeni katman ekler"},
        {LayerToolAction::LOCK,           "Kilitle##Layers",    Icon::Lock,          "Secili katmani kilitler"},
        {LayerToolAction::UNION_LAYERS,   "Birlestir##Layers",  Icon::Union,         "Secili katmanlari birlestirir"},
        {LayerToolAction::TOGGLE_VISIBLE, "Gorunurluk##Layers", Icon::Visible,       "Secili katmanin gorunurlugunu degistirir"}
    };
}

void QuickLayersToolbox::render(float displayWidth, float displayHeight) {
    const float panelWidth = getPanelWidth();
    const float panelTop = getPanelTop();
    const float requestedHeight = displayHeight - panelTop - 15.0f;
    const float panelHeight = requestedHeight > 220.0f
        ? requestedHeight
        : 220.0f;
    const float iconSide = std::clamp(displayHeight / 34.0f, 24.0f, 31.0f);
    const float padding = 10.0f;
    const float spacing = 5.0f;

    ImGui::SetCursorPos(ImVec2(displayWidth - panelWidth - 15.0f, panelTop));
    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.055f, 0.055f, 0.065f, 0.96f));
    ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0.85f, 0.45f, 0.0f, 0.62f));
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 12.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_ChildBorderSize, 1.5f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(padding, padding));
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 7.0f);

    ImGui::BeginChild(
        "LayersPanel",
        ImVec2(panelWidth, panelHeight),
        true,
        ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse
    );

    renderToolbar(iconSide, spacing);
    ImGui::Dummy(ImVec2(0.0f, 4.0f));
    ImGui::Separator();
    ImGui::Dummy(ImVec2(0.0f, 3.0f));

    ImGui::TextColored(ImVec4(0.96f, 0.63f, 0.25f, 1.0f), "KATMANLAR");
    ImGui::SameLine();
    ImGui::TextDisabled("%d", static_cast<int>(layers.size()));
    ImGui::Dummy(ImVec2(0.0f, 3.0f));

    renderLayerList();

    ImGui::EndChild();
    ImGui::PopStyleVar(4);
    ImGui::PopStyleColor(2);
}

void QuickLayersToolbox::renderToolbar(float iconSide, float spacing) {
    const float buttonWidth = iconSide + ImGui::GetStyle().FramePadding.x * 2.0f;
    const float totalWidth =
        buttonWidth * static_cast<float>(availableTools.size()) +
        spacing * static_cast<float>(availableTools.size() - 1);
    const float offsetX = (ImGui::GetContentRegionAvail().x - totalWidth) * 0.5f;
    if (offsetX > 0.0f) ImGui::SetCursorPosX(ImGui::GetCursorPosX() + offsetX);

    for (std::size_t i = 0; i < availableTools.size(); ++i) {
        const Tool& tool = availableTools[i];
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.0f, 0.0f, 0.0f, 0.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.85f, 0.45f, 0.0f, 0.40f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.85f, 0.45f, 0.0f, 0.72f));

        if (ToolboxUI::IconButton(tool.name.c_str(), tool.icon, ImVec2(iconSide, iconSide))) {
            lastAction = tool.id;

            auto selected = std::find_if(
                layers.begin(),
                layers.end(),
                [this](const LayerPanelItem& layer) {
                    return layer.id == selectedLayerId;
                }
            );

            if (selected != layers.end()) {
                lastChangedLayerId = selected->id;

                if (tool.id == LayerToolAction::LOCK) {
                    selected->locked = !selected->locked;
                } else if (tool.id == LayerToolAction::TOGGLE_VISIBLE) {
                    selected->visible = !selected->visible;
                } else if (tool.id == LayerToolAction::MOVE_DOWN) {
                    const auto next = selected + 1;
                    if (next != layers.end()) std::iter_swap(selected, next);
                }
            }
        }

        if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", tool.tooltip.c_str());
        ImGui::PopStyleColor(3);

        if (i + 1 < availableTools.size()) ImGui::SameLine(0.0f, spacing);
    }
}

void QuickLayersToolbox::renderLayerList() {
    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.025f, 0.025f, 0.03f, 0.82f));
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 7.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(5.0f, 5.0f));

    ImGui::BeginChild(
        "LayersScrollableList",
        ImVec2(0.0f, 0.0f),
        true,
        ImGuiWindowFlags_AlwaysVerticalScrollbar
    );

    if (layers.empty()) {
        ImGui::Dummy(ImVec2(0.0f, 12.0f));
        ImGui::TextDisabled("Henuz katman yok");
    } else {
        for (LayerPanelItem& layer : layers) renderLayerRow(layer);
    }

    ImGui::EndChild();
    ImGui::PopStyleVar(2);
    ImGui::PopStyleColor();
}

void QuickLayersToolbox::renderLayerRow(LayerPanelItem& layer) {
    ImGui::PushID(layer.id);

    const bool selected = layer.id == selectedLayerId;
    ImGui::PushStyleColor(
        ImGuiCol_ChildBg,
        selected
            ? ImVec4(0.24f, 0.12f, 0.025f, 0.92f)
            : ImVec4(0.09f, 0.09f, 0.105f, 0.90f)
    );
    ImGui::PushStyleColor(
        ImGuiCol_Border,
        selected
            ? ImVec4(0.95f, 0.52f, 0.08f, 0.90f)
            : ImVec4(0.20f, 0.20f, 0.22f, 0.75f)
    );
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 6.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_ChildBorderSize, selected ? 1.25f : 1.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));

    ImGui::BeginChild("LayerRow", ImVec2(0.0f, 56.0f), true,
                      ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);

    const float rowWidth = ImGui::GetWindowSize().x;

    ImGui::SetCursorPos(ImVec2(8.0f, 18.0f));
    if (ImGui::Checkbox("##Visible", &layer.visible)) {
        lastChangedLayerId = layer.id;
        lastAction = LayerToolAction::TOGGLE_VISIBLE;
    }

    const ImVec2 rowScreenPos = ImGui::GetWindowPos();
    const ImVec2 thumbMin(rowScreenPos.x + 34.0f, rowScreenPos.y + 8.0f);
    const ImVec2 thumbSize(40.0f, 40.0f);
    const ImVec2 thumbMax(thumbMin.x + thumbSize.x, thumbMin.y + thumbSize.y);
    ImGui::SetCursorPos(ImVec2(34.0f, 8.0f));
    if (layer.thumbnailTextureId != 0) {
        ImGui::Image(
            (ImTextureID)(intptr_t)layer.thumbnailTextureId,
            thumbSize,
            ImVec2(0.0f, 1.0f),
            ImVec2(1.0f, 0.0f)
        );
    } else {
        ImGui::Dummy(thumbSize);
        ImDrawList* drawList = ImGui::GetWindowDrawList();
        drawList->AddRectFilled(
            thumbMin,
            thumbMax,
            IM_COL32(30, 30, 35, 255),
            4.0f
        );
        drawList->AddRect(
            thumbMin,
            thumbMax,
            IM_COL32(105, 65, 25, 220),
            4.0f
        );
    }

    ImGui::SetCursorPos(ImVec2(82.0f, 9.0f));
    const float nameWidth = rowWidth - 124.0f;
    if (ImGui::Selectable(
            (layer.name + "##LayerName").c_str(),
            selected,
            ImGuiSelectableFlags_None,
            ImVec2(nameWidth > 40.0f ? nameWidth : 40.0f, 38.0f)
        )) {
        selectedLayerId = layer.id;
        lastChangedLayerId = layer.id;
        lastAction = LayerToolAction::SELECT_LAYER;
    }

    ImGui::SetCursorPos(ImVec2(rowWidth - 34.0f, 16.0f));
    ImGui::PushStyleColor(
        ImGuiCol_Button,
        layer.locked
            ? ImVec4(0.85f, 0.45f, 0.0f, 0.65f)
            : ImVec4(0.0f, 0.0f, 0.0f, 0.0f)
    );
    if (ToolboxUI::IconButton("Lock##LayerRow", Icon::Lock, ImVec2(22.0f, 22.0f))) {
        layer.locked = !layer.locked;
        lastChangedLayerId = layer.id;
        lastAction = LayerToolAction::LOCK;
    }
    ImGui::PopStyleColor();

    ImGui::EndChild();
    ImGui::PopStyleVar(3);
    ImGui::PopStyleColor(2);
    ImGui::Dummy(ImVec2(0.0f, 4.0f));
    ImGui::PopID();
}

void QuickLayersToolbox::setLayers(const std::vector<LayerPanelItem>& newLayers) {
    layers = newLayers;

    if (layers.empty()) {
        selectedLayerId = -1;
        return;
    }

    const auto selected = std::find_if(
        layers.begin(),
        layers.end(),
        [this](const LayerPanelItem& layer) {
            return layer.id == selectedLayerId;
        }
    );
    if (selected == layers.end()) selectedLayerId = layers.front().id;
}

LayerToolAction QuickLayersToolbox::consumeLastAction() {
    const LayerToolAction action = lastAction;
    lastAction = LayerToolAction::NONE;
    return action;
}

QuickLayersToolbox::~QuickLayersToolbox() = default;