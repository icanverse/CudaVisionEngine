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
    const float panelHeight = requestedHeight > 220.0f ? requestedHeight : 220.0f;
    const float iconSide = std::clamp(displayHeight / 34.0f, 24.0f, 31.0f);
    const float padding = 8.0f;
    const float spacing = 6.0f;

    ImGui::SetCursorPos(ImVec2(displayWidth - panelWidth - 15.0f, panelTop));

    // Modern koyu tema, sert hatlar ve sıfırlanmış/azaltılmış kenarlıklar
    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.08f, 0.08f, 0.09f, 0.98f));
    ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0.15f, 0.15f, 0.16f, 1.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 2.0f); // Sert hatlar
    ImGui::PushStyleVar(ImGuiStyleVar_ChildBorderSize, 1.0f); // İnce modern sınır
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(padding, padding));
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 2.0f);

    ImGui::BeginChild(
        "LayersPanel",
        ImVec2(panelWidth, panelHeight),
        true,
        ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse
    );

    renderToolbar(iconSide, spacing);
    ImGui::Dummy(ImVec2(0.0f, 2.0f));
    ImGui::PushStyleColor(ImGuiCol_Separator, ImVec4(0.18f, 0.18f, 0.20f, 1.0f));
    ImGui::Separator();
    ImGui::PopStyleColor();
    ImGui::Dummy(ImVec2(0.0f, 2.0f));

    // Daha şık bir başlık rengi (Çok patlayan turuncu yerine daha nötr/pastel bir ton)
    ImGui::TextColored(ImVec4(0.85f, 0.85f, 0.85f, 1.0f), "KATMANLAR");
    ImGui::SameLine();
    ImGui::TextDisabled("%d", static_cast<int>(layers.size()));
    ImGui::Dummy(ImVec2(0.0f, 2.0f));

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
        // Hover ve Active durumlarında modern vurgu rengi
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.25f, 0.25f, 0.27f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.35f, 0.35f, 0.38f, 1.0f));

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
    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.05f, 0.05f, 0.06f, 0.95f));
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 2.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(4.0f, 4.0f));

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

    // Modern seçim renkleri - Göz yormayan gri/mavi tonları
    ImGui::PushStyleColor(
        ImGuiCol_ChildBg,
        selected
            ? ImVec4(0.35f, 0.10f, 0.05f, 1.0f) // Seçili durum (daha modern bir ton)
            : ImVec4(0.90f, 0.10f, 0.11f, 0.90f)
    );
    ImGui::PushStyleColor(
        ImGuiCol_Border,
        selected
            ? ImVec4(0.90f, 0.50f, 0.10f, 0.90f)
            : ImVec4(0.0f, 0.0f, 0.0f, 0.0f) // Çerçeve kenarlığını seçili değilse tamamen kaldır
    );

    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 2.0f); // Keskin hatlar
    ImGui::PushStyleVar(ImGuiStyleVar_ChildBorderSize, selected ? 1.0f : 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));

    const float rowHeight = 36.0f; // Katman boyu çok uzundu, daraltıldı
    ImGui::BeginChild("LayerRow", ImVec2(0.0f, rowHeight), true,
                      ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);

    const float rowWidth = ImGui::GetWindowSize().x;

    // Görünürlük Checkbox (Dikeyde Ortalandı)
    float checkboxSize = ImGui::GetFrameHeight();
    ImGui::SetCursorPos(ImVec2(8.0f, (rowHeight - checkboxSize) * 0.5f));
    if (ImGui::Checkbox("##Visible", &layer.visible)) {
        lastChangedLayerId = layer.id;
        lastAction = LayerToolAction::TOGGLE_VISIBLE;
    }

    // --- Thumbnail (Orijinal Ölçeğe Sadık Kalma - Aspect Ratio) ---
    const float maxThumbSize = 28.0f; // Row height'a uygun max boyut
    ImVec2 actualThumbSize(maxThumbSize, maxThumbSize);

    // NOT: layer.thumbWidth ve layer.thumbHeight değerlerinin LayerPanelItem içinde
    // tanımlı olduğunu varsayıyoruz. Tanımlı değilse orantıyı 1:1 kabul eder.
    //float tWidth = layer.thumbWidth > 0 ? layer.thumbWidth : 1.0f;
    //float tHeight = layer.thumbHeight > 0 ? layer.thumbHeight : 1.0f;

    //float aspect = tWidth / tHeight;
    float aspect = 1.142;
    if (aspect > 1.0f) {
        actualThumbSize.y = maxThumbSize / aspect;
    } else {
        actualThumbSize.x = maxThumbSize * aspect;
    }

    // Thumbnail'i kutu içinde dikeyde ve yatayda ortala
    float thumbX = 34.0f + (maxThumbSize - actualThumbSize.x) * 0.5f;
    float thumbY = (rowHeight - actualThumbSize.y) * 0.5f;
    ImGui::SetCursorPos(ImVec2(thumbX, thumbY));

    if (layer.thumbnailTextureId != 0) {
        ImGui::Image(
            (ImTextureID)(intptr_t)layer.thumbnailTextureId,
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
        drawList->AddRectFilled(thumbMin, thumbMax, IM_COL32(40, 40, 45, 255), 0.0f); // Keskin
        drawList->AddRect(thumbMin, thumbMax, IM_COL32(70, 70, 75, 220), 0.0f);
    }

    // --- İsim Barı (Dikeyde Ortalandı ve Genişletildi) ---
    // Y pozisyonunu metin yüksekliğine göre hesaplıyoruz
    float textHeight = ImGui::GetTextLineHeight();
    ImGui::SetCursorPos(ImVec2(72.0f, (rowHeight - textHeight) * 0.5f));

    const float nameWidth = rowWidth - 110.0f;

    // Fontu biraz daha okunaklı yapmak için geçici ölçekleme
    ImGui::SetWindowFontScale(1.05f);
    ImGui::SetNextItemAllowOverlap();
    if (ImGui::Selectable(
            (layer.name + "##LayerName").c_str(),
            selected,
            ImGuiSelectableFlags_None, // Flag'i None yaptık
            ImVec2(nameWidth > 40.0f ? nameWidth : 40.0f, textHeight)
        )) {
        selectedLayerId = layer.id;
        lastChangedLayerId = layer.id;
        lastAction = LayerToolAction::SELECT_LAYER;
        }
    ImGui::SetWindowFontScale(1.0f);

    // Kilit Butonu (Ortalandı)
    float lockBtnSize = 20.0f;
    ImGui::SetCursorPos(ImVec2(rowWidth - 30.0f, (rowHeight - lockBtnSize) * 0.5f));
    ImGui::PushStyleColor(
        ImGuiCol_Button,
        layer.locked
            ? ImVec4(0.6f, 0.6f, 0.6f, 0.45f) // Çok parlak turuncu yerine modern kilit rengi
            : ImVec4(0.0f, 0.0f, 0.0f, 0.0f)
    );
    if (ToolboxUI::IconButton("Lock##LayerRow", Icon::Lock, ImVec2(lockBtnSize, lockBtnSize))) {
        layer.locked = !layer.locked;
        lastChangedLayerId = layer.id;
        lastAction = LayerToolAction::LOCK;
    }
    ImGui::PopStyleColor();

    ImGui::EndChild();
    ImGui::PopStyleVar(3);
    ImGui::PopStyleColor(2);
    ImGui::Dummy(ImVec2(0.0f, 3.0f)); // Satırlar arası boşluk azaltıldı
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