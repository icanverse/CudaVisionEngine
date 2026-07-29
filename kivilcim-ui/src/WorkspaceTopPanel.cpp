#include "WorkspaceTopPanel.h"
#include "imgui.h"

#ifdef _WIN32
#include <windows.h>
#endif

#include <GLFW/glfw3.h>
#include <string>

namespace {
constexpr float kWorkspacePanelHeight = 48.0f;
constexpr float kWindowButtonSize = 32.0f;
constexpr float kButtonGap = 4.0f;
constexpr float kRightPadding = 8.0f;
constexpr float kMenuPopupOffsetX = 10.0f;
constexpr float kMenuPopupOffsetY = 4.0f;

bool menuButton(const char* label, const char* popupId, float width) {
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.0f, 0.0f, 0.0f, 0.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.90f, 0.50f, 0.15f, 0.45f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.95f, 0.55f, 0.18f, 0.65f));

    const bool clicked = ImGui::Button(label, ImVec2(width, kWindowButtonSize));
    ImGui::PopStyleColor(3);

    if (clicked) ImGui::OpenPopup(popupId);

    const ImVec2 buttonMin = ImGui::GetItemRectMin();
    const ImVec2 buttonMax = ImGui::GetItemRectMax();
    ImGui::SetNextWindowPos(
        ImVec2(buttonMin.x + kMenuPopupOffsetX, buttonMax.y + kMenuPopupOffsetY),
        ImGuiCond_Appearing
    );

    return clicked;
}
}

bool WorkspaceTopPanel::render(
    Kdata::WorkspaceStateData* state,
    GLFWwindow* window,
    float displayWidth,
    unsigned int logoTextureId
) {
    // Güvenlik kontrolü
    if (window == nullptr || state == nullptr) return false;

    if (logoTextureId == 0) logoTextureId = sharedLogoTextureId;

    ImGui::SetCursorPos(ImVec2(0.0f, 0.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_ChildBorderSize, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 2.0f);
    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.08f, 0.08f, 0.09f, 1.0f));

    const ImGuiWindowFlags panelFlags =
        ImGuiWindowFlags_NoScrollbar |
        ImGuiWindowFlags_NoScrollWithMouse;

    ImGui::BeginChild("WorkspaceTopPanel", ImVec2(displayWidth, kWorkspacePanelHeight), false, panelFlags);

    const ImVec2 minPos = ImGui::GetWindowPos();
    const ImVec2 maxPos(minPos.x + ImGui::GetWindowWidth(), minPos.y + ImGui::GetWindowHeight());

    ImGui::GetWindowDrawList()->AddRectFilledMultiColor(
        minPos, maxPos,
        IM_COL32(32, 32, 36, 255), IM_COL32(32, 32, 36, 255),
        IM_COL32(18, 18, 20, 255), IM_COL32(18, 18, 20, 255)
    );

    float nextX = 10.0f;

    if (logoTextureId != 0) {
        constexpr float logoPadding = 12.0f;
        const float logoHeight = kWorkspacePanelHeight - logoPadding;
        float logoWidth = logoHeight;

        glBindTexture(GL_TEXTURE_2D, logoTextureId);
        int textureWidth = 0;
        int textureHeight = 0;
        glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, &textureWidth);
        glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT, &textureHeight);

        if (textureHeight > 0) {
            logoWidth = logoHeight * (static_cast<float>(textureWidth) / static_cast<float>(textureHeight));
        }

        ImGui::SetCursorPos(ImVec2(12.0f, logoPadding * 0.5f));
        ImGui::Image(
            (ImTextureID)(intptr_t)logoTextureId,
            ImVec2(logoWidth, logoHeight),
            ImVec2(0.0f, 1.0f),
            ImVec2(1.0f, 0.0f)
        );
        nextX = 12.0f + logoWidth + 12.0f;
    }

    const float buttonY = (kWorkspacePanelHeight - kWindowButtonSize) * 0.5f;

    ImGui::PushStyleColor(ImGuiCol_PopupBg, ImVec4(0.10f, 0.10f, 0.12f, 0.98f));
    ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0.20f, 0.20f, 0.22f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_HeaderHovered, ImVec4(0.90f, 0.50f, 0.15f, 0.45f));
    ImGui::PushStyleColor(ImGuiCol_HeaderActive, ImVec4(0.95f, 0.55f, 0.18f, 0.65f));

    bool closeWorkspace = false;

    ImGui::SetCursorPos(ImVec2(nextX, buttonY));
    menuButton("Dosya", "WorkspaceFileMenu", 58.0f);
    if (ImGui::BeginPopup("WorkspaceFileMenu")) {
        // İleride dosya modallarını tetikleyeceğin yerler
        if (ImGui::MenuItem("Ice Aktar...")) state->editors.activeEditor = Kdata::ExclusiveEditor::IMPORT_DIALOG;
        ImGui::Separator();
        if (ImGui::MenuItem("Farkli Kaydet...", "Ctrl+Shift+S")) state->editors.activeEditor = Kdata::ExclusiveEditor::SAVE_AS_DIALOG;
        if (ImGui::MenuItem("Disa Aktar...")) state->editors.activeEditor = Kdata::ExclusiveEditor::EXPORT_SETTINGS;
        ImGui::Separator();
        if (ImGui::MenuItem("Workspace'i Kapat")) closeWorkspace = true;
        ImGui::EndPopup();
    }

    ImGui::SameLine(0.0f, 2.0f);
    menuButton("Duzen", "WorkspaceEditMenu", 58.0f); //[cite: 8]
    if (ImGui::BeginPopup("WorkspaceEditMenu")) {
        if (ImGui::MenuItem("Geri Al", "Ctrl+Z")) state->tools.lastFiredAction = Kdata::InstantAction::UNDO; //[cite: 8]
        if (ImGui::MenuItem("Yinele", "Ctrl+Y")) state->tools.lastFiredAction = Kdata::InstantAction::REDO; //[cite: 8]

        ImGui::Separator();

        // YENI EKLENEN KISIM: Ayarlar menusunu tetikler
        // Eger WorkspaceStateData (state) icerisinde baska bir Editor yonetimin varsa onu kullanabilirsin.
        // Ornek: state->editors.activeEditor = Kdata::ExclusiveEditor::PREFERENCES;
        if (ImGui::MenuItem("Motor Tercihleri...")) {
            // Main loop icinde PreferencesPanel::render() fonksiyonunu cagiracak bayragi kaldir.
            state->showPreferences = true;
        }

        ImGui::EndPopup();
    }

    ImGui::SameLine(0.0f, 2.0f);
    menuButton("Katmanlar", "WorkspaceLayersMenu", 76.0f);
    if (ImGui::BeginPopup("WorkspaceLayersMenu")) {
        if (ImGui::MenuItem("Yeni Katman Ekle")) {
            state->tools.lastFiredAction = Kdata::InstantAction::LAYER_ADD;
        }

        ImGui::Separator();

        // YENİ MİMARİ: Katmanları doğrudan merkez (state) yapıdan çekiyoruz
        if (state->layers.layers.empty()) {
            ImGui::TextDisabled("Gosterilecek katman yok");
        } else {
            for (auto& layer : state->layers.layers) {
                const std::string itemId = layer.name + "##WorkspaceLayer" + std::to_string(layer.id);
                // boolean değeri doğrudan layer.isVisible'a bağlandı
                ImGui::MenuItem(itemId.c_str(), nullptr, &layer.isVisible);
            }
        }
        ImGui::EndPopup();
    }

    ImGui::SameLine(0.0f, 2.0f);
    menuButton("Filtreler", "WorkspaceFiltersMenu", 70.0f);
    if (ImGui::BeginPopup("WorkspaceFiltersMenu")) {
        if (ImGui::MenuItem("Parlaklık / Kontrast")) state->tools.activeAdjustment = Kdata::AdjustmentTool::BRIGHTNESS_CONTRAST;
        if (ImGui::MenuItem("Renk Dengesi")) state->tools.activeAdjustment = Kdata::AdjustmentTool::COLOR_BALANCE;
        if (ImGui::MenuItem("Ton / Doygunluk")) state->tools.activeAdjustment = Kdata::AdjustmentTool::HUE_SATURATION;
        ImGui::Separator();
        if (ImGui::MenuItem("Bulaniklastir")) state->tools.activeAdjustment = Kdata::AdjustmentTool::BLUR_SHARPEN;
        if (ImGui::MenuItem("Pozlama / Gama")) state->tools.activeAdjustment = Kdata::AdjustmentTool::EXPOSURE_GAMMA;
        ImGui::EndPopup();
    }

    ImGui::SameLine(0.0f, 2.0f);
    menuButton("Araçlar", "WorkspaceToolsMenu", 68.0f);
    if (ImGui::BeginPopup("WorkspaceToolsMenu")) {
        if (ImGui::BeginMenu("Oluştur")) {
            if (ImGui::MenuItem("Fiziksel Derinlik Katmanı")) {
                state->editors.activeEditor = Kdata::ExclusiveEditor::ISO_DEPTH;
            }
            // if (ImGui::MenuItem("Sanal Işık")) { ... } (Gelecekte eklenecek)
            ImGui::EndMenu();
        }
        ImGui::EndPopup();
    }

    ImGui::PopStyleColor(4);

    const float minimizeX = displayWidth - kRightPadding - (kWindowButtonSize * 2.0f) - kButtonGap;
    const float closeX = displayWidth - kRightPadding - kWindowButtonSize;
    const float dragAreaStartX = nextX + 58.0f + 2.0f + 58.0f + 2.0f + 76.0f + 2.0f + 70.0f + 2.0f + 68.0f + 8.0f;
    const float availableDragWidth = minimizeX - dragAreaStartX;
    const float dragAreaWidth = availableDragWidth > 10.0f ? availableDragWidth : 10.0f;

    ImGui::SetCursorPos(ImVec2(dragAreaStartX, 0.0f));
    ImGui::InvisibleButton("WorkspaceDragArea", ImVec2(dragAreaWidth, kWorkspacePanelHeight));

    if (ImGui::IsItemHovered() && ImGui::IsMouseClicked(0)) {
        isDragging = true;
        double mouseX = 0.0;
        double mouseY = 0.0;
        glfwGetCursorPos(window, &mouseX, &mouseY);
        dragOffsetX = static_cast<int>(mouseX);
        dragOffsetY = static_cast<int>(mouseY);
    }

    if (isDragging && ImGui::IsMouseDown(0)) {
#ifdef _WIN32
        POINT cursorPosition;
        GetCursorPos(&cursorPosition);
        glfwSetWindowPos(window, cursorPosition.x - dragOffsetX, cursorPosition.y - dragOffsetY);
#endif
    } else {
        isDragging = false;
    }

    ImGui::SetCursorPos(ImVec2(minimizeX, buttonY));
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.0f, 0.0f, 0.0f, 0.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.3f, 0.35f, 0.6f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.4f, 0.4f, 0.45f, 0.8f));
    if (ImGui::Button("-##WorkspaceMinimize", ImVec2(kWindowButtonSize, kWindowButtonSize))) {
        glfwIconifyWindow(window);
    }
    ImGui::PopStyleColor(3);

    ImGui::SetCursorPos(ImVec2(closeX, buttonY));
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.0f, 0.0f, 0.0f, 0.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.9f, 0.2f, 0.2f, 0.8f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.8f, 0.15f, 0.15f, 1.0f));
    const bool closeButtonClicked = ImGui::Button("X##WorkspaceClose", ImVec2(kWindowButtonSize, kWindowButtonSize));
    ImGui::PopStyleColor(3);

    ImGui::EndChild();
    ImGui::PopStyleColor();
    ImGui::PopStyleVar(4);

    return closeButtonClicked || closeWorkspace;
}

float WorkspaceTopPanel::getPanelHeight() {
    return kWorkspacePanelHeight;
}