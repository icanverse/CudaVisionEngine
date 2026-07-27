#include "../WorkspaceTopPanel.h"
#include "imgui.h"

#ifdef _WIN32
#include <windows.h>
#endif

#include <GLFW/glfw3.h>

#include <cstdint>
#include <utility>

namespace {
constexpr float kWorkspacePanelHeight = 48.0f;
constexpr float kWindowButtonSize = 32.0f;
constexpr float kButtonGap = 4.0f;
constexpr float kRightPadding = 8.0f;
constexpr float kMenuPopupOffsetX = 10.0f;
constexpr float kMenuPopupOffsetY = 4.0f;

bool menuButton(const char* label, const char* popupId, float width) {
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.0f, 0.0f, 0.0f, 0.0f));

    // İstediğin turuncu hover ve active (tıklanma) renk vurguları
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.90f, 0.50f, 0.15f, 0.45f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.95f, 0.55f, 0.18f, 0.65f));

    const bool clicked = ImGui::Button(label, ImVec2(width, kWindowButtonSize));
    ImGui::PopStyleColor(3);

    if (clicked) ImGui::OpenPopup(popupId);

    const ImVec2 buttonMin = ImGui::GetItemRectMin();
    const ImVec2 buttonMax = ImGui::GetItemRectMax();
    ImGui::SetNextWindowPos(
        ImVec2(
            buttonMin.x + kMenuPopupOffsetX,
            buttonMax.y + kMenuPopupOffsetY
        ),
        ImGuiCond_Appearing
    );

    return clicked;
}
}

bool WorkspaceTopPanel::render(
    GLFWwindow* window,
    float displayWidth,
    unsigned int logoTextureId
) {
    if (window == nullptr) return false;

    if (logoTextureId == 0) logoTextureId = sharedLogoTextureId;

    ImGui::SetCursorPos(ImVec2(0.0f, 0.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_ChildBorderSize, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));

    // Modern tasarım: Kenarlar sivriltildi (6.0f -> 2.0f)
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 2.0f);

    // Panel arka planı daha nötr bir koyu gri yapıldı
    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.08f, 0.08f, 0.09f, 1.0f));

    const ImGuiWindowFlags panelFlags =
        ImGuiWindowFlags_NoScrollbar |
        ImGuiWindowFlags_NoScrollWithMouse;

    ImGui::BeginChild(
        "WorkspaceTopPanel",
        ImVec2(displayWidth, kWorkspacePanelHeight),
        false,
        panelFlags
    );

    const ImVec2 minPos = ImGui::GetWindowPos();
    const ImVec2 maxPos(
        minPos.x + ImGui::GetWindowWidth(),
        minPos.y + ImGui::GetWindowHeight()
    );

    // Eski kahverengi tonlar yerine modern, profesyonel antrasit renk geçişi
    ImGui::GetWindowDrawList()->AddRectFilledMultiColor(
        minPos,
        maxPos,
        IM_COL32(32, 32, 36, 255),
        IM_COL32(32, 32, 36, 255),
        IM_COL32(18, 18, 20, 255),
        IM_COL32(18, 18, 20, 255)
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
            logoWidth = logoHeight *
                (static_cast<float>(textureWidth) / static_cast<float>(textureHeight));
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

    // Menü popup'larının görünümünü modernleştirmek için ufak bir stil müdahalesi
    ImGui::PushStyleColor(ImGuiCol_PopupBg, ImVec4(0.10f, 0.10f, 0.12f, 0.98f));
    ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0.20f, 0.20f, 0.22f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_HeaderHovered, ImVec4(0.90f, 0.50f, 0.15f, 0.45f)); // Menü içi turuncu hover
    ImGui::PushStyleColor(ImGuiCol_HeaderActive, ImVec4(0.95f, 0.55f, 0.18f, 0.65f));

    ImGui::SetCursorPos(ImVec2(nextX, buttonY));
    menuButton("Dosya", "WorkspaceFileMenu", 58.0f);
    if (ImGui::BeginPopup("WorkspaceFileMenu")) {
        if (ImGui::MenuItem("Yeni", "Ctrl+N")) lastAction = WorkspaceMenuAction::NEW_FILE;
        if (ImGui::MenuItem("Dosya Ac...", "Ctrl+O")) lastAction = WorkspaceMenuAction::OPEN_FILE;
        if (ImGui::MenuItem("Ice Aktar...")) lastAction = WorkspaceMenuAction::IMPORT_FILE;
        ImGui::Separator();
        if (ImGui::MenuItem("Kaydet", "Ctrl+S")) lastAction = WorkspaceMenuAction::SAVE_FILE;
        if (ImGui::MenuItem("Farkli Kaydet...", "Ctrl+Shift+S")) lastAction = WorkspaceMenuAction::SAVE_AS;
        if (ImGui::MenuItem("Disa Aktar...")) lastAction = WorkspaceMenuAction::EXPORT_FILE;
        ImGui::Separator();
        if (ImGui::MenuItem("Workspace'i Kapat")) {
            lastAction = WorkspaceMenuAction::CLOSE_WORKSPACE;
        }
        ImGui::EndPopup();
    }

    ImGui::SameLine(0.0f, 2.0f);
    menuButton("Duzen", "WorkspaceEditMenu", 58.0f);
    if (ImGui::BeginPopup("WorkspaceEditMenu")) {
        if (ImGui::MenuItem("Geri Al", "Ctrl+Z")) lastAction = WorkspaceMenuAction::UNDO;
        if (ImGui::MenuItem("Yinele", "Ctrl+Y")) lastAction = WorkspaceMenuAction::REDO;
        ImGui::EndPopup();
    }

    ImGui::SameLine(0.0f, 2.0f);
    menuButton("Katmanlar", "WorkspaceLayersMenu", 76.0f);
    if (ImGui::BeginPopup("WorkspaceLayersMenu")) {
        if (ImGui::MenuItem("Yeni Katman Ekle")) {
            lastAction = WorkspaceMenuAction::ADD_LAYER;
        }

        ImGui::Separator();

        if (layers.empty()) {
            ImGui::TextDisabled("Gosterilecek katman yok");
        } else {
            for (WorkspaceLayerMenuItem& layer : layers) {
                const std::string itemId = layer.name + "##WorkspaceLayer" +
                                           std::to_string(layer.id);
                if (ImGui::MenuItem(itemId.c_str(), nullptr, &layer.visible)) {
                    lastChangedLayerId = layer.id;
                    lastAction = WorkspaceMenuAction::LAYER_VISIBILITY_CHANGED;
                }
            }
        }
        ImGui::EndPopup();
    }

    ImGui::SameLine(0.0f, 2.0f);
    menuButton("Filtreler", "WorkspaceFiltersMenu", 70.0f);
    if (ImGui::BeginPopup("WorkspaceFiltersMenu")) {
        if (ImGui::MenuItem("Kontrast")) lastAction = WorkspaceMenuAction::CONTRAST_FILTER;
        if (ImGui::MenuItem("Renk Sicakligi")) lastAction = WorkspaceMenuAction::TEMPERATURE_FILTER;
        if (ImGui::MenuItem("Siyah Beyaz")) lastAction = WorkspaceMenuAction::GRAYSCALE_FILTER;
        ImGui::Separator();
        if (ImGui::MenuItem("Bulaniklastir")) lastAction = WorkspaceMenuAction::BLUR_FILTER;
        if (ImGui::MenuItem("Keskinlestir")) lastAction = WorkspaceMenuAction::SHARPEN_FILTER;
        ImGui::EndPopup();
    }

    ImGui::SameLine(0.0f, 2.0f);
    menuButton("Araçlar", "WorkspaceToolsMenu", 68.0f);

    if (ImGui::BeginPopup("WorkspaceToolsMenu")) {
        if (ImGui::BeginMenu("Oluştur")) {
            if (ImGui::MenuItem("Fiziksel Derinlik Katmanı")) {
                lastAction =
                    WorkspaceMenuAction::CREATE_PHYSICAL_DEPTH_LAYER;

            }

            if (ImGui::MenuItem("Sanal Işık")) {
                lastAction =
                    WorkspaceMenuAction::CREATE_VIRTUAL_LIGHT;
            }

            ImGui::EndMenu();
        }

        ImGui::EndPopup();
    }

    // Popup renk stillerini temizle
    ImGui::PopStyleColor(4);

    const float minimizeX = displayWidth - kRightPadding -
                            (kWindowButtonSize * 2.0f) - kButtonGap;
    const float closeX = displayWidth - kRightPadding - kWindowButtonSize;
    const float dragAreaStartX =
    nextX +
    58.0f + 2.0f +   // Dosya
    58.0f + 2.0f +   // Duzen
    76.0f + 2.0f +   // Katmanlar
    70.0f + 2.0f +   // Filtreler
    68.0f +           // Araclar
    8.0f;
    const float availableDragWidth = minimizeX - dragAreaStartX;
    const float dragAreaWidth = availableDragWidth > 10.0f
        ? availableDragWidth
        : 10.0f;

    ImGui::SetCursorPos(ImVec2(dragAreaStartX, 0.0f));
    ImGui::InvisibleButton(
        "WorkspaceDragArea",
        ImVec2(dragAreaWidth, kWorkspacePanelHeight)
    );

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
        glfwSetWindowPos(
            window,
            cursorPosition.x - dragOffsetX,
            cursorPosition.y - dragOffsetY
        );
#endif
    } else {
        isDragging = false;
    }

    // Küçültme butonu renklerini modernize etme
    ImGui::SetCursorPos(ImVec2(minimizeX, buttonY));
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.0f, 0.0f, 0.0f, 0.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.3f, 0.35f, 0.6f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.4f, 0.4f, 0.45f, 0.8f));
    if (ImGui::Button("-##WorkspaceMinimize", ImVec2(kWindowButtonSize, kWindowButtonSize))) {
        glfwIconifyWindow(window);
    }
    ImGui::PopStyleColor(3);

    // Kapatma butonu renklerini modernize etme
    ImGui::SetCursorPos(ImVec2(closeX, buttonY));
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.0f, 0.0f, 0.0f, 0.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.9f, 0.2f, 0.2f, 0.8f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.8f, 0.15f, 0.15f, 1.0f));
    const bool closeButtonClicked = ImGui::Button(
        "X##WorkspaceClose",
        ImVec2(kWindowButtonSize, kWindowButtonSize)
    );
    ImGui::PopStyleColor(3);

    ImGui::EndChild();
    ImGui::PopStyleColor();
    ImGui::PopStyleVar(4);

    const bool closeFromMenu = lastAction == WorkspaceMenuAction::CLOSE_WORKSPACE;
    return closeButtonClicked || closeFromMenu;
}

float WorkspaceTopPanel::getPanelHeight() {
    return kWorkspacePanelHeight;
}

void WorkspaceTopPanel::setLayers(
    const std::vector<WorkspaceLayerMenuItem>& newLayers
) {
    layers = newLayers;
}

WorkspaceMenuAction WorkspaceTopPanel::consumeLastAction() {
    const WorkspaceMenuAction action = lastAction;
    lastAction = WorkspaceMenuAction::NONE;
    return action;
}