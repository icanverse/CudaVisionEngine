#include "UI/Layers/QuickLeftToolbar.h"

#include "imgui.h"
#include "io/AssetsManager/IconManager.h"

QuickLeftToolbar::QuickLeftToolbar()
    : currentTool(ActiveTool::NONE)
{
    availableTools.push_back({
        ActiveTool::REGION_SELECT,
        "Secim",
        "Dikdortgen Secim Araci (ROI)",
        Icon::Select_Region_Rectangle,
        "Dikdortgen seklinde alan secer"
    });

    availableTools.push_back({
        ActiveTool::REGION_CROP,
        "Kirp",
        "Secili Alani Kirp",
        Icon::Crop,
        "Secili alani keser ve ayirir"
    });

    availableTools.push_back({
        ActiveTool::PAN,
        "Kaydir",
        "Tuvali Kaydir (El Araci)",
        Icon::Move,
        "Calisma alaninda gezinmeyi saglar"
    });
}

void QuickLeftToolbar::render(
    float displayWidth,
    float displayHeight
) {
    const float toolbarWidth = 50.0f;
    const float toolbarHeight =
        20.0f + (75.0f * static_cast<float>(availableTools.size()));

    const float xPos = 15.0f;
    const float yPos =
        (displayHeight - toolbarHeight) * 0.5f;

    ImGui::SetNextWindowPos(
        ImVec2(xPos, yPos),
        ImGuiCond_Always
    );

    ImGui::SetNextWindowSize(
        ImVec2(toolbarWidth, toolbarHeight),
        ImGuiCond_Always
    );

    ImGui::PushStyleColor(
        ImGuiCol_WindowBg,
        ImVec4(0.08f, 0.08f, 0.09f, 0.75f)
    );

    ImGui::PushStyleColor(
        ImGuiCol_Border,
        ImVec4(0.85f, 0.45f, 0.0f, 0.4f)
    );

    ImGui::PushStyleVar(
        ImGuiStyleVar_WindowRounding,
        15.0f
    );

    ImGui::PushStyleVar(
        ImGuiStyleVar_WindowBorderSize,
        1.5f
    );

    ImGui::PushStyleVar(
        ImGuiStyleVar_WindowPadding,
        ImVec2(10.0f, 10.0f)
    );

    ImGui::PushStyleVar(
        ImGuiStyleVar_FrameRounding,
        12.0f
    );

    const ImGuiWindowFlags toolboxFlags =
        ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoScrollbar |
        ImGuiWindowFlags_NoCollapse |
        ImGuiWindowFlags_NoTitleBar;

    ImGui::Begin(
        "Sol Arac Kutusu",
        nullptr,
        toolboxFlags
    );

    for (const auto& tool : availableTools) {

        if (currentTool == tool.id) {
            ImGui::PushStyleColor(
                ImGuiCol_Button,
                ImVec4(0.85f, 0.45f, 0.0f, 1.0f)
            );

            ImGui::PushStyleColor(
                ImGuiCol_ButtonHovered,
                ImVec4(1.0f, 0.55f, 0.0f, 1.0f)
            );

            ImGui::PushStyleColor(
                ImGuiCol_ButtonActive,
                ImVec4(0.65f, 0.35f, 0.0f, 1.0f)
            );
        }
        else {
            ImGui::PushStyleColor(
                ImGuiCol_Button,
                ImVec4(0.0f, 0.0f, 0.0f, 0.0f)
            );

            ImGui::PushStyleColor(
                ImGuiCol_ButtonHovered,
                ImVec4(0.2f, 0.2f, 0.2f, 0.5f)
            );

            ImGui::PushStyleColor(
                ImGuiCol_ButtonActive,
                ImVec4(0.3f, 0.3f, 0.3f, 0.6f)
            );
        }

        // tool burada artık geçerli.
        const unsigned int textureID =
            IconManager::Get(tool.icon);

        const ImTextureID imguiTexID =
            static_cast<ImTextureID>(textureID);


        const ImVec2 iconSize(displayHeight/32.0f, displayHeight/32.0f);

        float buttonWidth = iconSize.x + (ImGui::GetStyle().FramePadding.x * 2.0f);
        float availWidth = ImGui::GetContentRegionAvail().x;
        float offsetX = (availWidth - buttonWidth) * 0.5f;
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + offsetX);
        bool clicked = false;

        if (textureID != 0) {
            clicked = ImGui::ImageButton(
                tool.name.c_str(),
                imguiTexID,
                iconSize,
                ImVec2(0.0f, 0.0f),
                ImVec2(1.0f, 1.0f),
                ImVec4(0.0f, 0.0f, 0.0f, 0.0f),
                ImVec4(1.0f, 1.0f, 1.0f, 1.0f)
            );
        }
        else {
            // Texture bulunamazsa siyah kutu yerine soru işareti.
            const std::string fallbackID =
                "?##" + tool.name;

            clicked = ImGui::Button(
                fallbackID.c_str(),
                iconSize
            );
        }

        // Hover kontrolü ImageButton/Button'dan hemen sonra yapılmalı.
        const bool hovered = ImGui::IsItemHovered();

        if (clicked) {
            currentTool = tool.id;
        }

        if (hovered) {
            ImGui::SetTooltip(
                "%s",
                tool.tooltip.c_str()
            );
        }

        ImGui::PopStyleColor(3);

        ImGui::Dummy(
            ImVec2(0.0f, 5.0f)
        );
    }

    ImGui::End();

    ImGui::PopStyleVar(4);
    ImGui::PopStyleColor(2);
}

QuickLeftToolbar::~QuickLeftToolbar() = default;