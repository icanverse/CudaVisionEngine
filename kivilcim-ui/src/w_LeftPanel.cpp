#include "w_LeftPanel.h"

#include <algorithm>
#include <cstdint>
#include <iostream>

#include "imgui.h"
#include "w_TopPanel.h"

// YENİ MİMARİ: Namespace Kdata'ya dönüştürüldü
#include "Persistence/KvlcmProjectParser.h"
#include "TextureUtility/TextureUtility.h"

static const std::string kWorkspaceFilePath =
    "C:/Users/Can/Desktop/sirca_workspace.kvlcm_proj";

void LeftPanel::render(
    float displayWidth,
    float displayHeight
) {
    ImGuiViewport* viewport = ImGui::GetMainViewport();
    float topPanelHeight = TopPanel::getPanelHeight();
    float realScreenHeight = ImGui::GetIO().DisplaySize.y;

    float panelWidth = 840.0f;
    float xPos = viewport->Pos.x + 15.0f;
    float yPos =
        viewport->Pos.y
        + 50.0f
        + topPanelHeight * 0.3f;
    float panelHeight =
        realScreenHeight
        - yPos
        - 15.0f;

    if (panelHeight < 100.0f) {
        panelHeight = 100.0f;
    }

    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.02f, 0.02f, 0.03f, 0.6f));

    ImGui::SetNextWindowViewport(viewport->ID);
    ImGui::SetNextWindowSize(ImVec2(panelWidth, panelHeight), ImGuiCond_Always);
    ImGui::SetNextWindowPos(ImVec2(xPos, yPos), ImGuiCond_Always);

    ImGuiWindowFlags leftPanelFlags =
        ImGuiWindowFlags_NoResize
        | ImGuiWindowFlags_NoCollapse
        | ImGuiWindowFlags_NoMove
        | ImGuiWindowFlags_NoTitleBar
        | ImGuiWindowFlags_NoDocking;

    ImGui::Begin("Hadi Baslayalim!", nullptr, leftPanelFlags);

    ImGui::SetWindowFontScale(1.8f);
    ImGui::TextColored(ImVec4(1.0f, 1.0f, 1.0f, 1.0f), "Hadi Baslayalim!");
    ImGui::SetWindowFontScale(1.0f);
    ImGui::Separator();
    ImGui::Dummy(ImVec2(0.0f, 15.0f));

    float windowVisibleX2 = ImGui::GetWindowPos().x + ImGui::GetWindowContentRegionMax().x;

    ImGuiStyle& style = ImGui::GetStyle();

    float tileWidth = 256.0f;
    float tileHeight = 144.0f;

    for (std::size_t i = 0; i < projectStack.size(); ++i) {
        ImGui::PushID(static_cast<int>(i));
        ImGui::BeginGroup();

        ImVec2 startPos = ImGui::GetCursorPos();

        if (projectStack[i].textureID > 0) {
            float originalWidth = static_cast<float>(projectStack[i].size.x);
            float originalHeight = static_cast<float>(projectStack[i].size.y);

            if (originalWidth <= 0.0f) originalWidth = tileWidth;
            if (originalHeight <= 0.0f) originalHeight = tileHeight;

            float scale = std::min(tileWidth / originalWidth, tileHeight / originalHeight);
            float renderWidth = originalWidth * scale;
            float renderHeight = originalHeight * scale;
            float offsetX = (tileWidth - renderWidth) * 0.5f;
            float offsetY = (tileHeight - renderHeight) * 0.5f;

            ImGui::SetCursorPos(ImVec2(startPos.x + offsetX, startPos.y + offsetY));

            const ImTextureID textureId = (ImTextureID)(intptr_t)projectStack[i].textureID;

            if (ImGui::ImageButton(
                    projectStack[i].name.c_str(),
                    textureId,
                    ImVec2(renderWidth, renderHeight),
                    ImVec2(0, 1),
                    ImVec2(1, 0)
                )) {
                std::cout << "[UI] Proje secildi: " << projectStack[i].name << std::endl;
                projectStack[i].isSelected = true;
            }

            if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
                std::cout << "[UI] Projeye CIFT TIKLANDI: " << projectStack[i].name << std::endl;
                if (onProjectDoubleClicked) {
                    onProjectDoubleClicked(projectStack[i].id);
                }
            }
        }
        else {
            if (ImGui::Button("Gorsel\nYok", ImVec2(tileWidth, tileHeight))) {
                std::cout << "[UI] Proje secildi: " << projectStack[i].name << std::endl;
            }

            if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
                std::cout << "[UI] Projeye CIFT TIKLANDI: " << projectStack[i].name << std::endl;
                if (onProjectDoubleClicked) {
                    onProjectDoubleClicked(projectStack[i].id);
                }
            }
        }

        ImGui::SetCursorPos(ImVec2(startPos.x, startPos.y + tileHeight + 5.0f));

        float textWidth = ImGui::CalcTextSize(projectStack[i].name.c_str()).x;
        float textIndent = (tileWidth - textWidth) * 0.5f;

        if (textIndent > 0.0f) {
            ImGui::SetCursorPosX(ImGui::GetCursorPosX() + textIndent);
        }

        ImGui::TextColored(ImVec4(0.8f, 0.8f, 0.8f, 1.0f), "%s", projectStack[i].name.c_str());

        ImGui::EndGroup();

        float lastGroupX2 = startPos.x + tileWidth;
        float nextGroupX2 = lastGroupX2 + style.ItemSpacing.x + tileWidth;

        if (i + 1 < projectStack.size() && nextGroupX2 < windowVisibleX2) {
            ImGui::SameLine();
        } else {
            ImGui::Dummy(ImVec2(0.0f, 20.0f));
        }

        ImGui::PopID();
    }

    ImGui::End();
    ImGui::PopStyleColor();
    ImGui::PopStyleVar(2);
}

void LeftPanel::addProjectToStack(Kdata::ProjectData newProject) {
    if (newProject.id == 0) {
        newProject.id = projectCounter++;
    }
    else if (newProject.id >= projectCounter) {
        projectCounter = newProject.id + 1;
    }

    if (newProject.name == "İsimsiz-1" || newProject.name.empty()) {
        newProject.name = "İsimsiz Proje " + std::to_string(newProject.id);
    }

    projectStack.insert(projectStack.begin(), newProject);
    std::cout << "[Kivilcim UI] Proje eklendi: " << newProject.name << std::endl;
}

void LeftPanel::loadWorkspace() {
    projectStack.clear();

    // DİKKAT: Parser sınıfın eğer Kivilcim namespace içindeyse burayı o şekilde bırakıyoruz.
    // Ancak Parser'ın döndürdüğü vektör Kdata::ProjectData tipinde olmalıdır!
    std::vector<Kdata::ProjectData> savedProjects = Kivilcim::KvlcmProjectParser::load(kWorkspaceFilePath);

    for (auto it = savedProjects.rbegin(); it != savedProjects.rend(); ++it) {
        Kdata::ProjectData& project = *it;

        if (!project.imagePath.empty()) {
            int originalWidth = 0;
            int originalHeight = 0;

            project.textureID = TextureUtility::LoadThumbnailFromFile(
                project.imagePath, 256, 144, originalWidth, originalHeight
            );

            if (project.textureID > 0) {
                project.size = { originalWidth, originalHeight };
            }
        } else {
            project.textureID = TextureUtility::CreateSolidColor(
                project.bgColor[0], project.bgColor[1], project.bgColor[2]
            );
        }
        addProjectToStack(project);
    }
}

void LeftPanel::saveWorkspace() {
    Kivilcim::KvlcmProjectParser::save(kWorkspaceFilePath, projectStack);
}