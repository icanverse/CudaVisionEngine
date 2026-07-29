#include "MainUI.h"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "TextureUtility/TextureUtility.h"
#include <iostream>

#include "AssetsManager/IconManager.h"
#include "TextureUtility/CudaDynamicTexture.cuh"

MainUI::MainUI(GLFWwindow* window) : windowHandle(window), logoTextureId(0), logFont(nullptr) {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();

    ImGuiIO& io = ImGui::GetIO(); (void)io;

    static const ImWchar turkishRanges[] = {
        0x0020, 0x00FF,
        0x0100, 0x017F,
        0,
    };

    ImFont* mainFont = io.Fonts->AddFontFromFileTTF(
        "C:/Users/Can/CLionProjects/CudVisionEngineX/lib-assets/font/Inter/static/Inter_24pt-Black.ttf",
        16.0f, nullptr, turkishRanges
    );

    if (mainFont == nullptr) {
        std::cout << "[Sirca UI - UYARI] Inter fontu bulunamadi!\n";
    }

    logFont = io.Fonts->AddFontFromFileTTF(
        "C:/Users/Can/CLionProjects/CudVisionEngineX/lib-assets/font/Inter/static/Inter_24pt-Black.ttf",
        14.0f, nullptr, turkishRanges
    );

    if (logFont != nullptr) {
        std::cout << "[Sirca UI] Inter ve JetBrains Mono fontlari basariyla yuklendi.\n";
    }

    ImGui::StyleColorsDark();
    ImGuiStyle& style = ImGui::GetStyle();

    style.Colors[ImGuiCol_TitleBg] = ImVec4(0.08f, 0.08f, 0.09f, 1.0f);
    style.Colors[ImGuiCol_TitleBgActive] = ImVec4(0.12f, 0.12f, 0.13f, 1.0f);
    style.Colors[ImGuiCol_WindowBg] = ImVec4(0.05f, 0.05f, 0.06f, 1.0f);

    style.Colors[ImGuiCol_Button] = ImVec4(0.2f, 0.2f, 0.25f, 1.0f);
    style.Colors[ImGuiCol_ButtonHovered] = ImVec4(0.7f, 0.35f, 0.05f, 1.0f);
    style.Colors[ImGuiCol_ButtonActive] = ImVec4(0.8f, 0.45f, 0.10f, 1.0f);

    style.WindowRounding = 2.0f;
    style.FrameRounding = 2.0f;
    style.PopupRounding = 2.0f;
    style.WindowBorderSize = 0.0f;
    style.FrameBorderSize = 0.0f;

    ImGui_ImplGlfw_InitForOpenGL(windowHandle, true);
    ImGui_ImplOpenGL3_Init("#version 130");

    IconManager::Initialize();

    logoTextureId = TextureUtility::LoadTextureFromFile("C:/Users/Can/CLionProjects/CudaVisionEngine/lib-assets/logo.png");
    liquidCanvas = new CudaDynamicTexture(512, 288);

    // ==========================================
    // SİNYAL KÖPRÜLERİ (CALLBACKS)
    // ==========================================

    // SAĞ PANELDEN SOL PANELE OLUŞTURMA SİNYALİ
    // NOT: Kivilcim namespace'i Kdata olarak güncellendi.
    rightPanel.setOnProjectCreatedCallback([this](const Kdata::ProjectData& newProjectData) {
        std::cout << "[Kivilcim DEBUG 4] Sinyal alindi! Sola ekleniyor..." << std::endl;
        leftPanel.addProjectToStack(newProjectData);
        leftPanel.saveWorkspace();
    });

    // SOL PANELDEN GELEN ÇİFT TIKLAMA (DÜZENLE) SİNYALİ
    // SOL PANELDEN GELEN ÇİFT TIKLAMA (DÜZENLE) SİNYALİ
    leftPanel.setOnProjectDoubleClickedCallback([this](const int& projectID) {
        Kdata::ProjectData* p = leftPanel.getProjectByID(projectID);
        if (p) {
            appState.resetState();
            appState.project = *p;

            // Motoru "Çalışma (Editör)" moduna geçir!
            currentMode = AppMode::WORKSPACE;
        }
    });


    // ÇALIŞMA ALANINDAN "ANA EKRANA DÖN" SİNYALİ
    workspaceUI.setOnCloseCallback([this]() {
        currentMode = AppMode::START_SCREEN;
        appState.resetState(); // Ana ekrana dönerken güvenli sıfırlama
    });

    leftPanel.loadWorkspace();
}

MainUI::~MainUI() {
    leftPanel.saveWorkspace();

    if (liquidCanvas) {
        delete liquidCanvas;
    }

    if (logoTextureId != 0) {
        glDeleteTextures(1, &logoTextureId);
    }

    IconManager::Shutdown();
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}

void MainUI::newFrame() {
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

void MainUI::renderPanels() {
    ImGuiIO& io = ImGui::GetIO();

    backgroundPanel.render(io.DisplaySize.x, io.DisplaySize.y);
    topPanel.render(windowHandle, io.DisplaySize.x, logoTextureId);
    leftPanel.render(io.DisplaySize.x, logoTextureId);
    rightPanel.render(io.DisplaySize.x, io.DisplaySize.y);

    // EĞER ÇALIŞMA ALANINDAYSAK, ONU DA ÇİZ
    if (currentMode == AppMode::WORKSPACE) {
        // YENİ MİMARİ: Workspace artık tamamen aptal. Sadece state pointer'ını alıp ekrana basacak.
        workspaceUI.render(&appState, io.DisplaySize.x, io.DisplaySize.y);
    }

    preferencesPanel.render(appState.showPreferences, userPrefs);
}

void MainUI::renderDrawData() {
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}