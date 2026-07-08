#include "UI/MainUI.h"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "io/TextureUtility/TextureUtility.h"
#include <iostream>

MainUI::MainUI(GLFWwindow* window) : windowHandle(window), logoTextureId(0), logFont(nullptr) {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();

    // 1. IO BAĞLAMININ ALINMASI
    ImGuiIO& io = ImGui::GetIO(); (void)io;

    // 2. TÜRKÇE KARAKTER DESTEĞİ VE FONT YÜKLEME
    static const ImWchar turkishRanges[] = {
        0x0020, 0x00FF, // Temel Latin
        0x0100, 0x017F, // Latin Genişletilmiş-A (Türkçe dahil)
        0,
    };

    // Ana Arayüz Fontu (Inter)
    ImFont* mainFont = io.Fonts->AddFontFromFileTTF(
        "C:/Users/Can/CLionProjects/CudVisionEngineX/lib-assets/font/Inter/static/Inter_24pt-Black.ttf",
        16.0f, nullptr, turkishRanges
    );

    if (mainFont == nullptr) {
        std::cout << "[Sirca UI - UYARI] Inter fontu bulunamadi!\n";
    }

    // Log / Terminal Fontu
    logFont = io.Fonts->AddFontFromFileTTF(
        "C:/Users/Can/CLionProjects/CudVisionEngineX/lib-assets/font/Inter/static/Inter_24pt-Black.ttf",
        14.0f, nullptr, turkishRanges
    );

    if (logFont != nullptr) {
        std::cout << "[Sirca UI] Inter ve JetBrains Mono fontlari basariyla yuklendi.\n";
    }

    // 3. RENK PALETİ VE ENDÜSTRİYEL TASARIM
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

    // 4. BACKEND BAŞLATMA
    ImGui_ImplGlfw_InitForOpenGL(windowHandle, true);
    ImGui_ImplOpenGL3_Init("#version 130");

    logoTextureId = TextureUtility::LoadTextureFromFile("C:/Users/Can/CLionProjects/CudVisionEngineX/src/UI/logo.png");

    // ==========================================
    // SİNYAL KÖPRÜLERİ (CALLBACKS)
    // ==========================================

    // SAĞ PANELDEN SOL PANELE OLUŞTURMA SİNYALİ
    rightPanel.setOnProjectCreatedCallback([this](const Kivilcim::ProjectData& newProjectData) {
        std::cout << "[Kivilcim DEBUG 4] Sinyal alindi! Sola ekleniyor..." << std::endl;
        leftPanel.addProjectToStack(newProjectData);
        leftPanel.saveWorkspace(); // Otomatik Kayıt
    });

    // SOL PANELDEN GELEN ÇİFT TIKLAMA (DÜZENLE) SİNYALİ
    leftPanel.setOnProjectDoubleClickedCallback([this](int projectID) {
        Kivilcim::ProjectData* p = leftPanel.getProjectByID(projectID);
        if (p) {
            workspaceUI.loadProject(p);            // Projeyi çalışma alanına yükle
            currentMode = AppMode::WORKSPACE;      // Motoru "Çalışma (Editör)" moduna geçir!
        }
    });

    // ÇALIŞMA ALANINDAN "ANA EKRANA DÖN" SİNYALİ
    workspaceUI.setOnCloseCallback([this]() {
        currentMode = AppMode::START_SCREEN;       // Motoru tekrar karşılama ekranına döndür
    });

    // x2 Hatanın asıl sebebi olan çift yükleme satırı düzeltildi (Sadece tek çağrı bırakıldı)
    leftPanel.loadWorkspace();
}

MainUI::~MainUI() {
    leftPanel.saveWorkspace();
    if (logoTextureId != 0) {
        glDeleteTextures(1, &logoTextureId);
    }
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

    if (currentMode == AppMode::START_SCREEN) {
        // KARŞILAMA EKRANI MODU
        backgroundPanel.render(io.DisplaySize.x, io.DisplaySize.y);
        topPanel.render(windowHandle, io.DisplaySize.x, logoTextureId);
        leftPanel.render(io.DisplaySize.x, logoTextureId);
        rightPanel.render(io.DisplaySize.x, io.DisplaySize.y);
    }
    else if (currentMode == AppMode::WORKSPACE) {
        // PROFESYONEL ÇALIŞMA ALANI MODU (PHOTOSHOP GİBİ)
        workspaceUI.render(io.DisplaySize.x, io.DisplaySize.y);
    }
}

void MainUI::renderDrawData() {
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}