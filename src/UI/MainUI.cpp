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

    // (Opsiyonel) Docking özelliğini aktif etmek istersen bu satırın yorumunu kaldırabilirsin
    // io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

    // 2. TÜRKÇE KARAKTER DESTEĞİ VE FONT YÜKLEME
    static const ImWchar turkishRanges[] = {
        0x0020, 0x00FF, // Temel Latin (İngilizce harfler ve semboller)
        0x0100, 0x017F, // Latin Genişletilmiş-A (Türkçe özel karakterlerin tamamı buradadır)
        0,              // Diziyi sonlandırmak için 0 şarttır
    };

    // Ana Arayüz Fontu (Inter) - Yüklenen ilk font varsayılan (default) olur
    ImFont* mainFont = io.Fonts->AddFontFromFileTTF(
        "C:/Users/Can/CLionProjects/CudVisionEngineX/lib-assets/font/Inter/static/Inter_24pt-Black.ttf",
        16.0f, nullptr, turkishRanges
    );

    if (mainFont == nullptr) {
        std::cout << "[Sirca UI - UYARI] Inter-Medium.ttf bulunamadi!\n";
    }

    // Log / Terminal Fontu (JetBrains Mono) - Daha küçük bir punto iyi durur
    logFont = io.Fonts->AddFontFromFileTTF(
        "C:/Users/Can/CLionProjects/CudVisionEngineX/lib-assets/font/Inter/static/Inter_24pt-Black.ttf",
        14.0f, nullptr, turkishRanges
    );

    if (logFont == nullptr) {
        std::cout << "[Sirca UI - UYARI] JetBrainsMono-Regular.ttf bulunamadi!\n";
    } else {
        std::cout << "[Sirca UI] Inter ve JetBrains Mono fontlari basariyla yuklendi.\n";
    }

    // 3. RENK PALETİ VE ENDÜSTRİYEL TASARIM
    ImGui::StyleColorsDark();
    ImGuiStyle& style = ImGui::GetStyle();

    style.Colors[ImGuiCol_TitleBg] = ImVec4(0.08f, 0.08f, 0.09f, 1.0f);
    style.Colors[ImGuiCol_TitleBgActive] = ImVec4(0.12f, 0.12f, 0.13f, 1.0f); // Odak dağıtmayan gri
    style.Colors[ImGuiCol_WindowBg] = ImVec4(0.05f, 0.05f, 0.06f, 1.0f);

    // Vurgu (Turuncu) sadece etkileşimli alanlarda
    style.Colors[ImGuiCol_Button] = ImVec4(0.2f, 0.2f, 0.25f, 1.0f);
    style.Colors[ImGuiCol_ButtonHovered] = ImVec4(0.7f, 0.35f, 0.05f, 1.0f);
    style.Colors[ImGuiCol_ButtonActive] = ImVec4(0.8f, 0.45f, 0.10f, 1.0f);

    // Çerçeveler ve Kenarlar (Keskin, teknik donanım hissiyatı)
    style.WindowRounding = 2.0f;
    style.FrameRounding = 2.0f;
    style.PopupRounding = 2.0f;
    style.WindowBorderSize = 0.0f;
    style.FrameBorderSize = 0.0f; // Butonların ve inputların etrafına ince bir sınır ekler

    // 4. BACKEND BAŞLATMA
    ImGui_ImplGlfw_InitForOpenGL(windowHandle, true);
    ImGui_ImplOpenGL3_Init("#version 130");

    // Logo Yükleme
    logoTextureId = TextureUtility::LoadTextureFromFile("C:/Users/Can/CLionProjects/CudVisionEngineX/src/UI/logo.png");

    // SAĞ PANELDEN SOL PANELE KÖPRÜ KURULUMU (Lambda Fonksiyonu ile)
    rightPanel.setOnImageImportedCallback([this](const std::string& imagePath) {
        leftPanel.addProjectToStack(imagePath);
    });
}

MainUI::~MainUI() {
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

    // Arka plan paneli
    backgroundPanel.render(io.DisplaySize.x, io.DisplaySize.y);

    // Ana paneller
    topPanel.render(windowHandle, io.DisplaySize.x, logoTextureId);
    leftPanel.render(io.DisplaySize.x, logoTextureId);
    rightPanel.render(io.DisplaySize.x, io.DisplaySize.y);

    // --- JETBRAINS MONO KULLANIM ÖRNEĞİ ---
    /* // Log penceresini çizerken fontu geçici olarak JetBrains Mono ile değiştiriyoruz
    if (logFont != nullptr) {
        ImGui::PushFont(logFont);
    }

    ImGui::Begin("Kivilcim Terminali");
    ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "[OK] VRAM Rezerve Edildi: 1920x1080");
    ImGui::Text("FPS: 60 | GPU Kullanimi: %d MB", 245);
    ImGui::End();

    if (logFont != nullptr) {
        ImGui::PopFont(); // Çizim bitince varsayılan fonta (Inter) geri döner
    }
    */
}

void MainUI::renderDrawData() {
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}