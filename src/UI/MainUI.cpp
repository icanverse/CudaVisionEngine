#include "UI/MainUI.h"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "io/TextureUtility/TextureUtility.h"

MainUI::MainUI(GLFWwindow* window) : windowHandle(window), logoTextureId(0) {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();

    ImGuiStyle& style = ImGui::GetStyle();

    // --- SIRÇA GLOBAL RENK PALETİ ---
    // Panel başlık çubuğu (Aktif değilken)
    style.Colors[ImGuiCol_TitleBg] = ImVec4(0.08f, 0.08f, 0.09f, 1.0f);
    // Panel başlık çubuğu (Tıklandığında/Aktifken) - Logondaki turuncuya yakın bir ton
    style.Colors[ImGuiCol_TitleBgActive] = ImVec4(0.7f, 0.35f, 0.05f, 1.0f);
    // Pencere arkaplanları
    style.Colors[ImGuiCol_WindowBg] = ImVec4(0.05f, 0.05f, 0.06f, 1.0f);
    // Butonlar
    style.Colors[ImGuiCol_Button] = ImVec4(0.2f, 0.2f, 0.25f, 1.0f);
    style.Colors[ImGuiCol_ButtonHovered] = ImVec4(0.7f, 0.35f, 0.05f, 1.0f); // Turuncu Hover

    style.WindowRounding = 12.0f;
    style.FrameRounding = 6.0f;
    style.WindowBorderSize = 0.0f;

    ImGui_ImplGlfw_InitForOpenGL(windowHandle, true);
    ImGui_ImplOpenGL3_Init("#version 130");

    // Logonun yolunu buraya yazıyoruz (Örn: assets klasörü içindeyse)
    // Şimdilik kök dizindeki "logo.png" olarak ayarlıyoruz.
    logoTextureId = TextureUtility::LoadTextureFromFile("C:/Users/Can/CLionProjects/CudVisionEngineX/src/UI/logo.png");}

MainUI::~MainUI() {
    // VRAM sızıntısını (Memory Leak) önlemek için logoyu ekran kartından siliyoruz
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

    // İLK OLARAK ZEMİN ÇİZİLİR (En arkada kalması için)
    // Eğer arka plan panelindeki render fonksiyonun bir sınıfa aitse bu şekilde çağrılır:
    backgroundPanel.render(io.DisplaySize.x, io.DisplaySize.y);

    // 2. DİĞER PANELLER ONUN ÜZERİNE BİNER
    topPanel.render(windowHandle, io.DisplaySize.x, logoTextureId);
    rightPanel.render(io.DisplaySize.x, io.DisplaySize.y);
}

void MainUI::renderDrawData() {
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}