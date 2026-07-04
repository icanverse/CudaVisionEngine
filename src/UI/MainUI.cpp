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

    // Yüklenen logo ID'sini TopPanel'e gönderiyoruz!
    topPanel.render(windowHandle, io.DisplaySize.x, logoTextureId);
    rightPanel.render(io.DisplaySize.x, io.DisplaySize.y);
}

void MainUI::renderDrawData() {
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}