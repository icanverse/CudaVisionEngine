#include <iostream>
#include <cuda_runtime.h>

#ifdef _WIN32
#include <windows.h>
extern "C" {
    __declspec(dllexport) unsigned long NvOptimusEnablement = 0x00000001;
    __declspec(dllexport) int AmdPowerXpressRequestHighPerformance = 1;
}
#endif

// MOTOR BAŞLIK DOSYALARI
#include "EngineFactory/EngineFactory.cuh"
#include "Graphics/Renderer3D.cuh"
#include "io/GlfwInteropTarget.h"
#include "Graphics/Scene.cuh"
#include "io/Graphics/SceneBuilder.h"
#include "Compute/ParticleSystem/ParticleSystem.cuh" // <-- KIVILCIM SİSTEMİ EKLENDİ

// DEAR IMGUI BAŞLIK DOSYALARI
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

int main() {
    std::cout << "[Kivilcim] Adim 1: Siber-Arayuz Penceresi Olusturuluyor...\n";

    int width = 1366;
    int height = 768;
    int channels = 3;
    GlfwInteropTarget target(width, height, channels, "Kivilcim - Sirca UI");

    std::cout << "[Kivilcim] Adim 2: CUDA Donanimi Uyandiriliyor...\n";
    cudaSetDevice(0);
    cudaFree(0);

    EngineFactory visionEngine(width, height, channels);
    Renderer3D graphicsRenderer(width, height, channels);

    std::cout << "[Kivilcim] Adim 3: 3D Sahne ve Materyaller Yukleniyor...\n";
    Scene myScene = SceneBuilder::build("assets-graphics/scenes/scene_ui.kvlcm");

    // PARÇACIK (KIVILCIM) SİSTEMİ BAŞLATILIYOR
    std::cout << "[Kivilcim] Adim 3.5: Parcacik Fiziği Baslatiliyor...\n";
    ParticleSystem kivilcimSistemi(5000);

    cudaMemset(visionEngine.getDeviceData(), 0, width * height * channels * sizeof(float));

    std::cout << "[Kivilcim] Adim 4: ImGui Sirca Arayuzu Kuruluyor...\n";
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();

    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowRounding = 12.0f;
    style.FrameRounding = 6.0f;
    style.WindowBorderSize = 0.0f;

    ImGui_ImplGlfw_InitForOpenGL(target.getWindow(), true);
    ImGui_ImplOpenGL3_Init("#version 130");

    bool isDragging = false;
    int dragOffsetX = 0, dragOffsetY = 0;
    float timeTracker = 0.0f;

    std::cout << "[Kivilcim] Motor Aktif. Sistem hazir.\n";

    // --- ANA DÖNGÜ ---
    while (!target.shouldClose()) {
        glfwPollEvents();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        ImGuiIO& io = ImGui::GetIO();

        // [ÜST BAR]
        ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(io.DisplaySize.x, 35.0f), ImGuiCond_Always);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);

        ImGui::Begin("UstPanel", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove);
        ImGui::Text("  KIVILCIM ENGINEX - Sirca UI");
        ImGui::SameLine();

        float dragAreaWidth = io.DisplaySize.x - ImGui::GetCursorPosX() - 95.0f;
        ImGui::InvisibleButton("DragArea", ImVec2(dragAreaWidth, 35.0f));
        if (ImGui::IsItemHovered() && ImGui::IsMouseClicked(0)) {
            isDragging = true;
            double mX, mY; glfwGetCursorPos(target.getWindow(), &mX, &mY);
            dragOffsetX = (int)mX; dragOffsetY = (int)mY;
        }
        if (isDragging && ImGui::IsMouseDown(0)) {
            POINT p; GetCursorPos(&p);
            glfwSetWindowPos(target.getWindow(), p.x - dragOffsetX, p.y - dragOffsetY);
        } else isDragging = false;

        ImGui::SameLine();
        if (ImGui::Button("-", ImVec2(35, 25))) glfwIconifyWindow(target.getWindow());
        ImGui::SameLine();
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.8f, 0.2f, 0.2f, 1.0f));
        if (ImGui::Button("X", ImVec2(35, 25))) glfwSetWindowShouldClose(target.getWindow(), 1);
        ImGui::PopStyleColor();
        ImGui::End();
        ImGui::PopStyleVar();

        // [SAĞ PANEL]
        ImGui::SetNextWindowSize(ImVec2(350, io.DisplaySize.y - 70), ImGuiCond_Always);
        ImGui::SetNextWindowPos(ImVec2(io.DisplaySize.x - 365, 50), ImGuiCond_Always);
        ImGui::Begin("SircaKontrol", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove);
        ImGui::Text("Siber-Optik Arayuz");
        ImGui::Separator();
        ImGui::Text("Motor Durumu: Stabil");
        ImGui::Text("Parcacik Sayisi: 5000");
        ImGui::End();

        // ==========================================
        // [RENDER DÖNGÜSÜ VE KIVILCIM FİZİĞİ]
        // ==========================================
        timeTracker += 0.016f;

        // 1. Sahnemizi (Kristali) RayTracer ile VRAM'e çiz
        graphicsRenderer.render(visionEngine.getDeviceData(), myScene, timeTracker);

        // 2. Parçacık fiziğini hesapla (Konumları güncelle)
        kivilcimSistemi.update(0.016f);

        // 3. VRAM'i al, görüntü katmanlarını birleştir
        unsigned char* d_pbo = target.mapVRAM();
        if (d_pbo) {
            // Kristali uchar formatına geçir
            visionEngine.copyToDeviceUchar(d_pbo);

            // Kıvılcımları PBO üzerine (Kristalin önüne) çiz
            kivilcimSistemi.draw(d_pbo, width, height);
        }
        target.unmapAndRender();

        // ImGui Arayüzünü en üste bas
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(target.getWindow());
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    return 0;
}