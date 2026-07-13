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
#include "Compute/ParticleSystem/ParticleSystem.cuh"

// YENİ UI SİSTEMİ
#include "UI/MainUI.h"

// STB IMAGE (Görsel okumak için)
#include <stb_image.h>

int main() {
    std::cout << "[Kivilcim] Adim 1: Siber-Arayuz Penceresi Olusturuluyor...\n";
    int width = 1366, height = 768, channels = 3;
    GlfwInteropTarget target(width, height, channels, "Kivilcim - Sirca UI");

    // ==========================================
    // ÖZEL KIVILCIM İMLECİ (HARDWARE CURSOR) YÜKLEMESİ
    // ==========================================
    int cursorWidth, cursorHeight, cursorChannels;
    unsigned char* cursorPixels = stbi_load("lib-assets/cursor/classic20.png", &cursorWidth, &cursorHeight, &cursorChannels, 4);

    if (cursorPixels) {
        GLFWimage image;
        image.width = cursorWidth;
        image.height = cursorHeight;
        image.pixels = cursorPixels;

        int hotX = 0;
        int hotY = 0;

        GLFWcursor* customCursor = glfwCreateCursor(&image, hotX, hotY);
        glfwSetCursor(target.getWindow(), customCursor);

        stbi_image_free(cursorPixels);
        std::cout << "[Kivilcim] Ozel imlec (Cursor) donanima yuklendi.\n";
    } else {
        std::cerr << "[Kivilcim] HATA: Cursor PNG dosyasi bulunamadi (assets-graphics/ui/custom_cursor.png)!\n";
    }

    std::cout << "[Kivilcim] Adim 2: CUDA Donanimi Uyandiriliyor...\n";
    cudaSetDevice(0);
    cudaFree(0);

    EngineFactory visionEngine(width, height, channels);
    Renderer3D graphicsRenderer(width, height, channels);

    std::cout << "[Kivilcim] Adim 3: 3D Sahne ve Materyaller Yukleniyor...\n";
    Scene myScene = SceneBuilder::build("assets-graphics/scenes/scene_ui.kvlcm");

    std::cout << "[Kivilcim] Adim 3.5: Parcacik Fiziği Baslatiliyor...\n";
    ParticleSystem kivilcimSistemi(50);

    cudaMemset(visionEngine.getDeviceData(), 0, width * height * channels * sizeof(float));

    std::cout << "[Kivilcim] Adim 4: ImGui Sirca Arayuzu Kuruluyor...\n";
    MainUI sircaUI(target.getWindow());

    // ==========================================
    // IMGUI YAPILANDIRMASI: VIEWPORT VE DOCKING
    // ==========================================
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NoMouseCursorChange;
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;     // Pencereleri kenetlemeyi aç
    io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;   // YENİ: Çoklu pencere desteğini aç
    // ==========================================

    float timeTracker = 0.0f;
    std::cout << "[Kivilcim] Motor Aktif. Sistem hazir.\n";

    double lastTime = glfwGetTime();
    int frameCount = 0;

    unsigned char* d_render_canvas;
    size_t canvasSize = width * height * channels * sizeof(unsigned char);
    cudaMalloc(&d_render_canvas, canvasSize);

    // --- ANA DÖNGÜ ---
    while (!target.shouldClose()) {
        glfwPollEvents();

        // 1. UI Yeni Kare
        sircaUI.newFrame();
        sircaUI.renderPanels();

        // 2. Motor Fiziği
        timeTracker += 0.016f;
        kivilcimSistemi.update(0.016f, timeTracker);

        // 3. RENDER PIPELINE (Çizim Hattı)
        graphicsRenderer.render(visionEngine.getDeviceData(), myScene, timeTracker);
        visionEngine.copyToDeviceUchar(d_render_canvas);
        kivilcimSistemi.draw(d_render_canvas, width, height);

        // E. SONUÇLARI EKRANA BASMA (PBO)
        unsigned char* d_pbo = target.mapVRAM();
        if (d_pbo) {
            cudaMemcpy(d_pbo, d_render_canvas, canvasSize, cudaMemcpyDeviceToDevice);
        }
        target.unmapAndRender();

        // 4. UI Verisini PBO'nun Üstüne Bas
        sircaUI.renderDrawData();

        // ==========================================
        // YENİ: VIEWPORT (ALT PENCERELERİ) ÇİZİM MOTORU
        // ==========================================
        if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
            GLFWwindow* backup_current_context = glfwGetCurrentContext();
            ImGui::UpdatePlatformWindows();
            ImGui::RenderPlatformWindowsDefault();
            glfwMakeContextCurrent(backup_current_context);
        }

        glfwSwapBuffers(target.getWindow());

        // 5. FPS VE VRAM TERMİNAL BİLGİLENDİRMESİ
        double currentTime = glfwGetTime();
        frameCount++;
        if (currentTime - lastTime >= 1.0) {
            size_t free_byte, total_byte;
            cudaMemGetInfo(&free_byte, &total_byte);

            double free_db = (double)free_byte / (1024.0 * 1024.0);
            double total_db = (double)total_byte / (1024.0 * 1024.0);
            double used_db = total_db - free_db;

            std::cout << "\r[Kivilcim] FPS: " << frameCount
                      << " | VRAM Kullanim: " << (int)used_db << " MB / " << (int)total_db << " MB   " << std::flush;

            frameCount = 0;
            lastTime = currentTime;
        }
    }

    cudaFree(d_render_canvas);
    std::cout << "\n[Kivilcim] Motor guvenli bir sekilde kapatiliyor...\n";

    return 0;
}