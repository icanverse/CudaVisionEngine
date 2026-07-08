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
// EĞER projende başka bir yerde "#define STB_IMAGE_IMPLEMENTATION" kullanmadıysan,
// bu satırın hemen üstüne o define'ı eklemen gerekebilir.
#include <stb_image.h>

int main() {
    std::cout << "[Kivilcim] Adim 1: Siber-Arayuz Penceresi Olusturuluyor...\n";
    int width = 1366, height = 768, channels = 3;
    GlfwInteropTarget target(width, height, channels, "Kivilcim - Sirca UI");

    // ==========================================
    // ÖZEL KIVILCIM İMLECİ (HARDWARE CURSOR) YÜKLEMESİ
    // ==========================================
    int cursorWidth, cursorHeight, cursorChannels;

    // NOT: İmleç PNG dosyanı bu yola koymalısın (Tercihen 32x32 boyutlarında)
    unsigned char* cursorPixels = stbi_load("lib-assets/cursor/classic20.png", &cursorWidth, &cursorHeight, &cursorChannels, 4);

    if (cursorPixels) {
        GLFWimage image;
        image.width = cursorWidth;
        image.height = cursorHeight;
        image.pixels = cursorPixels;

        // Tıklama noktasını (Hotspot) belirliyoruz (Sol üst köşe için 0,0)
        int hotX = 0;
        int hotY = 0;

        // GLFW cursor'ını oluştur ve hedef pencereye bağla
        GLFWcursor* customCursor = glfwCreateCursor(&image, hotX, hotY);
        glfwSetCursor(target.getWindow(), customCursor);

        stbi_image_free(cursorPixels); // RAM'i temizle
        std::cout << "[Kivilcim] Ozel imlec (Cursor) donanima yuklendi.\n";
    } else {
        std::cerr << "[Kivilcim] HATA: Cursor PNG dosyasi bulunamadi (assets-graphics/ui/custom_cursor.png)!\n";
    }
    // ==========================================

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
    // UI Sınıfımızı Başlatıyoruz
    MainUI sircaUI(target.getWindow());

    // ==========================================
    // IMGUI İMLEÇ KONTROLÜNÜ DEVRE DIŞI BIRAK
    // ==========================================
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NoMouseCursorChange;
    // ==========================================

    float timeTracker = 0.0f;
    std::cout << "[Kivilcim] Motor Aktif. Sistem hazir.\n";


    // >>> FPS ve VRAM Takibi İçin Değişkenler
    double lastTime = glfwGetTime();
    int frameCount = 0;

    // >>> Post-Process İçin Geçici Tuval (Buffer)
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
        timeTracker += 0.016f; // (İleride bunu sabit yerine deltaTime ile değiştirebilirsin)
        kivilcimSistemi.update(0.016f, timeTracker);

        // 3. RENDER PIPELINE (Çizim Hattı)

        // A. 3D Sahneyi visionEngine'in kendi Float belleğine (d_data) çiz
        graphicsRenderer.render(visionEngine.getDeviceData(), myScene, timeTracker);

        // B. 3D Sahneyi 8-bit'e çevirip "Geçici Tuval"e kopyala
        visionEngine.copyToDeviceUchar(d_render_canvas);

        // C. Parçacıkları (Ateş böceklerini) Geçici Tuvalin üstüne ekle (Additive Blend)
        kivilcimSistemi.draw(d_render_canvas, width, height);

        // D. POST-PROCESS (Kabul Köprüsü)
        // Parçacıklarla birleşmiş o güzel tuvali al, fabrikaya sok, işle!
        /* visionEngine.loadFromVRAM(d_render_canvas)
                    .applySmoothing2D(5)         // Örnek: Hafif Glow efekti
                    .applyLogTransformation();   // Örnek: Renkleri sinematik yap
        */

        // E. SONUÇLARI EKRANA BASMA (PBO)
        unsigned char* d_pbo = target.mapVRAM();
        if (d_pbo) {
            // Eğer Post-Process kapalıysa, doğrudan tuvali ekrana gönder:
            cudaMemcpy(d_pbo, d_render_canvas, canvasSize, cudaMemcpyDeviceToDevice);
        }
        target.unmapAndRender();

        // 4. UI Verisini PBO'nun Üstüne Bas
        sircaUI.renderDrawData();

        glfwSwapBuffers(target.getWindow());

        // ==========================================
        // 5. FPS VE VRAM TERMİNAL BİLGİLENDİRMESİ
        // ==========================================
        double currentTime = glfwGetTime();
        frameCount++;
        if (currentTime - lastTime >= 1.0) { // Her 1 saniyede bir ekranı güncelle
            size_t free_byte, total_byte;
            cudaMemGetInfo(&free_byte, &total_byte);

            // Baytları Megabayta (MB) çeviriyoruz
            double free_db = (double)free_byte / (1024.0 * 1024.0);
            double total_db = (double)total_byte / (1024.0 * 1024.0);
            double used_db = total_db - free_db;

            // '\r' karakteri terminalde yeni satıra geçmeden aynı satırı temizleyip üzerine yazmayı sağlar.
            std::cout << "\r[Kivilcim] FPS: " << frameCount
                      << " | VRAM Kullanim: " << (int)used_db << " MB / " << (int)total_db << " MB   " << std::flush;

            frameCount = 0;
            lastTime = currentTime;
        }
    }

    // Döngü bittiğinde bellek sızıntısını (Memory Leak) önlemek için geçici tuvali temizle
    cudaFree(d_render_canvas);
    std::cout << "\n[Kivilcim] Motor guvenli bir sekilde kapatiliyor...\n";

    // MainUI'nin Yıkıcısı (Destructor) kapanış işlerini otomatik halleder.

    return 0;
}