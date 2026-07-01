#include <iostream>
#include <cuda_runtime.h>

#ifdef _WIN32
#include <windows.h>
extern "C" {
    // NVIDIA ekran kartlarını zorlamak için:
    __declspec(dllexport) unsigned long NvOptimusEnablement = 0x00000001;
    // AMD Dahili GPU'lu sistemlerde NVIDIA'yı tetiklemek için:
    __declspec(dllexport) int AmdPowerXpressRequestHighPerformance = 1;
}
#endif

// MOTOR BAŞLIK DOSYALARI
#include "EngineFactory/EngineFactory.cuh"
#include "Graphics/Renderer3D.cuh"
#include "io/GlfwInteropTarget.h"
#include "Graphics/Scene.cuh"
#include "io/Graphics/SceneBuilder.h"

// DEAR IMGUI BAŞLIK DOSYALARI
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

int main() {
    std::cout << "[Kivilcim] Adim 1: Siber-Arayuz Penceresi Olusturuluyor...\n";
    int width = 1366;
    int height = 768;
    int channels = 3;

    // 1. ÖNCE OPENGL PENCERESİ OLUŞTURULMALI (CUDA bu köprüye bağlanacak)
    GlfwInteropTarget target(width, height, channels, "Kivilcim - Sirca UI");

    std::cout << "[Kivilcim] Adim 2: CUDA Donanimi Uyandiriliyor...\n";
    // 2. SONRA CUDA UYANDIRILMALI
    cudaSetDevice(0);
    cudaFree(0);

    EngineFactory visionEngine(width, height, channels);
    Renderer3D graphicsRenderer(width, height, channels);

    std::cout << "[Kivilcim] Adim 3: 3D Sahne ve Materyaller Yukleniyor...\n";
    // 3. SAHNE İNŞASI (CrystalSpark ve ui_panel.kvlcm burada devreye giriyor)
    Scene myScene = SceneBuilder::build("assets-graphics/scenes/scene_ui.kvlcm");

    cudaMemset(visionEngine.getDeviceData(), 0, width * height * channels * sizeof(float));

    std::cout << "[Kivilcim] Adim 4: ImGui Sirca Stili Uygulaniyor...\n";
    // 4. IMGUI KURULUMU
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();

    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowRounding = 12.0f;
    style.FrameRounding = 6.0f;
    style.WindowBorderSize = 0.0f;

    ImGui_ImplGlfw_InitForOpenGL(target.getWindow(), true);
    ImGui_ImplOpenGL3_Init("#version 130");

    // Sürükleme ve Animasyon değişkenleri
    bool isDragging = false;
    int dragOffsetX = 0;
    int dragOffsetY = 0;
    float timeTracker = 0.0f; // Shader efektleri için zaman sayacı

    std::cout << "[Kivilcim] Guc Aktif. Motor calisiyor!\n";

    while (!target.shouldClose()) {
        glfwPollEvents();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGuiIO& io = ImGui::GetIO();

        // ==========================================
        // ÜST PANEL (TOP BAR) & SÜRÜKLEME ALANI
        // ==========================================
        ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(io.DisplaySize.x, 35.0f), ImGuiCond_Always);

        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
        ImGuiWindowFlags topbar_flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                                        ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoMove |
                                        ImGuiWindowFlags_NoScrollbar;

        ImGui::Begin("UstPanel", nullptr, topbar_flags);
        ImGui::SetCursorPosY(8.0f);
        ImGui::Text("  KIVILCIM ENGINEX - Sirca UI");
        ImGui::SameLine();

        float dragAreaWidth = io.DisplaySize.x - ImGui::GetCursorPosX() - 95.0f;
        if (dragAreaWidth < 10.0f) dragAreaWidth = 10.0f;

        ImGui::InvisibleButton("DragArea", ImVec2(dragAreaWidth, 35.0f));

        if (ImGui::IsItemHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
            isDragging = true;
            double mouseX, mouseY;
            glfwGetCursorPos(target.getWindow(), &mouseX, &mouseY);
            dragOffsetX = (int)mouseX;
            dragOffsetY = (int)mouseY;
        }

        if (isDragging) {
            if (ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
#ifdef _WIN32
                POINT p;
                if (GetCursorPos(&p)) {
                    glfwSetWindowPos(target.getWindow(), p.x - dragOffsetX, p.y - dragOffsetY);
                }
#endif
            } else {
                isDragging = false;
            }
        }

        ImGui::SameLine();
        ImGui::SetCursorPosY(5.0f);

        if (ImGui::Button("-", ImVec2(35, 25))) {
            glfwIconifyWindow(target.getWindow());
        }
        ImGui::SameLine();

        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.8f, 0.2f, 0.2f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(1.0f, 0.3f, 0.3f, 1.0f));
        if (ImGui::Button("X", ImVec2(35, 25))) {
            glfwSetWindowShouldClose(target.getWindow(), GLFW_TRUE);
        }
        ImGui::PopStyleColor(2);

        ImGui::End();
        ImGui::PopStyleVar();

        // ==========================================
        // SAĞ KONTROL PANELİ
        // ==========================================
        float panelGenislik = 350.0f;
        float panelYukseklik = io.DisplaySize.y - 70.0f;
        if (panelYukseklik < 100.0f) panelYukseklik = 100.0f;

        float x_poz = io.DisplaySize.x - panelGenislik - 15.0f;
        float y_poz = 50.0f;

        ImGui::SetNextWindowSize(ImVec2(panelGenislik, panelYukseklik), ImGuiCond_Always);
        ImGui::SetNextWindowPos(ImVec2(x_poz, y_poz), ImGuiCond_Always);

        ImGuiWindowFlags sirca_flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                                       ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoMove;

        ImGui::Begin("SircaKontrol", nullptr, sirca_flags);
        ImGui::Text("Siber-Optik Arayuz");
        ImGui::Separator();
        ImGui::Text("Motor Durumu: Stabil");
        ImGui::End();

        // ==========================================
        // KIVILCIM ÇİZİM VE EKRANA BASMA
        // ==========================================

        // Zamanı akıtarak arkadaki Liquid (Sıvı) shader'ını canlandırıyoruz
        timeTracker += 0.016f;

        // 1. Sahnemizi (CrystalSpark) zaman değişkeniyle VRAM'e çiz
        graphicsRenderer.render(visionEngine.getDeviceData(), myScene, timeTracker);

        // 2. VRAM'i al ve güvenli bir şekilde PBO'ya geçir
        unsigned char* d_pbo_vram_address = target.mapVRAM();
        if (d_pbo_vram_address != nullptr) {
            visionEngine.copyToDeviceUchar(d_pbo_vram_address);
        }
        target.unmapAndRender();

        // 3. ImGui Sırça Arayüzünü sahnemizin tam üzerine yerleştir
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        // Tüm katmanları birleştir ve ekrana bas
        glfwSwapBuffers(target.getWindow());
    }

    std::cout << "[Kivilcim] Temizlik yapiliyor, motor kapandi.\n";
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    return 0;
}