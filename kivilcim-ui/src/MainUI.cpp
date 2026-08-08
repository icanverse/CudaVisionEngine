#include "MainUI.h"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "TextureUtility/TextureUtility.h"
#include <iostream>

#include "AssetsManager/IconManager.h"
#include "Cuda/HardwareDetector.h"
#include "Persistence/KvlcmSerializer.h"
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
    // AKILLI DONANIM TESPİTİ VE AYAR YÜKLEME
    // ==========================================
    const std::string prefPath = "C:/Users/Can/Desktop/user_prefs.kvlcm-user-pref";

    // Önce diskten ayarları okumayı deniyoruz[cite: 17]
    if (!Kivilcim::KvlcmSerializer::loadPreferences(prefPath, userPrefs)) {
        std::cout << "[Sirca UI] Ayar dosyasi bulunamadi. Ilk calistirma icin donanim taraniyor...\n";

        // Donanımı sadece bu ilk açılışta tarıyoruz[cite: 17]
        Kcore::HardwareInfoData hwInfo = Kcore::HardwareDetector::inspectSystem(windowHandle);

        // Sisteme özel optimum ayarları (RAM/VRAM limitlerini) belirliyoruz[cite: 17]
        userPrefs.ram_limit = static_cast<int>(hwInfo.totalSysRAM / 2); // Sistemin yarısı[cite: 17]

        if (hwInfo.dedicatedVRAM > 0) {
            userPrefs.vram_limit = static_cast<int>(hwInfo.dedicatedVRAM);
        } else {
            userPrefs.vram_limit = 4096; // Güvenli varsayılan[cite: 17]
        }

        userPrefs.enableHardwareAcceleration = hwInfo.cudaAvailable;
        userPrefs.enableHardwareCuda = hwInfo.cudaAvailable;

        // ==========================================
        // DONANIM BİLGİLERİNİ ÖNBELLEĞE YAZ
        // ==========================================
        userPrefs.hw_cpuModel = hwInfo.cpuModel;
        userPrefs.hw_gpuModel = hwInfo.gpuModel;
        userPrefs.hw_totalRamMB = static_cast<int>(hwInfo.totalSysRAM);
        userPrefs.hw_totalVramMB = static_cast<int>(hwInfo.dedicatedVRAM);

        // Eğer CUDA varsa ilk cihazın çekirdek sayısını alalım
        if (hwInfo.cudaAvailable && !hwInfo.cudaDevices.empty()) {
            userPrefs.hw_cudaCores = hwInfo.cudaDevices[0].totalCudaCores;
        } else {
            userPrefs.hw_cudaCores = 0;
        }

        userPrefs.isPreferencesChanged = true;

        // Tarama bittikten sonra bu ideal ayarları diske kaydediyoruz ki bir daha tarama yapmasın[cite: 17]
        std::vector<Kdata::PreferenceData> prefsToSave = { userPrefs };
        Kivilcim::KvlcmSerializer::savePreferences(".kvlcm-user-pref", prefPath, prefsToSave);

        std::cout << "[Sirca UI] Donanim tespit edildi ve optimum ayarlar kaydedildi.\n";
    } else {
        std::cout << "[Sirca UI] Ayarlar diskten basariyla yuklendi.\n";
    }

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
    leftPanel.setOnProjectDoubleClickedCallback([this](const int& projectID) {
        Kdata::ProjectData* p = leftPanel.getProjectByID(projectID);
        if (p) {
            appState.resetState();

            // Verileri State'e kopyala
            appState.project = *p;

            // ==========================================
            // YÜKSEK ÇÖZÜNÜRLÜKLÜ GÖRSELİ YÜKLE
            // ==========================================
            if (!appState.project.imagePath.empty() &&
                appState.project.textureID == 0)
            {
                std::cout << "[Sirca UI] Full Resolution Texture Yukleniyor : "
                          << appState.project.imagePath << "\n";

                appState.project.textureID =
                    TextureUtility::LoadTextureFromFile(
                        appState.project.imagePath.c_str());

                if (appState.project.textureID == 0)
                {
                    std::cout << "[Sirca UI - HATA] Full Resolution Texture Yuklenemedi.\n";
                }
            }

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