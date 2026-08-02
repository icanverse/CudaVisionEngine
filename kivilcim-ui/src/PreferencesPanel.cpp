#include "../include/PreferencesPanel.h"
#include "imgui.h"

#include "../../kivilcim-graphics/include/Shaders/LiquidShader.cuh"
#include "Persistence/KvlcmSerializer.h"

#include <cstring>
#include <cstdlib>

#ifdef _WIN32
#include <windows.h> // ÖNCE windows.h GELMELİDİR![cite: 18]
#include <shellapi.h>
#endif

namespace Kivilcim {
    namespace UI {

        PreferencesPanel::PreferencesPanel() {
            bgShaderTexture = new CudaDynamicTexture(512, 288); //[cite: 18]
            flowTime = 0.0f; //[cite: 18]
        }

        PreferencesPanel::~PreferencesPanel() {
            if (bgShaderTexture) {
                delete bgShaderTexture;
                bgShaderTexture = nullptr;
            }
        }

        void PreferencesPanel::render(bool& isOpen, Kdata::PreferenceData& userPrefs) {
            if (!isOpen) return;

            // Pencere Boyutu ve Ortalama
            ImGui::SetNextWindowSize(ImVec2(800.0f, 600.0f), ImGuiCond_FirstUseEver); //[cite: 18]
            ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_FirstUseEver, ImVec2(0.5f, 0.5f)); //[cite: 18]

            if (ImGui::IsWindowAppearing()) {
                ImGui::SetWindowFocus("Kivilcim Ayarlari");
            }

            ImGuiWindowFlags windowFlags =
                ImGuiWindowFlags_NoDocking |
                ImGuiWindowFlags_NoCollapse |
                ImGuiWindowFlags_NoSavedSettings; //[cite: 18]

            ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.0f, 0.0f, 0.0f, 0.0f)); //[cite: 18]
            ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0.85f, 0.45f, 0.0f, 0.0f)); //[cite: 18]

            ImGui::Begin("Kivilcim Ayarlari", &isOpen, windowFlags); //[cite: 18]

            // ==========================================
            // 1. CANLI LİKİT SHADER ARKA PLAN ÇİZİMİ
            // ==========================================
            ImDrawList* drawList = ImGui::GetWindowDrawList();

            // Çarpanı artırarak shader'ın çok daha canlı ve hızlı akmasını sağlıyoruz
            flowTime += ImGui::GetIO().DeltaTime * 1.8f; //[cite: 18]

            cudaSurfaceObject_t surface = bgShaderTexture->map(); //[cite: 18]

            Kivilcim::Shaders::launchLiquidFlowShader(
                surface,
                bgShaderTexture->getWidth(),
                bgShaderTexture->getHeight(),
                flowTime,
                1.75f, // scale biraz daha büyütüldü[cite: 18]
                2.2f, // speed canlandırıldı[cite: 18]
                make_float3(0.85f, 0.45f, 0.0f) //[cite: 18]
            );

            bgShaderTexture->unmap();

            ImVec2 winPos = ImGui::GetWindowPos(); //[cite: 18]
            ImVec2 winSize = ImGui::GetWindowSize(); //[cite: 18]
            drawList->AddImage(
                (ImTextureID)(intptr_t)bgShaderTexture->getTextureID(),
                winPos,
                ImVec2(winPos.x + winSize.x, winPos.y + winSize.y)
            );

            // Yazıların okunabilirliği için koyu yarı şeffaf katman
            drawList->AddRectFilled(winPos, ImVec2(winPos.x + winSize.x, winPos.y + winSize.y), IM_COL32(15, 15, 18, 220)); //[cite: 18]

            // ==========================================
            // 2. SEKMELİ YAPI VE "MERHABA" SEKMESİ
            // ==========================================
            ImGui::Dummy(ImVec2(0.0f, 10.0f));
            bool isChanged = false;

            ImGui::PushStyleColor(ImGuiCol_Tab, ImVec4(0.12f, 0.12f, 0.14f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_TabHovered, ImVec4(0.85f, 0.45f, 0.0f, 0.6f));
            ImGui::PushStyleColor(ImGuiCol_TabSelected, ImVec4(0.85f, 0.45f, 0.0f, 1.0f));

            if (ImGui::BeginTabBar("PreferencesTabs")) {

                // ------------------------------------------
                // SEKME 0: MERHABA (Hakkında & Donanım & Optimize)
                // ------------------------------------------
                if (ImGui::BeginTabItem(" Merhaba ")) {
                    ImGui::Dummy(ImVec2(0.0f, 15.0f));
                    ImGui::TextColored(ImVec4(1.0f, 0.7f, 0.2f, 1.0f), "KIVILCİM VİZYON MOTORUNA HOŞ GELDİNİZ!"); //[cite: 18]
                    ImGui::Separator();
                    ImGui::Dummy(ImVec2(0.0f, 10.0f));

                    ImGui::TextWrapped(
                        "Bu yazılım, yüksek performanslı CUDA ve modern C++ mimarisiyle "
                        "geliştirilmiş yeni nesil bir görsel işleme ve katman yönetim motorudur. "
                        "Arka planda akıp giden bu özel likit shader, motorun grafik gücünü simgelemektedir." //[cite: 18]
                    );

                    ImGui::Dummy(ImVec2(0.0f, 10.0f));

                    // TIKLANABİLİR BAĞLANTI (LINK)
                    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.3f, 0.6f, 1.0f, 1.0f));
                    ImGui::Text("-> Resmi GitHub Repository'sini Ziyaret Et"); //[cite: 18]
                    ImGui::PopStyleColor();
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
                        if (ImGui::IsItemClicked()) {
#ifdef _WIN32
                            ShellExecuteA(NULL, "open", "https://github.com", NULL, NULL, SW_SHOWNORMAL); //[cite: 18]
#endif
                        }
                    }

                    // --- YENİ BÖLÜM: DONANIM LİSTELEME ---
                    ImGui::Dummy(ImVec2(0.0f, 25.0f));
                    ImGui::TextColored(ImVec4(1.0f, 0.7f, 0.2f, 1.0f), "SİSTEM DONANIMINIZ");
                    ImGui::Separator();
                    ImGui::Dummy(ImVec2(0.0f, 10.0f));

                    float labelWidth = 160.0f;

                    // İşlemci
                    ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "İşlemci (CPU):");
                    ImGui::SameLine(labelWidth);
                    ImGui::Text("%s", userPrefs.hw_cpuModel.c_str());

                    // Ekran Kartı
                    ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Grafik Kartı (GPU):");
                    ImGui::SameLine(labelWidth);
                    ImGui::Text("%s", userPrefs.hw_gpuModel.c_str());

                    // CUDA Çekirdekleri
                    if (userPrefs.hw_cudaCores > 0) {
                        ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "CUDA Çekirdekleri:");
                        ImGui::SameLine(labelWidth);
                        ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f), "%d Core", userPrefs.hw_cudaCores);
                    }

                    // RAM ve VRAM
                    ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Sistem Belleği (RAM):");
                    ImGui::SameLine(labelWidth);
                    ImGui::Text("%d MB", userPrefs.hw_totalRamMB);

                    ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Video Belleği (VRAM):");
                    ImGui::SameLine(labelWidth);
                    ImGui::Text("%d MB", userPrefs.hw_totalVramMB);

                    // --- YENİ BÖLÜM: OPTİMİZE BUTONU ---
                    ImGui::Dummy(ImVec2(0.0f, 20.0f));

                    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.5f, 0.2f, 1.0f)); // Yeşil ton
                    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.7f, 0.3f, 1.0f));
                    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.15f, 0.4f, 0.15f, 1.0f));

                    if (ImGui::Button("Sistemi Optimize Et", ImVec2(200.0f, 40.0f))) {
                        // RAM Limitini sistemin yarısına çek
                        userPrefs.ram_limit = (userPrefs.hw_totalRamMB > 0) ? (userPrefs.hw_totalRamMB / 2) : 8192;

                        // VRAM Limitini sahip olunan tüm ayrılmış belleğe eşitle
                        userPrefs.vram_limit = (userPrefs.hw_totalVramMB > 0) ? userPrefs.hw_totalVramMB : 4096;

                        // CUDA varsa motorları aktifleştir
                        if (userPrefs.hw_cudaCores > 0) {
                            userPrefs.enableHardwareAcceleration = true;
                            userPrefs.enableHardwareCuda = true;
                        }

                        userPrefs.enableSharedMemory = true;
                        isChanged = true;
                    }
                    ImGui::PopStyleColor(3);

                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("Donanımınızın gücünü analiz ederek bellek ve ivmelendirme ayarlarını maksimum performansa ayarlar.");
                    }

                    ImGui::EndTabItem();
                }

                // ------------------------------------------
                // SEKME 1: GENEL VE KULLANICI AYARLARI
                // ------------------------------------------
                if (ImGui::BeginTabItem(" Genel ")) { //[cite: 18]
                    ImGui::Dummy(ImVec2(0.0f, 10.0f));
                    ImGui::TextColored(ImVec4(1.0f, 0.7f, 0.2f, 1.0f), "KULLANICI BILGILERI");
                    ImGui::Separator();
                    ImGui::Dummy(ImVec2(0.0f, 5.0f));

                    char nameBuf[128];
                    strncpy(nameBuf, userPrefs.userName.c_str(), sizeof(nameBuf));
                    nameBuf[sizeof(nameBuf) - 1] = '\0';
                    ImGui::PushItemWidth(250.0f);
                    if (ImGui::InputText("Kullanici Adi", nameBuf, sizeof(nameBuf))) {
                        userPrefs.userName = std::string(nameBuf);
                        isChanged = true;
                    }
                    ImGui::PopItemWidth();

                    const char* langs[] = { "tr_TR", "en_US" };
                    int currentLangIdx = (userPrefs.language == "tr_TR") ? 0 : 1;
                    ImGui::PushItemWidth(150.0f);
                    if (ImGui::Combo("Arayuz Dili", &currentLangIdx, langs, IM_ARRAYSIZE(langs))) {
                        userPrefs.language = (currentLangIdx == 0) ? "tr_TR" : "en_US";
                        isChanged = true;
                    }
                    ImGui::PopItemWidth();

                    ImGui::Dummy(ImVec2(0.0f, 15.0f));
                    ImGui::TextColored(ImVec4(1.0f, 0.7f, 0.2f, 1.0f), "MOTOR VE OTOMATIK KAYIT");
                    ImGui::Separator();
                    ImGui::Dummy(ImVec2(0.0f, 5.0f));

                    if (ImGui::Checkbox("Otomatik Kayit (Auto-Save) Aktif", &userPrefs.enableAutoSave)) isChanged = true;

                    if (userPrefs.enableAutoSave) {
                        ImGui::PushItemWidth(200.0f);
                        if (ImGui::SliderInt("Kayit Araligi (Dakika)", &userPrefs.autoSaveIntervalMinutes, 1, 60)) isChanged = true;
                        ImGui::PopItemWidth();
                    }

                    ImGui::EndTabItem();
                }

                // ------------------------------------------
                // SEKME 2: DONANIM VE PERFORMANS
                // ------------------------------------------
                if (ImGui::BeginTabItem(" Performans ")) { //[cite: 18]
                    ImGui::Dummy(ImVec2(0.0f, 10.0f));
                    ImGui::TextColored(ImVec4(1.0f, 0.7f, 0.2f, 1.0f), "DONANIM HIZLANDIRMASI");
                    ImGui::Separator();
                    ImGui::Dummy(ImVec2(0.0f, 5.0f));

                    if (ImGui::Checkbox("Donanim Hizlandirmasi Kullan", &userPrefs.enableHardwareAcceleration)) isChanged = true;
                    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Ekran karti (GPU) uzerinden islem yapilmasini saglar.");

                    if (userPrefs.enableHardwareAcceleration) {
                        ImGui::Indent(20.0f);
                        if (ImGui::Checkbox("NVIDIA CUDA Motoru", &userPrefs.enableHardwareCuda)) isChanged = true;
                        if (ImGui::Checkbox("OpenCL Motoru", &userPrefs.enableHardwareOpenCL)) isChanged = true;
                        ImGui::Unindent(20.0f);
                    } else {
                        userPrefs.enableHardwareCPU = true;
                        ImGui::Indent(20.0f);
                        ImGui::TextDisabled("Sadece CPU Modu Aktif (Yavas islem)");
                        ImGui::Unindent(20.0f);
                    }

                    ImGui::Dummy(ImVec2(0.0f, 15.0f));
                    ImGui::TextColored(ImVec4(1.0f, 0.7f, 0.2f, 1.0f), "BELLEK YONETIMI VE LIMITLER");
                    ImGui::Separator();
                    ImGui::Dummy(ImVec2(0.0f, 5.0f));

                    if (ImGui::Checkbox("Paylasimli Bellek (Shared Memory) Kullanimi", &userPrefs.enableSharedMemory)) isChanged = true;

                    ImGui::Dummy(ImVec2(0.0f, 10.0f));
                    ImGui::PushItemWidth(300.0f);
                    if (ImGui::SliderInt("Sistem RAM Limiti (MB)", &userPrefs.ram_limit, 1024, 65536)) isChanged = true;
                    if (ImGui::SliderInt("VRAM Limiti (MB)", &userPrefs.vram_limit, 512, 32768)) isChanged = true;
                    ImGui::PopItemWidth();

                    ImGui::EndTabItem();
                }

                // ------------------------------------------
                // SEKME 3: DOSYA YOLLARI VE YÖNETİM
                // ------------------------------------------
                if (ImGui::BeginTabItem(" Konumlar ")) { //[cite: 18]
                    ImGui::Dummy(ImVec2(0.0f, 10.0f));
                    ImGui::TextColored(ImVec4(1.0f, 0.7f, 0.2f, 1.0f), "ONBELLEK VE CIKTI");
                    ImGui::Separator();
                    ImGui::Dummy(ImVec2(0.0f, 5.0f));

                    char cacheBuf[512];
                    strncpy(cacheBuf, userPrefs.cache_path.c_str(), sizeof(cacheBuf));
                    cacheBuf[sizeof(cacheBuf) - 1] = '\0';
                    ImGui::PushItemWidth(450.0f);
                    if (ImGui::InputText("Onbellek (Cache) Klasoru", cacheBuf, sizeof(cacheBuf))) {
                        userPrefs.cache_path = std::string(cacheBuf);
                        isChanged = true;
                    }
                    ImGui::PopItemWidth();

                    ImGui::Dummy(ImVec2(0.0f, 10.0f));

                    char exportBuf[512];
                    strncpy(exportBuf, userPrefs.default_export_path.c_str(), sizeof(exportBuf));
                    exportBuf[sizeof(exportBuf) - 1] = '\0';
                    ImGui::PushItemWidth(450.0f);
                    if (ImGui::InputText("Varsayilan Disa Aktarma", exportBuf, sizeof(exportBuf))) {
                        userPrefs.default_export_path = std::string(exportBuf);
                        isChanged = true;
                    }
                    ImGui::PopItemWidth();

                    ImGui::EndTabItem();
                }

                ImGui::EndTabBar();
            }
            ImGui::PopStyleColor(3);

            if (isChanged) {
                userPrefs.isPreferencesChanged = true; //[cite: 18]
            }

            // ==========================================
            // 3. KAYDET VE KAPAT BUTONLARI
            // ==========================================
            ImGui::SetCursorPosY(winSize.y - 50.0f); //[cite: 18]
            ImGui::Separator();
            ImGui::Dummy(ImVec2(0.0f, 5.0f));

            ImGui::SetCursorPosX(winSize.x - 220.0f); //[cite: 18]

            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.2f, 0.22f, 1.0f)); //[cite: 18]
            if (ImGui::Button("Iptal", ImVec2(90.0f, 30.0f))) {
                isOpen = false;
            }
            ImGui::PopStyleColor();

            ImGui::SameLine();

            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.85f, 0.45f, 0.0f, 1.0f)); //[cite: 18]
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(1.0f, 0.55f, 0.0f, 1.0f)); //[cite: 18]
            if (ImGui::Button("Kaydet", ImVec2(90.0f, 30.0f))) {
                std::vector<Kdata::PreferenceData> prefsToSave = { userPrefs };
                Kivilcim::KvlcmSerializer::savePreferences(".kvlcm-user-pref", "C:/Users/Can/Desktop/user_prefs.kvlcm-user-pref", prefsToSave); //[cite: 18]

                userPrefs.isPreferencesChanged = false; //[cite: 18]
                isOpen = false;
            }
            ImGui::PopStyleColor(2);

            ImGui::End();
            ImGui::PopStyleColor(2);
        }

    }
}