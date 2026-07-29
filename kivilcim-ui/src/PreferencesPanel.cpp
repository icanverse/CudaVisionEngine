#include "../include/PreferencesPanel.h"
#include "imgui.h"

#include "../../kivilcim-graphics/include/Shaders/LiquidShader.cuh"
#include "Persistence/KvlcmSerializer.h"

#include <cstring>
#include <cstdlib>

#ifdef _WIN32
#include <windows.h> // ÖNCE windows.h GELMELİDİR!
#include <shellapi.h>
#endif

namespace Kivilcim {
    namespace UI {

        PreferencesPanel::PreferencesPanel() {
            bgShaderTexture = new CudaDynamicTexture(512, 288);
            flowTime = 0.0f;
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
            ImGui::SetNextWindowSize(ImVec2(800.0f, 600.0f), ImGuiCond_FirstUseEver);
            ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_FirstUseEver, ImVec2(0.5f, 0.5f));

            if (ImGui::IsWindowAppearing()) {
                ImGui::SetWindowFocus("Kivilcim Ayarlari");
            }

            ImGuiWindowFlags windowFlags =
                ImGuiWindowFlags_NoDocking |
                ImGuiWindowFlags_NoCollapse |
                ImGuiWindowFlags_NoSavedSettings;

            ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.0f, 0.0f, 0.0f, 0.0f));
            ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0.85f, 0.45f, 0.0f, 0.0f));

            ImGui::Begin("Kivilcim Ayarlari", &isOpen, windowFlags);

            // ==========================================
            // 1. CANLI LİKİT SHADER ARKA PLAN ÇİZİMİ
            // ==========================================
            ImDrawList* drawList = ImGui::GetWindowDrawList();

            // Çarpanı artırarak shader'ın çok daha canlı ve hızlı akmasını sağlıyoruz
            flowTime += ImGui::GetIO().DeltaTime * 1.8f;

            cudaSurfaceObject_t surface = bgShaderTexture->map();

            Kivilcim::Shaders::launchLiquidFlowShader(
                surface,
                bgShaderTexture->getWidth(),
                bgShaderTexture->getHeight(),
                flowTime,
                1.75f, // scale biraz daha büyütüldü
                2.2f, // speed canlandırıldı
                make_float3(0.85f, 0.45f, 0.0f)
            );

            bgShaderTexture->unmap();

            ImVec2 winPos = ImGui::GetWindowPos();
            ImVec2 winSize = ImGui::GetWindowSize();
            drawList->AddImage(
                (ImTextureID)(intptr_t)bgShaderTexture->getTextureID(),
                winPos,
                ImVec2(winPos.x + winSize.x, winPos.y + winSize.y)
            );

            // Yazıların okunabilirliği için koyu yarı şeffaf katman
            drawList->AddRectFilled(winPos, ImVec2(winPos.x + winSize.x, winPos.y + winSize.y), IM_COL32(15, 15, 18, 220));

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
                // SEKME 0: MERHABA (Hakkında & Bağlantı)
                // ------------------------------------------
                if (ImGui::BeginTabItem(" Merhaba ")) {
                    ImGui::Dummy(ImVec2(0.0f, 15.0f));
                    ImGui::TextColored(ImVec4(1.0f, 0.7f, 0.2f, 1.0f), "KIVILCİM VİZYON MOTORUNA HOŞ GELDİNİZ!");
                    ImGui::Separator();
                    ImGui::Dummy(ImVec2(0.0f, 10.0f));

                    ImGui::TextWrapped(
                        "Bu yazılım, yüksek performanslı CUDA ve modern C++ mimarisiyle "
                        "geliştirilmiş yeni nesil bir görsel işleme ve katman yönetim motorudur. "
                        "Arka planda akıp giden bu özel likit shader, motorun grafik gücünü simgelemektedir."
                    );

                    ImGui::Dummy(ImVec2(0.0f, 20.0f));
                    ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "Proje Kaynağı ve Dokümantasyon:");
                    ImGui::Dummy(ImVec2(0.0f, 5.0f));

                    // TIKLANABİLİR BAĞLANTI (LINK) TASARIMI
                    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.3f, 0.6f, 1.0f, 1.0f)); // Mavi link rengi
                    ImGui::Text("-> Resmi GitHub Repository'sini Ziyaret Et");
                    ImGui::PopStyleColor();

                    // Kullanıcı bu yazının üzerine tıklarsa tarayıcıda açılır
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetMouseCursor(ImGuiMouseCursor_Hand); // Fareyi el işaretine çevir
                        if (ImGui::IsItemClicked()) {
#ifdef _WIN32
                            ShellExecuteA(NULL, "open", "https://github.com", NULL, NULL, SW_SHOWNORMAL);
#endif
                        }
                    }

                    ImGui::EndTabItem();
                }

                // ------------------------------------------
                // SEKME 1: GENEL VE KULLANICI AYARLARI
                // ------------------------------------------
                if (ImGui::BeginTabItem(" Genel ")) {
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
                if (ImGui::BeginTabItem(" Performans ")) {
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
                    if (ImGui::SliderInt("Sistem RAM Limiti (MB)", &userPrefs.ram_limit, 1024, 32768)) isChanged = true;
                    if (ImGui::SliderInt("VRAM Limiti (MB)", &userPrefs.vram_limit, 512, 16384)) isChanged = true;
                    ImGui::PopItemWidth();

                    ImGui::EndTabItem();
                }

                // ------------------------------------------
                // SEKME 3: DOSYA YOLLARI VE YÖNETİM
                // ------------------------------------------
                if (ImGui::BeginTabItem(" Konumlar ")) {
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
                userPrefs.isPreferencesChanged = true;
            }

            // ==========================================
            // 3. KAYDET VE KAPAT BUTONLARI
            // ==========================================
            ImGui::SetCursorPosY(winSize.y - 50.0f);
            ImGui::Separator();
            ImGui::Dummy(ImVec2(0.0f, 5.0f));

            ImGui::SetCursorPosX(winSize.x - 220.0f);

            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.2f, 0.22f, 1.0f));
            if (ImGui::Button("Iptal", ImVec2(90.0f, 30.0f))) {
                isOpen = false;
            }
            ImGui::PopStyleColor();

            ImGui::SameLine();

            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.85f, 0.45f, 0.0f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(1.0f, 0.55f, 0.0f, 1.0f));
            if (ImGui::Button("Kaydet", ImVec2(90.0f, 30.0f))) {
                std::vector<Kdata::PreferenceData> prefsToSave = { userPrefs };
                Kivilcim::KvlcmSerializer::savePreferences(".kvlcm-user-pref", "C:/Users/Can/Desktop/user_prefs.kvlcm-user-pref", prefsToSave);

                userPrefs.isPreferencesChanged = false;
                isOpen = false;
            }
            ImGui::PopStyleColor(2);

            ImGui::End();
            ImGui::PopStyleColor(2);
        }

    }
}