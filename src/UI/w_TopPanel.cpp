#include "UI/w_TopPanel.h"
#include "imgui.h"

#ifdef _WIN32
#include <windows.h>
#endif

#include <GLFW/glfw3.h> // OpenGL sorguları için eklendi

void TopPanel::render(GLFWwindow* window, float displayWidth, unsigned int logoTextureId) {
    float panelHeight = 60.0f; // Panel yüksekliği artırıldı ve daha ferah yapıldı

    ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(displayWidth, panelHeight), ImGuiCond_Always);

    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);

    // --- RENK AYARI ---
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.05f, 0.05f, 0.06f, 1.0f));

    // --- KAYDIRMA ÇUBUĞU İPTALİ ---
    ImGuiWindowFlags flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                             ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar |
                             ImGuiWindowFlags_NoScrollWithMouse;

    ImGui::Begin("UstPanel", nullptr, flags);

    // --- LOGO ÇİZİMİ (Orijinal Oran ve Dinamik Boyutlandırma) ---
    if (logoTextureId != 0) {
        float padding = 16.0f; // Logonun alt/üst boşluk payı
        float logoHeight = panelHeight - padding;
        float logoWidth = logoHeight; // Hata durumunda varsayılan olarak kare kabul et

        // Logonun VRAM'deki orijinal piksel genişlik/yükseklik değerlerini çekiyoruz
        glBindTexture(GL_TEXTURE_2D, logoTextureId);
        int texWidth = 0, texHeight = 0;
        glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, &texWidth);
        glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT, &texHeight);

        // Orijinal en-boy oranını (aspect ratio) koruyarak yeni genişliği hesapla
        if (texHeight > 0) {
            logoWidth = logoHeight * ((float)texWidth / (float)texHeight);
        }

        // Logoyu dikeyde ortala ve 15px sol boşluk bırak
        ImGui::SetCursorPos(ImVec2(15.0f, padding * 0.5f));
        ImGui::Image((void*)(intptr_t)logoTextureId, ImVec2(logoWidth, logoHeight), ImVec2(0.0f, 1.0f), ImVec2(1.0f, 0.0f));
        ImGui::SameLine();
    }

    ImGui::SetCursorPosY(0.0f);
    // Butonlar ve logo payı için sağdan yeterli boşluk bırakıldı
    float dragAreaWidth = displayWidth - ImGui::GetCursorPosX() - 110.0f;
    if (dragAreaWidth < 10.0f) dragAreaWidth = 10.0f;

    ImGui::InvisibleButton("DragArea", ImVec2(dragAreaWidth, panelHeight));

    if (ImGui::IsItemHovered() && ImGui::IsMouseClicked(0)) {
        isDragging = true;
        double mX, mY;
        glfwGetCursorPos(window, &mX, &mY);
        dragOffsetX = (int)mX;
        dragOffsetY = (int)mY;
    }

    if (isDragging && ImGui::IsMouseDown(0)) {
#ifdef _WIN32
        POINT p;
        GetCursorPos(&p);
        glfwSetWindowPos(window, p.x - dragOffsetX, p.y - dragOffsetY);
#endif
    } else {
        isDragging = false;
    }

    // --- PENCERE KONTROL BUTONLARI (Hizalama Kusursuzlaştırıldı) ---
    ImGui::SameLine();

    float buttonWidth = 40.0f;
    float buttonHeight = 30.0f;
    // Her iki butonun da aynı hizada olması için kesin Y koordinatı
    float buttonYPos = (panelHeight - buttonHeight) * 0.5f;

    // "-" (Simge Durumuna Küçült) Butonu
    ImGui::SetCursorPosY(buttonYPos);
    if (ImGui::Button("-", ImVec2(buttonWidth, buttonHeight))) glfwIconifyWindow(window);

    ImGui::SameLine();

    // "X" (Kapat) Butonu
    ImGui::SetCursorPosY(buttonYPos); // Kesinlikle aynı yükseklik zorlanıyor
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.8f, 0.2f, 0.2f, 1.0f));
    if (ImGui::Button("X", ImVec2(buttonWidth, buttonHeight))) glfwSetWindowShouldClose(window, 1);
    ImGui::PopStyleColor();

    ImGui::End();
    ImGui::PopStyleColor();
    ImGui::PopStyleVar();
}