#include "UI/w_TopPanel.h"
#include "imgui.h"

#ifdef _WIN32
#include <windows.h>
#endif

#include <GLFW/glfw3.h> // OpenGL sorguları için eklendi

float panelHeight = 60.0f; // Panel yüksekliği

void TopPanel::render(GLFWwindow* window, float displayWidth, unsigned int logoTextureId) {

    ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(displayWidth, panelHeight), ImGuiCond_Always);

    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);

    // --- RENK AYARI ---
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.05f, 0.05f, 0.06f, 1.0f));

    // --- KAYDIRMA ÇUBUĞU İPTALİ ---
    ImGuiWindowFlags flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                             ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar |
                             ImGuiWindowFlags_NoScrollWithMouse;

    ImGui::Begin("TopPanel", nullptr, flags);

    // Pencerenin köşe koordinatlarını al
    ImVec2 minPos = ImGui::GetWindowPos(); // Sol üst köşe
    ImVec2 maxPos = ImVec2(minPos.x + ImGui::GetWindowWidth(), minPos.y + ImGui::GetWindowHeight()); // Sağ alt köşe

    // Renkleri belirle (Üstten alta doğru koyulaşan bir turuncu/siyah gradyanı)
    ImU32 colorTopLeft  = IM_COL32(55, 30, 10, 255);
    ImU32 colorTopRight = IM_COL32(55, 30, 10, 255);
    ImU32 colorBotLeft  = IM_COL32(0, 0, 0, 255);
    ImU32 colorBotRight = IM_COL32(0, 0, 0, 255);

    // Arka plana gradyan dikdörtgeni çiz
    ImDrawList* drawList = ImGui::GetWindowDrawList();
    // Ekrana ilk bu çizileceği için arkada kalacak, UI elemanları bunun üstüne binecek
    drawList->AddRectFilledMultiColor(minPos, maxPos, colorTopLeft, colorTopRight, colorBotRight, colorBotLeft);

    // --- LOGO ÇİZİMİ ---
    if (logoTextureId != 0) {
        float padding = 16.0f;
        float logoHeight = panelHeight - padding;
        float logoWidth = logoHeight;

        glBindTexture(GL_TEXTURE_2D, logoTextureId);
        int texWidth = 0, texHeight = 0;
        glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, &texWidth);
        glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT, &texHeight);

        if (texHeight > 0) {
            logoWidth = logoHeight * ((float)texWidth / (float)texHeight);
        }

        ImGui::SetCursorPos(ImVec2(15.0f, padding * 0.5f));
        ImGui::Image((void*)(intptr_t)logoTextureId, ImVec2(logoWidth, logoHeight), ImVec2(0.0f, 1.0f), ImVec2(1.0f, 0.0f));
        ImGui::SameLine();
    }

    // --- SÜRÜKLEME ALANI (DRAG AREA) ---
    ImGui::SetCursorPosY(0.0f);

    float buttonWidth = panelHeight;
    float buttonHeight = panelHeight;

    // Butonların kaplayacağı alanı tam olarak hesaplayıp sürükleme alanını sınırlandırıyoruz
    float dragAreaWidth = displayWidth - ImGui::GetCursorPosX() - (buttonWidth * 2.0f);
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

    // --- PENCERE KONTROL BUTONLARI (Kusursuz Windows Davranışı) ---
    ImGui::SameLine(0, 0); // Sürükleme alanıyla butonlar arasındaki boşluğu sıfırla

    // Butonların kenar yuvarlatmasını sıfırla (Tam dikdörtgen yap)
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 0.0f);

    // "-" (Simge Durumuna Küçült) Butonu
    ImGui::SetCursorPosY(0.0f); // Panelin tam en üstünden başla
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.0f, 0.0f, 0.0f, 0.0f)); // Varsayılan: Şeffaf
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(1.0f, 1.0f, 1.0f, 0.1f)); // Üzerine gelince: Hafif saydam beyaz
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(1.0f, 1.0f, 1.0f, 0.2f));  // Tıklanırken: Biraz daha belirgin beyaz

    if (ImGui::Button("-", ImVec2(buttonWidth, buttonHeight))) glfwIconifyWindow(window);

    ImGui::PopStyleColor(3); // "-" butonu için açılan 3 renk kuralını kapat

    ImGui::SameLine(0, 0); // "-" butonuyla "X" butonu arasındaki boşluğu sıfırla

    // "X" (Kapat) Butonu
    ImGui::SetCursorPosY(0.0f); // Panelin tam en üstünden başla
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.0f, 0.0f, 0.0f, 0.0f)); // Varsayılan: Şeffaf
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.9f, 0.1f, 0.15f, 1.0f)); // Üzerine gelince: Kırmızı
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.7f, 0.1f, 0.1f, 1.0f));  // Tıklanırken: Koyu Kırmızı

    if (ImGui::Button("X", ImVec2(buttonWidth, buttonHeight))) glfwSetWindowShouldClose(window, 1);

    ImGui::PopStyleColor(3); // "X" butonu için açılan 3 renk kuralını kapat

    ImGui::PopStyleVar(); // FrameRounding sıfırlamasını geri al

    ImGui::End();
    ImGui::PopStyleColor(); // Pencere arka planı Pop
    ImGui::PopStyleVar();   // Pencere kenar yuvarlatması Pop
}

float TopPanel::getPanelHeight() {
    return panelHeight;
}
