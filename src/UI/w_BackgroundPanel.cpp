#include "../../include/UI/w_BackgroundPanel.h"
#include "imgui.h"

void BackgroundPanel::render(float displayWidth, float displayHeight) {
    // Pencereyi tam ekran yap ve sol üst köşeye oturt
    ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(displayWidth, displayHeight), ImGuiCond_Always);

    // Kenar boşluklarını ve yuvarlatmaları sıfırla ki ekrana tam yapışsın
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));

    // Bu paneli "hayalet" yapacak olan sihirli bayraklar (Flags)
    ImGuiWindowFlags bgFlags = ImGuiWindowFlags_NoTitleBar |
                               ImGuiWindowFlags_NoCollapse |
                               ImGuiWindowFlags_NoResize |
                               ImGuiWindowFlags_NoMove |
                               ImGuiWindowFlags_NoBringToFrontOnFocus |
                               ImGuiWindowFlags_NoNavFocus;

    // Arka planı transparan yap
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.0f, 0.0f, 0.0f, 0.0f));

    ImGui::Begin("AnaZemin", nullptr, bgFlags);

    // ==========================================
    // SOL ALT KÖŞE METİN MATEMATİĞİ
    // ==========================================

    const char* altMetin = "Kıvılcım Görüntü Motoru ile destekleniyor";

    // A. Metnin ekranda kaplayacağı piksel boyutunu (Genişlik ve Yükseklik) hesapla
    ImVec2 textSize = ImGui::CalcTextSize(altMetin);

    // B. Kenarlardan ne kadar boşluk bırakacağımızı belirle (Padding)
    float paddingX = 15.0f; // Soldan boşluk
    float paddingY = 15.0f; // Alttan boşluk

    // C. İmleci tam olarak sol alt noktaya gönder
    // Y Koordinatı = Ekranın En Altı (displayHeight) - Metnin Boyu - İstediğimiz Boşluk
    ImGui::SetCursorPos(ImVec2(paddingX, displayHeight - textSize.y - paddingY));

    // D. Metni ekrana bas (Arka planda çok parlamaması için hafif gri/soluk bir renk veriyoruz)
    ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "%s", altMetin);

    ImGui::End();

    ImGui::PopStyleColor();
    ImGui::PopStyleVar(3);
}