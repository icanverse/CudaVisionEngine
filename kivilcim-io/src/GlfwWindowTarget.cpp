#include "GlfwWindowTarget.h"
#include <iostream>

GlfwWindowTarget::GlfwWindowTarget(int width, int height, const std::string& title) {
    // GLFW'yi başlat
    if (!glfwInit()) {
        std::cerr << "[GlfwTarget] HATA: GLFW baslatilamadi!" << std::endl;
        exit(1);
    }

    // Pencereyi oluştur (Donanım hızlandırmalı OpenGL bağlamı yaratır)
    window = glfwCreateWindow(width, height, title.c_str(), NULL, NULL);
    if (!window) {
        glfwTerminate();
        std::cerr << "[GlfwTarget] HATA: Pencere olusturulamadi!" << std::endl;
        exit(1);
    }

    // OpenGL komutlarını bu pencereye yönlendir
    glfwMakeContextCurrent(window);
}

GlfwWindowTarget::~GlfwWindowTarget() {
    glfwDestroyWindow(window);
    glfwTerminate();
}

void GlfwWindowTarget::present(unsigned char* data, int width, int height, int channels) {
    // Ekranı temizle
    glClear(GL_COLOR_BUFFER_BIT);

    // DİKKAT: Şu an VRAM'den RAM'e çektiğimiz veriyi (data) ekrana basıyoruz.
    // İleride burayı tamamen sıfır gecikmeli "CUDA-GL Interop" yapısına çevireceğiz!
    // OpenGL'in başlangıç noktası sol alt köşedir, o yüzden resmi çizerken yönüne dikkat eder.
    glDrawPixels(width, height, GL_RGB, GL_UNSIGNED_BYTE, data);

    // Arka planda çizileni öne getir (Double Buffering)
    glfwSwapBuffers(window);

    // Klavyeden / fareden gelen tepkileri dinle
    glfwPollEvents();
}

bool GlfwWindowTarget::shouldClose() {
    return glfwWindowShouldClose(window);
}