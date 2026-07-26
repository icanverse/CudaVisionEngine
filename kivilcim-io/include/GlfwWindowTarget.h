#ifndef CUDAVISIONENGINE_GLFWWINDOWTARGET_H
#define CUDAVISIONENGINE_GLFWWINDOWTARGET_H

#include "IRenderTarget.h"
#include <GLFW/glfw3.h> // Sadece bu hafif kütüphaneyi kullanacağız
#include <string>

class GlfwWindowTarget : public IRenderTarget {
private:
    GLFWwindow* window;

public:
    GlfwWindowTarget(int width, int height, const std::string& title);
    ~GlfwWindowTarget() override;

    void present(unsigned char* data, int width, int height, int channels) override;

    // Oyun motoru döngüsünü kontrol etmek için
    bool shouldClose();
};


#endif //CUDAVISIONENGINE_GLFWWINDOWTARGET_H