#ifndef CUDAVISIONENGINE_GLFWINTEROPTARGET_H
#define CUDAVISIONENGINE_GLFWINTEROPTARGET_H

#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h> // CUDA-GL Köprüsü!
#include <string>

class GlfwInteropTarget {
private:
    GLFWwindow* window;
    unsigned int pbo; // Pixel Buffer Object (VRAM Havuzu)
    struct cudaGraphicsResource* cuda_pbo_resource;
    int width, height, channels;

public:
    GlfwInteropTarget(int w, int h, int c, const std::string& title);
    ~GlfwInteropTarget();

    // 1. CUDA'nın içine yazabilmesi için VRAM'in kapısını açar ve adresi verir
    unsigned char* mapVRAM();

    // 2. Kapıyı kapatır ve o VRAM havuzunu doğrudan ekrana çizer
    void unmapAndRender();

    bool shouldClose();
};


#endif //CUDAVISIONENGINE_GLFWINTEROPTARGET_H