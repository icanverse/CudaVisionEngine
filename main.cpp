#include <algorithm>
#include <cstddef>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

#include "GLFW/glfw3.h"
#include "imgui.h"
#include "lib/stb_image.h"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

extern "C" {
__declspec(dllexport) DWORD NvOptimusEnablement = 0x00000001;
__declspec(dllexport) int AmdPowerXpressRequestHighPerformance = 1;
}
#endif

// kivilcim-vision
#include "ParticleSystem/ParticleSystem.cuh"
#include "EngineFactory/EngineFactory.cuh"

// kivilcim-graphics
#include "Renderer3D.cuh"
#include "Scene.cuh"

// kivilcim-io
#include "GlfwInteropTarget.h"
#include "kivilcim-graphics/include/SceneBuilder.h"

// kivilcim-ui
#include "MainUI.h"

namespace {

constexpr int WINDOW_WIDTH = 1366;
constexpr int WINDOW_HEIGHT = 768;
constexpr int IMAGE_CHANNELS = 3;
constexpr int PARTICLE_COUNT = 50;
constexpr float MAX_FRAME_DELTA_SECONDS = 0.1F;

void checkCuda(cudaError_t result, const char* operation) {
    if (result == cudaSuccess) {
        return;
    }

    throw std::runtime_error(
        std::string(operation) + ": " + cudaGetErrorString(result)
    );
}

struct CudaBufferDeleter {
    void operator()(unsigned char* pointer) const noexcept {
        if (pointer != nullptr) {
            cudaFree(pointer);
        }
    }
};

using CudaBuffer = std::unique_ptr<unsigned char, CudaBufferDeleter>;
using ImagePixels = std::unique_ptr<stbi_uc, decltype(&stbi_image_free)>;

class CursorHandle {
public:
    explicit CursorHandle(GLFWwindow* window) noexcept
        : window_(window) {
    }

    CursorHandle(const CursorHandle&) = delete;
    CursorHandle& operator=(const CursorHandle&) = delete;

    ~CursorHandle() {
        reset(nullptr);
    }

    void reset(GLFWcursor* cursor) noexcept {
        if (cursor_ != nullptr) {
            if (window_ != nullptr) {
                glfwSetCursor(window_, nullptr);
            }
            glfwDestroyCursor(cursor_);
        }

        cursor_ = cursor;

        if (window_ != nullptr && cursor_ != nullptr) {
            glfwSetCursor(window_, cursor_);
        }
    }

private:
    GLFWwindow* window_ = nullptr;
    GLFWcursor* cursor_ = nullptr;
};

void loadCustomCursor(GLFWwindow* window, CursorHandle& cursor) {
    int width = 0;
    int height = 0;
    int channels = 0;

    ImagePixels pixels(
        stbi_load(
            "lib-assets/cursor/classic20.png",
            &width,
            &height,
            &channels,
            STBI_rgb_alpha
        ),
        &stbi_image_free
    );

    if (!pixels) {
        std::cerr
            << "[Kivilcim] Uyari: Ozel imlec yuklenemedi: "
            << "lib-assets/cursor/classic20.png\n";
        return;
    }

    GLFWimage image{};
    image.width = width;
    image.height = height;
    image.pixels = pixels.get();

    GLFWcursor* createdCursor = glfwCreateCursor(&image, 0, 0);
    if (createdCursor == nullptr) {
        std::cerr << "[Kivilcim] Uyari: GLFW ozel imleci olusturamadi.\n";
        return;
    }

    cursor.reset(createdCursor);
    std::cout << "[Kivilcim] Ozel imlec yuklendi.\n";
}

} // namespace

int main() {
    try {
        std::cout << "[Kivilcim] Pencere olusturuluyor...\n";

        GlfwInteropTarget target(
            WINDOW_WIDTH,
            WINDOW_HEIGHT,
            IMAGE_CHANNELS,
            "Kivilcim - Sirca UI"
        );

        CursorHandle customCursor(target.getWindow());
        loadCustomCursor(target.getWindow(), customCursor);

        std::cout << "[Kivilcim] CUDA aygiti hazirlaniyor...\n";
        checkCuda(cudaSetDevice(0), "cudaSetDevice");
        checkCuda(cudaFree(nullptr), "CUDA context initialization");

        EngineFactory visionEngine(
            WINDOW_WIDTH,
            WINDOW_HEIGHT,
            IMAGE_CHANNELS
        );

        Renderer3D graphicsRenderer(
            WINDOW_WIDTH,
            WINDOW_HEIGHT,
            IMAGE_CHANNELS
        );

        std::cout << "[Kivilcim] Sahne yukleniyor...\n";
        Scene scene =
            SceneBuilder::build("assets-graphics/scenes/scene_ui.kvlcm");

        ParticleSystem particleSystem(PARTICLE_COUNT);

        const std::size_t pixelCount =
            static_cast<std::size_t>(WINDOW_WIDTH) *
            static_cast<std::size_t>(WINDOW_HEIGHT) *
            static_cast<std::size_t>(IMAGE_CHANNELS);

        const std::size_t floatCanvasSize =
            pixelCount * sizeof(float);

        const std::size_t byteCanvasSize =
            pixelCount * sizeof(unsigned char);

        if (visionEngine.getDeviceData() == nullptr) {
            throw std::runtime_error(
                "EngineFactory gecersiz CUDA goruntu tamponu dondurdu."
            );
        }

        checkCuda(
            cudaMemset(
                visionEngine.getDeviceData(),
                0,
                floatCanvasSize
            ),
            "cudaMemset(visionEngine)"
        );

        unsigned char* rawRenderCanvas = nullptr;
        checkCuda(
            cudaMalloc(
                reinterpret_cast<void**>(&rawRenderCanvas),
                byteCanvasSize
            ),
            "cudaMalloc(renderCanvas)"
        );

        CudaBuffer renderCanvas(rawRenderCanvas);

        std::cout << "[Kivilcim] ImGui arayuzu kuruluyor...\n";
        MainUI sircaUI(target.getWindow());

        ImGuiIO& io = ImGui::GetIO();
        io.ConfigFlags |= ImGuiConfigFlags_NoMouseCursorChange;
        io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
        io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;

        float animationTime = 0.0F;
        double previousFrameTime = glfwGetTime();
        double statisticsStartTime = previousFrameTime;
        int renderedFrameCount = 0;

        std::cout << "[Kivilcim] Sistem hazir.\n";

        while (!target.shouldClose()) {
            glfwPollEvents();

            const double frameTime = glfwGetTime();
            const float deltaTime = std::clamp(
                static_cast<float>(frameTime - previousFrameTime),
                0.0F,
                MAX_FRAME_DELTA_SECONDS
            );
            previousFrameTime = frameTime;
            animationTime += deltaTime;

            sircaUI.newFrame();
            sircaUI.renderPanels();

            particleSystem.update(deltaTime, animationTime);

            graphicsRenderer.render(
                visionEngine.getDeviceData(),
                scene,
                animationTime
            );

            visionEngine.copyToDeviceUchar(renderCanvas.get());

            particleSystem.draw(
                renderCanvas.get(),
                WINDOW_WIDTH,
                WINDOW_HEIGHT
            );

            checkCuda(cudaGetLastError(), "CUDA render islemi");

            unsigned char* mappedPbo = target.mapVRAM();
            if (mappedPbo == nullptr) {
                throw std::runtime_error(
                    "OpenGL PBO, CUDA bellegine eslenemedi."
                );
            }

            const cudaError_t copyResult = cudaMemcpy(
                mappedPbo,
                renderCanvas.get(),
                byteCanvasSize,
                cudaMemcpyDeviceToDevice
            );

            target.unmapAndRender();
            checkCuda(copyResult, "cudaMemcpy(renderCanvas -> PBO)");
            sircaUI.renderDrawData();

            if ((io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) != 0) {
                GLFWwindow* previousContext = glfwGetCurrentContext();
                ImGui::UpdatePlatformWindows();
                ImGui::RenderPlatformWindowsDefault();
                glfwMakeContextCurrent(previousContext);
            }

            glfwSwapBuffers(target.getWindow());
            ++renderedFrameCount;

            const double statisticsTime = glfwGetTime();
            const double statisticsInterval =
                statisticsTime - statisticsStartTime;

            if (statisticsInterval >= 1.0) {
                std::size_t freeBytes = 0;
                std::size_t totalBytes = 0;
                checkCuda(
                    cudaMemGetInfo(&freeBytes, &totalBytes),
                    "cudaMemGetInfo"
                );

                const auto usedMegabytes = static_cast<unsigned long long>(
                    (totalBytes - freeBytes) / (1024ULL * 1024ULL)
                );
                const auto totalMegabytes = static_cast<unsigned long long>(
                    totalBytes / (1024ULL * 1024ULL)
                );
                const int framesPerSecond = static_cast<int>(
                    static_cast<double>(renderedFrameCount) /
                    statisticsInterval
                );

                std::cout
                    << "\r[Kivilcim] FPS: " << framesPerSecond
                    << " | VRAM: " << usedMegabytes
                    << " MB / " << totalMegabytes
                    << " MB   "
                    << std::flush;

                renderedFrameCount = 0;
                statisticsStartTime = statisticsTime;
            }
        }

        checkCuda(cudaDeviceSynchronize(), "Final CUDA synchronization");

        std::cout
            << "\n[Kivilcim] Motor guvenli sekilde kapatiliyor...\n";

        return 0;
    } catch (const std::exception& error) {
        std::cerr
            << "\n[Kivilcim] Kritik hata: "
            << error.what()
            << '\n';
        return 1;
    } catch (...) {
        std::cerr << "\n[Kivilcim] Bilinmeyen kritik hata.\n";
        return 1;
    }
}
// //
// #include "Cuda/HardwareDetector.h"
//
// int main() {
//      Kcore::HardwareInfoData info = Kcore::HardwareDetector::inspectSystem(nullptr);
//      Kcore::HardwareDetector::printHardwareReport(info);
//      return 0;
// }