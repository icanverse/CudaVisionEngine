#pragma once
#include "HardwareInfoData.h"
struct GLFWwindow; // GLFW bağımlılığı için ön bildirim

namespace Kcore {

    class HardwareDetector {
    public:
        // Tüm sistemi tarar ve tek bir nesnede toplar
        static HardwareInfoData inspectSystem(GLFWwindow* window = nullptr);

        // Konsola veya log dosyasına şık bir rapor basar
        static void printHardwareReport(const HardwareInfoData& info);
    };

}