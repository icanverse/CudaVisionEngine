// main.cpp
#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include "EngineFactory.cuh"

int main() {
    EngineFactory engine("assets/holiday.jpeg");

    // Zincirleme Filtre Şovu
    engine.applyTemperature(0.15f)
          .rgbToHsv()
          .applyShadowsHighlights(0.2f, -0.1f)
          .applyGamma(0.8f)
          .hsvToRgb()
          .saveImage("assets/holiday_output_fluent.jpg");

    std::cout << "Islem Tamamlandi!" << std::endl;
    return 0;
}