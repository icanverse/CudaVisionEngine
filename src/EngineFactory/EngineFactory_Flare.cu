#include "../../include/EngineFactory/EngineFactory.cuh"
#include "OperationWrapper.cuh"
#include <iostream>

EngineFactory& EngineFactory::blendTexture(cudaTextureObject_t tex, int texW, int texH,
                                           float targetX, float targetY,
                                           float opacity, bool isAdditive) {

    OperationWrapper::applyTextureBlendKernel(d_data, width, height, channels,
                                              tex, texW, texH,
                                              targetX, targetY, opacity, isAdditive);
    return *this;
}

EngineFactory& EngineFactory::renderProceduralFlare(float x, float y, float hue, float opacity) {

    // Flare texture çözünürlüğü >> daha detaylı ışık için değeri büyüt
    const int FLARE_SIZE = 512;
    const int FLARE_CHANNELS = 4; // Texture donanımı hizalama için 4 kanal (RGBA) sever

    if (d_flareArray == nullptr) {
        std::cout << "[EngineFactory] Flare dokusu ilk kez hesaplanıyor. Fırınlama başladı." << std::endl;

        initTextureMemory(d_flareArray, flareTexture, FLARE_SIZE, FLARE_SIZE);

        float* d_tempFlare;
        size_t flareBytes = FLARE_SIZE * FLARE_SIZE * FLARE_CHANNELS * sizeof(float);
        cudaMalloc(&d_tempFlare, flareBytes);

        OperationWrapper::generateFlareHSV(d_tempFlare, FLARE_SIZE, FLARE_SIZE, FLARE_CHANNELS,
                                           FLARE_SIZE / 2.0f, FLARE_SIZE / 2.0f,
                                           hue, 0.8f, 0.02f);

        OperationWrapper::hsvToRgb(d_tempFlare, d_tempFlare, FLARE_SIZE, FLARE_SIZE, FLARE_CHANNELS);

        size_t pitch = FLARE_SIZE * FLARE_CHANNELS * sizeof(float);
        cudaMemcpy2DToArray(d_flareArray, 0, 0,                 // Hedef Array ve offset (0,0)
                            d_tempFlare, pitch,                 // Kaynak VRAM ve satır uzunluğu (Pitch)
                            pitch, FLARE_SIZE,                  // Kopyalanacak Genişlik ve Yükseklik
                            cudaMemcpyDeviceToDevice);          // GPU içi transfer

        cudaFree(d_tempFlare);

        std::cout << "[EngineFactory] Fırınlama tamamlandi. Flare donanima yuklendi " << std::endl;
    }

    blendTexture(flareTexture, FLARE_SIZE, FLARE_SIZE, x, y, opacity, true);

    return *this;
}
