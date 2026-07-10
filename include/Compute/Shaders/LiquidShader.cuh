#pragma once
#include <cuda_runtime.h>

namespace Kivilcim {
    namespace Shaders {
        void launchLiquidShader(cudaSurfaceObject_t surface, 
                                int width, int height, 
                                float time, float mouseX, float mouseY,
                                float waveFrequency = 40.0f,
                                float waveSpeed = 3.0f,
                                float waveAmplitude = 0.05f,
                                float3 backgroundColor = make_float3(0.05f, 0.05f, 0.06f),
                                float3 liquidColor = make_float3(0.85f, 0.45f, 0.00f),
                                float liquidAlpha = 1.0f);

        void launchLiquidFlowShader(cudaSurfaceObject_t surface, int width, int height,
                            float time, float flowSpeed, float freq, float3 liquidColor);
    }
}