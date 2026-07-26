#include "EngineFactory/EngineFactory.cuh"
#include "OperationWrapper.cuh"

EngineFactory& EngineFactory::applyTemperature(float temperature) {
    OperationWrapper::temperatureAdjustment(d_data, width, height, channels, temperature);
    return *this;
}

EngineFactory& EngineFactory::applyShadowsHighlights(float shadowAmount, float highlightAmount) {
    OperationWrapper::shadowsHighlightsAdjustment(d_data, width, height, channels, shadowAmount, highlightAmount);
    return *this;
}

EngineFactory& EngineFactory::applyGamma(float gamma) {
    OperationWrapper::gammaCorrectionAdjustment(d_data, width, height, channels, gamma);
    return *this;
}

