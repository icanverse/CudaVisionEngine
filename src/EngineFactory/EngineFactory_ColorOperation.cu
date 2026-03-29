#include "OperationWrapper.cuh"
#include "EngineFactory/EngineFactory.cuh"

EngineFactory& EngineFactory::isolateColor(float targetHue, float tolerance) {
    OperationWrapper::isolateColor(d_data, width, height, channels, targetHue, tolerance);
    return *this;
}

EngineFactory &EngineFactory::colorReplacement(float targetHue, float tolerance, float replacementHue) {
    OperationWrapper::colorReplacement(d_data, width, height, channels, targetHue, tolerance,  replacementHue);
    return *this;
}