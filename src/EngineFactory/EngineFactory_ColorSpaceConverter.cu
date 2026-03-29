#include "EngineFactory/EngineFactory.cuh"
#include "OperationWrapper.cuh"

EngineFactory& EngineFactory::rgbToHsv() {
    OperationWrapper::rgbToHsv(d_data, d_temp_data, width, height, channels);
    std::swap(d_data, d_temp_data);
    return *this;
}

EngineFactory& EngineFactory::hsvToRgb() {
    OperationWrapper::hsvToRgb(d_data, d_temp_data, width, height, channels);
    std::swap(d_data, d_temp_data);
    return *this;
}

EngineFactory& EngineFactory::rgbToYuv() {
    OperationWrapper::rgbToYuv(d_data, d_temp_data, width, height, channels);
    std::swap(d_data, d_temp_data);
    return *this;
}

EngineFactory& EngineFactory::yuvToRgb() {
    OperationWrapper::yuvToRgb(d_data, d_temp_data, width, height, channels);
    std::swap(d_data, d_temp_data);
    return *this;
}