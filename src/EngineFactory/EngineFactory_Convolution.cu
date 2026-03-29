#include "OperationWrapper.cuh"
#include "EngineFactory/EngineFactory.cuh"

EngineFactory& EngineFactory::applyBoxBlur() {
    OperationWrapper::applyBoxBlur(d_data, d_temp_data, width, height, channels);
    std::swap(d_data, d_temp_data); // Sıfır maliyetle kopyalama!
    return *this;
}

EngineFactory& EngineFactory::applySharpen() {
    OperationWrapper::applySharpen(d_data, d_temp_data, width, height, channels);
    std::swap(d_data, d_temp_data);
    return *this;
}

EngineFactory& EngineFactory::applyEdgeDetection() {
    OperationWrapper::applyEdgeDetection(d_data, d_temp_data, width, height, channels);
    std::swap(d_data, d_temp_data);
    return *this;
}

EngineFactory& EngineFactory::applyGaussianBlur5x5() {
    OperationWrapper::applyGaussianBlur5x5(d_data, d_temp_data, width, height, channels);
    std::swap(d_data, d_temp_data);
    return *this;
}

EngineFactory& EngineFactory::applySobelX() {
    OperationWrapper::applySobelX(d_data, d_temp_data, width, height, channels);
    std::swap(d_data, d_temp_data);
    return *this;
}

EngineFactory& EngineFactory::applySobelY() {
    OperationWrapper::applySobelY(d_data, d_temp_data, width, height, channels);
    std::swap(d_data, d_temp_data);
    return *this;
}

EngineFactory& EngineFactory::applyEmboss() {
    OperationWrapper::applyEmboss(d_data, d_temp_data, width, height, channels);
    std::swap(d_data, d_temp_data);
    return *this;
}

EngineFactory &EngineFactory::applyGaussianBlurVChannel() {
    OperationWrapper::applyGaussianBlurVChannel(d_data, d_temp_data, width, height, channels);
    std::swap(d_data, d_temp_data);
    return *this;
}
