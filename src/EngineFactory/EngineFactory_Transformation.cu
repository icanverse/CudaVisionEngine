#include "OperationWrapper.cuh"
#include "EngineFactory/EngineFactory.cuh"

EngineFactory &EngineFactory::logTransformation() {
    OperationWrapper::logTransformation(d_data, d_temp_data, width, height, channels);
    std::swap(d_data, d_temp_data);
    return *this;
}
