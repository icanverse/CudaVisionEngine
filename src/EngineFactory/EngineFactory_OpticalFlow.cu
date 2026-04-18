#include "../../include/EngineFactory/EngineFactory.cuh"
#include "OperationWrapper.cuh"

EngineFactory& EngineFactory::applyOpticalFlowLucasKanade(float strength) {

    OperationWrapper::opticalFlowLucasKanade(
        d_data,
        d_prev_data,
        width, height, channels,
        d_flow_u,
        d_flow_v
    );

    // Bir sonraki frame için şimdiki frame'i "eski" olarak kaydet
    saveCurrentFrameAsPrevious();

    return *this;
}