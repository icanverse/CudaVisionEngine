#include "EngineFactory/EngineFactory.cuh"
#include "OperationWrapper.cuh"

EngineFactory& EngineFactory::applyVectorFieldColoring(float intensity) {
    OperationWrapper::applyVectorFieldColoring(d_data, d_flow_u, d_flow_v, width, height, channels, intensity);
    return *this;
}

EngineFactory& EngineFactory::applyNormalMapVisualization(float intensity) {
    OperationWrapper::applyNormalMapVisualization(d_data, d_flow_u, d_flow_v, width, height, channels, intensity);
    return *this;
}

EngineFactory& EngineFactory::applyQuiverPlotVisualization(float intensity) {
    OperationWrapper::applyQuiverPlotVisualization(d_data, d_flow_u, d_flow_v, width, height, channels, intensity);
    return *this;
}

EngineFactory& EngineFactory::applyJetScalarColorPalette(float maxSpeed) {
    OperationWrapper::applyJetScalarColorPalette(d_data, d_flow_u, d_flow_v, width, height, channels, maxSpeed);
    return *this;
}

EngineFactory &EngineFactory::applyLineIntegralConvolution(int steps) {
    OperationWrapper::applyLineIntegralConvolution(d_data, d_flow_u, d_flow_v, width, height, channels, steps);
    return *this;
}

