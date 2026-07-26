//
// Created by Can on 22.03.2026.
//

#ifndef CUDAVISIONENGINE_LOGTRANSFORMATION_CUH
#define CUDAVISIONENGINE_LOGTRANSFORMATION_CUH


__global__ void logTransformationVChannel(float* input, float* output, int width, int height, int channels);


#endif //CUDAVISIONENGINE_LOGTRANSFORMATION_CUH