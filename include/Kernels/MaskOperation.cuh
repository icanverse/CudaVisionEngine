
#ifndef CUDAVISIONENGINE_MASKOPERATION_CUH
#define CUDAVISIONENGINE_MASKOPERATION_CUH


__global__ void blendVChannel(const float* base, const float* detail, float* output, int width, int height, int channels, float strength);


#endif //CUDAVISIONENGINE_MASKOPERATION_CUH