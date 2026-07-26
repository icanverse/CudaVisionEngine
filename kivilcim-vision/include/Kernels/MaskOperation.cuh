
#ifndef CUDAVISIONENGINE_MASKOPERATION_CUH
#define CUDAVISIONENGINE_MASKOPERATION_CUH


__global__ void blendVChannel(const float* base, const float* detail, float* output, int width, int height, int channels, float strength);
__global__ void applyTextureBlend(float* data, int width, int height, int channels,
                                        cudaTextureObject_t overlayTex, int texWidth, int texHeight,
                                        float targetX, float targetY, float opacity, bool isAdditive);

#endif //CUDAVISIONENGINE_MASKOPERATION_CUH