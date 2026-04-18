#ifndef CUDAVISIONENGINE_OPTICALFLOW_CUH
#define CUDAVISIONENGINE_OPTICALFLOW_CUH

#define TILE_SIZE 16
#define RADIUS 2
#define PADDING (RADIUS + 1)
#define SHARED_WIDTH (TILE_SIZE + 2 * PADDING)

__global__ void opticalFlowLucasKanade(const float* currentFrame, const float* previousFrame,
                                       int width, int height, int channels, float* flowU, float* flowV);
__device__ float getLuma(const float* d_rgb, int width, int height, int channels, int x, int y );

#endif //CUDAVISIONENGINE_OPTICALFLOW_CUH