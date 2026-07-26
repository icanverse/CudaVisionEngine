#ifndef CUDAVISIONENGINE_LUT_3D_CUH
#define CUDAVISIONENGINE_LUT_3D_CUH


__global__ void apply3DLUT(float* d_rgb, int width, int height, int channels, cudaTextureObject_t lutTexture);


#endif //CUDAVISIONENGINE_LUT_3D_CUH