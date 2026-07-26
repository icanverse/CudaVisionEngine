#ifndef CUDAVISIONENGINE_FLARE_CUH
#define CUDAVISIONENGINE_FLARE_CUH



__global__ void generateFlareHSV(float* data, int width, int height, int channels,
                                 float flareX, float flareY,
                                 float baseHue, float baseSaturation, float falloff);


#endif //CUDAVISIONENGINE_FLARE_CUH