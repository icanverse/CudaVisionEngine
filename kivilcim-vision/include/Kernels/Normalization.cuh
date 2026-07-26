#ifndef CUDAVISIONENGINE_NORMALIZATION_CUH
#define CUDAVISIONENGINE_NORMALIZATION_CUH

__global__ void k_normalizeImage(unsigned char* input, float* output, int totalElements);
__global__ void k_denormalizeImage(float* input, unsigned char* output, int totalElements);
__global__ void retinexNormalize(float* input, const float* global_min, const float* global_max, int total_pixels, int channels);


#endif //CUDAVISIONENGINE_NORMALIZATION_CUH