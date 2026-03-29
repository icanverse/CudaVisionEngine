
#ifndef CUDAVISIONENGINE_REDUCTION_CUH
#define CUDAVISIONENGINE_REDUCTION_CUH


__global__ void k_InitGlobalMinMax(float* global_min, float* global_max);
__global__ void k_MinMaxReduction(const float* input, float* d_output_min, float* d_output_max, int width, int height, int channels);



#endif //CUDAVISIONENGINE_REDUCTION_CUH