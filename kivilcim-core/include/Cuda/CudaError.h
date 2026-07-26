#ifndef KIVILCIM_CORE_CUDA_CUDAERROR_H
#define KIVILCIM_CORE_CUDA_CUDAERROR_H

#include <cuda_runtime_api.h>

#include <stdexcept>
#include <string_view>

namespace Kivilcim::Core::Cuda {

class CudaException final : public std::runtime_error {
public:
    CudaException(cudaError_t error, std::string_view operation);

    [[nodiscard]] cudaError_t error() const noexcept {
        return error_;
    }

private:
    cudaError_t error_;
};

void throwIfFailed(cudaError_t result, std::string_view operation);

} // namespace Kivilcim::Core::Cuda

#endif // KIVILCIM_CORE_CUDA_CUDAERROR_H
