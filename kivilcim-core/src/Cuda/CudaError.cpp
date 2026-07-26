#include "../../kivilcim-core/include/Cuda/CudaError.h"
#include <string>

namespace Kivilcim::Core::Cuda {
namespace {

std::string createMessage(cudaError_t error, std::string_view operation) {
    std::string message(operation);
    message += ": ";
    message += cudaGetErrorString(error);
    return message;
}

} // namespace

CudaException::CudaException(
    cudaError_t error,
    std::string_view operation
)
    : std::runtime_error(createMessage(error, operation)),
      error_(error) {
}

void throwIfFailed(cudaError_t result, std::string_view operation) {
    if (result != cudaSuccess) {
        throw CudaException(result, operation);
    }
}

} // namespace Kivilcim::Core::Cuda
