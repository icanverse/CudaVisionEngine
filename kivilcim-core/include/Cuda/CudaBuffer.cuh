/**
 * Include Guards
 *      Projenin birden fazla yerinde include edilirse
 *      içindeki kodların birden fazla kez derlenmesini önler.
 */
#ifndef KIVILCIM_CORE_CUDA_CUDABUFFER_CUH
#define KIVILCIM_CORE_CUDA_CUDABUFFER_CUH

#include "../Cuda/CudaError.h"
#include <cuda_runtime_api.h>

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

/**
 * Namespace
 *      isim çakışmalarını önler
 */
namespace Kivilcim::Core::Cuda {

/**
 * template<typename T>
 *      veri tipi serbest çalışabilmeyi sağlar
 */
template<typename T>


// Cuda Belleğinin adresini ve boyutunu tutan hafif bir gösterge sınıfı
class CudaBufferView {
public:
    using value_type = T;

    /**
    * 1. Kurucu boş bir view oluşturur.
    *      data_ ve size_ sıfırlanırç
    *
    * constexpr
    *      derleme zamanı çalıştırılabileceğini belirtir
    */
    constexpr CudaBufferView() noexcept = default;

    /**
    * 2. Kurucu dışarıdan
    *       T* data  (bellek adresi) ve
    *       std::size_t size (eleman sayısı)
    *       alarak view nesnesini başlatır.
    *
    * noexcept
    *       hata fırlatmayacağını garanti eder
    */
    constexpr CudaBufferView(T* data, std::size_t size) noexcept
        : data_(data),
          size_(size) {
    }

    /**
    * data() ham bellek adresini döndürür
    * size() toplam eleman sayısını döndürür
    * sizeBytes() = eleman sayısı * tipin boyutu
    * empty() size == 0 mı söyler
    * explicit operator bool()
    * [[nodiscard]] fonksiyonların döndürdüğü sonuçlar görmezden gelinirse derleyici uyarır.
    */

    [[nodiscard]] constexpr T* data() const noexcept {
        return data_;
    }

    [[nodiscard]] constexpr std::size_t size() const noexcept {
        return size_;
    }

    [[nodiscard]] constexpr std::size_t sizeBytes() const noexcept {
        return size_ * sizeof(std::remove_const_t<T>);
    }

    [[nodiscard]] constexpr bool empty() const noexcept {
        return size_ == 0;
    }

    [[nodiscard]] constexpr explicit operator bool() const noexcept {
        return data_ != nullptr;
    }

private:
    T* data_ = nullptr;
    std::size_t size_ = 0;
};

/**
 * Derleme aşamasında tip kontrolü yapar
 * Buraya T olarak kopyalanamayacak karmaşık bir c++ sınıfı (örn std::string vb)
 * kod derlenmez hata mesajı verir
 * GPU'ya atılan verilerin saf byte verisi olmasını istiyoruz
 */
template<typename T>
class CudaBuffer {
    static_assert(
        std::is_trivially_copyable_v<T>,
        "CudaBuffer only supports trivially copyable element types."
    );

    /**
     * Boş buffer oluşturur
     *
     * explicit CudaBuffer(std::size_t elementCount)
     *      Belirtilen eleman sayısında yer açar
     *      explicit : istem dışı tür dönüşümlerini engeller
     *
     */

public:
    using value_type = T;

    CudaBuffer() noexcept = default;

    explicit CudaBuffer(std::size_t elementCount) {
        allocate(elementCount);
    }

    ~CudaBuffer() {
        release();
    }

    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;

    CudaBuffer(CudaBuffer&& other) noexcept {
        swap(other);
    }

    CudaBuffer& operator=(CudaBuffer&& other) noexcept {
        if (this != &other) {
            release();
            swap(other);
        }
        return *this;
    }

    // Mevcut veriyi korumaz. Yeni tahsis basarili olduktan sonra eski
    // allocation serbest birakilir.
    void resizeDiscard(std::size_t elementCount) {
        if (elementCount == size_) {
            return;
        }

        CudaBuffer replacement(elementCount);
        swap(replacement);
    }

    void reset() noexcept {
        release();
    }

    void zero() {
        if (empty()) {
            return;
        }

        throwIfFailed(
            cudaMemset(data_, 0, sizeBytes()),
            "cudaMemset(CudaBuffer)"
        );
    }

    void copyFromHost(
        const T* source,
        std::size_t elementCount
    ) {
        validateCopy(source, elementCount, "copyFromHost");

        throwIfFailed(
            cudaMemcpy(
                data_,
                source,
                checkedByteSize(elementCount),
                cudaMemcpyHostToDevice
            ),
            "cudaMemcpy(host -> device)"
        );
    }

    void copyToHost(
        T* destination,
        std::size_t elementCount
    ) const {
        validateCopy(destination, elementCount, "copyToHost");

        throwIfFailed(
            cudaMemcpy(
                destination,
                data_,
                checkedByteSize(elementCount),
                cudaMemcpyDeviceToHost
            ),
            "cudaMemcpy(device -> host)"
        );
    }

    void copyFromDevice(
        const T* source,
        std::size_t elementCount
    ) {
        validateCopy(source, elementCount, "copyFromDevice");

        throwIfFailed(
            cudaMemcpy(
                data_,
                source,
                checkedByteSize(elementCount),
                cudaMemcpyDeviceToDevice
            ),
            "cudaMemcpy(device -> device)"
        );
    }

    void copyFromDevice(const CudaBuffer& source) {
        if (source.size() != size_) {
            throw std::invalid_argument(
                "CudaBuffer device copy requires equal buffer sizes."
            );
        }

        copyFromDevice(source.data(), source.size());
    }

    [[nodiscard]] T* data() noexcept {
        return data_;
    }

    [[nodiscard]] const T* data() const noexcept {
        return data_;
    }

    [[nodiscard]] std::size_t size() const noexcept {
        return size_;
    }

    [[nodiscard]] std::size_t sizeBytes() const noexcept {
        return size_ * sizeof(T);
    }

    [[nodiscard]] bool empty() const noexcept {
        return size_ == 0;
    }

    [[nodiscard]] explicit operator bool() const noexcept {
        return data_ != nullptr;
    }

    [[nodiscard]] CudaBufferView<T> view() noexcept {
        return {data_, size_};
    }

    [[nodiscard]] CudaBufferView<const T> view() const noexcept {
        return {data_, size_};
    }

    void swap(CudaBuffer& other) noexcept {
        std::swap(data_, other.data_);
        std::swap(size_, other.size_);
    }

private:
    static std::size_t checkedByteSize(std::size_t elementCount) {
        if (elementCount > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
            throw std::length_error("CudaBuffer byte size overflow.");
        }

        return elementCount * sizeof(T);
    }

    void allocate(std::size_t elementCount) {
        if (elementCount == 0) {
            return;
        }

        T* allocation = nullptr;
        throwIfFailed(
            cudaMalloc(
                reinterpret_cast<void**>(&allocation),
                checkedByteSize(elementCount)
            ),
            "cudaMalloc(CudaBuffer)"
        );

        data_ = allocation;
        size_ = elementCount;
    }

    template<typename Pointer>
    void validateCopy(
        Pointer pointer,
        std::size_t elementCount,
        std::string_view operation
    ) const {
        if (elementCount > size_) {
            throw std::out_of_range(
                std::string(operation) + " exceeds CudaBuffer capacity."
            );
        }

        if (elementCount > 0 && pointer == nullptr) {
            throw std::invalid_argument(
                std::string(operation) + " received a null pointer."
            );
        }
    }

    void release() noexcept {
        if (data_ != nullptr) {
            cudaFree(data_);
            data_ = nullptr;
        }
        size_ = 0;
    }

    T* data_ = nullptr;
    std::size_t size_ = 0;
};

template<typename T>
void swap(CudaBuffer<T>& left, CudaBuffer<T>& right) noexcept {
    left.swap(right);
}

} // namespace Kivilcim::Core::Cuda

#endif // KIVILCIM_CORE_CUDA_CUDABUFFER_CUH
