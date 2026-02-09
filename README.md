# CudaVisionEngine 🚀

Yüksek performanslı görüntü işleme motoru. **CUDA C++** kullanılarak geliştirilmiştir.

## Third-Party Libraries
Bu proje aşağıdaki açık kaynak kütüphaneleri kullanmaktadır:
* [stb](https://github.com/nothings/stb) - Resim yükleme ve kaydetme işlemleri için (Public Domain / MIT).

## Özellikler
- **GPU Hızlandırma:** Görüntü yükleme, işleme ve kaydetme süreçleri optimize edildi.
- **Kernel Yönetimi:** `src/ElementaryMatrixOp.cu` üzerinden özel CUDA çekirdekleri.
- **Bellek Yönetimi:** Pinned Memory (cudaMallocHost) ve Device Memory senkronizasyonu.
- **Architecture:** Modüler C++ Sınıf yapısı.

## Kurulum
Bu proje **CLion** ve **CMake** ile geliştirilmiştir.
- NVIDIA CUDA Toolkit v13.x gerektirir.
- C++20 Standardı kullanılır.

## Kullanım
```cpp
// GPU üzerinde Normalize & Denormalize işlemleri otomatik yapılır.

GeneralOperations myImage("assets/input.jpg");
myImage.saveImage("assets/output.png");
```
