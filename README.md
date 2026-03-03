# CudaVisionEngine 🚀
Yüksek performanslı, donanım hızlandırmalı (Hardware-Accelerated) görüntü işleme ve matris hesaplama motoru. Pikselleri ve karmaşık matematiksel operasyonları doğrudan GPU VRAM üzerinde işlemek için CUDA C++ ile sıfırdan geliştirilmiştir.

## 🛠 Teknik Mimari
#### GPU Hızlandırma: Görüntü işleme süreçleri CPU darboğazından kurtarılarak paralel iş parçacıklarına (Thread/Block) dağıtılmıştır.

#### Bellek Yönetimi: Yüksek hızlı bellek transferleri ve Shared Memory optimizasyonları.

#### Modüler Tasarım: Çekirdek (Kernel) fonksiyonları güvenli ve ölçeklenebilir bir OperationWrapper sınıfı üzerinden yönetilir. Dinamik grid/block hesaplamaları ve otomatik hata ayıklama (Error Checking) mimarisi mevcuttur.

## Third-Party Libraries
Bu proje aşağıdaki açık kaynak kütüphaneleri kullanmaktadır:
* [stb](https://github.com/nothings/stb) - Resim yükleme ve kaydetme işlemleri için (Public Domain / MIT).

## ✨ Özellikler ve Yetenekler
Motorun sunduğu yetenekler OperationWrapper üzerinden aşağıdaki kategorilere ayrılmıştır:

```cpp
// Temel İşlemler
// Görüntü verilerini float formatına normalize/denormalize etme
OperationWrapper::normalize(d_img_in, d_img_out, w, h);
OperationWrapper::denormalize(d_img_in, d_img_out, w, h);

// Renk Uzayı 
// Kayıpsız VRAM içi renk uzayı dönüşümleri (RGB ⇄ HSV)
OperationWrapper::rgbToHsv(d_rgb, d_hsv, w, h, c);

// Ton Ayarlamaları 
// Işık, Doygunluk ve Orta Nokta (Midpoint) destekli Kontrast
OperationWrapper::brightnessAdjustment(d_hsv, w, h, c, 0.2f);
OperationWrapper::saturationAdjustment(d_hsv, w, h, c, 1.5f);
OperationWrapper::contrastAdjustment(d_hsv, w, h, c, 1.2f, 0.5f);

// Renk Operasyonları -- Sinematik Filtreler
// Rengi izole etme veya başka bir renge (örn: kırmızıdan maviye) kaydırma
OperationWrapper::isolateColor(d_hsv, w, h, c, 0.0f, 30.0f);
OperationWrapper::colorReplacement(d_hsv, w, h, c, 0.0f, 30.0f, 240.0f);

// Uzamsal Filtreler
// Komşu pikseller arası ilişkiye dayalı 2D Blur (Yumuşatma)
OperationWrapper::smoothing2D(d_in, d_out, w, h, c, 3);

// Matris Operasyonları
// Shared Memory destekli yüksek hızlı matris matematiği
OperationWrapper::add(d_A, d_B, d_C, size, true);
OperationWrapper::multiply(d_A, d_B, d_C, size);
OperationWrapper::getSubMatrix(d_in, d_out, col, row, size);


```

## 📦 Kurulum ve Gereksinimler
Bu proje CLion ve CMake modern C++ standartlarına göre yapılandırılmıştır.

Dil: C++20

Platform: NVIDIA CUDA Toolkit v13.x

Üçüncü Parti Kütüphaneler: - stb - Resim okuma/yazma işlemleri için (Public Domain / MIT)

## 💻 Kullanım Örneği
Motor, VRAM üzerindeki işlemleri arka arkaya zincirlemenize (Pipeline) olanak tanır. Aşağıda bir görüntünün yüklenip, HSV uzayında parlaklığının artırılması ve tekrar RGB'ye çevrilerek kaydedilmesi örneğini görebilirsiniz:

```cpp
// GPU üzerinde Normalize & Denormalize işlemleri otomatik yapılır.

#include "CudaVisionEngine.h"
#include "OperationWrapper.cuh"

int main() {
    // 1. Motoru başlat ve veriyi belleğe al
    GeneralOperations myImage("assets/input.jpg");
    
    int width = myImage.getWidth();
    int height = myImage.getHeight();
    int channels = myImage.getChannels();

    // 2. RGB'den HSV'ye dönüşüm
    OperationWrapper::rgbToHsv(d_rgb_input, d_hsv_temp, width, height, channels);

    // 3. Ton Ayarı: Parlaklığı %20 artır
    OperationWrapper::brightnessAdjustment(d_hsv_temp, width, height, channels, 0.2f);

    // 4. Tekrar RGB'ye dönüştür
    OperationWrapper::hsvToRgb(d_hsv_temp, d_rgb_output, width, height, channels);

    // 5. Kaydet
    myImage.updateDeviceData(d_rgb_output);
    myImage.saveImage("assets/output_bright.png");

    return 0;
}
```
