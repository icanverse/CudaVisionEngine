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
CudaVisionEngineX, Bağımlılıkların Tersine Çevrilmesi (Dependency Inversion) prensibiyle çalışır. Motor, verinin nereden geldiğini bilmez; sadece donanım hızlandırmalı (GPU) zincirleme işlemlere odaklanır.

Aşağıda bir görüntünün diskten okunup, zincirleme (Fluent) metotlarla sinematik bir renk filtresinden geçirilerek tekrar diske kaydedilmesi örneğini görebilirsiniz:

```cpp

#include <vector>
#include "io/StbImageSource.h"
#include "io/StbImageTarget.h"
#include "EngineFactory.cuh"

int main() {
    // 1. I/O Modüllerini Başlat (Kaynak ve Hedef)
    StbImageSource source("assets/input.jpg");
    StbImageTarget target("assets/output_cinematic.png");

    unsigned char* rawFrame = source.grabNextFrame();
    if (!rawFrame) return -1;

    // 2. Motoru Başlat (Sadece boyutlara göre VRAM rezerve eder)
    EngineFactory engine(source.getWidth(), source.getHeight(), source.getChannels());
    
    // CPU'da işlenmiş veri için geçici alan
    std::vector<unsigned char> processedData(source.getWidth() * source.getHeight() * source.getChannels());

    // 3. Akıcı Arayüz (Fluent Pipeline) ile Sıfır Kesintili GPU İşlemleri
    engine.uploadFrame(rawFrame)
          .applyTemperature(0.15f)             // Sıcaklığı artır
          .rgbToHsv()                          // İşlemler için HSV uzayına geç
          .applyShadowsHighlights(0.2f, -0.1f) // Gölgeleri aç, parlamaları kıs
          .applyGamma(1.1f)                    // Kontrastı ayarla
          .hsvToRgb()                          // Ekrana/Diske basmak için RGB'ye dön
          .downloadFrame(processedData.data());// Sonucu VRAM'den RAM'e çek

    // 4. Hedefe Gönder ve Temizle
    target.present(processedData.data(), engine.getWidth(), engine.getHeight(), engine.getChannels());
    source.releaseFrame(rawFrame);

    return 0;
}
```

## 💻 Kullanım Örneği 2: Gerçek Zamanlı Zero-Copy Interop Döngüsü (Real-Time Game Loop)

Motor, CPU-GPU arası veri transferi dar boğazını aşmak için CUDA-GL Interoperability (Zero-Copy) mimarisini destekler. Aşağıdaki örnekte, VRAM'de işlenen pikseller işlemciye (RAM) hiç uğramadan doğrudan OpenGL PBO (Pixel Buffer Object) üzerinden monitöre fırlatılır.

Bu sayede saniyede yüzlerce kare (FPS) işlenirken gecikme süreleri milisaniye seviyesine iner:

```cpp
#include <iostream>
#include <chrono>
#include <cmath>
#include <iomanip> // std::setprecision için eklendi
#include "io/StbImageSource.h"
#include "io/GlfwInteropTarget.h"
#include "EngineFactory.cuh"

int main() {
    std::cout << "[Main] ZERO-COPY Interop Motoru Baslatiliyor..." << std::endl;

    StbImageSource source("assets/starwars.jpg");
    unsigned char* rawFrame = source.grabNextFrame();
    if (!rawFrame) return -1;

    // Sıfır Gecikmeli Interop Monitörünü Başlat
    GlfwInteropTarget target(source.getWidth(), source.getHeight(), source.getChannels(), "CudaVisionEngine - Zero Copy");

    EngineFactory engine(source.getWidth(), source.getHeight(), source.getChannels());

    auto t_start = std::chrono::high_resolution_clock::now();
    int frameCount = 0;
    float timeTracker = 0.0f;

    // THE GAME LOOP
    while (!target.shouldClose()) {
        float dynamicTemp = std::sin(timeTracker) * 0.5f;
        timeTracker += 0.02f;

        // 1. İşlemleri GPU'da Yap (Fluent Interface)
        engine.uploadFrame(rawFrame)
              .applyTemperature(dynamicTemp)
              .rgbToHsv()
              .applyGamma(1.1f)
              .hsvToRgb();

        // 2. VRAM Kapısını Aç ve Hedef Adresi Al
        unsigned char* d_pbo_vram_address = target.mapVRAM();

        // 3. Pikselleri VRAM'den VRAM'e YAZ (CPU'ya kopyalamak yok!)
        engine.copyToDeviceUchar(d_pbo_vram_address);

        // 4. Kapıyı Kapat ve Monitöre Çiz
        target.unmapAndRender();

        // Performans ve Gecikme (Latency) Ölçümü
        frameCount++;
        if (frameCount % 100 == 0) {
            auto t_end = std::chrono::high_resolution_clock::now();
            double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
            
            double avg_latency_ms = elapsed_ms / 100.0; // Kare başına düşen gecikme
            double fps = 1000.0 / avg_latency_ms;       // Saniyedeki kare sayısı
            
            std::cout << "Guncel FPS: " << std::fixed << std::setprecision(1) << fps 
                      << " | Gecikme: " << std::fixed << std::setprecision(2) << avg_latency_ms << " ms    \r" << std::flush;
            
            t_start = std::chrono::high_resolution_clock::now();
        }
    }

    source.releaseFrame(rawFrame);
    std::cout << "\nMotor basariyla kapatildi." << std::endl;
    return 0;
}

```