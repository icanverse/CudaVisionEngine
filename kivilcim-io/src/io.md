# CudaVisionEngine - Giriş / Çıkış (I/O) Mimarisi

Bu belge, motorun dış dünya ile (Dosyalar, Kameralar, Videolar ve Monitörler) nasıl iletişim kurduğunu açıklamaktadır.

## 1. Mimari Fikir: "Decoupling" (Bağımsızlaştırma)
CudaVisionEngineX, **Bağımlılıkların Tersine Çevrilmesi (Dependency Inversion)** prensibiyle tasarlanmıştır. Motorun kalbi olan `EngineFactory`, verinin nereden geldiğini veya işlendikten sonra nereye gideceğini **kesinlikle bilmez**.

Sadece `(Genişlik, Yükseklik, Kanal)` boyutlarını alır, VRAM'de yer ayırır ve pikselleri işler. Dış dünya ile iletişim, tamamen soyutlanmış **Arayüzler (Interfaces)** üzerinden sağlanır.

## 2. Soyut Arayüzler (Interfaces)

Tüm I/O modülleri bu iki temel arayüzden türer:

* **`IDataSource` (Gözler):** Motorun veri okuduğu kaynaktır. Her döngüde `grabNextFrame()` fonksiyonu ile yeni bir kareyi (Frame) ham bayt dizisi (unsigned char*) olarak motora teslim etmekle yükümlüdür.

* **`IRenderTarget` (Monitör):** Motorun işlediği veriyi teslim ettiği hedeftir. `present()` fonksiyonu ile işlenmiş VRAM veya RAM verisini alır ve ilgili donanıma (Disk, Ekran, Ağ) yönlendirir.

## 3. Mevcut I/O Modülleri

Şu an projemizde aktif olarak çalışan modüller şunlardır:

### Girdi Kaynakları (Sources)
* **`StbImageSource`:** `stb_image.h` kütüphanesini kullanarak disktedeki `.jpg` ve `.png` dosyalarını okur. Motoru tek bir kare ile besler.

### Çıktı Hedefleri (Targets)
* **`StbImageTarget`:** İşlenmiş pikselleri CPU RAM'ine çeker ve `stb_image_write.h` ile diske `.png` olarak kaydeder.
* **`GlfwWindowTarget` (Yedek/Fallback):** Standart bir OpenGL penceresi açar. İşlenmiş veriyi önce CPU'ya (`downloadFrame`), oradan da OpenGL API'si üzerinden ekrana yollar. Güvenlidir ancak CPU-GPU arası kopyalama maliyeti nedeniyle gecikme (Latency) yaratır.
* **`GlfwInteropTarget` (Sıfır Gecikme / Zero-Copy):** Motorun amiral gemisi çıktı modülüdür. **CUDA-GL Interoperability** ve **Pixel Buffer Object (PBO)** mimarisini kullanır. İşlenen pikseller, işlemciye (CPU) hiç uğramadan VRAM içinde OpenGL'in bellek havuzuna haritalanır ve doğrudan monitöre fırlatılır. V-Sync kısıtlaması kapalıdır ve maksimum donanım limitlerinde (130+ FPS) çalışır.

## 4. Evrensel Render Döngüsü (The Pipeline)

Sistemin `main.cpp` içindeki standart çalışma akışı şu şekildedir:

1. Girdi ve Çıktı modülleri başlatılır.
2. `EngineFactory` bu modüllerin boyutlarına göre VRAM'i ayarlar.
3. Sonsuz döngü (Game Loop) başlar:
    - Kaynak yeni kareyi verir (`grabNextFrame`).
    - Motor veriyi VRAM'e alır (`uploadFrame`) ve CUDA Kernellerini (Temperature, Gamma vb.) sırayla in-place olarak çalıştırır.
    - İşlenmiş veri Zero-Copy ile veya CPU üzerinden hedefe gönderilir (`present`).
    - Kaynağa belleği temizlemesi söylenir (`releaseFrame`).

## 5. Gelecek Planları (Roadmap)
* `FFmpegVideoSource`: Saf C/C++ kütüphaneleri ile donanım hızlandırmalı (H.264/HEVC) video çözme yeteneği eklenecek.
* `V4L2 / Windows Media Foundation Source`: Web kameralarından canlı ve gecikmesiz görüntü alma (Real-time Vision) modülü entegre edilecek.