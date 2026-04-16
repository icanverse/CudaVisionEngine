#include "../../include/EngineFactory/EngineFactory.cuh"

#include <fstream>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

#include "OperationWrapper.cuh"

// Dosya yolunu alır, okuduğu renkleri RGBA formatında 'lutData' vektörüne doldurur.
// Küpün bir kenar uzunluğunu (örneğin 33) 'lutSize' referansına yazar.
bool loadCubeLUT(const std::string& filepath, std::vector<float>& lutData, int& lutSize) {

    // Dosyayı Aç (ifstream)
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "[HATA] LUT dosyasi bulunamadi veya acilamadi: " << filepath << std::endl;
        return false;
    }

    std::string line;

    // Satır Satır Oku (getline)
    while (std::getline(file, line)) {

        // Boş satırları atla
        if (line.empty()) continue;

        // Yorum satırlarını (#) ve TITLE gibi gereksiz metinleri atla
        if (line[0] == '#' || line.find("TITLE") == 0) continue;

        // Küpün Boyutunu Bul
        // Eğer satır "LUT_3D_SIZE" kelimesiyle başlıyorsa...
        if (line.find("LUT_3D_SIZE") == 0) {
            std::stringstream ss(line);
            std::string tempText;

            // Satırdaki ilk kelimeyi tempText'e ("LUT_3D_SIZE"), sayıyı lutSize'a (örn: 33) at
            ss >> tempText >> lutSize;
            continue; // Bu satırla işimiz bitti, sonraki satıra geç
        }

        // Renkleri Oku ve Diziye Ekle!
        // Eğer yukarıdaki if'lere takılmadıysa, demek ki bu satır bir renk verisidir.
        std::stringstream ss(line);
        float r, g, b;

        // ss ayrıştırıcısı 3 tane sayıyı başarıyla bulabildiyse...
        if (ss >> r >> g >> b) {
            lutData.push_back(r);    // Kırmızı
            lutData.push_back(g);    // Yeşil
            lutData.push_back(b);    // Mavi
            lutData.push_back(1.0f); // GPU KURALI :: Donanım çökmesin diye Alpha kanalı ekliyoruz!
        }
    }

    file.close();

    // Güvenlik kontrolü: Gerçekten veri okuyabildik mi?
    if (lutData.empty() || lutSize == 0) {
        std::cerr << "[HATA] LUT dosyasi okundu ama icinde gecerli veri yok!" << std::endl;
        return false;
    }

    std::cout << "[LUT Parser] " << filepath << " basariyla RAM'e yuklendi. Boyut: "
              << lutSize << "x" << lutSize << "x" << lutSize << std::endl;

    return true;
}

void EngineFactory::init3DTextureMemory(const float* h_lutData, int lutSize, cudaArray_t& targetArray, cudaTextureObject_t& targetTexture) {

    // Temizlik (Eğer önceden yüklü bir LUT varsa sil)
    if (targetTexture) { cudaDestroyTextureObject(targetTexture); targetTexture = 0; }
    if (targetArray) { cudaFreeArray(targetArray); targetArray = nullptr; }

    // 3 Boyutlu Hacmi (Volume) Tanımla (Genişlik, Yükseklik, Derinlik)
    cudaExtent extent = make_cudaExtent(lutSize, lutSize, lutSize);

    // Kanal Formatı (Yine donanımın sevdiği 4 Kanal: RGBA)
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();

    // GPU'da 3 Boyutlu Array Ayır
    cudaError_t err = cudaMalloc3DArray(&targetArray, &channelDesc, extent);
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc3DArray Hatasi: " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }

    // Kopyalama Parametreleri (Canavarın Midesi)
    // CPU'daki tek boyutlu diziyi, GPU'daki 3 boyutlu küpe nasıl yerleştireceğini tarif ediyoruz.
    cudaMemcpy3DParms copyParams = {0};

    // Kaynak: RAM'deki (Host) dizimiz. Pitch (Satır genişliği) = X ekseni x float4 boyutu
    copyParams.srcPtr = make_cudaPitchedPtr((void*)h_lutData, lutSize * sizeof(float4), lutSize, lutSize);

    // Hedef: GPU'daki (Device) Array'imiz.
    copyParams.dstArray = targetArray;
    copyParams.extent = extent;
    copyParams.kind = cudaMemcpyHostToDevice;

    // FIRINLA! (RAM'den VRAM'e 3 Boyutlu Transfer)
    err = cudaMemcpy3D(&copyParams);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy3D Hatasi: " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }

    // Kaynak (Resource) Tanımı
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray; // Array kullanıyoruz
    resDesc.res.array.array = targetArray;

    // Doku (Texture) Okuma Kuralları
    cudaTextureDesc texDesc = {};

    // Küpün sınırına (Örn: Tam kırmızıya) gelirsek, en uçtaki rengi kullan (Clamp)
    texDesc.addressMode[0] = cudaAddressModeClamp; // R Eksenini Sınırla
    texDesc.addressMode[1] = cudaAddressModeClamp; // G Eksenini Sınırla
    texDesc.addressMode[2] = cudaAddressModeClamp; // B Eksenini Sınırla

    texDesc.filterMode = cudaFilterModeLinear;  // TRILINEAR INTERPOLATION AÇIK! Donanım pürüzsüzleştirir.
    texDesc.readMode = cudaReadModeElementType; // Float oku
    texDesc.normalizedCoords = 1;               // Renk koordinatlarını 0.0 - 1.0 arası gönder

    // Objeyi Yarat ve Bize Verilen Referansa Bağla
    err = cudaCreateTextureObject(&targetTexture, &resDesc, &texDesc, nullptr);
    if (err != cudaSuccess) {
        std::cerr << "cudaCreateTextureObject (3D) Hatasi: " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }

    std::cout << "[EngineFactory] 3D LUT Donanimi Basariyla Kuruldu! Boyut: " << lutSize << "^3" << std::endl;
}

EngineFactory& EngineFactory::apply3DLUT(cudaTextureObject_t lutTexture) {
    OperationWrapper::apply3DLUT(d_data, width, height, channels, lutTexture);
    return *this;
}
