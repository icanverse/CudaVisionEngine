#ifndef CUDAVISIONENGINE_ENGINEMANAGEMENT_H
#define CUDAVISIONENGINE_ENGINEMANAGEMENT_H

#include <cuda.h>
#include <utility> // std::swap için eklendi
#include <cuda_runtime.h>
#include <vector>
#include <string>

class EngineFactory {
private:
    // Görsel Özellikleri
    int width;
    int height;
    int channels;
    size_t totalElementCount; // w * h * c

    // Pointerlar
    float* d_data;      // Device (GPU) - İşlenmiş Float Veri (0.0 - 1.0 arası)
    float* d_temp_data; // Çift Bellek Mimarisi için geçici VRAM alanı
    float* d_mask_data;
    float* d_global_min;
    float* d_global_max;

    // Texture Belleği Kullanmak için
    cudaArray_t d_flareArray = nullptr;        // Fiziksel bellek ~doku belleği
    cudaTextureObject_t flareTexture = 0;      // Doku okuma objesi

    // Optical Flow Belleği
    float* d_prev_data;   // Bir önceki kare (t-1)
    float* d_flow_u;      // X yönündeki hareket vektörleri (Horizontal Flow)
    float* d_flow_v;      // Y yönündeki hareket vektörleri (Vertical Flow)

    // 3D Mesh (Ağ) Verileri için
    float3* d_vertices;
    int3* d_indices;
    int numTriangles;

    // Yardımcı Fonksiyonlar
    void allocateMemory();
    void cleanUp();

    void saveCurrentFrameAsPrevious();

public:

    // LUT Texture Belleği dışarıdan erişilebilmesi için public'te
    cudaArray_t d_lutArray = nullptr;
    cudaTextureObject_t lutTexture = 0;

    // Motor boyutları alıp VRAM'de yer ayırır.
    EngineFactory(int w, int h, int c);

    // Destructor: Belleği temizler
    ~EngineFactory();

    // RAM'den VRAM'e veri atar (Char -> Float normalizasyonu yapar)
    EngineFactory& uploadFrame(const unsigned char* cpu_data);

    // VRAM'den RAM'e işlenmiş veriyi çeker (Float -> Char denormalizasyonu yapar)
    void downloadFrame(unsigned char* cpu_data);

    // Hangi array ve objeyi verirsen, onu o boyutlarda texture yapar.
    void initTextureMemory(cudaArray_t& targetArray, cudaTextureObject_t& targetTexture, int texWidth, int texHeight);

    void init3DTextureMemory(const float* h_lutData, int lutSize, cudaArray_t& targetArray, cudaTextureObject_t& targetTexture);

    // Getterlar
    int getWidth() const { return width; }
    int getHeight() const { return height; }
    int getChannels() const { return channels; }
    float* getDeviceData() const { return d_data; }

    void updateDeviceData(const float* newData) {
        if (d_data && newData) {
            size_t size = width * height * channels * sizeof(float);
            cudaMemcpy(d_data, newData, size, cudaMemcpyDeviceToDevice);
        }
    }

    // doğrudan başka bir VRAM adresine (Interop için) yazar
    void copyToDeviceUchar(unsigned char* d_dest_uchar);

    // Geriye referans dönüyoruz ki zincirleme (fluent) devam edebilsin
    EngineFactory& loadNV12DevicePointer(CUdeviceptr d_nv12, int pitch);

    EngineFactory& loadFromVRAM(unsigned char* d_source_uchar);

    // Renk Uzayı Dönüşümleri (Ping-Pong kullanır)
    EngineFactory& rgbToHsv();
    EngineFactory& hsvToRgb();
    EngineFactory& rgbToYuv();
    EngineFactory& yuvToRgb();
    EngineFactory& kernelNV12toRGB();
    EngineFactory& loadNV12DevicePointer();
    EngineFactory& retinexNormalize();

    //
    EngineFactory& subVCh();

    // Filtreler (In-place çalışır)
    EngineFactory& applyTemperature(float temperature);
    EngineFactory& applyShadowsHighlights(float shadowAmount, float highlightAmount);
    EngineFactory& applyGamma(float gamma);

    //
    EngineFactory& logTransformation();

    /// Convolution
    // EngineFactory& applyConvolution();
    // EngineFactory& applyConvolutionVChannel();

    ///
    EngineFactory& applyBoxBlur();
    EngineFactory& applySharpen();
    EngineFactory& applyEdgeDetection();
    EngineFactory& applyGaussianBlur5x5();
    EngineFactory& applySobelX();
    EngineFactory& applySobelY();
    EngineFactory& applyEmboss();

    EngineFactory& applyGaussianBlurVChannel();

    // Renk Uzayına Bağlı Hazır Gelişmiş İşlemler
    EngineFactory& isolateColor(float targetHue, float tolerance);
    EngineFactory& colorReplacement(float targetHue, float tolerance, float replacementHue);

    // Kompleks İşlemler
    EngineFactory& applyRetinex();

    // --- PROCEDURAL EFEKTLER VE TEXTURE MAPPING ---
    EngineFactory& blendTexture(cudaTextureObject_t tex, int texW, int texH, float targetX, float targetY, float opacity, bool isAdditive);
    EngineFactory& renderProceduralFlare(float x, float y, float hue, float opacity);


    EngineFactory& apply3DLUT(cudaTextureObject_t lutTexture);

    EngineFactory& applyOpticalFlowLucasKanade(float strength = 1.0f);

    EngineFactory& applyVectorFieldColoring(float intensity = 1.0f);
    EngineFactory& applyNormalMapVisualization(float intensity = 1.0f);
    EngineFactory& applyQuiverPlotVisualization(float intensity = 1.0f);
    EngineFactory& applyJetScalarColorPalette(float maxSpeed);
    EngineFactory& applyLineIntegralConvolution(int steps);

    EngineFactory& loadMesh(const float3* cpu_vertices, int numVerts, const int3* cpu_indices, int numTris);
    EngineFactory& render3DScene(float time);

};

// Yardımcı Fonksiyonlar
bool loadCubeLUT(const std::string& filepath, std::vector<float>& lutData, int& lutSize);



#endif //CUDAVISIONENGINE_ENGINEMANAGEMENT_H