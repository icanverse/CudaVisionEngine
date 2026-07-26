#ifndef CUDAVISIONENGINE_DEPTHMAPKERNEL_CUH
#define CUDAVISIONENGINE_DEPTHMAPKERNEL_CUH

struct IsoLineData {
    float2* points;       // Tüm çizgilerin noktaları peş peşe (X, Y)
    float* depthValues;   // Her çizginin Z derinlik değeri (0.0f - 1.0f)
    int* pointLineIndices; // Hangi nokta hangi çizgiye ait (O(1) erişim için)
    int* isSegmentEnd;     // Bu nokta çizginin son noktası mı? (1 veya 0)
    int* lineLengths;     // Hangi çizgi kaç noktadan oluşuyor?
    int totalPoints;      // Toplam nokta sayısı
    int totalLines;       // Toplam çizgi sayısı
};

void launchIsoDepthKernel(
        cudaSurfaceObject_t outputDepthMap, // Üzerine yazacağımız boş (siyah) derinlik dokusu
        int width,
        int height,
        const IsoLineData& lineData
    );

class DepthMapKernel {
};


#endif