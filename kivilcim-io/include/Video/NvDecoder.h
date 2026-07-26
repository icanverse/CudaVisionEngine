#ifndef CUDAVISIONENGINE_NVDECODER_H
#define CUDAVISIONENGINE_NVDECODER_H


#pragma once

#include <cuda.h>
#include <nv_sdk_interface/cuviddec.h>
#include <nv_sdk_interface/nvcuvid.h>
#include <mutex>
#include <queue>

class NvDecoder {
private:
    CUcontext m_cuContext = nullptr;
    CUvideoparser m_hParser = nullptr;
    CUvideodecoder m_hDecoder = nullptr;

    int m_width = 0;
    int m_height = 0;

    // VRAM'deki çözülmüş resimlerin adreslerini (pointer) tutacağımız kuyruk
    std::queue<int> m_frameQueue;
    std::mutex m_queueMutex;

    // NVDEC Callback Fonksiyonları (Statik olmak zorundalar)
    static int CUDAAPI HandleVideoSequenceProc(void* pUserData, CUVIDEOFORMAT* pVideoFormat);
    static int CUDAAPI HandlePictureDecodeProc(void* pUserData, CUVIDPICPARAMS* pPicParams);
    static int CUDAAPI HandlePictureDisplayProc(void* pUserData, CUVIDPARSERDISPINFO* pDispInfo);

public:
    NvDecoder();
    ~NvDecoder();

    // Kuryeden (Demuxer) gelen paketi GPU'ya fırlatır
    void decodePacket(const uint8_t* data, int size);

    // Çözülmüş bir kare varsa VRAM adresini döner (Zero-Copy!)
    bool getDecodedFrame(CUdeviceptr* pFrameData, unsigned int* pPitch);

    int getWidth() const { return m_width; }
    int getHeight() const { return m_height; }

    // İşlem bittiğinde kilidi açmak için:
    void releaseFrame(CUdeviceptr pFrameData) {
        if (m_hDecoder && pFrameData) {
            cuvidUnmapVideoFrame(m_hDecoder, pFrameData);
        }
    }
};


#endif //CUDAVISIONENGINE_NVDECODER_H