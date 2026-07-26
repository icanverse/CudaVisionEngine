#pragma once
#include <string>
#include <thread>
#include <mutex>
#include <atomic>
#include <iostream>

extern "C" {
#include <libavformat/avformat.h>
}

#include "Video/NvDecoder.h" // NvDecoder başlığını dahil etmeliyiz

class NetworkStream {
private:
    AVFormatContext* formatContext = nullptr;
    int videoStreamIndex = -1;

    // Asenkron Altyapı Değişkenleri
    std::thread streamThread;
    std::atomic<bool> isStreaming{false};
    std::mutex frameMutex;

    // Vitrin (Son Bilinen Kare)
    CUdeviceptr currentNV12Frame = 0;
    unsigned int currentPitch = 0;

    // Çözücü Referansı
    NvDecoder* pDecoder = nullptr;

    // Arka planda dönecek gizli işçi fonksiyon
    void decodeLoop() {
        AVPacket packet;
        av_init_packet(&packet);
        CUdeviceptr newFrame = 0;
        unsigned int newPitch = 0;
        CUdeviceptr oldFrameToRelease = 0;

        // Bu ipliğe de ekran kartını tanıtıyoruz (Siyah ekran hatasını çözer)
        cudaSetDevice(0);
        cudaFree(0);

        while (isStreaming) {
            if (av_read_frame(formatContext, &packet) >= 0) {
                pDecoder->decodePacket(packet.data, packet.size);
                av_packet_unref(&packet);

                while (pDecoder->getDecodedFrame(&newFrame, &newPitch)) {
                    // Vitrini Güncelle (Kilitli Bölge)
                    frameMutex.lock();
                    oldFrameToRelease = currentNV12Frame;
                    currentNV12Frame = newFrame;
                    currentPitch = newPitch;
                    frameMutex.unlock();

                    // Eskiyi İade Et
                    if (oldFrameToRelease != 0 && oldFrameToRelease != newFrame) {
                        pDecoder->releaseFrame(oldFrameToRelease);
                    }
                }
            }
        }
    }

public:
    NetworkStream(const std::string& rtspUrl) {
        avformat_network_init();
        AVDictionary* options = nullptr;
        av_dict_set(&options, "rtsp_transport", "udp", 0);
        av_dict_set(&options, "fflags", "nobuffer", 0);
        av_dict_set(&options, "flags", "low_delay", 0);

        if (avformat_open_input(&formatContext, rtspUrl.c_str(), nullptr, &options) != 0) {
            std::cerr << "[NetworkStream] Telefona baglanilamadi!" << std::endl;
            exit(1);
        }
        avformat_find_stream_info(formatContext, nullptr);
    }

    ~NetworkStream() {
        stopStream();
        if (formatContext) {
            avformat_close_input(&formatContext);
            avformat_free_context(formatContext);
        }

    }

    // Dışarıdan Motoru Ateşleme Komutu
    void startStream(NvDecoder* decoder) {
        if (!isStreaming) {
            pDecoder = decoder;
            isStreaming = true;
            streamThread = std::thread(&NetworkStream::decodeLoop, this);
            std::cout << "[NetworkStream] Asenkron ag isleyicisi arka planda baslatildi!" << std::endl;
        }
    }

    // Güvenli Kapanış
    void stopStream() {
        if (isStreaming) {
            isStreaming = false;
            if (streamThread.joinable()) {
                streamThread.join();
            }
            std::cout << "[NetworkStream] Ag isleyicisi guvenle kapatildi." << std::endl;
        }
    }

    // Ana döngünün çok temiz bir şekilde vitrinden kareyi almasını sağlayan fonksiyon
    bool getLatestFrame(CUdeviceptr& outFrame, unsigned int& outPitch) {
        std::lock_guard<std::mutex> lock(frameMutex); // Otomatik kilit mekanizması
        if (currentNV12Frame == 0) return false;

        outFrame = currentNV12Frame;
        outPitch = currentPitch;
        return true;
    }
};