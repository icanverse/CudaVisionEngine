#pragma once
#include <string>
extern "C" {
#include <libavformat/avformat.h>
}

class NetworkStream {
private:
    AVFormatContext* formatContext = nullptr;
    int videoStreamIndex = -1;
public:
    NetworkStream(const std::string& rtspUrl) {
        avformat_network_init(); // FFmpeg Ağ modülünü uyandır

        // RTSP akışlarında gecikmeyi sıfıra indirmek için kritik FFmpeg parametreleri
        AVDictionary* options = nullptr;
        av_dict_set(&options, "rtsp_transport", "udp", 0); // UDP üzerinden hızlı iletim
        av_dict_set(&options, "fflags", "nobuffer", 0);    // Tamponlamayı (buffering) kapat
        av_dict_set(&options, "flags", "low_delay", 0);   // Düşük gecikme modunu aç

        if (avformat_open_input(&formatContext, rtspUrl.c_str(), nullptr, &options) != 0) {
            std::cerr << "[NetworkStream] Telefona baglanilamadi!" << std::endl;
            exit(1);
        }

        avformat_find_stream_info(formatContext, nullptr);
        // Video stream indeksini bul...
    }

    // Demuxer'daki readPacket'in aynısı, ama ağdan canlı veri çeker!
    bool readLivePacket(AVPacket* packet) {
        return av_read_frame(formatContext, packet) >= 0;
    }
};