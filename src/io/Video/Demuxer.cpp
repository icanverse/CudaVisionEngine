#include "../../include/io/Video/Demuxer.h"
#include <iostream>

Demuxer::Demuxer(const std::string& filepath) {
    // 1. FFmpeg Format Context oluştur
    fmtCtx = avformat_alloc_context();

    // 2. MP4/MKV dosyasını aç
    if (avformat_open_input(&fmtCtx, filepath.c_str(), nullptr, nullptr) != 0) {
        std::cerr << "[Demuxer] HATA: Video dosyasi acilamadi -> " << filepath << std::endl;
        return;
    }

    // 3. İçindeki stream (video/ses) bilgilerini ayrıştır
    if (avformat_find_stream_info(fmtCtx, nullptr) < 0) {
        std::cerr << "[Demuxer] HATA: Stream bilgileri okunamadi!" << std::endl;
        return;
    }

    // 4. Bize sadece Video stream'i lazım (Sesleri ve alt yazıları pas geçiyoruz)
    for (unsigned int i = 0; i < fmtCtx->nb_streams; i++) {
        if (fmtCtx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            videoStreamIndex = i;
            break;
        }
    }

    if (videoStreamIndex == -1) {
        std::cerr << "[Demuxer] HATA: Dosyada video akisi bulunamadi!" << std::endl;
        return;
    }

    // 5. Paketleri tutacağımız belleği ayır
    packet = av_packet_alloc();

    // 6. YENİ: NVDEC'in anlaması için MP4 paketlerini Annex-B'ye çeviren filtreyi kur!
    const AVBitStreamFilter* bsf = av_bsf_get_by_name("h264_mp4toannexb");
    av_bsf_alloc(bsf, &bsfContext);
    avcodec_parameters_copy(bsfContext->par_in, fmtCtx->streams[videoStreamIndex]->codecpar);
    av_bsf_init(bsfContext);
    // ------------------------------------------------------------------------

    std::cout << "[Demuxer] Video basariyla yuklendi! Cozunurluk: "
              << getWidth() << "x" << getHeight() << std::endl;
}

Demuxer::~Demuxer() {
    // C tabanlı kütüphanelerde bellek temizliği hayati önem taşır
    if (packet) av_packet_free(&packet);
    if (fmtCtx) avformat_close_input(&fmtCtx);
}

int Demuxer::getWidth() const {
    if (videoStreamIndex == -1 || !fmtCtx) return 0;
    return fmtCtx->streams[videoStreamIndex]->codecpar->width;
}

int Demuxer::getHeight() const {
    if (videoStreamIndex == -1 || !fmtCtx) return 0;
    return fmtCtx->streams[videoStreamIndex]->codecpar->height;
}

bool Demuxer::readPacket(uint8_t** data, int* size) {
    while (av_read_frame(fmtCtx, packet) >= 0) {
        if (packet->stream_index == videoStreamIndex) {

            // 1. Paketi Çevirmene Yolla
            av_bsf_send_packet(bsfContext, packet);

            // 2. Çevrilmiş halini (Annex-B) geri al
            if (av_bsf_receive_packet(bsfContext, packet) == 0) {
                *data = packet->data;
                *size = packet->size;
                return true;
            }
        }
        av_packet_unref(packet);
    }
    return false;
}

void Demuxer::freePacket() {
    if (packet) {
        av_packet_unref(packet);
    }
}