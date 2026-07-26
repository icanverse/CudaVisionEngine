#ifndef CUDAVISIONENGINE_DEMUXER_H
#define CUDAVISIONENGINE_DEMUXER_H

#pragma once
#include <string>

// FFmpeg saf C kütüphanesi olduğu için C++ projesinde extern "C" ile sarmalamak zorundayız
extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavcodec/bsf.h>
}

class Demuxer {
private:
    AVFormatContext* fmtCtx = nullptr;
    AVBSFContext* bsfContext = nullptr;
    int videoStreamIndex = -1;
    AVPacket* packet = nullptr;

public:
    Demuxer(const std::string& filepath);
    ~Demuxer();

    // Bir sonraki video paketini (AVPacket) okur
    bool readPacket(uint8_t** data, int* size);
    void freePacket();

    // Video çözünürlüğünü motorumuza bildirmek için
    int getWidth() const;
    int getHeight() const;
};


#endif //CUDAVISIONENGINE_DEMUXER_H