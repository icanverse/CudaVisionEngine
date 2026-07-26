//
// Created by Can on 11.03.2026.
//

#ifndef CUDAVISIONENGINE_STBIMAGESOURCE_H
#define CUDAVISIONENGINE_STBIMAGESOURCE_H

#include "IDataSource.h"

class StbImageSource : public IDataSource {
private:
    const char* filename;
    int width;
    int height;
    int channels;
    bool isLoaded; // Fotoğraf olduğu için sonsuz döngüye girmesini engeller

public:
    StbImageSource(const char* filepath);
    ~StbImageSource() override = default;

    unsigned char* grabNextFrame() override;
    void releaseFrame(unsigned char* data) override;

    int getWidth() const override { return width; }
    int getHeight() const override { return height; }
    int getChannels() const override { return channels; }
};


#endif //CUDAVISIONENGINE_STBIMAGESOURCE_H