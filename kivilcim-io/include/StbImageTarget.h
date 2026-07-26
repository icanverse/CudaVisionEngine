#ifndef CUDAVISIONENGINE_STBIMAGETARGET_H
#define CUDAVISIONENGINE_STBIMAGETARGET_H

#include "IRenderTarget.h"

class StbImageTarget : public IRenderTarget {
private:
    const char* filename;

public:
    StbImageTarget(const char* filepath);
    ~StbImageTarget() override = default;

    void present(unsigned char* data, int width, int height, int channels) override;
};


#endif //CUDAVISIONENGINE_STBIMAGETARGET_H