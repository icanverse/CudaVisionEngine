#ifndef CUDAVISIONENGINE_IRENDERTARGET_H
#define CUDAVISIONENGINE_IRENDERTARGET_H

class IRenderTarget {
public:
    virtual ~IRenderTarget() = default;

    // Motor GPU'da işlediği veriyi tekrar CPU'ya (unsigned char*) çekip buraya teslim edecek.
    // Bu sınıf da ister diske yazacak, ister OpenGL ile ekrana basacak.
    virtual void present(unsigned char* data, int width, int height, int channels) = 0;
};


#endif //CUDAVISIONENGINE_IRENDERTARGET_H