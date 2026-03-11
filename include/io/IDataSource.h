#ifndef CUDAVISIONENGINE_IDATASOURCE_H
#define CUDAVISIONENGINE_IDATASOURCE_H

class IDataSource {
public:
    virtual ~IDataSource() = default;

    // Motor her döngüde bu fonksiyonu çağırıp yeni kareyi (Frame) isteyecek.
    // Geriye CPU'daki ham piksel verisini (unsigned char*) döndürecek.
    virtual unsigned char* grabNextFrame() = 0;

    // Motor işi bitince belleği temizlemek için bu fonksiyonu çağıracak.
    virtual void releaseFrame(unsigned char* data) = 0;

    // Motorun VRAM'de ne kadar yer ayıracağını bilmesi için boyutlar:
    virtual int getWidth() const = 0;
    virtual int getHeight() const = 0;
    virtual int getChannels() const = 0;
};


#endif //CUDAVISIONENGINE_IDATASOURCE_H