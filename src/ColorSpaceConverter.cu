__global__ void rgbToHsv(const float* A, float* Result, int width, int height, int chanel) {

    unsigned int dx = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int dy = threadIdx.y + blockDim.y * blockIdx.y;

    if (dx < width && dy < height) {
        unsigned int index = (dx + dy * width) * chanel;

        // 1. Veriyi en hızlı belleğe (Register) çek
        float r = A[index];
        float g = A[index + 1];
        float b = A[index + 2];

        float c_max = fmaxf(r, fmaxf(g, b)); // V (Value - Parlaklık)
        float c_min = fminf(r, fminf(g, b));
        float delta = c_max - c_min;

        // 2. Hesaplamaları Global Bellek yerine Yerel Değişkenlerde yap
        float h = 0.0f;
        float s = 0.0f;
        float v = c_max; // V doğrudan c_max'a eşittir

        // 3. Saturation (Doygunluk)
        if (c_max > 0.0f) {
            s = delta / c_max;
        }

        // 4. Hue (Renk Tonu - Açısal Hesaplama)
        if (delta > 0.0f) {
            if (c_max == r) {
                h = 60.0f * fmodf(((g - b) / delta), 6.0f);
            } else if (c_max == g) {
                h = 60.0f * (((b - r) / delta) + 2.0f);
            } else if (c_max == b) {
                h = 60.0f * (((r - g) / delta) + 4.0f);
            }

            // Açı negatifse pozitife tamamla
            if (h < 0.0f) {
                h += 360.0f;
            }
        }

        // 5. EN SON ADIM: Global belleğe tek seferde yaz
        Result[index]     = h;
        Result[index + 1] = s;
        Result[index + 2] = v;

        // RGBA (4 Kanallı) ise Alfa kanalını koru
        if (chanel == 4) {
            Result[index + 3] = A[index + 3];
        }
    }
}

__global__ void hsvToRgb(const float* A, float* Result, int width, int height, int chanel) {
    unsigned int dx = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int dy = threadIdx.y + blockDim.y * blockIdx.y;

    if (dx < width && dy < height) {
        unsigned int index = (dx + dy * width) * chanel;

        float h = A[index];
        float s = A[index + 1];
        float v = A[index + 2];

        // Temel bileşenleri hesapla
        float chroma = v * s;
        // DİKKAT: 1.0f - kısmı eklendi!
        float x = chroma * (1.0f - fabsf(fmodf(h / 60.0f, 2.0f) - 1.0f));
        float m = v - chroma;

        float r = 0.0f;
        float g = 0.0f;
        float b = 0.0f;

        // Renk tekerleğindeki (Color Wheel) 6 dilim
        if (h >= 0.0f && h < 60.0f) {
            r = chroma;
            g = x;
            b = 0.0f;
        } else if (h >= 60.0f && h < 120.0f) {
            r = x;
            g = chroma;
            b = 0.0f;
        } else if (h >= 120.0f && h < 180.0f) {
            r = 0.0f;
            g = chroma;
            b = x;
        } else if (h >= 180.0f && h < 240.0f) {
            r = 0.0f;
            g = x;
            b = chroma;
        } else if (h >= 240.0f && h < 300.0f) {
            r = x;
            g = 0.0f;
            b = chroma;
        } else if (h >= 300.0f && h < 360.0f) {
            r = chroma;
            g = 0.0f;
            b = x;
        }

        // Parlaklık tabanını (m) ekleyerek belleğe yaz
        Result[index]     = r + m;
        Result[index + 1] = g + m;
        Result[index + 2] = b + m;

        // Alfa kanalı koruması
        if (chanel == 4) {
            Result[index + 3] = A[index + 3];
        }
    }
}

__global__ void rgbToYuv(const float* A, float* Result, int width, int height, int channel) {
    unsigned int tx = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int ty = threadIdx.y + blockDim.y * blockIdx.y;

    unsigned int dx = tx + blockDim.x * blockIdx.x;
    unsigned int dy = ty + blockDim.y * blockIdx.y;

    if (dx < width && dy < height) {
        unsigned int index = (dx + dy * width) * channel;

        float r = A[index];
        float g = A[index + 1];
        float b = A[index + 2];

        float y = 0.299*r + 0.587*g + 0.114*b;
        float u = -0.1687*r - 0.3313*g + 0.5*b + 0.5;
        float v = 0.5*r - 0.4187*g - 0.0813*b + 0.5;

        y = fminf(fmaxf(y, 0.0f), 255.0f);
        u = fminf(fmaxf(u, 0.0f), 255.0f);
        v = fminf(fmaxf(v, 0.0f), 255.0f);

        Result[index]     = y;
        Result[index + 1] = u;
        Result[index + 2] = v;


    }
}

__global__ void yuvToRgb(const float* A, float* Result, int width, int height, int channels) {
    unsigned int dx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int dy = blockIdx.y * blockDim.y + threadIdx.y;

    if (dx < width && dy < height) {
        // 2. Bellek indeksi (Interleaved YUVYUV...)
        unsigned int index = (dy * width + dx) * channels;

        float y = A[index];
        float u = A[index + 1] - 0.5f;
        float v = A[index + 2] - 0.5f;

        float r = y + 1.402f * v;
        float g = y - 0.344136f * u - 0.714136f * v;
        float b = y + 1.772f * u;

        // Clamping
        Result[index]     = fminf(fmaxf(r, 0.0f), 255.0f);
        Result[index + 1] = fminf(fmaxf(g, 0.0f), 255.0f);
        Result[index + 2] = fminf(fmaxf(b, 0.0f), 255.0f);
    }
}