#include "../../include/io/Video/NvDecoder.h"
#include <iostream>

// =========================================================================
// 1. SEQUENCE CALLBACK: Parser videonun formatını çözdüğünde otomatik tetiklenir
// =========================================================================
int CUDAAPI NvDecoder::HandleVideoSequenceProc(void* pUserData, CUVIDEOFORMAT* pVideoFormat) {
    NvDecoder* pDec = static_cast<NvDecoder*>(pUserData);

    pDec->m_width = pVideoFormat->coded_width;
    pDec->m_height = pVideoFormat->coded_height;

    // VRAM üzerinde donanımsal çözücüyü (Decoder) kuruyoruz
    CUVIDDECODECREATEINFO videoDecodeCreateInfo = {};
    videoDecodeCreateInfo.CodecType = pVideoFormat->codec;
    videoDecodeCreateInfo.ulWidth = pVideoFormat->coded_width;
    videoDecodeCreateInfo.ulHeight = pVideoFormat->coded_height;
    videoDecodeCreateInfo.ulNumDecodeSurfaces = pVideoFormat->min_num_decode_surfaces + 3;
    videoDecodeCreateInfo.ChromaFormat = pVideoFormat->chroma_format;
    videoDecodeCreateInfo.OutputFormat = cudaVideoSurfaceFormat_NV12; // Ekrana çıkacak format!
    videoDecodeCreateInfo.ulTargetWidth = pVideoFormat->coded_width;
    videoDecodeCreateInfo.ulTargetHeight = pVideoFormat->coded_height;
    videoDecodeCreateInfo.ulNumOutputSurfaces = 2;
    videoDecodeCreateInfo.ulCreationFlags = cudaVideoCreate_PreferCUVID;

    cuvidCreateDecoder(&pDec->m_hDecoder, &videoDecodeCreateInfo);

    std::cout << "[NVDEC] Donanim Cozucu Ateslendi! Format: NV12 | Cozunurluk: "
              << pDec->m_width << "x" << pDec->m_height << std::endl;

    return videoDecodeCreateInfo.ulNumDecodeSurfaces;
}

// =========================================================================
// 2. DECODE CALLBACK: Donanıma "Pikselleri VRAM'de Çöz!" emrini verdiğimiz yer
// =========================================================================
int CUDAAPI NvDecoder::HandlePictureDecodeProc(void* pUserData, CUVIDPICPARAMS* pPicParams) {
    NvDecoder* pDec = static_cast<NvDecoder*>(pUserData);
    cuvidDecodePicture(pDec->m_hDecoder, pPicParams);
    return 1;
}

// =========================================================================
// 3. DISPLAY CALLBACK: Çözülmüş kare VRAM'de hazır olduğunda tetiklenir
// =========================================================================
int CUDAAPI NvDecoder::HandlePictureDisplayProc(void* pUserData, CUVIDPARSERDISPINFO* pDispInfo) {
    NvDecoder* pDec = static_cast<NvDecoder*>(pUserData);

    std::lock_guard<std::mutex> lock(pDec->m_queueMutex);

    // ZORAKİ DÖNÜŞÜMÜ KALDIRDIK, DİREKT İNDEKSİ ATIYORUZ
    pDec->m_frameQueue.emplace(pDispInfo->picture_index);

    return 1;
}

// =========================================================================
// CLASS METOTLARI
// =========================================================================
NvDecoder::NvDecoder() {
    // 1. Düşük seviye CUDA (Driver API) başlatma
    cuInit(0);
    CUdevice device;
    cuDeviceGet(&device, 0);       // 0 numaralı GPU


    // YENİ VE MODERN KOD: GPU'nun ana bağlamına (Primary Context) bağlanıyoruz
    cuDevicePrimaryCtxRetain(&m_cuContext, device);

    // 2. Parser'ı (Paket Anlamlandırıcı) Kur
    CUVIDPARSERPARAMS parserParams = {};
    parserParams.CodecType = cudaVideoCodec_H264;
    parserParams.ulMaxNumDecodeSurfaces = 10;
    parserParams.pUserData = this;
    parserParams.pfnSequenceCallback = HandleVideoSequenceProc;
    parserParams.pfnDecodePicture = HandlePictureDecodeProc;
    parserParams.pfnDisplayPicture = HandlePictureDisplayProc;

    cuvidCreateVideoParser(&m_hParser, &parserParams);
}

NvDecoder::~NvDecoder() {
    if (m_hParser) cuvidDestroyVideoParser(m_hParser);
    if (m_hDecoder) cuvidDestroyDecoder(m_hDecoder);

    // ESKİ VE SORUNLU KOD (SİL):
    // if (m_cuContext) cuCtxDestroy(m_cuContext);

    // YENİ KOD: Bağlamı güvenle serbest bırak
    if (m_cuContext) {
        CUdevice device;
        cuDeviceGet(&device, 0);
        cuDevicePrimaryCtxRelease(device);
    }
}

void NvDecoder::decodePacket(const uint8_t* data, int size) {
    CUVIDSOURCEDATAPACKET packet = {};
    packet.payload = data;
    packet.payload_size = size;

    // Bu fonksiyon arka planda otomatik olarak
    // Sequence, Decode ve Display Callback'lerini tetikler!
    cuvidParseVideoData(m_hParser, &packet);
}

bool NvDecoder::getDecodedFrame(CUdeviceptr* pFrameData, unsigned int* pPitch) {
    std::lock_guard<std::mutex> lock(m_queueMutex);
    if (m_frameQueue.empty()) return false;

    int picIndex = m_frameQueue.front();
    m_frameQueue.pop();

    CUVIDPROCPARAMS vpp = {};
    vpp.progressive_frame = 1;

    // Donanımdaki kareyi CUDA'nın erişebileceği bir adrese (pFrameData) haritalıyoruz
    cuvidMapVideoFrame(m_hDecoder, picIndex, pFrameData, pPitch, &vpp);

    //İşlem bittikten sonra cuvidUnmapVideoFrame(m_hDecoder, *pFrameData)


    return true;
}