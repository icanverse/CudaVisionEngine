#pragma once
#include <string>
// Gerekirse Node veya IsoLine yapılarını buraya include edebilirsin

namespace Kdata {

    // ==========================================
    // BAĞIMSIZ EDİTÖRLER VE MODALLAR
    // ==========================================
    enum class ExclusiveEditor {
        NONE,
        ISO_DEPTH,              // İzohips Derinlik Editörü
        NODE_GRAPH,             // Düğüm Grafiği Editörü
        TEXT_EDITOR,            // Gelişmiş Metin Editörü
        SAVE_AS_DIALOG,         // Farklı Kaydet Modalı
        EXPORT_SETTINGS,        // Çıktı Ayarları Penceresi
        IMPORT_DIALOG,          // İçe Aktarma Modalı
        PREFERENCES             // Motor/Proje Ayarları
    };

    // ==========================================
    // EDİTÖR BAĞLAMLARI (Gelecekte şişecek kısımlar)
    // ==========================================
    
    // İzohips editörünün kendi karmaşık verileri
    struct IsoDepthContext {
        // İleride std::vector<IsoLine> isoLines; gibi veriler buraya gelecek
        float currentDepthValue = 0.5f;
    };

    // Çıktı ayarlarının verileri
    struct ExportContext {
        int format = 0;         // 0: PNG, 1: JPG, 2: KVLCM
        int quality = 100;
        std::string exportPath = "";
    };

    // Node tabanlı sistem verileri
    struct NodeGraphContext {
        // Node bağlantıları, aktif node vb.
    };

    // ==========================================
    // EDİTÖR YÖNETİCİSİ
    // ==========================================
    struct EditorData {
        ExclusiveEditor activeEditor = ExclusiveEditor::NONE;

        IsoDepthContext isoDepthCtx;
        ExportContext exportCtx;
        NodeGraphContext nodeGraphCtx;

        bool isEditorActive() const {
            return activeEditor != ExclusiveEditor::NONE;
        }
    };

}