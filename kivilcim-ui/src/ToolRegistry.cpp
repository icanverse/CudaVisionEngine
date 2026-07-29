#include "../include/Data/ToolRegistry.h"

namespace UIRegistry {

    const std::vector<ToolUI_Info>& ToolRegistry::GetCanvasTools() {
        // static anahtar kelimesi sayesinde bu liste bellekte sadece 1 kez oluşturulur.
        static const std::vector<ToolUI_Info> canvasTools = {
            { Kdata::CanvasTool::MOVE,              "Kaydir",       "Tuvali Kaydir (El Araci)",         Icon::Move,                    "Calisma alaninda gezinmeyi saglar" },
            { Kdata::CanvasTool::SELECT_RECTANGLE,  "Kare Secim",   "Dikdortgen Secim Araci (ROI)",     Icon::Select_Region_Rectangle, "Dikdortgen seklinde alan secer" },
            { Kdata::CanvasTool::SELECT_FREE,       "Serbest Secim","Serbest Alan Secimi (Lasso)",      Icon::Select_Region_Free,      "Tuvalde serbest sekilde alan secer" },
            { Kdata::CanvasTool::BRUSH,             "Firca",        "Serbest Boyama Araci",             Icon::Brush,                   "Pikselleri serbestce boyar" },
            { Kdata::CanvasTool::COLOR_PICKER,      "Renk",         "Renk Secici",                      Icon::Color,                   "Firca ve metin rengini belirler" },
            { Kdata::CanvasTool::TEXT,              "Metin",        "Metin Araci",                      Icon::Text,                    "Tuvale yazi katmani ekler" }
        };
        
        return canvasTools;
    }

    // ToolRegistry.cpp içerisine eklenecek kısım:
    const std::vector<AdjustmentUI_Info>& ToolRegistry::GetAdjustmentTools() {
        static const std::vector<AdjustmentUI_Info> adjustmentTools = {
            { Kdata::AdjustmentTool::BRIGHTNESS_CONTRAST, "Kontrast##QuickRight", "Kontrast ayarlarini acar", Icon::Contrast, "Renk kontrastını ayarlar" },
            { Kdata::AdjustmentTool::COLOR_BALANCE,       "Sicaklik##QuickRight", "Renk sicakligi ayarlarini acar", Icon::Temperature, "Sıcaklık ve ton ayarları" }
            // İleride buraya HUE_SATURATION, BLUR_SHARPEN vs. eklenebilir
        };
        return adjustmentTools;
    }


    const std::vector<ToolUI_Info>& ToolRegistry::GetCenterTools() {
        static const std::vector<ToolUI_Info> centerTools = {
            { Kdata::CanvasTool::SHAPE_CIRCLE, "Daire##TopCenter",      "Daire ekler",             Icon::Circle,     "" },
            { Kdata::CanvasTool::SHAPE_LINE,   "Cizgi##TopCenter",      "Cizgi ekler",             Icon::Line,       "" },
            { Kdata::CanvasTool::SHAPE_SQUARE, "Dikdortgen##TopCenter", "Dikdortgen ekler",        Icon::Square,     "" },
            { Kdata::CanvasTool::VECTOR_PATH,  "Vektor##TopCenter",     "Vektor yolu ekler",       Icon::Vector,     "" },
            { Kdata::CanvasTool::BRUSH,        "Firca##TopCenter",      "Firca aracini secer",     Icon::Brush,      "" },
            { Kdata::CanvasTool::ERASER,       "Silgi##TopCenter",      "Silgi aracini secer",     Icon::Erase,      "" },
            { Kdata::CanvasTool::COLOR_PICKER, "Renk##TopCenter",       "Cizim rengini secer",     Icon::Color,      "" },
            { Kdata::CanvasTool::TEXT,         "Metin##TopCenter",      "Metin ekler",             Icon::Text,       "" },
            { Kdata::CanvasTool::TEXT_SIZE,    "MetinBoyutu##TopCenter","Metin boyutunu ayarlar",  Icon::Text_Size,  "" },
            { Kdata::CanvasTool::TEXT_COLOR,   "MetinRengi##TopCenter", "Metin rengini ayarlar",   Icon::Text_Color, "" }
        };
        return centerTools;
    }

    const std::vector<InstantActionUI_Info>& ToolRegistry::GetTopRightTools() {
        static const std::vector<InstantActionUI_Info> topRightTools = {
            {Kdata::InstantAction::UNDO,              "Geri Al",      "Son islemi geri alir",      Icon::Undo,              "Islem gecmisinde bir adim geriye gider"},
            {Kdata::InstantAction::REDO,              "Yinele",       "Geri alinan islemi yineler",Icon::Redo,              "Islem gecmisinde bir adim ileri gider"},
            {Kdata::InstantAction::CANVAS_TURN_LEFT,  "Sola Dondur",  "90 Derece Sola Dondur",     Icon::Turn_Left,         "Tuvali saat yonunun tersine cevirir"},
            {Kdata::InstantAction::CANVAS_TURN_RIGHT, "Saga Dondur",  "90 Derece Saga Dondur",     Icon::Turn_Right,        "Tuvali saat yonunde cevirir"},
            {Kdata::InstantAction::CANVAS_ZOOM_IN,    "Yakinlastir",  "Tuvali Yakinlastir (+)",    Icon::Zoom_In,           "Calisma alanina yaklasir"},
            {Kdata::InstantAction::CANVAS_ZOOM_OUT,   "Uzaklastir",   "Tuvali Uzaklastir (-)",     Icon::Zoom_Out,          "Calisma alanindan uzaklasir"},
            {Kdata::InstantAction::MIRROR_HORIZONTAL, "Yatay Aynala", "Yatay Eksende Aynala",      Icon::Mirror_Horizontal, "Goruntuyu yatay eksende cevirir"},
            {Kdata::InstantAction::MIRROR_VERTICAL,   "Dikey Aynala", "Dikey Eksende Aynala",      Icon::Mirror_Vertical,   "Goruntuyu dikey eksende cevirir"}
        };
        return topRightTools;
    }

}