#pragma once

#include "Data/WorkspaceStateData.h"
#include "AssetsManager/IconManager.h"
#include <vector>
#include <string>

namespace UIRegistry {

    // Sol ve Orta paneller için (Canvas araçları)
    struct ToolUI_Info {
        Kdata::CanvasTool id;
        std::string name;
        std::string tooltip;
        Icon icon;
        std::string info;
    };

    // Sağ panel için (Ayar araçları)
    struct AdjustmentUI_Info {
        Kdata::AdjustmentTool id;
        std::string name;
        std::string tooltip;
        Icon icon;
        std::string info;
    };

    struct InstantActionUI_Info {
        Kdata::InstantAction id;
        std::string name;
        std::string tooltip;
        Icon icon;
        std::string info;
    };

    class ToolRegistry {
    public:
        // Tüm panellerin veri çekme fonksiyonları (Sembol hatalarını çözer)
        static const std::vector<ToolUI_Info>& GetCanvasTools();
        static const std::vector<AdjustmentUI_Info>& GetAdjustmentTools();
        static const std::vector<ToolUI_Info>& GetCenterTools();
        static const std::vector<InstantActionUI_Info>& GetTopRightTools();
    };

}