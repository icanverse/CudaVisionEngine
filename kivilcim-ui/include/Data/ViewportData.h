#pragma once

namespace Kdata {

    struct ViewportData {
        float zoomLevel = 1.0f;         // 1.0 = %100, 2.0 = %200 vb.
        float cameraPosX = 0.0f;        // X ekseninde kaydırma (Pan)
        float cameraPosY = 0.0f;        // Y ekseninde kaydırma (Pan)
        
        bool showGrid = false;          // Izgara görünürlüğü
        bool showRulers = true;         // Cetvel görünürlüğü
        bool showGuideLines = false;    // Kılavuz çizgileri
    };

}