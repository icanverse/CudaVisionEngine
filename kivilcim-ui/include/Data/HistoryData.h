#pragma once

namespace Kdata {

    struct HistoryData {
        int currentStepIndex = -1;      // Şu anki geçmiş adımı
        int totalSteps = 0;             // Toplam kaydedilmiş adım sayısı
        int maxHistorySize = 50;        // RAM'i şişirmemek için maksimum limit
        
        bool canUndo = false;
        bool canRedo = false;
    };

}