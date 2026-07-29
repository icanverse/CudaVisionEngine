#pragma once
#include "ProjectData.h"
#include "LayerData.h"
#include "MaskData.h"
#include "ToolData.h"
#include "EditorData.h"
#include "ViewportData.h"
#include "HistoryData.h"
#include "ResourceData.h"
#include "PreferenceData.h"

namespace Kdata {

    struct WorkspaceStateData {

        // 1. Proje ve Katman Hiyerarşisi (Diske Kaydedilenler)
        ProjectData project;
        LayerData layers;

        // 2. Arayüz ve Görünüm (Ekranda Görülenler)
        ViewportData viewport;

        // 3. Etkileşim ve Araçlar (Kullanıcının O An Yaptıkları)
        ToolData tools;
        EditorData editors;
        MaskData activeSelection;

        // 4. Sistem ve Bellek (Motorun Arka Planı)
        HistoryData history;
        ResourceData resources;
        PreferenceData preferences;


        bool showPreferences = false;

        // Proje ayağa kalktığında ilk verilerle başlatılır
        WorkspaceStateData()
            : project(0, "Yeni Kıvılcım Projesi") {

            // Varsayılan bir arkaplan katmanı oluştur
            Layer bgLayer;
            bgLayer.id = 0;
            bgLayer.name = "Background";
            layers.layers.push_back(bgLayer);
            layers.activeLayerIndex = 0;
        }

        // Uygulama genelinde durumu sıfırlamak için
        void resetState() {
            // Araçları ve Editörleri Sıfırla
            tools.activeCanvasTool = CanvasTool::NONE;
            tools.activeAdjustment = AdjustmentTool::NONE;
            editors.activeEditor = ExclusiveEditor::NONE;

            // Katmanları Sıfırla
            layers.activeLayerIndex = -1;
            layers.layers.clear();

            // Seçim ve Maskeleri Sıfırla
            activeSelection.clear();

            // (Opsiyonel olarak viewport ve history de burada sıfırlanabilir)
            viewport.zoomLevel = 1.0f;
            history.currentStepIndex = -1;
        }
    };

}