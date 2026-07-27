#pragma once

#include "AssetsManager/IconManager.h"

#include <string>
#include <vector>

enum class LayerToolAction {
    NONE,
    MOVE_DOWN,
    OPEN_LAYERS,
    ADD_LAYER,
    LOCK,
    UNION_LAYERS,
    TOGGLE_VISIBLE,
    SELECT_LAYER
};

struct LayerPanelItem {
    int id;
    std::string name;
    unsigned int thumbnailTextureId = 0;
    bool visible = true;
    bool locked = false;
};

class QuickLayersToolbox {
public:
    QuickLayersToolbox();
    ~QuickLayersToolbox();

    void render(float displayWidth, float displayHeight);
    LayerToolAction consumeLastAction();

    void setLayers(const std::vector<LayerPanelItem>& newLayers);
    const std::vector<LayerPanelItem>& getLayers() const { return layers; }

    int getSelectedLayerId() const { return selectedLayerId; }
    int getLastChangedLayerId() const { return lastChangedLayerId; }

    static float getPanelWidth() { return 400.0f; }
    static float getPanelTop() { return 288.0f; }

private:
    struct Tool {
        LayerToolAction id;
        std::string name;
        Icon icon;
        std::string tooltip;
    };

    void renderToolbar(float iconSide, float spacing);
    void renderLayerList();
    void renderLayerRow(LayerPanelItem& layer);

    LayerToolAction lastAction;
    int selectedLayerId = -1;
    int lastChangedLayerId = -1;
    std::vector<Tool> availableTools;
    std::vector<LayerPanelItem> layers;
};