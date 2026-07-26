#pragma once

#include <string>
#include <vector>

struct GLFWwindow;

enum class WorkspaceMenuAction {
    NONE,
    NEW_FILE,
    OPEN_FILE,
    IMPORT_FILE,
    SAVE_FILE,
    SAVE_AS,
    EXPORT_FILE,
    CLOSE_WORKSPACE,
    UNDO,
    REDO,
    ADD_LAYER,
    CONTRAST_FILTER,
    TEMPERATURE_FILTER,
    GRAYSCALE_FILTER,
    BLUR_FILTER,
    SHARPEN_FILTER,
    LAYER_VISIBILITY_CHANGED,


    /// Araçlar
    // >>> Oluştur
    CREATE_PHYSICAL_DEPTH_LAYER,
    CREATE_VIRTUAL_LIGHT
};

struct WorkspaceLayerMenuItem {
    int id;
    std::string name;
    bool visible;
};

// Yalnizca ayrik Workspace viewport'u icin kullanilir. Ana pencerenin
// TopPanel sinifindan bagimsizdir; bu nedenle mevcut panel davranisini degistirmez.
class WorkspaceTopPanel {
public:
    bool render(GLFWwindow* window, float displayWidth, unsigned int logoTextureId = 0);
    static float getPanelHeight();
    static void setSharedLogoTexture(unsigned int logoTextureId) {
        if (logoTextureId != 0) sharedLogoTextureId = logoTextureId;
    }

    void setLayers(const std::vector<WorkspaceLayerMenuItem>& newLayers);
    const std::vector<WorkspaceLayerMenuItem>& getLayers() const { return layers; }

    WorkspaceMenuAction consumeLastAction();
    int getLastChangedLayerId() const { return lastChangedLayerId; }

private:
    bool isDragging = false;
    int dragOffsetX = 0;
    int dragOffsetY = 0;
    int lastChangedLayerId = -1;
    WorkspaceMenuAction lastAction = WorkspaceMenuAction::NONE;
    std::vector<WorkspaceLayerMenuItem> layers;
    inline static unsigned int sharedLogoTextureId = 0;
};