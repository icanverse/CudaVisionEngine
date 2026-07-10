#pragma once
#include "UI/Data/ProjectData.h"
#include <functional>

class Workspace {
public:
    Workspace();
    void render(float displayWidth, float displayHeight);

    void loadProject(Kivilcim::ProjectData* project);

    void setOnCloseCallback(std::function<void()> callback) {
        onClose = callback;
    }

    // GPU'dan gelen taze Framebuffer dokusunu (Texture) arayüze aktarır
    void updateShaderTexture(unsigned int texID) {
        if (activeProject) activeProject->textureID = texID;
    }

    // --- YENİ: CUDA SHADER KONTROL PARAMETRELERİ ---
    float waveFrequency;
    float waveSpeed;
    float waveAmplitude;
    float liquidColor[3];
    float shaderBgColor[3];
    float liquidAlpha;

private:
    Kivilcim::ProjectData* activeProject;
    std::function<void()> onClose;

    float bgAlpha;
};