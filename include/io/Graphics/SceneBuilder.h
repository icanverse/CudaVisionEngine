#pragma once
#include <string>
#include <iostream>
#include "KvlcmParser.h"
#include "ObjectLoader.h"
#include "../Graphics/Scene.cuh"

class SceneBuilder {
public:
    static Scene build(const std::string& kvlcmFilepath) {
        Scene finalScene;

        // Dosyayı okut ve reçeteyi al
        SceneDescription blueprint = KvlcmParser::parse(kvlcmFilepath);

        // Eksik dosyaları VRAM'e yüklet
        ObjectLoader::loadManifest(blueprint.manifest);

        // Montaj Hattı: Kamerayı Kur
        finalScene.setCamera(blueprint.camPos, blueprint.camRot);

        // Montaj Hattı: Işıkları Kur
        for (const auto& light : blueprint.lights) {
            finalScene.addLight(light.position, light.color, light.intensity);
        }

        // Montaj Hattı: Objeleri RAM'den çekip Sahneye bas
        for (const auto& objDesc : blueprint.objects) {
            auto meshData = ObjectLoader::getMesh(objDesc.meshPath);
            auto matData = ObjectLoader::getMaterial(objDesc.matPath);

            finalScene.addObject(meshData.first.data(), meshData.first.size(),
                                 meshData.second.data(), meshData.second.size(),
                                 objDesc.position, objDesc.rotation, matData);
        }

        std::cout << "[SceneBuilder] Sahne kusursuz insa edildi: " << kvlcmFilepath << std::endl;
        return finalScene;
    }
};