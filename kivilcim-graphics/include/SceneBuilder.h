#pragma once

#include <iostream>
#include <string>

#include "../../kivilcim-ui/include/Persistence/KvlcmProjectParser.h"
#include "../../kivilcim-io/include/kvlcm/MaterialLoader.h"
#include "../../kivilcim-io/include/kvlcm/ObjLoader.h"
#include "Scene.cuh"
#include "kvlcm/KvlcmSceneParser.h"

class SceneBuilder {
public:
    static Scene build(const std::string& kvlcmFilepath) {
        Scene finalScene;

        SceneDescription blueprint =
            KvlcmSceneParser::parse(kvlcmFilepath);

        ObjLoader::loadManifest(blueprint.manifest);
        MaterialLoader::loadManifest(blueprint.manifest);

        std::cout
            << "[SceneBuilder] Tum asset'ler bellege alindi."
            << std::endl;

        finalScene.setCamera(
            blueprint.camPos,
            blueprint.camRot
        );

        for (const auto& light : blueprint.lights) {
            finalScene.addLight(
                light.position,
                light.color,
                light.intensity
            );
        }

        for (const auto& objDesc : blueprint.objects) {
            auto meshData =
                ObjLoader::getMesh(objDesc.meshPath);

            auto matData =
                MaterialLoader::getMaterial(objDesc.matPath);

            finalScene.addObject(
                meshData.first.data(),
                meshData.first.size(),
                meshData.second.data(),
                meshData.second.size(),
                objDesc.position,
                objDesc.rotation,
                matData
            );
        }

        std::cout
            << "[SceneBuilder] Sahne kusursuz insa edildi: "
            << kvlcmFilepath
            << std::endl;

        return finalScene;
    }
};
