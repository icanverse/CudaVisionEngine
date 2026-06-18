#pragma once
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include "SceneDescription.h"

class KvlcmParser {
public:
    static SceneDescription parse(const std::string& filepath) {
        SceneDescription desc;
        std::ifstream file(filepath);

        if (!file.is_open()) {
            std::cerr << "[KvlcmParser] HATA: Sahne dosyasi acilamadi -> " << filepath << std::endl;
            return desc;
        }

        std::string line;
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;

            std::istringstream iss(line);
            std::string type, trash, token;
            iss >> type;

            if (type == "CAMERA") {
                iss >> trash >> desc.camPos.x >> desc.camPos.y >> desc.camPos.z
                    >> trash >> desc.camRot.x >> desc.camRot.y >> desc.camRot.z;

                // Varsa kamera filtrelerini listeye ekle
                while (iss >> token) {
                    if (token == "filters") {
                        iss >> token;
                        desc.manifest.cameraFilters.insert(token);
                    }
                }
            }
            else if (type == "LIGHT") {
                LightBlueprint lb;
                iss >> trash >> lb.position.x >> lb.position.y >> lb.position.z
                    >> trash >> lb.color.x >> lb.color.y >> lb.color.z
                    >> trash >> lb.intensity;
                desc.lights.push_back(lb);
            }
            else if (type == "OBJECT") {
                ObjectBlueprint ob;
                iss >> trash >> ob.meshPath >> trash >> ob.matPath
                    >> trash >> ob.position.x >> ob.position.y >> ob.position.z
                    >> trash >> ob.rotation.x >> ob.rotation.y >> ob.rotation.z;

                // Objeyi listeye ekle
                desc.objects.push_back(ob);

                // Gerekli dosyaları alışveriş listesine (Manifest) yaz
                desc.manifest.meshes.insert(ob.meshPath);
                desc.manifest.materials.insert(ob.matPath);
            }
        }

        std::cout << "[KvlcmParser] Reçete cikarildi (" << desc.objects.size() << " obje)." << std::endl;
        return desc;
    }
};