#pragma once

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>

#include "../../../kivilcim-core/include/SceneDescription.h"
#include "Scene.cuh"

class MaterialLoader {
private:
    static inline std::unordered_map<std::string, Material> materialCache;

    static bool loadMaterialFile(
        const std::string& filepath,
        Material& outMat
    ) {
        std::ifstream file(filepath);
        if (!file.is_open()) return false;

        outMat = {
            {1.0f, 0.0f, 1.0f},
            0.1f,
            0.8f,
            0.2f,
            32.0f
        };

        outMat.effectFlags = 0;
        outMat.glowSpeed = 0.0f;
        outMat.scanFreq = 0.0f;
        outMat.scanSpeed = 0.0f;
        outMat.tronGridSize = 0.0f;
        outMat.tronThickness = 0.0f;
        outMat.radarFreq = 0.0f;
        outMat.radarSpeed = 0.0f;
        outMat.jitterIntensity = 0.0f;
        outMat.dissolveSpeed = 0.0f;

        std::string line;
        std::string token;

        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;

            std::istringstream iss(line);
            iss >> token;

            if (token == "COLOR") {
                iss >> token
                    >> outMat.color.x
                    >> token
                    >> outMat.color.y
                    >> token
                    >> outMat.color.z;
            }
            else if (token == "AMBIENT") {
                iss >> outMat.ambient;
            }
            else if (token == "DIFFUSE") {
                iss >> outMat.diffuse;
            }
            else if (token == "SPECULAR") {
                iss >> outMat.specular;
            }
            else if (token == "SHININESS") {
                iss >> outMat.shininess;
            }
            else if (token == "OPACITY") {
                iss >> outMat.opacity;
            }
            else if (token == "EFFECT") {
                std::string effectName;
                iss >> effectName;

                if (effectName == "GLOW") {
                    outMat.effectFlags |= 1;
                    iss >> outMat.glowSpeed;
                }
                else if (effectName == "SCANLINE") {
                    outMat.effectFlags |= 2;
                    iss >> outMat.scanFreq;
                    iss >> outMat.scanSpeed;
                }
                else if (effectName == "TRON_GRID") {
                    outMat.effectFlags |= 4;
                    iss >> outMat.tronGridSize;
                    iss >> outMat.tronThickness;
                }
                else if (effectName == "RADAR_PING") {
                    outMat.effectFlags |= 8;
                    iss >> outMat.radarFreq;
                    iss >> outMat.radarSpeed;
                }
                else if (effectName == "MATRIX_JITTER") {
                    outMat.effectFlags |= 16;
                    iss >> outMat.jitterIntensity;
                }
                else if (effectName == "DISSOLVE") {
                    outMat.effectFlags |= 32;
                    iss >> outMat.dissolveSpeed;
                }
                else if (effectName == "NEGATIVE_ZONE") {
                    outMat.effectFlags |= 64;
                }
                else if (effectName == "RGB_DISCO") {
                    outMat.effectFlags |= 128;
                }
                else if (effectName == "NORMAL_DEBUG") {
                    outMat.effectFlags |= 256;
                }
                else if (effectName == "CEL_SHADING") {
                    outMat.effectFlags |= 512;
                    iss >> outMat.celBands;
                }
                else if (effectName == "LINEAR_FOG") {
                    outMat.effectFlags |= 1024;
                    iss
                        >> outMat.fogStart
                        >> outMat.fogEnd
                        >> outMat.fogColor.x
                        >> outMat.fogColor.y
                        >> outMat.fogColor.z;
                }
                else if (effectName == "EXP_FOG") {
                    outMat.effectFlags |= 2048;
                    iss
                        >> outMat.fogDensity
                        >> outMat.fogColor.x
                        >> outMat.fogColor.y
                        >> outMat.fogColor.z;
                }
                else if (effectName == "FRESNEL") {
                    outMat.effectFlags |= 4096;
                    iss
                        >> outMat.shieldColor.x
                        >> outMat.shieldColor.y
                        >> outMat.shieldColor.z
                        >> outMat.rimPower
                        >> outMat.rimIntensity;
                }
                else if (effectName == "LIDAR") {
                    outMat.effectFlags |= 16384;
                }
                else if (effectName == "STATICTV") {
                    outMat.effectFlags |= 32768;
                    iss >> outMat.noiseScale;
                }
                else if (effectName == "GLITCH") {
                    outMat.effectFlags |= 65536;
                    iss >> outMat.noiseScale;
                }
                else if (effectName == "WHITENOISE") {
                    outMat.effectFlags |= 131072;
                    iss >> outMat.noiseScale;
                }
                else if (effectName == "LIQUID") {
                    outMat.effectFlags |= 262144;
                    iss >> outMat.liquidSpeed;
                    iss >> outMat.liquidFreq;
                }
            }
        }

        return true;
    }

public:
    static void loadManifest(const SceneManifest& manifest) {
        for (const auto& matPath : manifest.materials) {
            if (materialCache.find(matPath) != materialCache.end()) {
                continue;
            }

            Material material;

            if (loadMaterialFile(matPath, material)) {
                materialCache[matPath] = material;
            }
            else {
                std::cerr
                    << "[MaterialLoader] Materyal bulunamadi -> "
                    << matPath
                    << std::endl;
            }
        }
    }

    static Material getMaterial(const std::string& path) {
        return materialCache[path];
    }
};
