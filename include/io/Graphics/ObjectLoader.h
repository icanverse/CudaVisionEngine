#pragma once
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <vector_types.h>
#include "../Graphics/Scene.cuh"
#include "SceneDescription.h"

class ObjectLoader {
private:
    static inline std::unordered_map<std::string, std::pair<std::vector<float3>, std::vector<int3>>> meshCache;
    static inline std::unordered_map<std::string, Material> materialCache;

    static int parseObjIndex(const std::string& token) {
        size_t slashPos = token.find('/');
        if (slashPos != std::string::npos) return std::stoi(token.substr(0, slashPos)) - 1;
        return std::stoi(token) - 1;
    }

    static bool loadObj(const std::string& filepath, std::vector<float3>& outVerts, std::vector<int3>& outInds) {
        // C++ ifstream yerine donanım seviyesine daha yakın olan C FILE pointer kullanıyoruz
        FILE* file = fopen(filepath.c_str(), "r");
        if (!file) return false;

        // RAM'i baştan ayır! (Milyonlarca push_back'in RAM'i kitlemesini engeller)
        outVerts.reserve(200000);
        outInds.reserve(200000);

        char line[256];
        // Dosyayı satır satır ve kopyalamadan oku
        while (fgets(line, sizeof(line), file)) {
            // Vertex satırı mı?
            if (line[0] == 'v' && line[1] == ' ') {
                float x, y, z;
                sscanf(line, "v %f %f %f", &x, &y, &z);
                outVerts.push_back({x, y, z});
            }
            // Face (Üçgen) satırı mı?
            else if (line[0] == 'f' && line[1] == ' ') {
                char t1[32], t2[32], t3[32];
                if (sscanf(line, "f %31s %31s %31s", t1, t2, t3) == 3) {
                    // MUAZZAM BİR C NUMARASI:
                    // atoi("15/20/30") komutu, '/' işaretini gördüğü an durur ve bize sadece '15'i verir!
                    // Bu sayede string parçalama (split) maliyetinden tamamen kurtuluruz.
                    outInds.push_back({atoi(t1) - 1, atoi(t2) - 1, atoi(t3) - 1});
                }
            }
        }

        fclose(file);
        return true;
    }

    static bool loadMaterialFile(const std::string& filepath, Material& outMat) {
        std::ifstream file(filepath);
        if (!file.is_open()) return false;

        outMat = {{1.0f, 0.0f, 1.0f}, 0.1f, 0.8f, 0.2f, 32.0f}; // Hata durumu için pembe

        // --- YENİ: Efekt değişkenlerini sıfırla ---
        outMat.effectFlags = 0;
        outMat.glowSpeed = 0.0f;
        outMat.scanFreq = 0.0f; outMat.scanSpeed = 0.0f;
        outMat.tronGridSize = 0.0f; outMat.tronThickness = 0.0f;
        outMat.radarFreq = 0.0f; outMat.radarSpeed = 0.0f;
        outMat.jitterIntensity = 0.0f;
        outMat.dissolveSpeed = 0.0f;

        std::string line, token;
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream iss(line); iss >> token;

            if (token == "COLOR") iss >> token >> outMat.color.x >> token >> outMat.color.y >> token >> outMat.color.z;
            else if (token == "AMBIENT") iss >> outMat.ambient;
            else if (token == "DIFFUSE") iss >> outMat.diffuse;
            else if (token == "SPECULAR") iss >> outMat.specular;
            else if (token == "SHININESS") iss >> outMat.shininess;

            // --- YENİ BİTMASK EFEKT OKUYUCU BLOK ---
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

                // --- YENİ EKLENEN EFEKTLER ---
                else if (effectName == "NEGATIVE_ZONE") { outMat.effectFlags |= 64; }
                else if (effectName == "RGB_DISCO") { outMat.effectFlags |= 128; }
                else if (effectName == "NORMAL_DEBUG") { outMat.effectFlags |= 256; }
                else if (effectName == "CEL_SHADING") {
                    outMat.effectFlags |= 512;
                    iss >> outMat.celBands;
                }
                else if (effectName == "LINEAR_FOG") {
                    outMat.effectFlags |= 1024;
                    iss >> outMat.fogStart >> outMat.fogEnd >> outMat.fogColor.x >> outMat.fogColor.y >> outMat.fogColor.z;
                }
                else if (effectName == "EXP_FOG") {
                    outMat.effectFlags |= 2048;
                    iss >> outMat.fogDensity >> outMat.fogColor.x >> outMat.fogColor.y >> outMat.fogColor.z;
                }
                else if (effectName == "FRESNEL") {
                    outMat.effectFlags |= 4096;
                    iss >> outMat.shieldColor.x >> outMat.shieldColor.y >> outMat.shieldColor.z >> outMat.rimPower >> outMat.rimIntensity;
                }
                else if (effectName == "LIDAR") {
                    outMat.effectFlags |= 16384; // 15. Bit
                }
                else if (effectName == "STATICTV") {
                    outMat.effectFlags |= 32768; // 16. Bit
                    iss >> outMat.noiseScale;
                }
                else if (effectName == "GLITCH") {
                    outMat.effectFlags |= 65536; // 17. Bit
                    iss >> outMat.noiseScale;
                }
                else if (effectName == "WHITENOISE") {
                    outMat.effectFlags |= 131072; // 18. Bit
                    iss >> outMat.noiseScale;
                }
            }
        }
        return true;
    }

public:
    static void loadManifest(const SceneManifest& manifest) {
        for (const auto& meshPath : manifest.meshes) {
            if (meshCache.find(meshPath) == meshCache.end()) {
                std::vector<float3> verts; std::vector<int3> inds;
                if (loadObj(meshPath, verts, inds)) meshCache[meshPath] = {verts, inds};
                else std::cerr << "[ObjectLoader] Mesh bulunamadi -> " << meshPath << std::endl;
            }
        }

        for (const auto& matPath : manifest.materials) {
             if (materialCache.find(matPath) == materialCache.end()) {
                 Material mat;
                 if (loadMaterialFile(matPath, mat)) materialCache[matPath] = mat;
                 else std::cerr << "[ObjectLoader] Materyal bulunamadi -> " << matPath << std::endl;
             }
        }
        std::cout << "[ObjectLoader] Tum asset'ler bellege alindi." << std::endl;
    }

    static auto getMesh(const std::string& path) { return meshCache[path]; }
    static Material getMaterial(const std::string& path) { return materialCache[path]; }
};