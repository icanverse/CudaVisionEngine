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

        // --- Efekt değişkenleri için varsayılan atamalar ---
        outMat.effectType = 0;
        outMat.effectParam1 = 0.0f;
        outMat.effectParam2 = 0.0f;

        std::string line, token;
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream iss(line); iss >> token;

            if (token == "COLOR") iss >> token >> outMat.color.x >> token >> outMat.color.y >> token >> outMat.color.z;
            else if (token == "AMBIENT") iss >> outMat.ambient;
            else if (token == "DIFFUSE") iss >> outMat.diffuse;
            else if (token == "SPECULAR") iss >> outMat.specular;
            else if (token == "SHININESS") iss >> outMat.shininess;

            // --- EFEKT OKUYUCU BLOK ---
            else if (token == "EFFECT") {
                std::string effectName;
                iss >> effectName;
                if (effectName == "GLOW") {
                    outMat.effectType = 1;      // Glow ID'si
                    iss >> outMat.effectParam1; // Glow Hızı
                }
                else if (effectName == "SCANLINE") {
                    outMat.effectType = 2;      // Scanline ID'si
                    iss >> outMat.effectParam1; // Frekans
                    iss >> outMat.effectParam2; // Hız
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