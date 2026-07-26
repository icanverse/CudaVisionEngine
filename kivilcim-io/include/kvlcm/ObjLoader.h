#pragma once

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <vector_types.h>

#include "../../../kivilcim-core/include/SceneDescription.h"

class ObjLoader {
private:
    using MeshData =
        std::pair<std::vector<float3>, std::vector<int3>>;

    static inline std::unordered_map<std::string, MeshData> meshCache;

    static bool loadObj(
        const std::string& filepath,
        std::vector<float3>& outVerts,
        std::vector<int3>& outInds
    ) {
        FILE* file = fopen(filepath.c_str(), "r");
        if (!file) return false;

        outVerts.reserve(200000);
        outInds.reserve(200000);

        char line[256];
        while (fgets(line, sizeof(line), file)) {
            if (line[0] == 'v' && line[1] == ' ') {
                float x, y, z;
                sscanf(line, "v %f %f %f", &x, &y, &z);
                outVerts.push_back({x, y, z});
            }
            else if (line[0] == 'f' && line[1] == ' ') {
                char t1[32], t2[32], t3[32];
                if (sscanf(
                        line,
                        "f %31s %31s %31s",
                        t1,
                        t2,
                        t3
                    ) == 3) {
                    outInds.push_back({
                        atoi(t1) - 1,
                        atoi(t2) - 1,
                        atoi(t3) - 1
                    });
                }
            }
        }

        fclose(file);
        return true;
    }

public:
    static void loadManifest(const SceneManifest& manifest) {
        for (const auto& meshPath : manifest.meshes) {
            if (meshCache.find(meshPath) != meshCache.end()) {
                continue;
            }

            std::vector<float3> verts;
            std::vector<int3> inds;

            if (loadObj(meshPath, verts, inds)) {
                meshCache[meshPath] = {verts, inds};
            }
            else {
                std::cerr
                    << "[ObjLoader] Mesh bulunamadi -> "
                    << meshPath
                    << std::endl;
            }
        }
    }

    static auto getMesh(const std::string& path) {
        return meshCache[path];
    }
};
