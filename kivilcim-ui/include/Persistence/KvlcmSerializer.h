#pragma once
#include <filesystem>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm> // string temizliği için

#include "Data/LayerData.h"
#include "Data/PreferenceData.h"
#include "Data/ProjectData.h"

using namespace std;

namespace Kivilcim {
    class KvlcmSerializer {
    private:
        static void trimLine(std::string& line) {
            while (!line.empty() && (line.back() == '\r' || line.back() == ' ' || line.back() == '\t')) {
                line.pop_back();
            }
        }

        static ifstream readFile(const std::string& filepath) {
            return ifstream(filepath);
        }

        static ofstream writeFile(const std::string& filepath) {
            ofstream file(filepath);
            if (!file.is_open()) cout << "Kütüphane Dosyası Açılamadı\n";
            return file;
        }

        static bool checkFileExtension(const string& expectedExtension, const std::string& filepath) {
            if (filesystem::path(filepath).extension() == expectedExtension) return true;
            cout << "[UNEXPECTED FILE EXTENSION] Dosya beklenen uzantıda değil\n";
            return false;
        }

    public:

        // ==========================================
        // PREFERENCES YÖNETİMİ
        // ==========================================
        static void savePreferences(const string& expectedExtension, const std::string& filepath, const std::vector<Kdata::PreferenceData>& users) {
            if (!checkFileExtension(expectedExtension, filepath)) return;

            ofstream file = writeFile(filepath);
            file << "### .kvlcm-user-pref ###\n## ~bu dosyayı manuel düzenlemeyiniz ##\n\n";

            for (const auto& user : users) {
                if (user.isPreferencesChanged) { // Eşitlik kontrolü düzeltildi
                    file << "PREFERENCES_BEGIN\n";
                    file << "$" << user.userName << "ıd" << user.userID << endl;
                    file << "H_ACC \n" << user.enableHardwareAcceleration << "\n";
                    file << "SH_MEM :\n"
                         << "sh_m " << user.enableSharedMemory << "\n"
                         << "r  " << user.ram_limit << "\n"
                         << "vr " << user.vram_limit << "\n";
                }
            }
        }

        // ==========================================
        // >>> KÜTÜPHANE YÖNETİMİ
        // ==========================================
        static void saveLibrary(const string& expectedExtension, const std::string& filepath, const std::vector<Kdata::ProjectData>& projects) {
            if (!checkFileExtension(expectedExtension, filepath)) return;
            std::ofstream file = writeFile(filepath);

            file << "### .kvlcm-project-library ###\n## ~bu dosyayı manuel düzenlemeyiniz ##\n\n";

            for (const auto& project : projects) {
                file << "PROJECT_BEGIN\n";
                file << "ID: " << project.id << "\n";
                file << "NAME: " << project.name << "\n";
                file << "DATE: " << project.date << "\n";
                file << "IS_FAV: " << project.isFavorite << "\n";

                file << "ORIGINAL_IMAGE_SIZE: \n" << "o_w: " << project.size.x << "\no_h: " << project.size.y << "\n";
                file << "ORIGINAL_CHANELS: " << project.channels << "\n";
                file << "ORIGINAL_FILE_SIZE: " << project.fileSize << "\n";

                file << "CANVAS_SCALE: \n" << "p_w: " << project.projectWidth << "\np_h: " << project.projectHeight
                     << "\ndim: " << project.dimMetric << "\nori: " << project.orientation << "\n";

                file << "RESOLUTION: \n" << "res: " << project.resolution << "\nres_metric: " << project.resMetric
                     << "\noriginal_size: " << project.keepOriginalSize << "\n";

                file << "BACKGROUND \n" << "res: " << project.bgContentMode << "\n"
                     << "[R]: " << project.bgColor[0] << "\n"
                     << "[G]: " << project.bgColor[1] << "\n"
                     << "[B]: " << project.bgColor[2] << "\n";

                file << "IMAGE_PATH ~" << (project.imagePath.empty() ? "NONE" : project.imagePath) << "\n";
                file << "DEPENDENCY_PATH ~" << (project.kvlcmDir.empty() ? "NONE" : project.kvlcmDir) << "\n";
                file << "PROJECT_END\n\n";
            }
            std::cout << "[KvlcmProjectParser] " << projects.size() << " proje library diske kaydedildi.\n";
        }

        static std::vector<Kdata::ProjectData> loadLibrary(const std::string& filepath) {
            std::vector<Kdata::ProjectData> loadedProjects;
            std::ifstream file = readFile(filepath);
            if (!file.is_open()) return loadedProjects;

            std::string line;
            bool inProject = false;
            Kdata::ProjectData temp(0, "", ""); // Geçiçi proje nesnesi

            while (std::getline(file, line)) {
                trimLine(line);
                if (line.empty() || line[0] == '#') continue;

                if (line == "PROJECT_BEGIN") {
                    inProject = true;
                    temp = Kdata::ProjectData(0, "", "");
                    continue;
                }
                if (line == "PROJECT_END" && inProject) {
                    loadedProjects.push_back(temp);
                    inProject = false;
                    continue;
                }

                if (inProject) {
                    if (line.find("ID: ") == 0) temp.id = std::stoi(line.substr(4));
                    else if (line.find("NAME: ") == 0) temp.name = line.substr(6);
                    else if (line.find("DATE: ") == 0) temp.date = line.substr(6);
                    else if (line.find("IS_FAV: ") == 0) temp.isFavorite = (line.substr(8) == "1");
                    else if (line.find("o_w: ") == 0) temp.size.x = std::stoi(line.substr(5));
                    else if (line.find("o_h: ") == 0) temp.size.y = std::stoi(line.substr(5));
                    else if (line.find("p_w: ") == 0) temp.projectWidth = std::stoi(line.substr(5));
                    else if (line.find("p_h: ") == 0) temp.projectHeight = std::stoi(line.substr(5));
                    else if (line.find("dim: ") == 0) temp.dimMetric = std::stoi(line.substr(5));
                    else if (line.find("ori: ") == 0) temp.orientation = std::stoi(line.substr(5));
                    else if (line.find("res: ") == 0) temp.resolution = std::stoi(line.substr(5));
                    else if (line.find("[R]: ") == 0) temp.bgColor[0] = std::stof(line.substr(5));
                    else if (line.find("[G]: ") == 0) temp.bgColor[1] = std::stof(line.substr(5));
                    else if (line.find("[B]: ") == 0) temp.bgColor[2] = std::stof(line.substr(5));
                    else if (line.find("IMAGE_PATH ~") == 0) {
                        std::string path = line.substr(12);
                        temp.imagePath = (path == "NONE") ? "" : path;
                    }
                    else if (line.find("DEPENDENCY_PATH ~") == 0) {
                        std::string path = line.substr(17);
                        temp.kvlcmDir = (path == "NONE") ? "" : path;
                    }
                }
            }
            return loadedProjects;
        }

        static void findProjectEnd(int projectId, int2& index, const std::string& filepath) {
            ifstream file = readFile(filepath);
            string line;
            string targetIdStr = "ID: " + std::to_string(projectId);
            bool insideTargetProject = false;
            std::streampos lastBeginPos = 0;

            while (std::getline(file, line)) {
                trimLine(line);
                std::streampos currentPos = file.tellg();

                if (line == "PROJECT_BEGIN") {
                    lastBeginPos = currentPos - static_cast<std::streampos>(line.length() + 1);
                }

                if (!insideTargetProject && line == targetIdStr) {
                    index.x = static_cast<long long>(lastBeginPos);
                    insideTargetProject = true;
                }

                if (insideTargetProject && line == "PROJECT_END") {
                    index.y = static_cast<long long>(currentPos);
                    break;
                }
            }
            file.close();
        }

        static bool deleteFromLibrary(int projectId, const std::string& filepath, const std::vector<Kdata::ProjectData>& projects) {
            int2 index = {-1, -1};
            findProjectEnd(projectId, index, filepath);

            if (index.x == -1 || index.y == -1) return false;

            ifstream inFile = readFile(filepath);
            string tempFilepath = filepath + ".tmp";
            ofstream outFile = writeFile(tempFilepath);

            if (!outFile.is_open()) return false;

            string line;
            while (true) {
                std::streampos lineStartPos = inFile.tellg();
                if (!std::getline(inFile, line)) break;
                streampos lineEndPos = inFile.tellg();

                bool isInsideTargetRange = (lineStartPos >= index.x && lineEndPos <= index.y);
                if (!isInsideTargetRange) outFile << line << "\n";
            }
            inFile.close();
            outFile.close();

            try {
                std::filesystem::remove(filepath);
                std::filesystem::rename(tempFilepath, filepath);
                return true;
            } catch (const std::filesystem::filesystem_error& e) { return false; }
        }

        // ==========================================
        // PROJECT SNAPSHOT
        // ==========================================
        static bool saveProjectData(const std::string& filepath, const Kdata::ProjectData& project, const Kdata::LayerData& layerData) {
            // DOSYAYI BİNARY AÇIYORUZ
            ofstream file(filepath, std::ios::binary | std::ios::out);
            if (!file.is_open()) return false;

            const char magic[8] = {'K','V','L','C','M','_','V','1'};
            file.write(magic, 8);

            std::stringstream meta;
            meta << "### .kvlcm-project SNAPSHOT ###\n";
            meta << "PROJECT_META_BEGIN\n";
            meta << "ID: " << project.id << "\n";
            meta << "CANVAS_W: " << project.projectWidth << "\n";
            meta << "CANVAS_H: " << project.projectHeight << "\n";
            meta << "CANVAS_RES: " << project.resolution << "\n";
            meta << "BG_MODE: " << project.bgContentMode << "\n";
            meta << "PROJECT_META_END\n\n";

            meta << "LAYER_COUNT: " << layerData.layers.size() << "\n\n";

            for (const auto& layer : layerData.layers) {
                meta << "LAYER_BEGIN\n";
                meta << "L_ID: " << layer.id << "\n";
                meta << "L_NAME: " << layer.name << "\n";
                meta << "L_TYPE: " << static_cast<int>(layer.type) << "\n";
                meta << "L_BLEND: " << static_cast<int>(layer.blendMode) << "\n";
                meta << "L_VISIBLE: " << layer.isVisible << "\n";
                meta << "L_OPACITY: " << layer.opacity << "\n";
                meta << "L_TRANSFORM: " << layer.transform.posX << " " << layer.transform.posY << " "
                                        << layer.transform.scaleX << " " << layer.transform.scaleY << " "
                                        << layer.transform.rotation << "\n";
                meta << "MASK_ACTIVE: " << layer.layerMask.isActive << "\n";
                if(layer.layerMask.isActive) {
                    meta << "MASK_BOUNDS: " << layer.layerMask.boundsWidth << " " << layer.layerMask.boundsHeight << "\n";
                }
                meta << "LAYER_END\n\n";
            }

            std::string metaStr = meta.str();
            uint32_t metaSize = static_cast<uint32_t>(metaStr.size());
            file.write(reinterpret_cast<const char*>(&metaSize), sizeof(uint32_t));
            file.write(metaStr.c_str(), metaSize);

            for (const auto& layer : layerData.layers) {
                if (layer.layerMask.isActive && layer.layerMask.d_maskData != nullptr) {
                    uint32_t pixelDataSize = layer.layerMask.boundsWidth * layer.layerMask.boundsHeight;
                    if (pixelDataSize > 0) {
                        file.write(reinterpret_cast<const char*>(&pixelDataSize), sizeof(uint32_t));
                        file.write(reinterpret_cast<const char*>(layer.layerMask.d_maskData), pixelDataSize);
                    }
                }
            }
            file.close();
            return true;
        }

        // BİNARY VE CHUNK DESTEKLİ PROJE OKUMA FONKSİYONU
        static bool loadProject(const std::string& filepath, Kdata::ProjectData& outProject, Kdata::LayerData& outLayerData) {
            std::ifstream file(filepath, std::ios::binary | std::ios::in);
            if (!file.is_open()) return false;

            char magic[8];
            file.read(magic, 8);
            if (std::string(magic, 8) != "KVLCM_V1") return false; // Hatalı dosya

            uint32_t metaSize = 0;
            file.read(reinterpret_cast<char*>(&metaSize), sizeof(uint32_t));

            std::string metaStr;
            metaStr.resize(metaSize);
            file.read(&metaStr[0], metaSize);

            std::istringstream metaParser(metaStr);
            std::string line;
            bool inLayer = false;
            Kdata::Layer tempLayer;
            outLayerData.layers.clear();

            // METADATA AYRIŞTIRMA (STRINGSTREAM)
            while (std::getline(metaParser, line)) {
                trimLine(line);
                if(line.empty()) continue;

                if (line.find("CANVAS_W: ") == 0) outProject.projectWidth = std::stoi(line.substr(10));
                else if (line.find("CANVAS_H: ") == 0) outProject.projectHeight = std::stoi(line.substr(10));

                else if (line == "LAYER_BEGIN") {
                    inLayer = true;
                    tempLayer = Kdata::Layer(); // Yenile
                }
                else if (line == "LAYER_END" && inLayer) {
                    outLayerData.layers.push_back(tempLayer);
                    inLayer = false;
                }
                else if (inLayer) {
                    if (line.find("L_ID: ") == 0) tempLayer.id = std::stoi(line.substr(6));
                    else if (line.find("L_NAME: ") == 0) tempLayer.name = line.substr(8);
                    else if (line.find("L_OPACITY: ") == 0) tempLayer.opacity = std::stof(line.substr(11));
                    else if (line.find("MASK_ACTIVE: ") == 0) tempLayer.layerMask.isActive = (line.substr(13) == "1");
                    else if (line.find("MASK_BOUNDS: ") == 0) {
                        std::istringstream boundsStream(line.substr(13));
                        boundsStream >> tempLayer.layerMask.boundsWidth >> tempLayer.layerMask.boundsHeight;
                    }
                    // Not: L_TRANSFORM vs eklenebilir, parse mantığı aynıdır.
                }
            }

            // METİN BİTTİ. ŞİMDİ SIRADA BİNARY PİKSEL VERİLERİ (CHUNKS) VAR
            for (auto& layer : outLayerData.layers) {
                if (layer.layerMask.isActive) {
                    uint32_t pixelDataSize = 0;
                    file.read(reinterpret_cast<char*>(&pixelDataSize), sizeof(uint32_t));

                    if (pixelDataSize > 0) {
                        // RAM'de yer ayır (Motor tarafında bu VRAM'e veya CUDA'ya yüklenecek)
                        layer.layerMask.d_maskData = new unsigned char[pixelDataSize];
                        file.read(reinterpret_cast<char*>(layer.layerMask.d_maskData), pixelDataSize);
                    }
                }
            }

            file.close();
            return true;
        }
    };
}