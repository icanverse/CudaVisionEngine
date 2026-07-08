#pragma once
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include "UI/Data/ProjectData.h"

namespace Kivilcim {

    class KvlcmProjectParser {
    public:
        // --- PROJELERİ DİSKE KAYDET ---
        static void save(const std::string& filepath, const std::vector<ProjectData>& projects) {
            std::ofstream file(filepath);
            if (!file.is_open()) return;

            file << "# Kivilcim Engine - Project Save File\n";
            file << "# Lutfen bu dosyayi elle duzenlemeyin.\n\n";

            for (const auto& p : projects) {
                file << "PROJECT_BEGIN\n";
                file << "ID " << p.id << "\n";
                file << "NAME|" << p.name << "\n";
                file << "IMAGE|" << (p.imagePath.empty() ? "NONE" : p.imagePath) << "\n";

                file << "CANVAS " << p.projectWidth << " " << p.projectHeight << " " << p.dimMetric << " " << p.orientation << "\n";
                file << "RES " << p.resolution << " " << p.resMetric << " " << p.keepOriginalSize << "\n";
                file << "BG " << p.bgContentMode << " " << p.bgColor[0] << " " << p.bgColor[1] << " " << p.bgColor[2] << "\n";

                file << "PROJECT_END\n\n";
            }

            file.close();
            std::cout << "[KvlcmProjectParser] " << projects.size() << " proje diske YAZILDI." << std::endl;
        }

        // --- PROJELERİ DİSKTEN OKU (AGRESİF DEBUG) ---
        static std::vector<ProjectData> load(const std::string& filepath) {
            std::vector<ProjectData> loadedProjects;
            std::ifstream file(filepath);

            if (!file.is_open()) {
                std::cout << "[KvlcmProjectParser DEBUG] HATA: Kayit dosyasi YOK veya ACILAMIYOR!" << std::endl;
                return loadedProjects;
            }

            std::cout << "[KvlcmProjectParser DEBUG] Dosya acildi. Icerik satir satir okunuyor..." << std::endl;
            std::string line;
            bool inProject = false;
            ProjectData tempData(0, "", "");
            int lineCount = 0;

            while (std::getline(file, line)) {
                lineCount++;

                // Gizli karakterleri temizle
                while (!line.empty() && (line.back() == '\r' || line.back() == ' ' || line.back() == '\t')) {
                    line.pop_back();
                }

                if (line.empty() || line[0] == '#') continue;

                // AJAN: Okuduğu her satırı ekrana bassın!
                std::cout << "[KvlcmProjectParser DEBUG] Satir " << lineCount << ": '" << line << "'" << std::endl;

                if (line == "PROJECT_BEGIN") {
                    std::cout << "[KvlcmProjectParser DEBUG] ---> PROJECT_BEGIN BULUNDU!" << std::endl;
                    inProject = true;
                    tempData = ProjectData(0, "", "");
                    continue;
                }

                if (line == "PROJECT_END") {
                    std::cout << "[KvlcmProjectParser DEBUG] ---> PROJECT_END BULUNDU!" << std::endl;
                    if (inProject) {
                        loadedProjects.push_back(tempData);
                        inProject = false;
                        std::cout << "[KvlcmProjectParser DEBUG] ---> Proje listeye basariyla eklendi!" << std::endl;
                    }
                    continue;
                }

                if (inProject) {
                    if (line.find("NAME|") == 0) {
                        tempData.name = line.substr(5);
                    }
                    else if (line.find("IMAGE|") == 0) {
                        std::string path = line.substr(6);
                        tempData.imagePath = (path == "NONE") ? "" : path;
                    }
                    else {
                        std::istringstream iss(line);
                        std::string type;
                        iss >> type;

                        if (type == "ID") iss >> tempData.id;
                        else if (type == "CANVAS") iss >> tempData.projectWidth >> tempData.projectHeight >> tempData.dimMetric >> tempData.orientation;
                        else if (type == "RES") iss >> tempData.resolution >> tempData.resMetric >> tempData.keepOriginalSize;
                        else if (type == "BG") iss >> tempData.bgContentMode >> tempData.bgColor[0] >> tempData.bgColor[1] >> tempData.bgColor[2];
                    }
                }
            }

            file.close();
            std::cout << "[KvlcmProjectParser DEBUG] Toplam " << lineCount << " satir islendi." << std::endl;
            std::cout << "[KvlcmProjectParser] " << loadedProjects.size() << " proje diskten OKUNDU." << std::endl;
            return loadedProjects;
        }
    };
}