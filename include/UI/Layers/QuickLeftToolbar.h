#ifndef CUDAVISIONENGINE_QUICKTOOLBAR_H
#define CUDAVISIONENGINE_QUICKTOOLBAR_H
#include <string>
#include <vector>

enum class Icon;
enum class ActiveTool;

struct ToolUI_Info {
    ActiveTool id;
    std::string name;
    std::string tooltip;
    Icon icon;
    std::string info;

};
enum class ActiveTool {
    NONE,
    REGION_SELECT,  // Serbest Seçim Aracı
    REGION_CROP,    // Kesim
    PAN,            // Gezinmek için
    ZOOM,            // Büyütmek için
    MASK_BRUSH
};

class QuickLeftToolbar {
public:
    QuickLeftToolbar();
    ~QuickLeftToolbar();

    void render(float displayWidth, float displayHeight);

    ActiveTool getCurrentTool() const { return currentTool; }
private:
    ActiveTool currentTool;
    std::vector<ToolUI_Info> availableTools; // Araç listemiz
};


#endif //CUDAVISIONENGINE_QUICKTOOLBAR_H