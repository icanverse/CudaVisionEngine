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
    BRUSH,
    COLOR,
    MOVE,
    SELECT_REGION_FREE,
    SELECT_REGION_RECTANGLE,
    TEXT,
};

class QuickLeftToolbox {
public:
    QuickLeftToolbox();
    ~QuickLeftToolbox();

    void render(float displayWidth, float displayHeight);

    ActiveTool getCurrentTool() const { return currentTool; }
private:
    ActiveTool currentTool;
    std::vector<ToolUI_Info> availableTools; // Araç listemiz
};


#endif //CUDAVISIONENGINE_QUICKTOOLBAR_H