#pragma once
#include <unordered_map>
#include <string>

enum class Icon {
    /// Baz İkonlar
    Copy,
    Delete,
    Folder,
    Download,
    Export,
    File,
    Horizontal_3_Dot,
    Import,
    Recently,
    Save,
    Oversave,
    Share,
    Star,
    Upload,
    Vertical_3_Dot,

    /// Üst Araç Paneli

    // Baz Bar
    Undo,
    Redo,
    Turn_Left,
    Turn_Right,
    Zoom_In,
    Zoom_Out,
    Mirror_Horizontal,
    Mirror_Vertical,

    // Şekil Bar
    Circle,
    Line,
    Square,
    Vector,

    // Çizim Bar
    Erase,

    // Metin Bar
    Text_Size,
    Text_Color,

    /// Sol Araç Paneli
    Brush,
    Color,
    Crop,
    Move,
    Select_Region_Free,
    Select_Region_Rectangle,
    Text,

    /// Sağ Araç Paneli
    // Temel Araçlar
    Contrast,
    Temperature,

    /// Katmanlar Paneli
    // Katmanlar
    Layers_toDown,
    Layers,
    Layers_Add,
    Lock,
    Union,
    Visible,


};

class IconManager {
public:
    static void Initialize();
    static void Shutdown();
    static unsigned int Get(Icon id);

private:
    static std::unordered_map<Icon, unsigned int> iconCache;
    static void RegisterIcon(Icon id, const std::string& filepath);
};