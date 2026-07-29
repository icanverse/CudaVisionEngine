#pragma once

namespace Kdata {

    enum class CanvasTool {
        NONE,
        MOVE,
        SELECT_RECTANGLE,
        SELECT_FREE,
        CROP,
        BRUSH,
        ERASE,
        SHAPES,
        TEXT,
        COLOR_PICKER,

        // Yeni eklenen merkez panel araçları:
        SHAPE_CIRCLE,
        SHAPE_LINE,
        SHAPE_SQUARE,
        VECTOR_PATH,
        ERASER,
        TEXT_SIZE,
        TEXT_COLOR
    };

    enum class AdjustmentTool {
        NONE,
        BRIGHTNESS_CONTRAST,
        HUE_SATURATION,
        COLOR_BALANCE,
        EXPOSURE_GAMMA,
        SHADOWS_HIGHLIGHTS,
        BLUR_SHARPEN
    };

    enum class InstantAction {
        NONE, UNDO, REDO,
        COPY, ACTION_DELETE, // ÇAKIŞMA ÖNLENDİ: DELETE yerine ACTION_DELETE yapıldı
        ZOOM_IN, ZOOM_OUT,
        TURN_LEFT, TURN_RIGHT,
        MIRROR_HOR, MIRROR_VER,

        LAYER_ADD, LAYER_DOWN,
        LAYER_LOCK, LAYER_VIS,

        ACCEPT_ACTION, CANCEL_ACTION,

        // yeni
        CANVAS_TURN_LEFT,
        CANVAS_TURN_RIGHT,
        CANVAS_ZOOM_IN,
        CANVAS_ZOOM_OUT,
        MIRROR_HORIZONTAL,
        MIRROR_VERTICAL
    };

    enum class TextAlignment { LEFT, CENTER, RIGHT };

    struct TextToolContext {
        int fontSize = 24;
        float textColor[4] = {1.0f, 1.0f, 1.0f, 1.0f};
        TextAlignment alignment = TextAlignment::LEFT;
        int fontIndex = 0;
        bool isBold = false;
        bool isItalic = false;
    };

    struct BrushToolContext {
        float radius = 15.0f;
        float hardness = 0.5f;
        float color[4] = {1.0f, 0.45f, 0.0f, 1.0f};
    };

    struct EraseToolContext {
        float radius = 15.0f;
        float opacity = 1.0f;
    };

    enum class ShapeType { CIRCLE, LINE, SQUARE, VECTOR };
    enum class ShapeDrawMode { STROKE, FILL };

    struct ShapeToolContext {
        ShapeType currentShape = ShapeType::SQUARE;
        float thickness = 2.0f;
        ShapeDrawMode drawMode = ShapeDrawMode::STROKE;
        float color[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    };

    struct AdjustmentContext {
        float brightness = 0.0f;
        float contrast = 1.0f;
        float temperature = 0.0f;
        float tint = 0.0f;
        float hue = 0.0f;
        float saturation = 1.0f;
        float lightness = 1.0f;
        float exposure = 0.0f;
        float gamma = 1.0f;
        float shadows = 0.0f;
        float highlights = 0.0f;
        float blur = 0.0f;
        float sharpen = 0.0f;
    };

    enum class SelectionMode { NEW, ADD, SUBTRACT, INTERSECT };

    struct SelectionToolContext {
        SelectionMode mode = SelectionMode::NEW;
        float feather = 0.0f;
    };

    struct CropToolContext {
        bool keepAspectRatio = false;
        float aspectRatioX = 1.0f;
        float aspectRatioY = 1.0f;
    };

    struct ToolData {
        CanvasTool activeCanvasTool = CanvasTool::NONE;
        AdjustmentTool activeAdjustment = AdjustmentTool::NONE;
        InstantAction lastFiredAction = InstantAction::NONE;

        TextToolContext textCtx;
        BrushToolContext brushCtx;
        EraseToolContext eraseCtx;
        ShapeToolContext shapeCtx;
        SelectionToolContext selectionCtx;
        CropToolContext cropCtx;
        AdjustmentContext adjustmentCtx;

        bool isAdjustmentActive() const {
            return activeAdjustment != AdjustmentTool::NONE;
        }
    };

} // namespace Kdata