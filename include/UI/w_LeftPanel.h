//
// Created by Can on 7.07.2026.
//

#ifndef CUDAVISIONENGINE_W_LEFTPANEL_H
#define CUDAVISIONENGINE_W_LEFTPANEL_H
#include <string>


class LeftPanel {

public:

    void render(float displayWidth, float displayHeight);
    void addPhotoToStack(const std::string& photoPath);

};


#endif //CUDAVISIONENGINE_W_LEFTPANEL_H