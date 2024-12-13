// include/FeatureControl.hpp
#pragma once
#include <opencv2/opencv.hpp>

class FeatureControl {
public:
    // Özellik durumları - cpp dosyasındaki tüm özellikleri yansıtacak şekilde
    bool showFPS;
    bool showTrajectory;
    bool showDistance;
    bool showDirection;
    bool showWaterLevel;
    bool showSkinDetection;
    bool showSeaDetection;
    bool showHumanDetection;
    bool enableNotifications;

    // Constructor - cpp'de tüm özellikleri true olarak başlattığımız için böyle olmalı
    FeatureControl();

    // Özellik kontrolleri - cpp'deki switch-case yapısına uygun
    void toggleFeature(char key);

    // Görselleştirme fonksiyonları - cpp'deki detaylı implementasyonlara uygun
    void displayHelp(cv::Mat& frame);
    void displayFeatureStatus(cv::Mat& frame);
};
