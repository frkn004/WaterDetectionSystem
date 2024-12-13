#include "FeatureControl.hpp"

FeatureControl::FeatureControl() :
    showFPS(true),
    showTrajectory(true),
    showDistance(true),
    showDirection(true),
    showWaterLevel(true),
    showSkinDetection(true),
    showSeaDetection(true),
    showHumanDetection(true),
    enableNotifications(true) {}

void FeatureControl::toggleFeature(char key) {
    switch(key) {
        case '1': showFPS = !showFPS; break;
        case '2': showTrajectory = !showTrajectory; break;
        case '3': showDistance = !showDistance; break;
        case '4': showDirection = !showDirection; break;
        case '5': showWaterLevel = !showWaterLevel; break;
        case '6': showSkinDetection = !showSkinDetection; break;
        case '7': showSeaDetection = !showSeaDetection; break;
        case '8': showHumanDetection = !showHumanDetection; break;
        case '9': enableNotifications = !enableNotifications; break;
    }
}

void FeatureControl::displayHelp(cv::Mat& frame) {
    int lineHeight = 20;
    int startY = frame.rows - 200;
    cv::Point startPoint(10, startY);
    cv::Scalar textColor(255, 255, 255);
    cv::Scalar activeColor(0, 255, 0);
    cv::Scalar inactiveColor(0, 0, 255);
    double fontScale = 0.5;
    int thickness = 1;

    // Yarı saydam siyah arka plan
    cv::Mat overlay = frame.clone();
    cv::rectangle(overlay, cv::Point(5, startY - 25),
                 cv::Point(250, startY + 200),
                 cv::Scalar(0, 0, 0), -1);
    cv::addWeighted(overlay, 0.7, frame, 0.3, 0, frame);

    // Başlık
    cv::putText(frame, "Kontroller:", startPoint,
               cv::FONT_HERSHEY_SIMPLEX, fontScale, textColor, thickness);

    // Özellik durumları
    startPoint.y += lineHeight;
    cv::putText(frame, "1: FPS " + std::string(showFPS ? "(AÇIK)" : "(KAPALI)"),
               startPoint, cv::FONT_HERSHEY_SIMPLEX, fontScale,
               showFPS ? activeColor : inactiveColor, thickness);

    startPoint.y += lineHeight;
    cv::putText(frame, "2: Hareket İzleme " + std::string(showTrajectory ? "(AÇIK)" : "(KAPALI)"),
               startPoint, cv::FONT_HERSHEY_SIMPLEX, fontScale,
               showTrajectory ? activeColor : inactiveColor, thickness);

    startPoint.y += lineHeight;
    cv::putText(frame, "3: Mesafe " + std::string(showDistance ? "(AÇIK)" : "(KAPALI)"),
               startPoint, cv::FONT_HERSHEY_SIMPLEX, fontScale,
               showDistance ? activeColor : inactiveColor, thickness);

    startPoint.y += lineHeight;
    cv::putText(frame, "4: Yön " + std::string(showDirection ? "(AÇIK)" : "(KAPALI)"),
               startPoint, cv::FONT_HERSHEY_SIMPLEX, fontScale,
               showDirection ? activeColor : inactiveColor, thickness);

    startPoint.y += lineHeight;
    cv::putText(frame, "5: Su Seviyesi " + std::string(showWaterLevel ? "(AÇIK)" : "(KAPALI)"),
               startPoint, cv::FONT_HERSHEY_SIMPLEX, fontScale,
               showWaterLevel ? activeColor : inactiveColor, thickness);

    startPoint.y += lineHeight;
    cv::putText(frame, "6: Cilt Tespiti " + std::string(showSkinDetection ? "(AÇIK)" : "(KAPALI)"),
               startPoint, cv::FONT_HERSHEY_SIMPLEX, fontScale,
               showSkinDetection ? activeColor : inactiveColor, thickness);

    startPoint.y += lineHeight;
    cv::putText(frame, "7: Deniz Tespiti " + std::string(showSeaDetection ? "(AÇIK)" : "(KAPALI)"),
               startPoint, cv::FONT_HERSHEY_SIMPLEX, fontScale,
               showSeaDetection ? activeColor : inactiveColor, thickness);

    startPoint.y += lineHeight;
    cv::putText(frame, "8: İnsan Tespiti " + std::string(showHumanDetection ? "(AÇIK)" : "(KAPALI)"),
               startPoint, cv::FONT_HERSHEY_SIMPLEX, fontScale,
               showHumanDetection ? activeColor : inactiveColor, thickness);

    startPoint.y += lineHeight;
    cv::putText(frame, "9: Bildirimler " + std::string(enableNotifications ? "(AÇIK)" : "(KAPALI)"),
               startPoint, cv::FONT_HERSHEY_SIMPLEX, fontScale,
               enableNotifications ? activeColor : inactiveColor, thickness);

    startPoint.y += lineHeight;
    cv::putText(frame, "H: Yardımı Gizle/Göster",
               startPoint, cv::FONT_HERSHEY_SIMPLEX, fontScale, textColor, thickness);

    startPoint.y += lineHeight;
    cv::putText(frame, "Q: Çıkış",
               startPoint, cv::FONT_HERSHEY_SIMPLEX, fontScale, textColor, thickness);
}

void FeatureControl::displayFeatureStatus(cv::Mat& frame) {
    int startY = 30;
    int lineHeight = 20;
    cv::Scalar activeColor(0, 255, 0);
    cv::Scalar inactiveColor(0, 0, 255);
    
    auto displayStatus = [&](const std::string& name, bool isActive, int& y) {
        cv::putText(frame, name + ": " + (isActive ? "AÇIK" : "KAPALI"),
                   cv::Point(frame.cols - 200, y),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5,
                   isActive ? activeColor : inactiveColor, 1);
        y += lineHeight;
    };

    displayStatus("FPS", showFPS, startY);
    displayStatus("Hareket", showTrajectory, startY);
    displayStatus("Mesafe", showDistance, startY);
    displayStatus("Yon", showDirection, startY);
    displayStatus("Su", showWaterLevel, startY);
    displayStatus("Cılt", showSkinDetection, startY);
    displayStatus("Denız", showSeaDetection, startY);
    displayStatus("Insan", showHumanDetection, startY);
    displayStatus("Bildirim", enableNotifications, startY);
}
