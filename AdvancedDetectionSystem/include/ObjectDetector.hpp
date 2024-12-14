#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include "NotificationSystem.hpp"
#include "FeatureControl.hpp"
#include "HumanDetector.hpp"
#include "TrackedObject.hpp"
#include "SeaDetector.hpp"
#include "SkinDetector.hpp"
#include "BoatDetector.hpp"

class ObjectDetector {
private:
    cv::CascadeClassifier faceCascade;
    std::vector<TrackedObject> trackers;
    HumanDetector humanDetector;
    SkinDetector skinDetector;
    NotificationSystem notifications;
    SeaDetector seaDetector;
    BoatDetector boatDetector;
    FeatureControl features;
    
    bool showHelp;
    int frameCount;
    float fps;
    std::chrono::steady_clock::time_point lastTime;

    // Kamera parametreleri
    struct CameraParams {
        float focalLength = 615.0f;
        float knownWidth = 600.0f;  // mm cinsinden ortalama insan genişliği
        float distanceThreshold = 5000.0f;  // mm cinsinden
        cv::Mat cameraMatrix;
        cv::Mat distCoeffs;
    } params;

    int nextTrackerId = 0;  // Yeni nesne takibi için benzersiz ID'ler

private:
    float calculateDistance(const cv::Rect& box);
    void updateFPS();
    void detectWaterLine(cv::Mat& frame);
    void drawFeatureStatus(cv::Mat& frame);
    void detectAndTrackBoats(cv::Mat& frame);
    void detectAndTrackHumans(cv::Mat& frame);
    void detectSea(cv::Mat& frame);
    void detectSkin(cv::Mat& frame);

public:
    ObjectDetector(const std::string& cascadePath);
    void detectAndTrack(cv::Mat& frame);
    void handleKeyPress(char key);
};
