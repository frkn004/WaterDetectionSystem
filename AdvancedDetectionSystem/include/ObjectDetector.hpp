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
#include <chrono>
#include <memory>

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
    SeaDetector::Config seaConfig;
    
    bool showHelp;
    int frameCount;
    float fps;
    std::chrono::steady_clock::time_point lastTime;

    struct CameraParams {
        float focalLength = 615.0f;
        float knownWidth = 600.0f;
        float distanceThreshold = 5000.0f;
        cv::Mat cameraMatrix;
        cv::Mat distCoeffs;
    } params;

    int nextTrackerId;

    // Private yardımcı fonksiyonlar
    float calculateDistance(const cv::Rect& box);
    void updateFPS();
    void detectWaterLine(cv::Mat& frame);
    void drawFeatureStatus(cv::Mat& frame);
    void detectAndTrackBoats(cv::Mat& frame);
    void detectAndTrackHumans(cv::Mat& frame);
    void detectSea(cv::Mat& frame);
    void detectSkin(cv::Mat& frame);
    void drawGrid(cv::Mat& frame);
    void drawCoordinates(cv::Mat& frame);
    void drawTimestamp(cv::Mat& frame);

public:
    explicit ObjectDetector(const std::string& cascadePath);
    ~ObjectDetector() = default;
    
    // Copy/Move prevention
    ObjectDetector(const ObjectDetector&) = delete;
    ObjectDetector& operator=(const ObjectDetector&) = delete;
    ObjectDetector(ObjectDetector&&) = delete;
    ObjectDetector& operator=(ObjectDetector&&) = delete;

    void detectAndTrack(cv::Mat& frame);
    void handleKeyPress(char key);
    
    // Getters
    FeatureControl& getFeatureControl() { return features; }
    SeaDetector& getSeaDetector() { return seaDetector; }
    HumanDetector& getHumanDetector() { return humanDetector; }
    BoatDetector& getBoatDetector() { return boatDetector; }
    NotificationSystem& getNotificationSystem() { return notifications; }
};