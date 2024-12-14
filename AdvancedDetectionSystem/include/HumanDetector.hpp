#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

class TrackedObject {
public:
    cv::Rect boundingBox;
    float confidence;
    float distance;
    cv::Point2f direction;
    bool isMoving;
};

class HumanDetector {
private:
    cv::HOGDescriptor hog;
    cv::dnn::Net net;
    std::vector<std::string> classes;
    
    const float CONFIDENCE_THRESHOLD = 0.5f;
    const float NMS_THRESHOLD = 0.4f;
    const float CONF_THRESHOLD = 0.5f;
    const int INPUT_WIDTH = 416;
    const int INPUT_HEIGHT = 416;
    
    cv::Rect lastDetectedBox;
    
    struct CameraParams {
        float focalLength;
        float realHeight;
        float verticalFOV;
        cv::Mat cameraMatrix;
        cv::Mat distCoeffs;
    } params;

    std::vector<std::string> getOutputLayerNames();
    float calculateDistance(const cv::Rect& box);
    cv::Point2f calculateDirection(const cv::Rect& currentBox, const cv::Rect& previousBox);
    bool isPersonMoving(const cv::Rect& currentBox, const cv::Rect& previousBox, float threshold = 10.0f);
    
public:
    HumanDetector();
    
    void loadYOLOModel(const std::string& modelPath, const std::string& configPath);
    std::vector<TrackedObject> detectWithYOLO(const cv::Mat& frame);
    std::vector<TrackedObject> detect(const cv::Mat& frame);
    bool detectHumanPose(const cv::Mat& frame, std::vector<cv::Point>& keypoints);
    float calculateConfidenceScore(const cv::Rect& detection);
    
    void setCameraParameters(float focalLength, float realHeight, float verticalFOV);
};

