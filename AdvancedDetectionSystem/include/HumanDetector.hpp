#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <fstream>
#include <vector>
#include <string>

class HumanDetector {
public:
    // DetectionInfo struct'ı public alanda ve sınıfın başında olmalı
    struct DetectionInfo {
        cv::Rect box;
        float confidence;
        std::string label;
        float distance;
        cv::Point2f direction;
        bool isMoving;
    };

    // Constructor
    HumanDetector();

    // Main detection functions
    std::vector<DetectionInfo> detectHumans(const cv::Mat& frame);
    float calculateDistance(const cv::Rect& box);

    // Model loading and configuration
    void loadDeepLearningModel(const std::string& modelPath,
                             const std::string& configPath,
                             const std::string& classesPath);
    void setCameraParameters(float focalLength, float realHeight,
                           float verticalFOV = 58.0f);

private:
    cv::HOGDescriptor hog;
    cv::dnn::Net net;
    const float CONFIDENCE_THRESHOLD;
    const float NMS_THRESHOLD;
    const float CONF_THRESHOLD;
    std::vector<std::string> classes;
    
    struct CameraParams {
        float focalLength;
        float realHeight;
        float verticalFOV;
        cv::Mat cameraMatrix;
        cv::Mat distCoeffs;
    } params;

    // Private helper functions
    std::vector<std::string> getOutputsNames(const cv::dnn::Net& net);
    void postprocess(const cv::Mat& frame,
                    const std::vector<cv::Mat>& outs,
                    std::vector<DetectionInfo>& detections);
    cv::Point2f calculateDirection(const cv::Rect& currentBox,
                                 const cv::Rect& previousBox);
    bool isPersonMoving(const cv::Rect& currentBox,
                       const cv::Rect& previousBox,
                       float threshold = 10.0f);
};
