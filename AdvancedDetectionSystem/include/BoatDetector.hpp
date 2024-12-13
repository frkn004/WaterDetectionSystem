#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

class BoatDetector {
public:
    struct BoatInfo {
        cv::Rect boundingBox;
        float confidence;
        cv::Point2f position;
        float area;
        bool isMoving;
    };

    struct Config {
        cv::Scalar lowerColor{90, 40, 40};    // Koyu renkli tekneler için HSV
        cv::Scalar upperColor{130, 255, 255}; // Açık renkli tekneler için HSV
        float minArea = 1000.0f;              // Minimum tekne alanı
        float maxArea = 100000.0f;            // Maximum tekne alanı
        int morphSize = 3;                    // Morfolojik işlem boyutu
        bool useAdaptiveThreshold = true;     // Adaptif eşikleme kullan
    };

    BoatDetector();
    explicit BoatDetector(const Config& config);

    std::vector<BoatInfo> detectBoats(const cv::Mat& frame);
    void setConfig(const Config& config) { this->config = config; }
    Config getConfig() const { return config; }
    void visualizeResults(cv::Mat& frame, const std::vector<BoatInfo>& boats);

private:
    Config config;
    cv::Mat kernel;
    cv::Mat backgroundModel;
    bool isBackgroundInitialized;
    std::vector<cv::Point2f> previousPositions;

    cv::Mat preprocessFrame(const cv::Mat& frame);
    std::vector<std::vector<cv::Point>> findBoatContours(const cv::Mat& mask);
    bool validateBoatShape(const std::vector<cv::Point>& contour);
    float calculateConfidence(const std::vector<cv::Point>& contour, const cv::Mat& mask);
    bool isBoatMoving(const cv::Point2f& currentPos);
    void updateBackgroundModel(const cv::Mat& frame);
};
