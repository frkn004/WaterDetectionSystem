#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

class SeaDetector {
public:
    struct SeaInfo {
        bool isDetected;
        float waveIntensity;
        cv::Point2f centerPoint;
        std::vector<cv::Point> contour;
        float area;
        float depth;
    };

    struct Config {
        cv::Scalar lowerThreshold;
        cv::Scalar upperThreshold;
        float minArea;
        float waveThreshold;
        int morphSize;
        bool useAdaptiveThreshold;
    };

    SeaDetector();
    explicit SeaDetector(const Config& config);
    
    SeaInfo detectSea(cv::Mat& frame);
    void detectWaves(const cv::Mat& frame, SeaInfo& info);
    void setConfig(const Config& config) { this->config = config; }
    Config getConfig() const { return config; }
    
    float calculateWaveIntensity(const std::vector<cv::Point>& contour);
    float estimateDepth(const cv::Mat& frame, const cv::Point2f& point);
    void visualizeResults(cv::Mat& frame, const SeaInfo& info);
    void setHorizonLine(float value) { horizonLine = value; }

private:
    Config config;
    float horizonLine;
    cv::Mat kernel;
    
    cv::Mat preprocessFrame(const cv::Mat& frame);
    std::vector<cv::Point> findLargestContour(const cv::Mat& mask);
    cv::Point2f calculateCenterPoint(const std::vector<cv::Point>& contour);
};
