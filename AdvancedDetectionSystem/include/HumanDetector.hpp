#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <mutex>
#include <memory>

class HumanDetector {
public:
    struct DetectionInfo {
        cv::Rect box;
        float confidence;
        float distance;
        cv::Point2f direction;
        bool isMoving;
        std::string label;

        bool isValid() const {
            return confidence >= 0 && confidence <= 1 && 
                   distance >= 0 && !label.empty();
        }
    };

    struct CameraParams {
        float focalLength{615.0f};
        float realHeight{1700.0f};
        float verticalFOV{60.0f};
        cv::Mat cameraMatrix;
        cv::Mat distCoeffs;

        void validate() const {
            if (focalLength <= 0 || realHeight <= 0 || verticalFOV <= 0)
                throw std::invalid_argument("Invalid camera parameters");
        }
    };

private:
    mutable std::mutex mtx;
    cv::HOGDescriptor hog;
    cv::Rect lastDetectedBox;
    CameraParams params;
    
    // Buffer'lar
    std::vector<cv::Rect> foundLocations;
    std::vector<double> weights;
    cv::Mat tempImage;

public:
    class DetectionError : public std::runtime_error {
        using std::runtime_error::runtime_error;
    };

    HumanDetector();
    ~HumanDetector() = default;

    // Copy/Move operations are deleted due to mutex
    HumanDetector(const HumanDetector&) = delete;
    HumanDetector& operator=(const HumanDetector&) = delete;
    HumanDetector(HumanDetector&&) = delete;
    HumanDetector& operator=(HumanDetector&&) = delete;

    std::vector<DetectionInfo> detectHumans(const cv::Mat& frame);

    void setCameraParameters(const CameraParams& newParams) {
        std::lock_guard<std::mutex> lock(mtx);
        newParams.validate();
        params = newParams;
    }

    CameraParams getCameraParameters() const {
        std::lock_guard<std::mutex> lock(mtx);
        return params;
    }

private:
    float calculateConfidenceScore(const cv::Rect& detection) const;
    float calculateDistance(const cv::Rect& box) const;
    cv::Point2f calculateDirection(const cv::Rect& currentBox, 
                                 const cv::Rect& previousBox) const;
    bool isPersonMoving(const cv::Rect& currentBox,
                       const cv::Rect& previousBox,
                       float threshold = 10.0f) const;
    void preprocessFrame(const cv::Mat& frame);
};