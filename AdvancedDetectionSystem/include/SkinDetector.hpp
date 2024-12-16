#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <mutex>
#include <stdexcept>

class SkinDetector {
public:
    // Structs
    struct SkinRegion {
        cv::Rect boundingBox;
        double area;
        float confidence;
        cv::Point2f center;
        std::vector<cv::Point> contour;
        float aspectRatio;

        bool isValid() const {
            return area > 0 && confidence >= 0 && confidence <= 1;
        }
    };

    struct ColorRange {
        cv::Scalar lower;
        cv::Scalar upper;
        float weight;

        bool isValid() const {
            return weight >= 0 && weight <= 1;
        }
    };

    struct Config {
        static constexpr double DEFAULT_MIN_AREA = 500.0;
        static constexpr double DEFAULT_MAX_AREA = 100000.0;
        static constexpr float DEFAULT_MIN_ASPECT_RATIO = 0.2f;
        static constexpr float DEFAULT_MAX_ASPECT_RATIO = 5.0f;
        static constexpr float DEFAULT_CONFIDENCE_THRESHOLD = 0.5f;

        double minArea{DEFAULT_MIN_AREA};
        double maxArea{DEFAULT_MAX_AREA};
        float minAspectRatio{DEFAULT_MIN_ASPECT_RATIO};
        float maxAspectRatio{DEFAULT_MAX_ASPECT_RATIO};
        float confidenceThreshold{DEFAULT_CONFIDENCE_THRESHOLD};
        bool useHistogramBackProjection{false};
        bool useAdaptiveThreshold{false};
        int morphologyIterations{2};
        int blurSize{3};

        void validate() const {
            if (minArea <= 0 || maxArea <= minArea)
                throw std::invalid_argument("Invalid area configuration");
            if (minAspectRatio <= 0 || maxAspectRatio <= minAspectRatio)
                throw std::invalid_argument("Invalid aspect ratio configuration");
            if (confidenceThreshold < 0 || confidenceThreshold > 1)
                throw std::invalid_argument("Invalid confidence threshold");
            if (morphologyIterations < 0)
                throw std::invalid_argument("Invalid morphology iterations");
            if (blurSize < 0 || blurSize % 2 == 0)
                throw std::invalid_argument("Invalid blur size");
        }
    };

    // Exception class
    class DetectionError : public std::runtime_error {
    public:
        using std::runtime_error::runtime_error;
    };

private:
    mutable std::mutex mtx;
    Config config;
    std::vector<ColorRange> skinRanges;
    cv::Mat skinHistogram;
    cv::Mat kernel;

    // Private methods
    cv::Mat createSkinMask(const cv::Mat& frame);
    cv::Mat applyMorphology(const cv::Mat& mask);
    cv::Mat applyAdaptiveThreshold(const cv::Mat& mask);
    void findSkinRegions(const cv::Mat& mask, std::vector<SkinRegion>& regions);
    float calculateConfidence(const cv::Mat& roi) const;
    void applyHistogramBackProjection(const cv::Mat& frame, cv::Mat& mask);
    bool validateRegion(const SkinRegion& region) const;
    cv::Point2f calculateCenter(const std::vector<cv::Point>& contour) const;
    float calculateAspectRatio(const cv::Rect& box) const;

public:
    SkinDetector();
    explicit SkinDetector(const Config& config);
    ~SkinDetector() = default;

    // Delete copy constructors
    SkinDetector(const SkinDetector&) = delete;
    SkinDetector& operator=(const SkinDetector&) = delete;

    // Main detection methods
    void detectSkin(const cv::Mat& frame,
                   std::vector<SkinRegion>& regions,
                   cv::Mat& mask);

    void detectSkinRegions(const cv::Mat& frame,
                          std::vector<SkinRegion>& regions);

    // Configuration methods
    void setConfig(const Config& newConfig) {
        std::lock_guard<std::mutex> lock(mtx);
        newConfig.validate();
        config = newConfig;
    }

    Config getConfig() const {
        std::lock_guard<std::mutex> lock(mtx);
        return config;
    }

    void setSkinRanges(const std::vector<ColorRange>& ranges);
    std::vector<ColorRange> getSkinRanges() const;
    
    void trainHistogram(const cv::Mat& skinSample);

    // Static visualization method
    static cv::Mat visualizeSkinRegions(
        const cv::Mat& frame,
        const std::vector<SkinRegion>& regions,
        bool showConfidence = true,
        bool showArea = false);
};