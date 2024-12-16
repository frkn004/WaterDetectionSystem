#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>
#include <mutex>

class SeaDetector {
public:
    struct WaveAnalysis {
        float waveHeight{0.0f};
        float wavePeriod{0.0f};
        float waveIntensity{0.0f};
        bool isDangerous{false};
        cv::Point2f dominantDirection{0.0f, 0.0f};
        float turbulence{0.0f};

        bool isValid() const {
            return waveHeight >= 0 && wavePeriod >= 0 && 
                   waveIntensity >= 0 && waveIntensity <= 1 &&
                   turbulence >= 0 && turbulence <= 1;
        }
    };

    struct SeaInfo {
        bool isDetected{false};
        float waveIntensity{0.0f};
        cv::Point2f centerPoint{0.0f, 0.0f};
        std::vector<cv::Point> contour;
        float area{0.0f};
        float depth{0.0f};
        WaveAnalysis waveAnalysis;
        float visibility{0.0f};
        float surfaceTemperature{0.0f};

        bool isValid() const {
            return area >= 0 && depth >= 0 && 
                   visibility >= 0 && surfaceTemperature > -273.15f &&
                   waveAnalysis.isValid();
        }
    };

    struct Config {
        static constexpr float DEFAULT_MIN_AREA = 1000.0f;
        static constexpr float DEFAULT_WAVE_THRESHOLD = 0.5f;

        cv::Scalar waterLowerBound{100, 40, 40};
        cv::Scalar waterUpperBound{140, 255, 255};
        float minArea{DEFAULT_MIN_AREA};
        float waveThreshold{DEFAULT_WAVE_THRESHOLD};
        int morphSize{3};
        bool useAdaptiveThreshold{true};
        float maxWaveHeight{2.0f};
        float dangerousWavePeriod{5.0f};

        void validate() const {
            if (minArea <= 0) 
                throw std::invalid_argument("Invalid minArea");
            if (waveThreshold < 0 || waveThreshold > 1)
                throw std::invalid_argument("Invalid waveThreshold");
            if (morphSize <= 0 || morphSize % 2 == 0)
                throw std::invalid_argument("Invalid morphSize");
            if (maxWaveHeight <= 0)
                throw std::invalid_argument("Invalid maxWaveHeight");
            if (dangerousWavePeriod <= 0)
                throw std::invalid_argument("Invalid dangerousWavePeriod");
        }
    };

private:
    mutable std::mutex mtx;
    Config config;
    float horizonLine{0.0f};
    cv::Mat kernel;
    cv::Mat preprocessBuffer;
    cv::Mat waterMaskBuffer;
    std::vector<std::vector<cv::Point>> contoursBuffer;

    cv::Mat preprocessFrame(const cv::Mat& frame);
    cv::Mat detectWaterRegion(const cv::Mat& frame);
    WaveAnalysis analyzeWavePattern(const cv::Mat& frame);
    float calculateWaveHeight(const cv::Mat& waterMask) const;
    float calculateWavePeriod() const;
    bool evaluateWaveConditions(const WaveAnalysis& analysis) const;
    std::vector<cv::Point> findLargestContour(const cv::Mat& mask);
    cv::Point2f calculateCenterPoint(const std::vector<cv::Point>& contour) const;
    float calculateTurbulence(const cv::Mat& frame, const cv::Mat& waterMask);
    float estimateVisibility(const cv::Mat& frame, const cv::Mat& waterMask);
    float estimateSurfaceTemperature(const cv::Mat& frame, const cv::Mat& waterMask);
    cv::Point2f calculateDominantDirection(const std::vector<cv::Point>& contour);

public:
    class SeaDetectionError : public std::runtime_error {
        using std::runtime_error::runtime_error;
    };

    SeaDetector();
    explicit SeaDetector(const Config& config);
    ~SeaDetector() = default;

    SeaDetector(const SeaDetector&) = delete;
    SeaDetector& operator=(const SeaDetector&) = delete;
    SeaDetector(SeaDetector&&) = delete;
    SeaDetector& operator=(SeaDetector&&) = delete;

    SeaInfo detectSea(const cv::Mat& frame);

    void setConfig(const Config& newConfig) {
        std::lock_guard<std::mutex> lock(mtx);
        newConfig.validate();
        config = newConfig;
    }

    Config getConfig() const {
        std::lock_guard<std::mutex> lock(mtx);
        return config;
    }

    void setHorizonLine(float value) {
        std::lock_guard<std::mutex> lock(mtx);
        if (value < 0 || value > 1)
            throw std::invalid_argument("Invalid horizon line value");
        horizonLine = value;
    }

    void visualizeResults(cv::Mat& frame, const SeaInfo& info) const;
};