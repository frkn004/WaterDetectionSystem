// include/SkinDetector.hpp
#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <memory>

class SkinDetector {
public:
    struct SkinRegion {
        cv::Rect boundingBox;
        double area;
        float confidence;
        cv::Point2f center;
        std::vector<cv::Point> contour;
        float aspectRatio;
    };

    struct ColorRange {
        cv::Scalar lower;
        cv::Scalar upper;
        float weight;
    };

    struct Config {
        double minArea = 500.0;
        double maxArea = 100000.0;
        float minAspectRatio = 0.2f;
        float maxAspectRatio = 5.0f;
        float confidenceThreshold = 0.5f;
        bool useHistogramBackProjection = false;
        bool useAdaptiveThreshold = false;
        int morphologyIterations = 2;
        int blurSize = 3;
    };

private:
    Config config;
    std::vector<ColorRange> skinRanges;
    cv::Mat skinHistogram;
    cv::Mat kernel;

    // Private helper functions - cpp'de kullandığımız tüm yardımcı fonksiyonlar
    cv::Mat createSkinMask(const cv::Mat& frame);
    cv::Mat applyMorphology(const cv::Mat& mask);
    cv::Mat applyAdaptiveThreshold(const cv::Mat& mask);
    void findSkinRegions(const cv::Mat& mask, std::vector<SkinRegion>& regions);
    float calculateConfidence(const cv::Mat& roi);
    void applyHistogramBackProjection(const cv::Mat& frame, cv::Mat& mask);
    bool validateRegion(const SkinRegion& region) const;
    cv::Point2f calculateCenter(const std::vector<cv::Point>& contour);
    float calculateAspectRatio(const cv::Rect& box) const;

public:
    // Constructors
    SkinDetector();
    explicit SkinDetector(const Config& config);

    // Main detection functions
    void detectSkin(const cv::Mat& frame,
                   std::vector<SkinRegion>& regions,
                   cv::Mat& mask);
    
    void detectSkinRegions(const cv::Mat& frame,
                          std::vector<SkinRegion>& regions);

    // Training and configuration
    void trainHistogram(const cv::Mat& skinSample);
    void setConfig(const Config& config) { this->config = config; }
    Config getConfig() const { return config; }
    
    void setSkinRanges(const std::vector<ColorRange>& ranges) { skinRanges = ranges; }
    std::vector<ColorRange> getSkinRanges() const { return skinRanges; }
    void addSkinRange(const ColorRange& range) { skinRanges.push_back(range); }
    void clearSkinRanges() { skinRanges.clear(); }

    // Visualization
    static cv::Mat visualizeSkinRegions(const cv::Mat& frame,
                                      const std::vector<SkinRegion>& regions,
                                      bool showConfidence = true,
                                      bool showArea = false);
};
