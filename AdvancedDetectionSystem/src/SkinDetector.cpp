#include "SkinDetector.hpp"
#include <algorithm>

SkinDetector::SkinDetector() : SkinDetector(Config{}) {}

SkinDetector::SkinDetector(const Config& config) {
    try {
        config.validate();
        std::lock_guard<std::mutex> lock(mtx);
        this->config = config;
        kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3,3));
        
        // Varsayılan ten rengi aralıkları
        skinRanges = {
            {cv::Scalar(0, 20, 70), cv::Scalar(20, 255, 255), 1.0f},
            {cv::Scalar(0, 30, 30), cv::Scalar(20, 150, 255), 0.8f},
            {cv::Scalar(0, 10, 160), cv::Scalar(25, 150, 255), 0.7f}
        };

        for (const auto& range : skinRanges) {
            if (!range.isValid()) {
                throw std::invalid_argument("Invalid skin color range");
            }
        }
    }
    catch (const std::exception& e) {
        throw DetectionError("Failed to initialize SkinDetector: " + std::string(e.what()));
    }
}

void SkinDetector::detectSkin(const cv::Mat& frame,
                            std::vector<SkinRegion>& regions,
                            cv::Mat& mask) {
    if (frame.empty()) {
        throw DetectionError("Empty frame provided");
    }

    try {
        std::lock_guard<std::mutex> lock(mtx);
        mask = createSkinMask(frame);
        
        if (config.useHistogramBackProjection && !skinHistogram.empty()) {
            applyHistogramBackProjection(frame, mask);
        }
        
        mask = applyMorphology(mask);
        
        if (config.useAdaptiveThreshold) {
            mask = applyAdaptiveThreshold(mask);
        }
        
        findSkinRegions(mask, regions);
        
        // Bölgeleri validate et
        regions.erase(
            std::remove_if(regions.begin(), regions.end(),
                [this](const SkinRegion& region) {
                    return !validateRegion(region);
                }
            ),
            regions.end()
        );
    }
    catch (const cv::Exception& e) {
        throw DetectionError("OpenCV error in detectSkin: " + std::string(e.what()));
    }
}

cv::Mat SkinDetector::createSkinMask(const cv::Mat& frame) {
    cv::Mat hsv;
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
    
    cv::Mat finalMask = cv::Mat::zeros(frame.size(), CV_8UC1);
    
    // Her ten rengi aralığı için maskeleme yap ve birleştir
    for (const auto& range : skinRanges) {
        cv::Mat rangeMask;
        cv::inRange(hsv, range.lower, range.upper, rangeMask);
        cv::addWeighted(finalMask, 1.0, rangeMask, range.weight, 0.0, finalMask);
    }
    
    return finalMask;
}

void SkinDetector::applyHistogramBackProjection(const cv::Mat& frame, cv::Mat& mask) {
    if (skinHistogram.empty()) return;

    cv::Mat hsv;
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);

    int channels[] = {0, 1};
    float hranges[] = {0, 180};
    float sranges[] = {0, 256};
    const float* ranges[] = {hranges, sranges};

    cv::Mat backproj;
    cv::calcBackProject(&hsv, 1, channels, skinHistogram, backproj, ranges);

    cv::bitwise_and(mask, backproj, mask);
}

cv::Mat SkinDetector::applyMorphology(const cv::Mat& mask) {
    cv::Mat processed = mask.clone();
    cv::erode(processed, processed, kernel, cv::Point(-1,-1), config.morphologyIterations);
    cv::dilate(processed, processed, kernel, cv::Point(-1,-1), config.morphologyIterations);
    
    if (config.blurSize > 0) {
        cv::GaussianBlur(processed, processed, cv::Size(config.blurSize, config.blurSize), 0);
    }
    
    return processed;
}

cv::Mat SkinDetector::applyAdaptiveThreshold(const cv::Mat& mask) {
    cv::Mat thresholded;
    cv::adaptiveThreshold(mask, thresholded, 255,
                         cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                         cv::THRESH_BINARY, 11, 2);
    return thresholded;
}

void SkinDetector::findSkinRegions(const cv::Mat& mask,
                                 std::vector<SkinRegion>& regions) {
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    regions.clear();
    regions.reserve(contours.size());
    
    for (const auto& contour : contours) {
        SkinRegion region;
        region.area = cv::contourArea(contour);
        
        if (region.area < config.minArea || region.area > config.maxArea) {
            continue;
        }
        
        region.boundingBox = cv::boundingRect(contour);
        region.contour = contour;
        region.aspectRatio = calculateAspectRatio(region.boundingBox);
        region.center = calculateCenter(contour);
        region.confidence = calculateConfidence(mask(region.boundingBox));
        
        if (validateRegion(region)) {
            regions.push_back(region);
        }
    }
}

float SkinDetector::calculateConfidence(const cv::Mat& roi) const {
    if (roi.empty()) return 0.0f;
    int nonZeroPixels = cv::countNonZero(roi);
    return static_cast<float>(nonZeroPixels) / (roi.rows * roi.cols);
}

bool SkinDetector::validateRegion(const SkinRegion& region) const {
    return (region.area >= config.minArea && 
            region.area <= config.maxArea &&
            region.aspectRatio >= config.minAspectRatio && 
            region.aspectRatio <= config.maxAspectRatio &&
            region.confidence >= config.confidenceThreshold);
}

cv::Point2f SkinDetector::calculateCenter(const std::vector<cv::Point>& contour) const {
    cv::Moments m = cv::moments(contour);
    if (m.m00 == 0) return cv::Point2f(0, 0);
    return cv::Point2f(m.m10/m.m00, m.m01/m.m00);
}

float SkinDetector::calculateAspectRatio(const cv::Rect& box) const {
    return box.height > 0 ? static_cast<float>(box.width) / box.height : 0.0f;
}

void SkinDetector::trainHistogram(const cv::Mat& skinSample) {
    if (skinSample.empty()) {
        throw std::invalid_argument("Empty skin sample provided");
    }

    try {
        cv::Mat hsv;
        cv::cvtColor(skinSample, hsv, cv::COLOR_BGR2HSV);
        
        int histSize[] = {30, 32};
        float hranges[] = {0, 180};
        float sranges[] = {0, 256};
        const float* ranges[] = {hranges, sranges};
        int channels[] = {0, 1};
        
        std::lock_guard<std::mutex> lock(mtx);
        cv::calcHist(&hsv, 1, channels, cv::Mat(), skinHistogram, 2, histSize, ranges, true, false);
        cv::normalize(skinHistogram, skinHistogram, 0, 255, cv::NORM_MINMAX);
    }
    catch (const cv::Exception& e) {
        throw DetectionError("Failed to train histogram: " + std::string(e.what()));
    }
}

void SkinDetector::setSkinRanges(const std::vector<ColorRange>& ranges) {
    if (ranges.empty()) {
        throw std::invalid_argument("Empty skin ranges provided");
    }

    for (const auto& range : ranges) {
        if (!range.isValid()) {
            throw std::invalid_argument("Invalid skin color range detected");
        }
    }

    std::lock_guard<std::mutex> lock(mtx);
    skinRanges = ranges;
}

std::vector<SkinDetector::ColorRange> SkinDetector::getSkinRanges() const {
    std::lock_guard<std::mutex> lock(mtx);
    return skinRanges;
}

cv::Mat SkinDetector::visualizeSkinRegions(
    const cv::Mat& frame,
    const std::vector<SkinRegion>& regions,
    bool showConfidence,
    bool showArea) {
    
    cv::Mat output = frame.clone();
    
    for (const auto& region : regions) {
        cv::rectangle(output, region.boundingBox, cv::Scalar(0, 255, 0), 2);
        cv::drawContours(output, std::vector<std::vector<cv::Point>>{region.contour},
                        -1, cv::Scalar(0, 255, 0), 1);
        cv::circle(output, cv::Point(region.center.x, region.center.y), 3, cv::Scalar(0, 0, 255), -1);
        
        if (showConfidence) {
            std::string confText = cv::format("Conf: %.2f", region.confidence);
            cv::putText(output, confText,
                       cv::Point(region.boundingBox.x, region.boundingBox.y - 5),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
        }
        
        if (showArea) {
            std::string areaText = cv::format("Area: %.0f", region.area);
            cv::putText(output, areaText,
                       cv::Point(region.boundingBox.x, region.boundingBox.y - 25),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
        }
    }
    
    return output;
}