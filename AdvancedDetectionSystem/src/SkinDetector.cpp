#include "SkinDetector.hpp"

SkinDetector::SkinDetector() : SkinDetector(Config{}) {}

SkinDetector::SkinDetector(const Config& config) : config(config) {
    kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3,3));
    
    // Varsayılan cilt rengi aralıkları
    skinRanges = {
        {cv::Scalar(0, 20, 70), cv::Scalar(20, 255, 255), 1.0f},   // Normal ten
        {cv::Scalar(0, 30, 30), cv::Scalar(20, 150, 255), 0.8f},   // Koyu ten
        {cv::Scalar(0, 10, 160), cv::Scalar(25, 150, 255), 0.7f}   // Açık ten
    };
}

void SkinDetector::detectSkin(const cv::Mat& frame,
                            std::vector<SkinRegion>& regions,
                            cv::Mat& mask) {
    try {
        mask = createSkinMask(frame);
        
        if (config.useHistogramBackProjection && !skinHistogram.empty()) {
            applyHistogramBackProjection(frame, mask);
        }
        
        mask = applyMorphology(mask);
        
        if (config.useAdaptiveThreshold) {
            mask = applyAdaptiveThreshold(mask);
        }
        
        findSkinRegions(mask, regions);
        
    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV Error in detectSkin: " << e.what() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error in detectSkin: " << e.what() << std::endl;
    }
}

void SkinDetector::detectSkinRegions(const cv::Mat& frame,
                                   std::vector<SkinRegion>& regions) {
    cv::Mat mask;
    detectSkin(frame, regions, mask);
}

cv::Mat SkinDetector::createSkinMask(const cv::Mat& frame) {
    cv::Mat hsv;
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
    
    cv::Mat finalMask = cv::Mat::zeros(frame.size(), CV_8UC1);
    
    for (const auto& range : skinRanges) {
        cv::Mat rangeMask;
        cv::inRange(hsv, range.lower, range.upper, rangeMask);
        cv::addWeighted(finalMask, 1.0, rangeMask, range.weight, 0.0, finalMask);
    }
    
    return finalMask;
}

cv::Mat SkinDetector::applyMorphology(const cv::Mat& mask) {
    cv::Mat processed = mask.clone();
    
    cv::erode(processed, processed, kernel,
              cv::Point(-1,-1), config.morphologyIterations);
    cv::dilate(processed, processed, kernel,
               cv::Point(-1,-1), config.morphologyIterations);
    
    if (config.blurSize > 0) {
        cv::GaussianBlur(processed, processed,
                        cv::Size(config.blurSize, config.blurSize), 0);
    }
    
    return processed;
}

cv::Mat SkinDetector::applyAdaptiveThreshold(const cv::Mat& mask) {
    cv::Mat result;
    if (config.useAdaptiveThreshold) {
        cv::adaptiveThreshold(
            mask,
            result,
            255,
            cv::ADAPTIVE_THRESH_GAUSSIAN_C,
            cv::THRESH_BINARY,
            11,
            2
        );
    } else {
        result = mask.clone();
    }
    return result;
}

void SkinDetector::findSkinRegions(const cv::Mat& mask,
                                 std::vector<SkinRegion>& regions) {
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    regions.clear();
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

float SkinDetector::calculateConfidence(const cv::Mat& roi) {
    int nonZeroPixels = cv::countNonZero(roi);
    return static_cast<float>(nonZeroPixels) / (roi.rows * roi.cols);
}

void SkinDetector::trainHistogram(const cv::Mat& skinSample) {
    cv::Mat hsv;
    cv::cvtColor(skinSample, hsv, cv::COLOR_BGR2HSV);
    
    int histSize[] = {30, 32};
    float hranges[] = {0, 180};
    float sranges[] = {0, 256};
    const float* ranges[] = {hranges, sranges};
    int channels[] = {0, 1};
    
    cv::calcHist(&hsv, 1, channels, cv::Mat(),
                 skinHistogram, 2, histSize, ranges,
                 true, false);
    
    cv::normalize(skinHistogram, skinHistogram, 0, 255, cv::NORM_MINMAX);
}

void SkinDetector::applyHistogramBackProjection(const cv::Mat& frame,
                                              cv::Mat& mask) {
    cv::Mat hsv;
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
    
    cv::Mat backproj;
    int histSize[] = {30, 32};
    float hranges[] = {0, 180};
    float sranges[] = {0, 256};
    const float* ranges[] = {hranges, sranges};
    int channels[] = {0, 1};
    
    cv::calcBackProject(&hsv, 1, channels, skinHistogram,
                       backproj, ranges);
    
    cv::bitwise_and(mask, backproj, mask);
}

bool SkinDetector::validateRegion(const SkinRegion& region) const {
    if (region.confidence < config.confidenceThreshold) {
        return false;
    }
    
    if (region.aspectRatio < config.minAspectRatio ||
        region.aspectRatio > config.maxAspectRatio) {
        return false;
    }
    
    return true;
}

cv::Point2f SkinDetector::calculateCenter(const std::vector<cv::Point>& contour) {
    cv::Moments moments = cv::moments(contour);
    if (moments.m00 != 0) {
        return cv::Point2f(moments.m10 / moments.m00, moments.m01 / moments.m00);
    }
    return cv::Point2f(0, 0);
}

float SkinDetector::calculateAspectRatio(const cv::Rect& box) const {
    return static_cast<float>(box.width) / box.height;
}

cv::Mat SkinDetector::visualizeSkinRegions(const cv::Mat& frame,
                                         const std::vector<SkinRegion>& regions,
                                         bool showConfidence,
                                         bool showArea) {
    cv::Mat output = frame.clone();
    
    for (const auto& region : regions) {
        // Bölgeyi çerçeve içine al
        cv::rectangle(output, region.boundingBox, cv::Scalar(0, 255, 0), 2);
        // Merkez noktasını göster
        cv::circle(output, region.center, 3, cv::Scalar(0, 0, 255), -1);
        
        int y = region.boundingBox.y - 10;
        
        if (showConfidence) {
            std::string confText = "Conf: " +
                std::to_string(static_cast<int>(region.confidence * 100)) + "%";
            cv::putText(output, confText, cv::Point(region.boundingBox.x, y),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
            y -= 20;
        }
        
        if (showArea) {
            std::string areaText = "Area: " +
                std::to_string(static_cast<int>(region.area)) + "px²";
            cv::putText(output, areaText, cv::Point(region.boundingBox.x, y),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
        }
    }
    
    return output;
}
