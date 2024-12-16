#include "HumanDetector.hpp"
#include <cmath>
#include <algorithm>

HumanDetector::HumanDetector() {
    try {
        hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());
        
        // Kamera matrisini başlat
        params.cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
        params.cameraMatrix.at<double>(0,0) = params.focalLength;
        params.cameraMatrix.at<double>(1,1) = params.focalLength;
        params.distCoeffs = cv::Mat::zeros(5, 1, CV_64F);
        
    } catch (const cv::Exception& e) {
        throw DetectionError("Failed to initialize HOG detector: " + std::string(e.what()));
    }
}

void HumanDetector::preprocessFrame(const cv::Mat& frame) {
    if (frame.size().width > 1920 || frame.size().height > 1080) {
        double scale = std::min(1920.0 / frame.size().width, 
                              1080.0 / frame.size().height);
        cv::resize(frame, tempImage, cv::Size(), scale, scale);
    } else {
        frame.copyTo(tempImage);
    }

    if (frame.channels() == 3) {
        cv::GaussianBlur(tempImage, tempImage, cv::Size(3,3), 0);
    }
}

std::vector<HumanDetector::DetectionInfo> 
HumanDetector::detectHumans(const cv::Mat& frame) {
    if (frame.empty()) {
        throw DetectionError("Empty frame provided");
    }

    std::vector<DetectionInfo> detections;
    try {
        std::lock_guard<std::mutex> lock(mtx);
        
        preprocessFrame(frame);
        
        // HOG tespiti
        foundLocations.clear();
        weights.clear();
        
        hog.detectMultiScale(tempImage, foundLocations, weights, 
                            0,              // hit threshold
                            cv::Size(8,8),  // win stride
                            cv::Size(32,32),// padding
                            1.05,           // scale
                            2.0);           // group threshold

        // Sonuçları dönüştür
        detections.reserve(foundLocations.size());
        for (size_t i = 0; i < foundLocations.size(); i++) {
            DetectionInfo info;
            info.box = foundLocations[i];
            info.confidence = weights[i];
            info.label = "Person";
            info.distance = calculateDistance(foundLocations[i]);
            info.direction = calculateDirection(foundLocations[i], lastDetectedBox);
            info.isMoving = isPersonMoving(foundLocations[i], lastDetectedBox);
            
            if (info.isValid()) {
                detections.push_back(info);
            }
        }

        // Son tespit edilen bölgeyi güncelle
        if (!foundLocations.empty()) {
            lastDetectedBox = foundLocations[0];
        }
        
        // Detections'ı filtreleme ve sıralama
        std::sort(detections.begin(), detections.end(),
                 [](const DetectionInfo& a, const DetectionInfo& b) {
                     return a.confidence > b.confidence;
                 });

        // Örtüşen tespitleri birleştir
        for (size_t i = 0; i < detections.size(); i++) {
            for (size_t j = i + 1; j < detections.size();) {
                float overlap = static_cast<float>(
                    (detections[i].box & detections[j].box).area()) /
                    static_cast<float>(
                        (detections[i].box | detections[j].box).area());
                
                if (overlap > 0.3f) {
                    detections.erase(detections.begin() + j);
                } else {
                    ++j;
                }
            }
        }
        
    } catch (const cv::Exception& e) {
        throw DetectionError("OpenCV error in detectHumans: " + std::string(e.what()));
    }

    return detections;
}

float HumanDetector::calculateConfidenceScore(const cv::Rect& detection) const {
    if (detection.empty()) return 0.0f;
    
    float aspectRatio = static_cast<float>(detection.width) / detection.height;
    float area = detection.width * detection.height;
    
    // İdeal insan vücut oranları ile karşılaştır
    float aspectRatioScore = std::max(0.0f, 1.0f - std::abs(aspectRatio - 0.41f));
    float areaScore = std::min(1.0f, area / 40000.0f);
    
    return (aspectRatioScore * 0.6f + areaScore * 0.4f);
}

float HumanDetector::calculateDistance(const cv::Rect& box) const {
    if (box.empty() || box.height <= 0) return -1.0f;
    
    // Pinhole kamera modeli ile mesafe hesapla
    float distance = (params.realHeight * params.focalLength) / box.height;
    
    // Perspektif düzeltmesi
    float verticalPosition = box.y + box.height/2.0f;
    float imageHeight = params.cameraMatrix.at<double>(1,2) * 2;
    float angleCorrection = 1.0f + (verticalPosition - imageHeight/2) * 0.001f;
    
    return distance * angleCorrection;
}

cv::Point2f HumanDetector::calculateDirection(const cv::Rect& currentBox,
                                            const cv::Rect& previousBox) const {
    if (previousBox.empty()) return cv::Point2f(0, 0);
    
    cv::Point2f currentCenter(currentBox.x + currentBox.width/2.0f,
                            currentBox.y + currentBox.height/2.0f);
    cv::Point2f previousCenter(previousBox.x + previousBox.width/2.0f,
                             previousBox.y + previousBox.height/2.0f);
    
    return currentCenter - previousCenter;
}

bool HumanDetector::isPersonMoving(const cv::Rect& currentBox,
                                  const cv::Rect& previousBox,
                                  float threshold) const {
    if (previousBox.empty()) return false;
    
    cv::Point2f direction = calculateDirection(currentBox, previousBox);
    float distance = cv::norm(direction);
    
    return distance > threshold;
}
