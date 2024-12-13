#include "BoatDetector.hpp"
#include <iostream>

BoatDetector::BoatDetector() : isBackgroundInitialized(false) {
    kernel = cv::getStructuringElement(cv::MORPH_RECT,
                                     cv::Size(config.morphSize, config.morphSize));
}

BoatDetector::BoatDetector(const Config& config) :
    config(config), isBackgroundInitialized(false) {
    kernel = cv::getStructuringElement(cv::MORPH_RECT,
                                     cv::Size(config.morphSize, config.morphSize));
}

std::vector<BoatDetector::BoatInfo> BoatDetector::detectBoats(const cv::Mat& frame) {
    std::vector<BoatInfo> boats;
    try {
        // Arkaplan modelini güncelle
        updateBackgroundModel(frame);

        // Ön işleme
        cv::Mat mask = preprocessFrame(frame);
        
        // Konturları bul
        auto contours = findBoatContours(mask);
        
        // Her kontur için tekne bilgisi oluştur
        for (const auto& contour : contours) {
            float area = cv::contourArea(contour);
            
            if (area < config.minArea || area > config.maxArea) {
                continue;
            }
            
            if (!validateBoatShape(contour)) {
                continue;
            }
            
            BoatInfo boat;
            boat.boundingBox = cv::boundingRect(contour);
            boat.area = area;
            boat.position = cv::Point2f(
                boat.boundingBox.x + boat.boundingBox.width/2.0f,
                boat.boundingBox.y + boat.boundingBox.height/2.0f
            );
            boat.confidence = calculateConfidence(contour, mask);
            boat.isMoving = isBoatMoving(boat.position);
            
            boats.push_back(boat);
        }
        
        // Pozisyon geçmişini güncelle
        if (!boats.empty()) {
            previousPositions.push_back(boats.back().position);
            if (previousPositions.size() > 10) {
                previousPositions.erase(previousPositions.begin());
            }
        }
        
    } catch (const cv::Exception& e) {
        std::cerr << "Error in boat detection: " << e.what() << std::endl;
    }
    
    return boats;
}

cv::Mat BoatDetector::preprocessFrame(const cv::Mat& frame) {
    cv::Mat processed;
    cv::Mat hsv;
    
    // HSV'ye dönüştür
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
    
    // Renk filtresi uygula
    cv::Mat colorMask;
    cv::inRange(hsv, config.lowerColor, config.upperColor, colorMask);
    
    // Arkaplan çıkarma
    cv::Mat fgMask;
    if (isBackgroundInitialized) {
        cv::absdiff(frame, backgroundModel, fgMask);
        cv::cvtColor(fgMask, fgMask, cv::COLOR_BGR2GRAY);
        cv::threshold(fgMask, fgMask, 30, 255, cv::THRESH_BINARY);
        
        // Renk maskesi ile birleştir
        cv::bitwise_and(colorMask, fgMask, processed);
    } else {
        processed = colorMask;
    }
    
    // Morfolojik işlemler
    cv::morphologyEx(processed, processed, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(processed, processed, cv::MORPH_CLOSE, kernel);
    
    return processed;
}

void BoatDetector::updateBackgroundModel(const cv::Mat& frame) {
    if (!isBackgroundInitialized) {
        backgroundModel = frame.clone();
        isBackgroundInitialized = true;
    } else {
        // Arkaplan modelini yavaşça güncelle
        cv::accumulateWeighted(frame, backgroundModel, 0.01);
    }
}

std::vector<std::vector<cv::Point>> BoatDetector::findBoatContours(const cv::Mat& mask) {
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    return contours;
}

bool BoatDetector::validateBoatShape(const std::vector<cv::Point>& contour) {
    // Kontur analizi
    cv::RotatedRect minRect = cv::minAreaRect(contour);
    float width = minRect.size.width;
    float height = minRect.size.height;
    
    // En-boy oranı kontrolü
    float aspectRatio = width / height;
    if (aspectRatio < 0.2f || aspectRatio > 5.0f) {
        return false;
    }
    
    // Konvekslik kontrolü
    std::vector<cv::Point> hull;
    cv::convexHull(contour, hull);
    double contourArea = cv::contourArea(contour);
    double hullArea = cv::contourArea(hull);
    
    if (hullArea <= 0) return false;
    
    float solidity = static_cast<float>(contourArea / hullArea);
    return solidity > 0.7f;  // Tekne şekli genelde dışbükeydir
}

float BoatDetector::calculateConfidence(const std::vector<cv::Point>& contour,
                                      const cv::Mat& mask) {
    cv::Rect bbox = cv::boundingRect(contour);
    cv::Mat roi = mask(bbox);
    
    int nonZeroPixels = cv::countNonZero(roi);
    float confidence = static_cast<float>(nonZeroPixels) /
                      (bbox.width * bbox.height);
    
    return confidence;
}

bool BoatDetector::isBoatMoving(const cv::Point2f& currentPos) {
    if (previousPositions.empty()) {
        return false;
    }
    
    const float movementThreshold = 5.0f;  // piksel cinsinden hareket eşiği
    cv::Point2f lastPos = previousPositions.back();
    
    float movement = cv::norm(currentPos - lastPos);
    return movement > movementThreshold;
}

void BoatDetector::visualizeResults(cv::Mat& frame,
                                  const std::vector<BoatInfo>& boats) {
    for (const auto& boat : boats) {
        // Tekne çerçevesi
        cv::Scalar color = boat.isMoving ? cv::Scalar(0, 0, 255) :
                                         cv::Scalar(0, 255, 0);
        cv::rectangle(frame, boat.boundingBox, color, 2);
        
        // Güven skoru
        std::string confText = std::to_string(static_cast<int>(boat.confidence * 100)) + "%";
        cv::putText(frame, confText,
                   cv::Point(boat.boundingBox.x, boat.boundingBox.y - 5),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 2);
        
        // Hareket durumu
        if (boat.isMoving) {
            cv::putText(frame, "Moving",
                       cv::Point(boat.boundingBox.x, boat.boundingBox.y - 25),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 2);
        }
    }
}
