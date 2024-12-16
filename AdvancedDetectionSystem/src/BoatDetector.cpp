#include "BoatDetector.hpp"
#include <iostream>
#include <cmath>

// Performans optimizasyonu için Static üyeler
static const int MIN_CONTOUR_POINTS = 5;
static const float SOLIDITY_THRESHOLD = 0.7f;
static const float MOVEMENT_THRESHOLD = 5.0f;

BoatDetector::BoatDetector() : BoatDetector(Config{}) {}

BoatDetector::BoatDetector(const Config& config) :
    config(config), 
    isBackgroundInitialized(false) {
    // Morfolojik işlemler için kernel optimizasyonu
    kernel = cv::getStructuringElement(
        cv::MORPH_RECT,
        cv::Size(config.morphSize, config.morphSize)
    );
}

std::vector<BoatDetector::BoatInfo> BoatDetector::detectBoats(const cv::Mat& frame) {
    std::vector<BoatInfo> boats;
    try {
        // 1. Arkaplan modelini güncelle
        updateBackgroundModel(frame);

        // 2. Görüntü ön işleme
        cv::Mat mask = preprocessFrame(frame);
        
        // 3. Konturları bul
        auto contours = findBoatContours(mask);
        
        // 4. Her kontur için tekne bilgisi oluştur
        for (const auto& contour : contours) {
            // Temel alan filtrelemesi
            float area = cv::contourArea(contour);
            if (area < config.minArea || area > config.maxArea) {
                continue;
            }
            
            // Şekil validasyonu
            if (!validateBoatShape(contour)) {
                continue;
            }
            
            // Tekne bilgisi oluştur
            BoatInfo boat;
            boat.boundingBox = cv::boundingRect(contour);
            boat.area = area;
            
            // Merkez noktası hesapla
            boat.position = cv::Point2f(
                boat.boundingBox.x + boat.boundingBox.width/2.0f,
                boat.boundingBox.y + boat.boundingBox.height/2.0f
            );
            
            // Güven skoru ve hareket durumu hesapla
            boat.confidence = calculateConfidence(contour, mask);
            boat.isMoving = isBoatMoving(boat.position);
            
            boats.push_back(boat);
        }
        
        // 5. Pozisyon geçmişini güncelle
        if (!boats.empty()) {
            previousPositions.push_back(boats.back().position);
            while (previousPositions.size() > 10) {
                previousPositions.erase(previousPositions.begin());
            }
        }
        
    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV Error in boat detection: " << e.what() << std::endl;
    }
    
    return boats;
}

cv::Mat BoatDetector::preprocessFrame(const cv::Mat& frame) {
    cv::Mat processed;
    
    // HSV renk uzayına dönüşüm
    cv::Mat hsv;
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
    
    // Renk filtresi uygula
    cv::Mat colorMask;
    cv::inRange(hsv, config.lowerColor, config.upperColor, colorMask);
    
    // Arkaplan çıkarma işlemi
    cv::Mat fgMask;
    if (isBackgroundInitialized) {
        // Mutlak fark alma
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
        // Kayan ortalama ile arkaplan güncelleme
        cv::accumulateWeighted(frame, backgroundModel, 0.01);
    }
}

std::vector<std::vector<cv::Point>> BoatDetector::findBoatContours(const cv::Mat& mask) {
    std::vector<std::vector<cv::Point>> contours;
    // Harici konturları bul (hiyerarşi bilgisi olmadan)
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    return contours;
}

bool BoatDetector::validateBoatShape(const std::vector<cv::Point>& contour) {
    // Minimum nokta sayısı kontrolü
    if (contour.size() < MIN_CONTOUR_POINTS) {
        return false;
    }
    
    // Rotasyona dayanıklı sınırlayıcı dikdörtgen
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
    
    // Doluluk oranı kontrolü
    float solidity = static_cast<float>(contourArea / hullArea);
    return solidity > SOLIDITY_THRESHOLD;
}

float BoatDetector::calculateConfidence(const std::vector<cv::Point>& contour,
                                      const cv::Mat& mask) {
    // Sınırlayıcı kutu içindeki beyaz piksel oranı
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
    
    // Son konumla karşılaştırma
    const cv::Point2f& lastPos = previousPositions.back();
    float movement = cv::norm(currentPos - lastPos);
    
    return movement > MOVEMENT_THRESHOLD;
}

void BoatDetector::visualizeResults(cv::Mat& frame,
                                  const std::vector<BoatInfo>& boats) {
    for (const auto& boat : boats) {
        // Hareket durumuna göre renk seçimi
        cv::Scalar color = boat.isMoving ? cv::Scalar(0, 0, 255) :
                                         cv::Scalar(0, 255, 0);
        
        // Tekne çerçevesi
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