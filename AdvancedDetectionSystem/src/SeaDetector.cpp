#include "SeaDetector.hpp"
#include <iostream>

SeaDetector::SeaDetector() {
    // Deniz renk aralıkları optimize edildi
    config.lowerThreshold = cv::Scalar(90, 40, 40);  // Mavi ton aralığı daraltıldı
    config.upperThreshold = cv::Scalar(130, 255, 255);
    config.minArea = 5000.0f;  // Minimum alan arttırıldı
    config.waveThreshold = 0.8f;
    config.morphSize = 5;
    config.useAdaptiveThreshold = true;

    horizonLine = 0.45f; // Ufuk çizgisi varsayılan değeri
    kernel = cv::getStructuringElement(cv::MORPH_RECT,
                                     cv::Size(config.morphSize, config.morphSize));
}

SeaDetector::SeaDetector(const Config& config) : config(config), horizonLine(0.45f) {
    kernel = cv::getStructuringElement(cv::MORPH_RECT,
                                     cv::Size(config.morphSize, config.morphSize));
}

SeaDetector::SeaInfo SeaDetector::detectSea(cv::Mat& frame) {
    SeaDetector::SeaInfo info{false, 0.0f, cv::Point2f(0,0)};
    
    try {
        // Ön işleme
        cv::Mat mask = preprocessFrame(frame);
        
        // Ana kontur tespiti
        std::vector<cv::Point> mainContour = findLargestContour(mask);
        
        if (!mainContour.empty()) {
            float area = cv::contourArea(mainContour);
            if (area > config.minArea) {
                info.isDetected = true;
                info.contour = mainContour;
                info.area = area;
                info.centerPoint = calculateCenterPoint(mainContour);
                
                // Dalga analizi
                detectWaves(frame, info);
                info.depth = estimateDepth(frame, info.centerPoint);
            }
        }
    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV Error in detectSea: " << e.what() << std::endl;
    }
    
    return info;
}

cv::Mat SeaDetector::preprocessFrame(const cv::Mat& frame) {
    cv::Mat hsv;
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
    
    // ROI tanımla - sadece ufuk çizgisi altını al
    int roiHeight = frame.rows * (1.0f - horizonLine);
    cv::Mat roi = hsv(cv::Rect(0, frame.rows * horizonLine,
                              frame.cols, roiHeight));
    
    // Deniz maskelemesi
    cv::Mat mask;
    cv::inRange(roi, config.lowerThreshold, config.upperThreshold, mask);
    
    // Morfolojik işlemler
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
    
    // Tam boyutlu maske oluştur
    cv::Mat fullMask = cv::Mat::zeros(frame.size(), CV_8UC1);
    mask.copyTo(fullMask(cv::Rect(0, frame.rows * horizonLine,
                                 frame.cols, roiHeight)));
    
    return fullMask;
}

void SeaDetector::detectWaves(const cv::Mat& frame, SeaDetector::SeaInfo& info) {
    cv::Mat grayFrame;
    cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);
    
    // ROI için sadece deniz bölgesini al
    int roiY = frame.rows * horizonLine;
    cv::Mat roi = grayFrame(cv::Rect(0, roiY,
                                    frame.cols, frame.rows - roiY));
    
    // Gradyan hesaplama
    cv::Mat sobelX, sobelY, gradMag;
    cv::Sobel(roi, sobelX, CV_32F, 1, 0);
    cv::Sobel(roi, sobelY, CV_32F, 0, 1);
    
    cv::magnitude(sobelX, sobelY, gradMag);
    gradMag.convertTo(gradMag, CV_8UC1);
    
    // Dalga desenlerini tespit et
    cv::threshold(gradMag, gradMag, 50, 255, cv::THRESH_BINARY);
    
    std::vector<std::vector<cv::Point>> waveContours;
    cv::findContours(gradMag, waveContours, cv::RETR_EXTERNAL,
                     cv::CHAIN_APPROX_SIMPLE);
    
    // Dalga yoğunluğunu hesapla
    float totalIntensity = 0.0f;
    int validContours = 0;
    
    for (const auto& waveContour : waveContours) {
        if (cv::contourArea(waveContour) > 100) {
            totalIntensity += calculateWaveIntensity(waveContour);
            validContours++;
        }
    }
    
    info.waveIntensity = validContours > 0 ? totalIntensity / validContours : 0.0f;
}

float SeaDetector::calculateWaveIntensity(const std::vector<cv::Point>& contour) {
    if (contour.empty()) return 0.0f;
    
    double perimeter = cv::arcLength(contour, true);
    double area = cv::contourArea(contour);
    
    if (area < 1e-5) return 0.0f;
    
    // Dalga karmaşıklık ölçüsü
    float complexity = static_cast<float>((perimeter * perimeter) /
                                        (4 * CV_PI * area));
    
    // Kontur analizi
    std::vector<cv::Point> hull;
    cv::convexHull(contour, hull);
    
    // Hull/Contour oranı
    float hullArea = cv::contourArea(hull);
    float convexityRatio = hullArea > 0 ? area / hullArea : 0;
    
    return (complexity * 0.6f + (1.0f - convexityRatio) * 0.4f);
}

float SeaDetector::estimateDepth(const cv::Mat& frame, const cv::Point2f& point) {
    float relativeHeight = (point.y - frame.rows * horizonLine) /
                          (frame.rows * (1.0f - horizonLine));
    
    float depth = relativeHeight > 0 ?
                 -std::log(1 - relativeHeight * 0.9f) : 0;
    
    return std::min(std::max(depth, 0.0f), 1.0f);
}

std::vector<cv::Point> SeaDetector::findLargestContour(const cv::Mat& mask) {
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    if (contours.empty()) return std::vector<cv::Point>();
    
    return *std::max_element(contours.begin(), contours.end(),
        [](const auto& c1, const auto& c2) {
            return cv::contourArea(c1) < cv::contourArea(c2);
        });
}

cv::Point2f SeaDetector::calculateCenterPoint(const std::vector<cv::Point>& contour) {
    cv::Moments moments = cv::moments(contour);
    if (moments.m00 != 0) {
        return cv::Point2f(moments.m10/moments.m00, moments.m01/moments.m00);
    }
    return cv::Point2f(0, 0);
}

void SeaDetector::visualizeResults(cv::Mat& frame, const SeaDetector::SeaInfo& info) {
    if (!info.isDetected) return;
    
    try {
        // Ufuk çizgisini göster
        int horizonY = frame.rows * horizonLine;
        cv::line(frame, cv::Point(0, horizonY),
                cv::Point(frame.cols, horizonY),
                cv::Scalar(0, 255, 255), 2);
        
        // Kontur çizimi
        cv::polylines(frame, info.contour, true, cv::Scalar(0, 255, 0), 2);
        
        // Merkez noktası
        cv::circle(frame, info.centerPoint, 5, cv::Scalar(0, 0, 255), -1);
        
        // Dalga yoğunluğu gösterimi
        std::string waveText = "Dalga Yogunlugu: " +
            std::to_string(static_cast<int>(info.waveIntensity * 100)) + "%";
        cv::putText(frame, waveText, cv::Point(10, 30),
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
        
        // Derinlik gösterimi
        std::string depthText = "Tahmini Derinlik: " +
            std::to_string(static_cast<int>(info.depth * 100)) + "%";
        cv::putText(frame, depthText, cv::Point(10, 60),
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
        
        // Yüksek dalga uyarısı
        if (info.waveIntensity > config.waveThreshold) {
            cv::putText(frame, "YUKSEK DALGA AKTIVITESI!",
                       cv::Point(10, 90),
                       cv::FONT_HERSHEY_SIMPLEX, 0.7,
                       cv::Scalar(0, 0, 255), 2);
        }
    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV Error in visualizeResults: " << e.what() << std::endl;
    }
}
