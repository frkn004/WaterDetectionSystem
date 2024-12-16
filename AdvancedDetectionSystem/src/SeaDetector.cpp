#include "SeaDetector.hpp"
#include <algorithm>
#include <numeric>

SeaDetector::SeaDetector() : SeaDetector(Config{}) {}

SeaDetector::SeaDetector(const Config& config) : config(config) {
    try {
        config.validate();
        kernel = cv::getStructuringElement(cv::MORPH_RECT, 
                                         cv::Size(config.morphSize, config.morphSize));
    }
    catch (const std::exception& e) {
        throw SeaDetectionError("Failed to initialize SeaDetector: " + std::string(e.what()));
    }
}

// SeaInfo yerine SeaDetector::SeaInfo kullanıldı
SeaDetector::SeaInfo SeaDetector::detectSea(const cv::Mat& frame) {
    if (frame.empty()) {
        throw SeaDetectionError("Empty frame provided");
    }

    SeaInfo info;
    try {
        std::lock_guard<std::mutex> lock(mtx);
        
        // 1. Görüntü ön işleme
        cv::Mat processed = preprocessFrame(frame);
        
        // 2. Su bölgesi tespiti
        cv::Mat waterMask = detectWaterRegion(processed);
        
        // 3. En büyük konturu bul
        info.contour = findLargestContour(waterMask);
        
        if (!info.contour.empty()) {
            info.isDetected = true;
            info.centerPoint = calculateCenterPoint(info.contour);
            info.area = cv::contourArea(info.contour);
            
            // 4. Dalga analizi
            info.waveAnalysis = analyzeWavePattern(frame);
            info.waveIntensity = info.waveAnalysis.waveIntensity;
            
            // 5. Ek özellikler
            info.visibility = estimateVisibility(frame, waterMask);
            info.surfaceTemperature = estimateSurfaceTemperature(frame, waterMask);
            info.waveAnalysis.dominantDirection = calculateDominantDirection(info.contour);
        }

        if (!info.isValid()) {
            throw SeaDetectionError("Invalid sea detection results");
        }
        
        return info;
    }
    catch (const cv::Exception& e) {
        throw SeaDetectionError("OpenCV error in detectSea: " + std::string(e.what()));
    }
}

cv::Mat SeaDetector::preprocessFrame(const cv::Mat& frame) {
    if (preprocessBuffer.empty() || preprocessBuffer.size() != frame.size() ||
        preprocessBuffer.type() != frame.type()) {
        preprocessBuffer = cv::Mat(frame.size(), frame.type());
    }
    
    frame.copyTo(preprocessBuffer);
    cv::GaussianBlur(preprocessBuffer, preprocessBuffer, cv::Size(5, 5), 0);
    cv::convertScaleAbs(preprocessBuffer, preprocessBuffer, 1.2, 0);
    
    return preprocessBuffer;
}

cv::Mat SeaDetector::detectWaterRegion(const cv::Mat& frame) {
    cv::Mat hsv;
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
    
    if (waterMaskBuffer.empty() || waterMaskBuffer.size() != frame.size()) {
        waterMaskBuffer = cv::Mat(frame.size(), CV_8UC1);
    }
    
    cv::inRange(hsv, config.waterLowerBound, config.waterUpperBound, waterMaskBuffer);
    cv::morphologyEx(waterMaskBuffer, waterMaskBuffer, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(waterMaskBuffer, waterMaskBuffer, cv::MORPH_CLOSE, kernel);

    if (config.useAdaptiveThreshold) {
        cv::adaptiveThreshold(waterMaskBuffer, waterMaskBuffer, 255,
                            cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                            cv::THRESH_BINARY, 11, 2);
    }

    return waterMaskBuffer;
}

// WaveAnalysis yerine SeaDetector::WaveAnalysis kullanıldı
SeaDetector::WaveAnalysis SeaDetector::analyzeWavePattern(const cv::Mat& frame) {
    WaveAnalysis analysis;
    
    try {
        cv::Mat waterMask = detectWaterRegion(frame);
        analysis.waveHeight = calculateWaveHeight(waterMask);
        analysis.wavePeriod = calculateWavePeriod();
        analysis.turbulence = calculateTurbulence(frame, waterMask);
        analysis.waveIntensity = analysis.turbulence;
        analysis.isDangerous = evaluateWaveConditions(analysis);
        
    } catch (const std::exception& e) {
        throw SeaDetectionError("Error in wave analysis: " + std::string(e.what()));
    }
    
    return analysis;
}

float SeaDetector::calculateTurbulence(const cv::Mat& frame, const cv::Mat& waterMask) {
    if (frame.empty() || waterMask.empty()) return 0.0f;

    cv::Mat gray, masked;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    gray.copyTo(masked, waterMask);

    cv::Mat laplacian;
    cv::Laplacian(masked, laplacian, CV_32F);

    cv::Mat gradX, gradY;
    cv::Sobel(masked, gradX, CV_32F, 1, 0);
    cv::Sobel(masked, gradY, CV_32F, 0, 1);

    cv::Mat magnitude;
    cv::magnitude(gradX, gradY, magnitude);

    cv::Scalar laplacianScore, gradientScore;
    cv::meanStdDev(laplacian, cv::Scalar(), laplacianScore);
    cv::meanStdDev(magnitude, cv::Scalar(), gradientScore);

    float combinedScore = (laplacianScore[0] + gradientScore[0]) / 2.0f;
    return std::min(1.0f, combinedScore / 100.0f);
}

float SeaDetector::calculateWaveHeight(const cv::Mat& waterMask) const {
    if (waterMask.empty()) return 0.0f;
    
    cv::Mat gradY;
    cv::Sobel(waterMask, gradY, CV_32F, 0, 1);
    
    cv::Scalar mean, stddev;
    cv::meanStdDev(gradY, mean, stddev);
    
    float estimatedHeight = stddev[0] * 0.1f;
    return std::min(estimatedHeight, config.maxWaveHeight);
}

float SeaDetector::calculateWavePeriod() const {
    return 5.0f;
}

bool SeaDetector::evaluateWaveConditions(const WaveAnalysis& analysis) const {
    bool heightDanger = analysis.waveHeight > config.maxWaveHeight * 0.8f;
    bool periodDanger = analysis.wavePeriod < config.dangerousWavePeriod;
    bool turbulenceDanger = analysis.turbulence > config.waveThreshold;
    
    return heightDanger || periodDanger || turbulenceDanger;
}

std::vector<cv::Point> SeaDetector::findLargestContour(const cv::Mat& mask) {
    if (mask.empty()) return std::vector<cv::Point>();
    
    contoursBuffer.clear();
    cv::findContours(mask.clone(), contoursBuffer, cv::RETR_EXTERNAL, 
                     cv::CHAIN_APPROX_SIMPLE);
    
    if (contoursBuffer.empty()) return std::vector<cv::Point>();
    
    auto maxContour = std::max_element(contoursBuffer.begin(), contoursBuffer.end(),
        [](const std::vector<cv::Point>& c1, const std::vector<cv::Point>& c2) {
            return cv::contourArea(c1) < cv::contourArea(c2);
        });
    
    return *maxContour;
}

cv::Point2f SeaDetector::calculateCenterPoint(const std::vector<cv::Point>& contour) const {
    if (contour.empty()) return cv::Point2f(0, 0);
    
    cv::Moments m = cv::moments(contour);
    if (m.m00 == 0) return cv::Point2f(0, 0);
    
    return cv::Point2f(m.m10/m.m00, m.m01/m.m00);
}

float SeaDetector::estimateVisibility(const cv::Mat& frame, const cv::Mat& waterMask) {
    cv::Mat masked;
    frame.copyTo(masked, waterMask);
    
    cv::Mat gray;
    cv::cvtColor(masked, gray, cv::COLOR_BGR2GRAY);
    
    cv::Scalar mean, stddev;
    cv::meanStdDev(gray, mean, stddev);
    
    float contrast = stddev[0] / mean[0];
    return 100.0f * (1.0f - std::min(1.0f, contrast));
}

float SeaDetector::estimateSurfaceTemperature(const cv::Mat& frame, const cv::Mat& waterMask) {
    constexpr float BASELINE_TEMP = 20.0f;
    constexpr float MAX_VARIATION = 5.0f;
    
    cv::Mat masked;
    frame.copyTo(masked, waterMask);
    
    cv::Scalar meanColor = cv::mean(masked, waterMask);
    float blueComponent = meanColor[0] / 255.0f;
    
    return BASELINE_TEMP + (blueComponent - 0.5f) * MAX_VARIATION;
}

cv::Point2f SeaDetector::calculateDominantDirection(const std::vector<cv::Point>& contour) {
    if (contour.size() < 2) return cv::Point2f(0, 0);
    
    cv::Mat points(contour.size(), 2, CV_32F);
    for (size_t i = 0; i < contour.size(); i++) {
        points.at<float>(i, 0) = contour[i].x;
        points.at<float>(i, 1) = contour[i].y;
    }
    
    cv::PCA pca(points, cv::Mat(), cv::PCA::DATA_AS_ROW);
    return cv::Point2f(pca.eigenvectors.at<float>(0, 0),
                      pca.eigenvectors.at<float>(0, 1));
}

void SeaDetector::visualizeResults(cv::Mat& frame, const SeaInfo& info) const {
    if (!info.isDetected) return;

    try {
        if (!info.contour.empty()) {
            cv::polylines(frame, std::vector<std::vector<cv::Point>>{info.contour},
                         true, cv::Scalar(0, 255, 0), 2);
        }

        cv::circle(frame, info.centerPoint, 5, cv::Scalar(0, 0, 255), -1);

        int lineHeight = 30;
        int yPos = 30;
        
        auto drawText = [&](const std::string& text, const cv::Scalar& color) {
            cv::putText(frame, text, cv::Point(10, yPos),
                       cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
            yPos += lineHeight;
        };

        drawText(cv::format("Wave Height: %.1fm", info.waveAnalysis.waveHeight),
                cv::Scalar(0, 255, 0));
        drawText(cv::format("Wave Period: %.1fs", info.waveAnalysis.wavePeriod),
                cv::Scalar(0, 255, 0));
        drawText(cv::format("Visibility: %.1fm", info.visibility),
                cv::Scalar(0, 255, 0));
        drawText(cv::format("Temperature: %.1f°C", info.surfaceTemperature),
                cv::Scalar(0, 255, 0));

        if (info.waveAnalysis.isDangerous) {
            drawText("WARNING: High Wave Activity!", cv::Scalar(0, 0, 255));
        }

        int barWidth = 200;
        int barHeight = 20;

        cv::rectangle(frame,
                     cv::Point(frame.cols - barWidth - 20, 30),
                     cv::Point(frame.cols - 20, 30 + barHeight),
                     cv::Scalar(0, 0, 0), 1);

        int filledWidth = static_cast<int>(info.waveIntensity * barWidth);
        cv::rectangle(frame,
                     cv::Point(frame.cols - barWidth - 20, 30),
                     cv::Point(frame.cols - barWidth - 20 + filledWidth, 30 + barHeight),
                     cv::Scalar(0, 0, 255), -1);
    }
    catch (const cv::Exception& e) {
        throw SeaDetectionError("Visualization error: " + std::string(e.what()));
    }
}