#include "SeaDetector.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>

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

// SeaDetector.cpp içine eklenecek fonksiyonlar:

void SeaDetector::calibrate(const CalibrationData& calibData) {
    try {
        std::lock_guard<std::mutex> lock(mtx);
        
        // Kalibrasyon verilerinin validasyonu
        if (calibData.pixelToMeterRatio <= 0) {
            throw std::invalid_argument("Invalid pixel to meter ratio");
        }
        
        // Referans noktalarının validasyonu
        for (int i = 0; i < 4; ++i) {
            if (calibData.referencePoints[i].x < 0 || calibData.referencePoints[i].y < 0) {
                throw std::invalid_argument("Invalid reference point coordinates");
            }
        }
        
        // Kalibrasyon verilerini güncelle
        calibration = calibData;
        
    } catch (const std::exception& e) {
        throw SeaDetectionError("Calibration error: " + std::string(e.what()));
    }
}

void SeaDetector::setConfig(const Config& newConfig) {
    try {
        std::lock_guard<std::mutex> lock(mtx);
        
        // Yeni konfigürasyonu doğrula
        newConfig.validate();
        
        // Konfigürasyonu güncelle
        config = newConfig;
        
        // Kernel'i yeniden oluştur
        kernel = cv::getStructuringElement(
            cv::MORPH_RECT, 
            cv::Size(config.morphSize, config.morphSize)
        );
        
    } catch (const std::exception& e) {
        throw SeaDetectionError("Configuration error: " + std::string(e.what()));
    }
}

SeaDetector::Config SeaDetector::getConfig() const {
    std::lock_guard<std::mutex> lock(mtx);
    return config;
}

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
        
        // 3. Ana kontur tespiti
        info.contour = findLargestContour(waterMask);
        
        if (!info.contour.empty()) {
            info.isDetected = true;
            info.centerPoint = calculateCenterPoint(info.contour);
            info.area = cv::contourArea(info.contour);
            
            // 4. Dalga analizi
            info.waveMetrics = analyzeWaves(frame, waterMask);
            info.waveIntensity = info.waveMetrics.energy;
            
            // 5. Su seviyesi tespiti
            info.waterLevel = detectWaterLevel(frame, waterMask);
            
            // 6. Sıcaklık ölçümü
            info.temperature = measureTemperature(frame, waterMask);
            
            // 7. Görünürlük hesaplama
            info.visibility = estimateVisibility(frame, waterMask);
            
            // 8. Zaman damgası
            info.timestamp = std::chrono::system_clock::now();

            // 9. Geçmiş verileri güncelle
            updateHistories(info.waveMetrics, info.temperature, info.waterLevel);
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
    if (preprocessBuffer.empty() || preprocessBuffer.size() != frame.size()) {
        preprocessBuffer = cv::Mat(frame.size(), frame.type());
    }
    
    // Gürültü azaltma
    cv::GaussianBlur(frame, preprocessBuffer, 
                     cv::Size(config.blurSize, config.blurSize), 0);
    
    // Kontrast iyileştirme
    cv::convertScaleAbs(preprocessBuffer, preprocessBuffer, 
                       config.contrastAlpha, 0);
    
    return preprocessBuffer;
}

cv::Mat SeaDetector::detectWaterRegion(const cv::Mat& frame) {
    cv::Mat hsv;
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
    
    if (waterMaskBuffer.empty() || waterMaskBuffer.size() != frame.size()) {
        waterMaskBuffer = cv::Mat(frame.size(), CV_8UC1);
    }
    
    // Su rengi tespiti
    cv::inRange(hsv, config.waterLowerBound, config.waterUpperBound, waterMaskBuffer);
    
    // Morfolojik işlemler
    cv::morphologyEx(waterMaskBuffer, waterMaskBuffer, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(waterMaskBuffer, waterMaskBuffer, cv::MORPH_CLOSE, kernel);

    if (config.useAdaptiveThreshold) {
        cv::adaptiveThreshold(waterMaskBuffer, waterMaskBuffer, 255,
                            cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                            cv::THRESH_BINARY, 11, 2);
    }

    return waterMaskBuffer;
}

SeaDetector::WaveMetrics SeaDetector::analyzeWaves(const cv::Mat& frame, 
                                                  const cv::Mat& waterMask) {
    WaveMetrics metrics;
    
    try {
        // 1. Optik akış hesaplama
        cv::Mat flow = computeOpticalFlow(frame);
        
        if (!flow.empty()) {
            // 2. Dalga hızı ve yönü
            cv::Point2f velocity = calculateWaveVelocity(flow);
            metrics.velocity = cv::norm(velocity);
            metrics.direction = atan2(velocity.y, velocity.x) * 180 / CV_PI;
            if (metrics.direction < 0) metrics.direction += 360;
            
            // 3. Dalga frekansı
            metrics.frequency = calculateWaveFrequency(flow);
            metrics.period = metrics.frequency > 0 ? 1.0f / metrics.frequency : 0;
            
            // 4. Dalga yüksekliği (perspektif düzeltmeli)
            cv::Mat gradY;
            cv::Sobel(waterMask, gradY, CV_32F, 0, 1);
            cv::Scalar mean, stddev;
            cv::meanStdDev(gradY, mean, stddev);
            
            metrics.height = pixelsToMeters(stddev[0]);
            
            // 5. Türbülans
            metrics.turbulence = calculateTurbulence(frame, waterMask);
            
            // 6. Dalga enerjisi
            metrics.energy = calculateWaveEnergy(metrics.height, metrics.period);
            
            // 7. Tehlike durumu değerlendirmesi
            metrics.isDangerous = evaluateWaveConditions(metrics);
        }
    } catch (const std::exception& e) {
        throw SeaDetectionError("Error in wave analysis: " + std::string(e.what()));
    }
    
    return metrics;
}

cv::Mat SeaDetector::computeOpticalFlow(const cv::Mat& current) {
    cv::Mat currentGray;
    cv::cvtColor(current, currentGray, cv::COLOR_BGR2GRAY);
    
    cv::Mat flow;
    if (!prevGray.empty()) {
        cv::calcOpticalFlowFarneback(prevGray, currentGray, flow,
                                   0.5, 3, 15, 3, 5, 1.2, 0);
    }
    
    currentGray.copyTo(prevGray);
    return flow;
}

float SeaDetector::calculateWaveFrequency(const cv::Mat& flow) {
    cv::Mat magnitude, angle;
    cv::cartToPolar(flow.col(0), flow.col(1), magnitude, angle);
    
    // FFT analizi
    cv::Mat rowData = magnitude.row(magnitude.rows/2);
    performFFTAnalysis(rowData);
    
    cv::Mat magnitudes;
    cv::magnitude(fftBuffer.col(0), fftBuffer.col(1), magnitudes);
    
    // En yüksek frekans bileşenini bul
    cv::Point maxLoc;
    cv::minMaxLoc(magnitudes(cv::Rect(1, 0, magnitudes.rows/2, 1)), 
                  nullptr, nullptr, nullptr, &maxLoc);
    
    float freq = maxLoc.x * config.samplingRate / config.fftSize;
    return std::clamp(freq, config.minWaveFreq, config.maxWaveFreq);
}

void SeaDetector::performFFTAnalysis(const cv::Mat& data) {
    int optimalSize = cv::getOptimalDFTSize(data.cols);
    
    // Veriyi FFT için hazırla
    cv::Mat padded;
    cv::copyMakeBorder(data, padded, 0, 0, 0, 
                      optimalSize - data.cols, 
                      cv::BORDER_CONSTANT, cv::Scalar::all(0));
    
    std::vector<cv::Mat> planes = {cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F)};
    cv::merge(planes, fftBuffer);
    
    // FFT uygula
    cv::dft(fftBuffer, fftBuffer);
}

cv::Point2f SeaDetector::calculateWaveVelocity(const cv::Mat& flow) {
    cv::Scalar meanFlow = cv::mean(flow);
    return cv::Point2f(meanFlow[0], meanFlow[1]);
}

float SeaDetector::calculateWaveEnergy(float height, float period) {
    if (height <= 0 || period <= 0) return 0.0f;
    
    const float WATER_DENSITY = 1000.0f;  // kg/m³
    const float GRAVITY = 9.81f;          // m/s²
    
    // Dalga enerjisi formülü: E = (1/8) * ρ * g * H² * T
    return 0.125f * WATER_DENSITY * GRAVITY * height * height * period;
}

SeaDetector::TemperatureData SeaDetector::measureTemperature(const cv::Mat& frame, 
                                                           const cv::Mat& waterMask) {
    TemperatureData temp;
    temp.timestamp = std::chrono::system_clock::now();
    
    cv::Mat masked;
    frame.copyTo(masked, waterMask);
    
    // Termal analiz (basitleştirilmiş)
    cv::Scalar meanColor = cv::mean(masked, waterMask);
    
    // Referans sıcaklık ile karşılaştır
    float blueRatio = meanColor[0] / 255.0f;
    float greenRatio = meanColor[1] / 255.0f;
    
    temp.value = calibration.knownTemperature + 
                 (blueRatio - 0.5f) * config.temperatureThreshold;
    
    // Güvenilirlik skoru hesapla
    temp.confidence = 1.0f - std::abs(greenRatio - blueRatio);
    
    return temp;
}

SeaDetector::WaterLevel SeaDetector::detectWaterLevel(const cv::Mat& frame, 
                                                     const cv::Mat& waterMask) {
    WaterLevel level;
    level.lastUpdate = std::chrono::system_clock::now();
    
    // Yatay çizgi tespiti
    cv::Mat edges;
    cv::Canny(frame, edges, 50, 150);
    
    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(edges, lines, 1, CV_PI/180, 50, 50, 10);
    
    // En iyi su çizgisini bul
    float bestY = 0;
    int maxVotes = 0;
    
    for (const auto& line : lines) {
        float angle = std::atan2(line[3] - line[1], line[2] - line[0]) * 180 / CV_PI;
        
        // Yatay çizgileri filtrele (-5° ile +5° arası)
        if (std::abs(angle) < 5) {
            float y = (line[1] + line[3]) / 2.0f;
            int votes = 0;
            
            // Bu seviyedeki piksel sayısını say
            for (int x = 0; x < waterMask.cols; x++) {
                if (waterMask.at<uchar>(static_cast<int>(y), x) > 0) {
                    votes++;
                }
            }
            
            if (votes > maxVotes) {
                maxVotes = votes;
                bestY = y;
            }
        }
    }
    
    if (maxVotes > 0) {
        // Pikselden metreye dönüştür
        level.currentLevel = pixelsToMeters(frame.rows - bestY);
        
        // Değişim hızını hesapla
        if (!waterLevelHistory.empty()) {
            float timeDiff = std::chrono::duration<float>(
                level.lastUpdate - waterLevelHistory.back().lastUpdate).count();
            
            if (timeDiff > 0) {
                level.changeRate = (level.currentLevel - waterLevelHistory.back().currentLevel) 
                                 / timeDiff * 3600; // m/saat'e dönüştür
                level.isRising = level.changeRate > 0;
            }
        }
        
        level.referenceLevel = calibration.knownWaterLevel;
    }
    
    return level;
}

float SeaDetector::calculateTurbulence(const cv::Mat& frame, const cv::Mat& waterMask) {
    if (frame.empty() || waterMask.empty()) return 0.0f;

    cv::Mat gray, masked;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    gray.copyTo(masked, waterMask);

    // Laplacian ve gradyan analizleri
    cv::Mat laplacian;
    cv::Laplacian(masked, laplacian, CV_32F);

    cv::Mat gradX, gradY;
    cv::Sobel(masked, gradX, CV_32F, 1, 0);
    cv::Sobel(masked, gradY, CV_32F, 0, 1);

    cv::Mat magnitude;
    cv::magnitude(gradX, gradY, magnitude);

    // İstatistiksel analiz
    cv::Scalar laplacianScore, gradientScore;
    cv::meanStdDev(laplacian, cv::Scalar(), laplacianScore);
    cv::meanStdDev(magnitude, cv::Scalar(), gradientScore);

    float combinedScore = (laplacianScore[0] + gradientScore[0]) / 2.0f;
    return std::min(1.0f, combinedScore / 100.0f);
}

void SeaDetector::updateHistories(const WaveMetrics& wave, 
                                const TemperatureData& temp,
                                const WaterLevel& level) {
    // Dalga geçmişi
    waveHistory.push_back(wave);
    while (waveHistory.size() > MAX_HISTORY_SIZE) {
        waveHistory.pop_front();
    }
    
    // Sıcaklık geçmişi
    temperatureHistory.push_back(temp);
    while (temperatureHistory.size() > MAX_HISTORY_SIZE) {
        temperatureHistory.pop_front();
    }
    
    // Su seviyesi geçmişi
    waterLevelHistory.push_back(level);
    while (waterLevelHistory.size() > MAX_HISTORY_SIZE) {
        waterLevelHistory.pop_front();
    }
}

float SeaDetector::pixelsToMeters(float pixels) const {
    return pixels * calibration.pixelToMeterRatio;
}

float SeaDetector::metersToPixels(float meters) const { 
    return meters / calibration.pixelToMeterRatio; 
}

void SeaDetector::visualizeResults(cv::Mat& frame, const SeaInfo& info) const {
    if (!info.isDetected) return;

    try {
        // 1. Su konturunu çiz
        if (!info.contour.empty()) {
            cv::polylines(frame, std::vector<std::vector<cv::Point>>{info.contour},
                         true, cv::Scalar(0, 255, 0), 2);
            
            // Merkez noktası
            cv::circle(frame, info.centerPoint, 5, cv::Scalar(0, 0, 255), -1);
        }

        int lineHeight = 30;
        int yPos = 30;
        int xPos = 10;
        
        auto drawText = [&](const std::string& text, const cv::Scalar& color) {
            cv::putText(frame, text, cv::Point(xPos, yPos),
                       cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
            yPos += lineHeight;
        };

        // 2. Dalga bilgileri
        drawText(cv::format("Wave Height: %.2fm", info.waveMetrics.height),
                cv::Scalar(0, 255, 0));
        drawText(cv::format("Wave Period: %.2fs", info.waveMetrics.period),
                cv::Scalar(0, 255, 0));
        drawText(cv::format("Wave Direction: %.1f°", info.waveMetrics.direction),
                cv::Scalar(0, 255, 0));
        drawText(cv::format("Wave Energy: %.1f J/m²", info.waveMetrics.energy),
                cv::Scalar(0, 255, 0));
        drawText(cv::format("Wave Frequency: %.2f Hz", info.waveMetrics.frequency),
                cv::Scalar(0, 255, 0));
        drawText(cv::format("Wave Velocity: %.2f m/s", info.waveMetrics.velocity),
                cv::Scalar(0, 255, 0));
        
        // 3. Su seviyesi bilgileri
        drawText(cv::format("Water Level: %.2fm", info.waterLevel.currentLevel),
                cv::Scalar(255, 255, 0));
        if (info.waterLevel.isRising) {
            drawText(cv::format("Level Rising: %.2f m/h", info.waterLevel.changeRate),
                    cv::Scalar(255, 100, 0));
        } else {
            drawText(cv::format("Level Falling: %.2f m/h", std::abs(info.waterLevel.changeRate)),
                    cv::Scalar(100, 255, 0));
        }
        
        // 4. Sıcaklık ve görünürlük
        drawText(cv::format("Temperature: %.1f°C (%.0f%% conf)", 
                info.temperature.value, info.temperature.confidence * 100),
                cv::Scalar(255, 165, 0));
        drawText(cv::format("Visibility: %.1fm", info.visibility),
                cv::Scalar(255, 165, 0));
        
        // 5. Tehlike uyarıları
        if (info.waveMetrics.isDangerous) {
            cv::Scalar warningColor(0, 0, 255);
            if (info.waveMetrics.height > config.dangerousWaveHeight) {
                drawText("WARNING: Dangerous Wave Height!", warningColor);
            }
            if (info.waveMetrics.period < config.dangerousWavePeriod) {
                drawText("WARNING: Dangerous Wave Period!", warningColor);
            }
            if (info.waveMetrics.energy > 1000.0f) {
                drawText("WARNING: High Wave Energy!", warningColor);
            }
            if (info.waveMetrics.turbulence > 0.7f) {
                drawText("WARNING: High Turbulence!", warningColor);
            }
        }

        // 6. Türbülans göstergesi
        int barWidth = 200;
        int barHeight = 20;
        int margin = 20;
        
        // Dalga yoğunluğu göstergesi
        cv::Point barStart(frame.cols - barWidth - margin, margin);
        cv::Point barEnd(frame.cols - margin, margin + barHeight);
        
        // Bar çerçevesi
        cv::rectangle(frame, barStart, barEnd, cv::Scalar(0, 0, 0), 1);
        
        // Doluluk miktarı
        int filledWidth = static_cast<int>(info.waveMetrics.turbulence * barWidth);
        cv::rectangle(frame,
                     barStart,
                     cv::Point(barStart.x + filledWidth, barStart.y + barHeight),
                     cv::Scalar(0, 0, 255), -1);
                     
        // Bar etiketi
        cv::putText(frame, "Turbulence Level",
                   cv::Point(barStart.x, barStart.y - 5),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
                   
        // Zaman damgası
        auto now = std::chrono::system_clock::now();
        auto nowTime = std::chrono::system_clock::to_time_t(now);
        std::string timeStr = std::ctime(&nowTime);
        timeStr = timeStr.substr(0, timeStr.length()-1);  // '\n' karakterini kaldır
        
        cv::putText(frame, timeStr,
                   cv::Point(10, frame.rows - 10),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
                   
    } catch (const cv::Exception& e) {
        throw SeaDetectionError("Visualization error: " + std::string(e.what()));
    }
}

float SeaDetector::estimateVisibility(const cv::Mat& frame, const cv::Mat& waterMask) {
    if (frame.empty() || waterMask.empty()) return 0.0f;

    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    
    // Su bölgesindeki kontrast analizi
    cv::Mat masked;
    gray.copyTo(masked, waterMask);
    
    cv::Scalar mean, stddev;
    cv::meanStdDev(masked, mean, stddev);
    
    // Kontrast bazlı görünürlük tahmini
    float contrast = stddev[0] / mean[0];
    float visibility = 100.0f * std::exp(-2.0f * contrast);
    
    return std::min(100.0f, std::max(0.0f, visibility));
}

std::vector<cv::Point> SeaDetector::findLargestContour(const cv::Mat& mask) {
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    if (contours.empty()) return std::vector<cv::Point>();
    
    // En büyük konturu bul
    auto maxContour = std::max_element(contours.begin(), contours.end(),
        [](const std::vector<cv::Point>& c1, const std::vector<cv::Point>& c2) {
            return cv::contourArea(c1) < cv::contourArea(c2);
        });
        
    return *maxContour;
}

cv::Point2f SeaDetector::calculateCenterPoint(const std::vector<cv::Point>& contour) const {
    cv::Moments m = cv::moments(contour);
    if (m.m00 == 0) return cv::Point2f(0, 0);
    return cv::Point2f(m.m10/m.m00, m.m01/m.m00);
}

bool SeaDetector::evaluateWaveConditions(const WaveMetrics& metrics) const {
    return metrics.height > config.dangerousWaveHeight ||
           metrics.period < config.dangerousWavePeriod ||
           metrics.turbulence > 0.7f ||
           metrics.energy > 1000.0f;
}

std::vector<SeaDetector::WaveMetrics> SeaDetector::getWaveHistory() const {
    std::lock_guard<std::mutex> lock(mtx);
    return std::vector<WaveMetrics>(waveHistory.begin(), waveHistory.end());
}

std::vector<SeaDetector::TemperatureData> SeaDetector::getTemperatureHistory() const {
    std::lock_guard<std::mutex> lock(mtx);
    return std::vector<TemperatureData>(temperatureHistory.begin(), temperatureHistory.end());
}