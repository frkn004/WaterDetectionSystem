#include "ObjectDetector.hpp"
#include <iostream>

ObjectDetector::ObjectDetector(const std::string& cascadePath) :
    showHelp(true),
    frameCount(0),
    fps(0.0f),
    lastTime(std::chrono::steady_clock::now()),
    nextTrackerId(0) {
    
    try {
        std::vector<std::string> possiblePaths = {
            cascadePath,
            "./resources/" + cascadePath,
            "../resources/" + cascadePath
        };

        bool loaded = false;
        for (const auto& path : possiblePaths) {
            if (faceCascade.load(path)) {
                loaded = true;
                break;
            }
        }

        if (!loaded) {
            throw std::runtime_error("Cascade file could not be loaded from any location");
        }

        // Kamera kalibrasyonu
        params.cameraMatrix = (cv::Mat_<double>(3,3) << 
            params.focalLength, 0, 320.0,
            0, params.focalLength, 240.0,
            0, 0, 1);
        params.distCoeffs = cv::Mat::zeros(5, 1, CV_64F);

        // SeaDetector konfigürasyonu
        seaConfig.waterLowerBound = cv::Scalar(90, 40, 40);
        seaConfig.waterUpperBound = cv::Scalar(130, 255, 255);
        seaConfig.maxWaveHeight = 2.0f;
        seaConfig.dangerousWaveHeight = 1.5f;
        seaConfig.dangerousWavePeriod = 5.0f;
        seaConfig.waterLevelThreshold = 0.5f;
        seaConfig.temperatureThreshold = 1.0f;
        seaConfig.morphSize = 3;
        seaConfig.blurSize = 5;
        seaConfig.contrastAlpha = 1.2f;
        seaConfig.samplingRate = 30.0f;
        seaConfig.fftSize = 512;
        seaConfig.minWaveFreq = 0.1f;
        seaConfig.maxWaveFreq = 2.0f;
        seaDetector.setConfig(seaConfig);

        // SeaDetector kalibrasyonu
        SeaDetector::CalibrationData calibData;
        calibData.knownTemperature = 20.0f;
        calibData.knownWaterLevel = 0.0f;
        calibData.pixelToMeterRatio = 0.1f;
        calibData.timestamp = std::chrono::system_clock::now();
        seaDetector.calibrate(calibData);
        
    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV error: " << e.what() << std::endl;
        throw;
    }
}

void ObjectDetector::detectAndTrack(cv::Mat& frame) {
    try {
        updateFPS();
        
        // FPS gösterimi
        if (features.isEnabled(FeatureControl::Feature::SHOW_FPS)) {
            std::string fpsText = "FPS: " + std::to_string(static_cast<int>(fps));
            cv::putText(frame, fpsText, cv::Point(10, 30),
                       cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
        }

        // Temel tespit işlevleri
        if (features.isEnabled(FeatureControl::Feature::SHOW_SEA_DETECTION)) {
            detectSea(frame);
        }

        if (features.isEnabled(FeatureControl::Feature::SHOW_HUMAN_DETECTION)) {
            detectAndTrackHumans(frame);
        }

        if (features.isEnabled(FeatureControl::Feature::SHOW_SKIN_DETECTION)) {
            detectSkin(frame);
        }

        if (features.isEnabled(FeatureControl::Feature::SHOW_BOAT_DETECTION)) {
            detectAndTrackBoats(frame);
        }

        // Ek görsel özellikler
        if (features.isEnabled(FeatureControl::Feature::SHOW_WATER_LEVEL)) {
            detectWaterLine(frame);
        }

        if (features.isEnabled(FeatureControl::Feature::SHOW_GRID)) {
            drawGrid(frame);
        }

        if (features.isEnabled(FeatureControl::Feature::SHOW_COORDINATES)) {
            drawCoordinates(frame);
        }

        if (features.isEnabled(FeatureControl::Feature::SHOW_TIMESTAMP)) {
            drawTimestamp(frame);
        }

        // Yardım veya durum gösterimi
        if (showHelp) {
            features.displayHelp(frame);
        } else {
            features.displayFeatureStatus(frame);
        }

    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV error in detectAndTrack: " << e.what() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error in detectAndTrack: " << e.what() << std::endl;
    }
}

void ObjectDetector::detectSea(cv::Mat& frame) {
    try {
        auto seaInfo = seaDetector.detectSea(frame);
        if (seaInfo.isDetected) {
            seaDetector.visualizeResults(frame, seaInfo);

            if (features.isEnabled(FeatureControl::Feature::ENABLE_NOTIFICATIONS)) {
                // Dalga tehlikesi kontrolü
                if (seaInfo.waveMetrics.isDangerous) {
                    if (seaInfo.waveMetrics.height > seaConfig.dangerousWaveHeight) {
                        notifications.notify("Tehlikeli dalga yüksekliği: " + 
                            std::to_string(seaInfo.waveMetrics.height) + "m", true);
                    }
                    if (seaInfo.waveMetrics.turbulence > 0.7f) {
                        notifications.notify("Yüksek türbülans tespit edildi!", true);
                    }
                }

                // Su seviyesi değişimi kontrolü
                if (seaInfo.waterLevel.isRising && 
                    std::abs(seaInfo.waterLevel.changeRate) > seaConfig.waterLevelThreshold) {
                    notifications.notify("Hızlı su seviyesi değişimi: " + 
                        std::to_string(seaInfo.waterLevel.changeRate) + "m/h", true);
                }

                // Sıcaklık değişimi kontrolü
                if (seaInfo.temperature.confidence > 0.7f && 
                    std::abs(seaInfo.temperature.value - 20.0f) > seaConfig.temperatureThreshold) {
                    notifications.notify("Anormal su sıcaklığı: " + 
                        std::to_string(seaInfo.temperature.value) + "°C", true);
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error in sea detection: " << e.what() << std::endl;
    }
}

void ObjectDetector::detectAndTrackHumans(cv::Mat& frame) {
    auto humans = humanDetector.detectHumans(frame);
    for (const auto& human : humans) {
        // Çerçeve çizimi
        cv::rectangle(frame, human.box, cv::Scalar(0, 255, 0), 2);
        
        // Mesafe gösterimi
        if (features.isEnabled(FeatureControl::Feature::SHOW_DISTANCE)) {
            std::string distText = "Mesafe: " +
                std::to_string(static_cast<int>(human.distance)) + "m";
            cv::putText(frame, distText,
                       cv::Point(human.box.x, human.box.y - 10),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5,
                       cv::Scalar(0, 255, 0), 2);
        }
        
        // Hız gösterimi
        if (features.isEnabled(FeatureControl::Feature::SHOW_SPEED)) {
            std::string speedText = "Hız: " +
                std::to_string(static_cast<int>(cv::norm(human.direction) * 100)) + " px/s";
            cv::putText(frame, speedText,
                       cv::Point(human.box.x, human.box.y - 30),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5,
                       cv::Scalar(0, 255, 0), 2);
        }
        
        if (features.isEnabled(FeatureControl::Feature::ENABLE_NOTIFICATIONS)) {
            notifications.notify("İnsan tespit edildi", false);
        }

        if (features.isEnabled(FeatureControl::Feature::SHOW_TRAJECTORY) && 
            human.isMoving) {
            TrackedObject tracker(nextTrackerId++);
            tracker.update(cv::Point2f(human.box.x + human.box.width/2,
                                     human.box.y + human.box.height/2));
            tracker.drawTrajectory(frame);
        }
    }
}

void ObjectDetector::detectAndTrackBoats(cv::Mat& frame) {
    auto boats = boatDetector.detectBoats(frame);
    for (const auto& boat : boats) {
        if (boat.confidence > 0.5f) {
            cv::rectangle(frame, boat.boundingBox, cv::Scalar(255, 0, 0), 2);
            
            if (features.isEnabled(FeatureControl::Feature::SHOW_OBJECT_ID)) {
                std::string idText = "ID: " + std::to_string(nextTrackerId);
                cv::putText(frame, idText,
                           cv::Point(boat.boundingBox.x, boat.boundingBox.y - 45),
                           cv::FONT_HERSHEY_SIMPLEX, 0.5,
                           cv::Scalar(255, 0, 0), 2);
            }
            
            if (features.isEnabled(FeatureControl::Feature::SHOW_CONFIDENCE)) {
                std::string confText = "Güven: " +
                    std::to_string(static_cast<int>(boat.confidence * 100)) + "%";
                cv::putText(frame, confText,
                           cv::Point(boat.boundingBox.x, boat.boundingBox.y - 25),
                           cv::FONT_HERSHEY_SIMPLEX, 0.5,
                           cv::Scalar(255, 0, 0), 2);
            }
            
            if (features.isEnabled(FeatureControl::Feature::ENABLE_NOTIFICATIONS)) {
                notifications.notify("Tekne tespit edildi", false);
            }
            
            if (features.isEnabled(FeatureControl::Feature::SHOW_TRAJECTORY) && 
                boat.isMoving) {
                TrackedObject tracker(nextTrackerId++);
                tracker.update(boat.position);
                tracker.drawTrajectory(frame);
            }
        }
    }
}

void ObjectDetector::detectSkin(cv::Mat& frame) {
    std::vector<SkinDetector::SkinRegion> skinRegions;
    cv::Mat skinMask;
    skinDetector.detectSkin(frame, skinRegions, skinMask);
    
    if (!skinRegions.empty()) {
        frame = SkinDetector::visualizeSkinRegions(frame, skinRegions,
            features.isEnabled(FeatureControl::Feature::SHOW_DISTANCE),
            features.isEnabled(FeatureControl::Feature::SHOW_DIRECTION));
            
        if (features.isEnabled(FeatureControl::Feature::ENABLE_NOTIFICATIONS)) {
            notifications.notify("Cilt bölgesi tespit edildi", false);
        }
    }
}

void ObjectDetector::detectWaterLine(cv::Mat& frame) {
    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    
    // Kenar tespiti
    cv::Mat edges;
    cv::Canny(gray, edges, 50, 150);
    
    // Gaussian blur uygula
    cv::GaussianBlur(edges, edges, cv::Size(5, 5), 0);
    
    // Hough dönüşümü ile yatay çizgileri bul
    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(edges, lines, 1, CV_PI/180, 50, 100, 20);
    
    // En iyi su çizgisini bul
    float bestY = 0;
    float bestConf = 0;
    
    for(const auto& line : lines) {
        float dx = line[2] - line[0];
        float dy = line[3] - line[1];
        float angle = std::abs(std::atan2(dy, dx) * 180 / CV_PI);
        
        // Yatay çizgi kontrolü (eğim < 10 derece)
        if(angle < 10 || angle > 170) {
            float y = (line[1] + line[3]) / 2.0f;
            float length = std::sqrt(dx*dx + dy*dy);
            float conf = length * (1.0f - angle/180.0f);
            
            if(conf > bestConf) {
                bestConf = conf;
                bestY = y;
            }
        }
    }
    
    if(bestConf > 0) {
        // Su çizgisini çiz
        cv::line(frame, 
                cv::Point(0, static_cast<int>(bestY)),
                cv::Point(frame.cols, static_cast<int>(bestY)),
                cv::Scalar(0, 255, 255), 2);
        
        // Su seviyesi bilgisini yaz
        std::string levelText = "Su Seviyesi: " + 
                              std::to_string(frame.rows - static_cast<int>(bestY)) + "px";
        cv::putText(frame, levelText,
                   cv::Point(10, static_cast<int>(bestY) - 10),
                   cv::FONT_HERSHEY_SIMPLEX, 0.6,
                   cv::Scalar(0, 255, 255), 2);
    }
}

void ObjectDetector::drawGrid(cv::Mat& frame) {
    const int gridSize = 50; // Grid hücre boyutu
    const cv::Scalar gridColor(128, 128, 128); // Gri renk
    
    // Dikey çizgiler
    for (int x = gridSize; x < frame.cols; x += gridSize) {
        cv::line(frame, cv::Point(x, 0), cv::Point(x, frame.rows), gridColor, 1);
    }
    
    // Yatay çizgiler
    for (int y = gridSize; y < frame.rows; y += gridSize) {
        cv::line(frame, cv::Point(0, y), cv::Point(frame.cols, y), gridColor, 1);
    }
}

void ObjectDetector::drawCoordinates(cv::Mat& frame) {
    const cv::Scalar textColor(255, 255, 255); // Beyaz renk
    const int fontSize = 1;
    const int thickness = 2;
    
    // Fare pozisyonunu al
    cv::Point mousePos;
    mousePos.x = frame.cols / 2;
    mousePos.y = frame.rows / 2;
    
    // Koordinatları yaz
    std::string coordText = "X: " + std::to_string(mousePos.x) + 
                           " Y: " + std::to_string(mousePos.y);
    cv::putText(frame, coordText,
               cv::Point(10, frame.rows - 10),
               cv::FONT_HERSHEY_SIMPLEX, fontSize,
               textColor, thickness);
}

void ObjectDetector::drawTimestamp(cv::Mat& frame) {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    struct tm timeinfo;
#ifdef _WIN32
    localtime_s(&timeinfo, &time);
#else
    localtime_r(&time, &timeinfo);
#endif
    char buffer[80];
    strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", &timeinfo);
    
    cv::putText(frame, buffer,
               cv::Point(frame.cols - 200, frame.rows - 10),
               cv::FONT_HERSHEY_SIMPLEX, 0.5,
               cv::Scalar(255, 255, 255), 1);
}

float ObjectDetector::calculateDistance(const cv::Rect& box) {
    if (box.height <= 0) return -1.0f;
    return (params.knownWidth * params.focalLength) / box.width;
}

void ObjectDetector::updateFPS() {
    frameCount++;
    auto currentTime = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>
                   (currentTime - lastTime).count();
                   
    if (duration >= 1000) {
        fps = frameCount * (1000.0f / duration);
        frameCount = 0;
        lastTime = currentTime;
    }
}

void ObjectDetector::handleKeyPress(char key) {
    switch(key) {
        case 'h':
        case 'H':
            showHelp = !showHelp;
            break;
        default:
            features.toggleFeature(key);
            break;
    }
} // End of handleKeyPress