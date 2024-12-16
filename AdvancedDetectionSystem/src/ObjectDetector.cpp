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

        // Kamera kalibrasyonu için gerekli matris ayarları
        params.cameraMatrix = (cv::Mat_<double>(3,3) << 
            params.focalLength, 0, 320.0,
            0, params.focalLength, 240.0,
            0, 0, 1);
        params.distCoeffs = cv::Mat::zeros(5, 1, CV_64F);
        
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

// Yardımcı fonksiyonlar
void ObjectDetector::drawGrid(cv::Mat& frame) {
    int cellSize = 50;
    for(int x = 0; x < frame.cols; x += cellSize) {
        cv::line(frame, cv::Point(x, 0), cv::Point(x, frame.rows),
                 cv::Scalar(128, 128, 128), 1);
    }
    for(int y = 0; y < frame.rows; y += cellSize) {
        cv::line(frame, cv::Point(0, y), cv::Point(frame.cols, y),
                 cv::Scalar(128, 128, 128), 1);
    }
}

void ObjectDetector::drawCoordinates(cv::Mat& frame) {
    std::string coords = "(" + std::to_string(frame.cols) + "x" +
                        std::to_string(frame.rows) + ")";
    cv::putText(frame, coords, cv::Point(10, frame.rows - 10),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
}

void ObjectDetector::drawTimestamp(cv::Mat& frame) {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::string timestamp = std::ctime(&time);
    timestamp = timestamp.substr(0, timestamp.length()-1);
    
    cv::putText(frame, timestamp, cv::Point(10, frame.rows - 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
}


void ObjectDetector::detectSea(cv::Mat& frame) {
    auto seaInfo = seaDetector.detectSea(frame);
    if (seaInfo.isDetected) {
        seaDetector.visualizeResults(frame, seaInfo);
        
        // Tehlike durumu kontrolü ve bildirim
        if (seaInfo.waveIntensity > seaDetector.getConfig().waveThreshold && 
            features.isEnabled(FeatureControl::Feature::ENABLE_NOTIFICATIONS)) {
            notifications.notify("Yüksek dalga aktivitesi tespit edildi!", true);
        }
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
        
        // Bildirimler
        if (features.isEnabled(FeatureControl::Feature::ENABLE_NOTIFICATIONS)) {
            notifications.notify("İnsan tespit edildi", false);
        }
        
        // Yörünge takibi ve çizimi
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
            // Tekne çerçevesi
            cv::rectangle(frame, boat.boundingBox, cv::Scalar(255, 0, 0), 2);
            
            // ID gösterimi
            if (features.isEnabled(FeatureControl::Feature::SHOW_OBJECT_ID)) {
                std::string idText = "ID: " + std::to_string(nextTrackerId);
                cv::putText(frame, idText,
                           cv::Point(boat.boundingBox.x, boat.boundingBox.y - 45),
                           cv::FONT_HERSHEY_SIMPLEX, 0.5,
                           cv::Scalar(255, 0, 0), 2);
            }
            
            // Güven skoru
            if (features.isEnabled(FeatureControl::Feature::SHOW_CONFIDENCE)) {
                std::string confText = "Güven: " +
                    std::to_string(static_cast<int>(boat.confidence * 100)) + "%";
                cv::putText(frame, confText,
                           cv::Point(boat.boundingBox.x, boat.boundingBox.y - 25),
                           cv::FONT_HERSHEY_SIMPLEX, 0.5,
                           cv::Scalar(255, 0, 0), 2);
            }
            
            // Bildirimler
            if (features.isEnabled(FeatureControl::Feature::ENABLE_NOTIFICATIONS)) {
                notifications.notify("Tekne tespit edildi", false);
            }
            
            // Yörünge takibi
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
        frame = SkinDetector::visualizeSkinRegions(
            frame, 
            skinRegions,
            features.isEnabled(FeatureControl::Feature::SHOW_DISTANCE),
            features.isEnabled(FeatureControl::Feature::SHOW_DIRECTION)
        );
        
        if (features.isEnabled(FeatureControl::Feature::ENABLE_NOTIFICATIONS)) {
            notifications.notify("Cilt bölgesi tespit edildi", false);
        }
    }
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

float ObjectDetector::calculateDistance(const cv::Rect& box) {
    if (box.height <= 0) return -1.0f;
    
    // Mesafe hesaplama (focal length formülü)
    float distance = (params.knownWidth * params.focalLength) / box.width;
    
    // Perspektif düzeltmesi
    float verticalPosition = box.y + box.height/2.0f;
    float imageHeight = params.cameraMatrix.at<double>(1,2) * 2;
    float angleCorrection = 1.0f + (verticalPosition - imageHeight/2) * 0.001f;
    
    return distance * angleCorrection;
}

void ObjectDetector::detectWaterLine(cv::Mat& frame) {
    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    
    // Kenar tespiti
    cv::Mat edges;
    cv::Canny(gray, edges, 50, 150);
    
    // Hough dönüşümü ile yatay çizgileri bul
    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(edges, lines, 1, CV_PI/180, 50, 50, 10);
    
    // En uzun yatay çizgiyi bul ve çiz
    cv::Vec4i longestLine(0,0,0,0);
    int maxLength = 0;
    
    for(const auto& line : lines) {
        int dx = line[2] - line[0];
        int dy = line[3] - line[1];
        int length = std::sqrt(dx*dx + dy*dy);
        
        // Yatay çizgi kontrolü (eğim < 15 derece)
        if(std::abs(dy) < std::abs(dx) * 0.27 && length > maxLength) {
            maxLength = length;
            longestLine = line;
        }
    }
    
    if(maxLength > 0) {
        cv::line(frame, cv::Point(longestLine[0], longestLine[1]),
                cv::Point(longestLine[2], longestLine[3]),
                cv::Scalar(0,255,255), 2);
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
}