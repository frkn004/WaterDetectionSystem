#include "ObjectDetector.hpp"
#include <iostream>

ObjectDetector::ObjectDetector(const std::string& cascadePath) :
    showHelp(true),
    frameCount(0),
    fps(0.0f) {
    
    try {
        if (!faceCascade.load(cascadePath)) {
            throw std::runtime_error("Error loading face cascade classifier");
        }
        
        lastTime = std::chrono::steady_clock::now();
        
        // Kamera matrisi ayarla
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
        
        if (features.showFPS) {
            std::string fpsText = "FPS: " + std::to_string(static_cast<int>(fps));
            cv::putText(frame, fpsText, cv::Point(10, 30),
                       cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
        }

        if (features.showSeaDetection) {
            detectSea(frame);
        }

        if (features.showHumanDetection) {
            detectAndTrackHumans(frame);
        }

        if (features.showSkinDetection) {
            detectSkin(frame);
        }

        detectAndTrackBoats(frame);

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
    auto seaInfo = seaDetector.detectSea(frame);
    if (seaInfo.isDetected) {
        seaDetector.visualizeResults(frame, seaInfo);
        
        if (seaInfo.waveIntensity > seaDetector.getConfig().waveThreshold) {
            notifications.notify("Yüksek dalga aktivitesi tespit edildi!", true);
        }
    }
}

void ObjectDetector::detectAndTrackHumans(cv::Mat& frame) {
    auto humans = humanDetector.detectHumans(frame);
    for (const auto& human : humans) {
        cv::rectangle(frame, human.box, cv::Scalar(0, 255, 0), 2);
        
        if (features.showDistance) {
            std::string distText = "Mesafe: " +
                std::to_string(static_cast<int>(human.distance)) + "m";
            cv::putText(frame, distText,
                        cv::Point(human.box.x, human.box.y - 10),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5,
                        cv::Scalar(0, 255, 0), 2);
        }
        
        if (features.enableNotifications) {
            notifications.notify("İnsan tespit edildi", false);
        }
        
        if (features.showTrajectory && human.isMoving) {
            TrackedObject tracker;
            tracker.update(cv::Point2f(human.box.x + human.box.width/2,
                                     human.box.y + human.box.height/2));
            tracker.drawTrajectory(frame);
        }
    }
}

void ObjectDetector::detectAndTrackBoats(cv::Mat& frame) {
    auto boats = boatDetector.detectBoats(frame);
    for (const auto& boat : boats) {
        if (boat.confidence > 0.5f) {  // Minimum güven eşiği
            cv::rectangle(frame, boat.boundingBox, cv::Scalar(255, 0, 0), 2);
            
            std::string infoText = "Tekne (" +
                std::to_string(static_cast<int>(boat.confidence * 100)) + "%)";
            cv::putText(frame, infoText,
                       cv::Point(boat.boundingBox.x, boat.boundingBox.y - 5),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5,
                       cv::Scalar(255, 0, 0), 2);
            
            if (features.enableNotifications) {
                notifications.notify("Tekne tespit edildi", false);
            }
            
            if (features.showTrajectory && boat.isMoving) {
                TrackedObject tracker;
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
                                                 features.showDistance,
                                                 features.showDirection);
        
        if (features.enableNotifications) {
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
    
    // Mesafe hesaplama
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
    
    // En uzun yatay çizgiyi bul
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
    
    // Su çizgisini çiz
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
