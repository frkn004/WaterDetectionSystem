#include "TrackedObject.hpp"

TrackedObject::TrackedObject() : kf(4, 2, 0) {
    initialized = false;
    
    // Kalman filtresi matrislerini ayarla
    kf.transitionMatrix = (cv::Mat_<float>(4, 4) <<
        1,0,1,0,
        0,1,0,1,
        0,0,1,0,
        0,0,0,1);
    
    kf.measurementMatrix = (cv::Mat_<float>(2, 4) <<
        1,0,0,0,
        0,1,0,0);
    
    kf.processNoiseCov = cv::Mat::eye(4, 4, CV_32F) * 0.05;
    kf.measurementNoiseCov = cv::Mat::eye(2, 2, CV_32F) * 0.1;
    kf.errorCovPost = cv::Mat::eye(4, 4, CV_32F);
}

void TrackedObject::update(const cv::Point2f& pos) {
    if (!initialized) {
        // İlk frame ise Kalman filtresini başlat
        kf.statePost.at<float>(0) = pos.x;
        kf.statePost.at<float>(1) = pos.y;
        kf.statePost.at<float>(2) = 0;
        kf.statePost.at<float>(3) = 0;
        lastPosition = pos;
        initialized = true;
        return;
    }

    try {
        // Ölçüm vektörünü oluştur
        cv::Mat measurement = (cv::Mat_<float>(2, 1) << pos.x, pos.y);
        
        // Kalman tahminini yap
        cv::Mat prediction = kf.predict();
        cv::Mat estimated = kf.correct(measurement);

        // Pozisyon ve hızı güncelle
        cv::Point2f currentPos(estimated.at<float>(0), estimated.at<float>(1));
        velocity.x = estimated.at<float>(2);
        velocity.y = estimated.at<float>(3);
        
        // Geçmiş pozisyonları sakla
        history.push_back(currentPos);
        if (history.size() > HISTORY_LENGTH) {
            history.erase(history.begin());
        }

        lastPosition = currentPos;
        
    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV Error in TrackedObject::update: " << e.what() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error in TrackedObject::update: " << e.what() << std::endl;
    }
}

cv::Point2f TrackedObject::getPosition() const {
    return lastPosition;
}

cv::Point2f TrackedObject::getVelocity() const {
    return velocity;
}

std::string TrackedObject::getPredictedDirection() const {
    if (!initialized) return "Unknown";
    
    // Hız vektörünün açısını hesapla
    float angle = atan2(velocity.y, velocity.x) * 180 / CV_PI;
    if (angle < 0) angle += 360;
    
    // Açıyı yöne çevir
    if (angle >= 337.5 || angle < 22.5) return "Right";
    else if (angle >= 22.5 && angle < 67.5) return "Down-Right";
    else if (angle >= 67.5 && angle < 112.5) return "Down";
    else if (angle >= 112.5 && angle < 157.5) return "Down-Left";
    else if (angle >= 157.5 && angle < 202.5) return "Left";
    else if (angle >= 202.5 && angle < 247.5) return "Up-Left";
    else if (angle >= 247.5 && angle < 292.5) return "Up";
    else return "Up-Right";
}

void TrackedObject::drawTrajectory(cv::Mat& frame) {
    if (history.size() < 2) return;
    
    try {
        // Yörüngeyi çiz
        for (size_t i = 1; i < history.size(); i++) {
            // Hareket hızına göre renk değiştir
            float dx = history[i].x - history[i-1].x;
            float dy = history[i].y - history[i-1].y;
            float speed = std::sqrt(dx*dx + dy*dy);
            
            // Hıza göre renk belirle (yavaş=yeşil, hızlı=kırmızı)
            cv::Scalar color;
            if (speed < 5.0f) {
                color = cv::Scalar(0, 255, 0); // Yeşil
            } else if (speed < 10.0f) {
                color = cv::Scalar(0, 255, 255); // Sarı
            } else {
                color = cv::Scalar(0, 0, 255); // Kırmızı
            }
            
            cv::line(frame, history[i-1], history[i], color, 2);
        }
        
        // Son pozisyonda yön oku çiz
        if (history.size() >= 2) {
            cv::Point2f lastPos = history.back();
            cv::Point2f direction = lastPos + velocity * 20;
            cv::arrowedLine(frame, lastPos, direction, cv::Scalar(255, 0, 0), 2);
        }
        
    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV Error in drawTrajectory: " << e.what() << std::endl;
    }
}
