#include "TrackedObject.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>

TrackedObject::TrackedObject(int id, const KalmanConfig& config) : 
    kf(4, 2, 0),  // State: [x, y, vx, vy], Measure: [x, y]
    lastPosition(0, 0),
    velocity(0, 0),
    initialized(false),
    objectId(id),
    age(0),
    missedFrames(0),
    confidence(1.0f),
    kalmanConfig(config) {
    
    try {
        initializeKalmanFilter();
    } catch (const std::exception& e) {
        throw TrackingError("Failed to initialize Kalman filter: " + std::string(e.what()));
    }
}

void TrackedObject::initializeKalmanFilter() {
    kalmanConfig.validate();
    
    // Geçiş matrisi (Transition Matrix) ayarlanması
    kf.transitionMatrix = (cv::Mat_<float>(4, 4) << 
        1,0,1,0,
        0,1,0,1,
        0,0,1,0,
        0,0,0,1);
    
    // Ölçüm matrisi (Measurement Matrix)
    cv::setIdentity(kf.measurementMatrix);
    
    // Process Noise Covariance Matrix (Q)
    cv::setIdentity(kf.processNoiseCov, 
                   cv::Scalar::all(kalmanConfig.processNoise));
    
    // Measurement Noise Covariance Matrix (R)
    cv::setIdentity(kf.measurementNoiseCov, 
                   cv::Scalar::all(kalmanConfig.measurementNoise));
    
    // Error Covariance Matrix (P)
    cv::setIdentity(kf.errorCovPost, 
                   cv::Scalar::all(kalmanConfig.errorCovariance));
}

void TrackedObject::update(const cv::Point2f& pos) {
    try {
        if (!initialized) {
            // İlk çerçeve için Kalman filtresi başlatma
            kf.statePost.at<float>(0) = pos.x;
            kf.statePost.at<float>(1) = pos.y;
            kf.statePost.at<float>(2) = 0;  // initial vx
            kf.statePost.at<float>(3) = 0;  // initial vy
            lastPosition = pos;
            initialized = true;
            updateHistory(pos);
            return;
        }

        // Prediction step
        cv::Mat prediction = kf.predict();
        
        // Measurement update
        cv::Mat measurement = (cv::Mat_<float>(2, 1) << pos.x, pos.y);
        cv::Mat estimated = kf.correct(measurement);

        // Update state
        lastPosition = cv::Point2f(estimated.at<float>(0), estimated.at<float>(1));
        velocity = cv::Point2f(estimated.at<float>(2), estimated.at<float>(3));
        
        // Update history and state
        updateHistory(lastPosition);
        age++;
        missedFrames = 0;
        confidence = 1.0f;

    } catch (const cv::Exception& e) {
        throw TrackingError("OpenCV error in update: " + std::string(e.what()));
    } catch (const std::exception& e) {
        throw TrackingError("Error in update: " + std::string(e.what()));
    }
}

void TrackedObject::updateHistory(const cv::Point2f& pos) {
    history.push_back(pos);
    while (history.size() > MAX_HISTORY_SIZE) {
        history.pop_front();
    }
}

cv::Point2f TrackedObject::calculateAverageVelocity() const {
    if (history.size() < 2) return cv::Point2f(0, 0);
    
    cv::Point2f totalVelocity(0, 0);
    size_t count = 0;
    
    // Son N çerçevedeki hızları hesapla
    auto it = history.begin();
    auto prevIt = it++;
    
    for (; it != history.end(); ++it, ++prevIt) {
        totalVelocity += *it - *prevIt;
        count++;
    }
    
    if (count == 0) return cv::Point2f(0, 0);
    return totalVelocity * (1.0f / count);
}

std::string TrackedObject::getDirectionFromAngle(float angle) const {
    // Açıyı normalize et (0-360)
    angle = fmod(angle + 360.0f, 360.0f);
    
    struct DirectionRange {
        float minAngle;
        float maxAngle;
        std::string direction;
    };
    
    static const DirectionRange directions[] = {
        {337.5f, 22.5f,  "Right"},
        {22.5f,  67.5f,  "Down-Right"},
        {67.5f,  112.5f, "Down"},
        {112.5f, 157.5f, "Down-Left"},
        {157.5f, 202.5f, "Left"},
        {202.5f, 247.5f, "Up-Left"},
        {247.5f, 292.5f, "Up"},
        {292.5f, 337.5f, "Up-Right"}
    };
    
    for (const auto& dir : directions) {
        if (dir.minAngle > dir.maxAngle) {
            if (angle >= dir.minAngle || angle < dir.maxAngle) {
                return dir.direction;
            }
        } else {
            if (angle >= dir.minAngle && angle < dir.maxAngle) {
                return dir.direction;
            }
        }
    }
    
    return "Unknown";
}

std::string TrackedObject::getPredictedDirection() const {
    if (!initialized) return "Unknown";
    
    cv::Point2f avgVelocity = calculateAverageVelocity();
    if (cv::norm(avgVelocity) < 1e-5) return "Stationary";
    
    float angle = atan2(avgVelocity.y, avgVelocity.x) * 180 / CV_PI;
    return getDirectionFromAngle(angle);
}

void TrackedObject::drawTrajectory(cv::Mat& frame) const {
    if (history.size() < 2) return;
    
    try {
        static const cv::Scalar SLOW_COLOR(0, 255, 0);    // Green
        static const cv::Scalar MEDIUM_COLOR(0, 255, 255); // Yellow
        static const cv::Scalar FAST_COLOR(0, 0, 255);    // Red
        
        auto it = history.begin();
        auto prevIt = it++;
        
        for (; it != history.end(); ++it, ++prevIt) {
            float dx = it->x - prevIt->x;
            float dy = it->y - prevIt->y;
            float speed = std::sqrt(dx*dx + dy*dy);
            
            // Hıza göre renk seçimi
            cv::Scalar color;
            if (speed < 5.0f) {
                color = SLOW_COLOR;
            } else if (speed < 10.0f) {
                color = MEDIUM_COLOR;
            } else {
                color = FAST_COLOR;
            }
            
            // Yörünge çizgisi
            cv::line(frame, *prevIt, *it, color, 2);
        }
        
        // Tahmin edilen hareket yönü oku
        if (history.size() >= 2) {
            cv::Point2f avgVelocity = calculateAverageVelocity();
            if (cv::norm(avgVelocity) > 1e-5) {
                cv::Point2f direction = lastPosition + avgVelocity * 20.0f;
                cv::arrowedLine(frame, lastPosition, direction,
                              cv::Scalar(255, 0, 0), 2);
            }
        }
        
    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV Error in drawTrajectory: " << e.what() << std::endl;
    }
}

void TrackedObject::setKalmanConfig(const KalmanConfig& config) {
    config.validate();
    kalmanConfig = config;
    initializeKalmanFilter();
}