#include "TrackedObject.hpp"
#include <iostream>

TrackedObject::TrackedObject(int id) : 
    kf(4, 2, 0),
    lastPosition(0, 0),
    velocity(0, 0),
    initialized(false),
    history(),
    objectId(id),
    age(0),
    missedFrames(0),
    confidence(1.0f),
    id(id)
{
    // Kalman filtresi başlatma işlemleri
    kf.transitionMatrix = (cv::Mat_<float>(4, 4) << 
        1,0,1,0,
        0,1,0,1,
        0,0,1,0,
        0,0,0,1);
    
    cv::setIdentity(kf.measurementMatrix);
    cv::setIdentity(kf.processNoiseCov, cv::Scalar::all(1e-4));
    cv::setIdentity(kf.measurementNoiseCov, cv::Scalar::all(1e-1));
    cv::setIdentity(kf.errorCovPost, cv::Scalar::all(1));
}

void TrackedObject::update(const cv::Point2f& pos) {
    try {
        if (!initialized) {
            kf.statePost.at<float>(0) = pos.x;
            kf.statePost.at<float>(1) = pos.y;
            kf.statePost.at<float>(2) = 0;
            kf.statePost.at<float>(3) = 0;
            lastPosition = pos;
            initialized = true;
            return;
        }

        cv::Mat measurement = (cv::Mat_<float>(2, 1) << pos.x, pos.y);
        cv::Mat prediction = kf.predict();
        cv::Mat estimated = kf.correct(measurement);

        cv::Point2f currentPos(estimated.at<float>(0), estimated.at<float>(1));
        velocity.x = estimated.at<float>(2);
        velocity.y = estimated.at<float>(3);
        
        history.push_back(currentPos);
        if (history.size() > HISTORY_LENGTH) {
            history.erase(history.begin());
        }

        lastPosition = currentPos;
        age++;
        missedFrames = 0;
        confidence = 1.0f;
    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV Error in TrackedObject::update: " << e.what() << std::endl;
        confidence = 0.0f;
    } catch (const std::exception& e) {
        std::cerr << "Error in TrackedObject::update: " << e.what() << std::endl;
        confidence = 0.0f;
    }
}

cv::Point2f TrackedObject::getPosition() const {
    return lastPosition;
}

cv::Point2f TrackedObject::getVelocity() const {
    return velocity;
}

cv::Point2f TrackedObject::calculateAverageVelocity() const {
    if (history.size() < 2) return cv::Point2f(0, 0);
    
    cv::Point2f totalVelocity(0, 0);
    for (size_t i = 1; i < history.size(); ++i) {
        totalVelocity.x += history[i].x - history[i-1].x;
        totalVelocity.y += history[i].y - history[i-1].y;
    }
    
    float avgX = totalVelocity.x / (history.size() - 1);
    float avgY = totalVelocity.y / (history.size() - 1);
    return cv::Point2f(avgX, avgY);
}

std::string TrackedObject::getDirectionFromAngle(float angle) const {
    if (angle < 0) angle += 360;
    
    if (angle >= 337.5 || angle < 22.5) return "Right";
    else if (angle >= 22.5 && angle < 67.5) return "Down-Right";
    else if (angle >= 67.5 && angle < 112.5) return "Down";
    else if (angle >= 112.5 && angle < 157.5) return "Down-Left";
    else if (angle >= 157.5 && angle < 202.5) return "Left";
    else if (angle >= 202.5 && angle < 247.5) return "Up-Left";
    else if (angle >= 247.5 && angle < 292.5) return "Up";
    else return "Up-Right";
}

std::string TrackedObject::getPredictedDirection() const {
    if (!initialized) return "Unknown";
    
    cv::Point2f avgVelocity = calculateAverageVelocity();
    float angle = atan2(avgVelocity.y, avgVelocity.x) * 180 / CV_PI;
    return getDirectionFromAngle(angle);
}

void TrackedObject::drawTrajectory(cv::Mat& frame) {
    if (history.size() < 2) return;
    
    try {
        for (size_t i = 1; i < history.size(); i++) {
            float dx = history[i].x - history[i-1].x;
            float dy = history[i].y - history[i-1].y;
            float speed = std::sqrt(dx*dx + dy*dy);
            
            cv::Scalar color;
            if (speed < 5.0f) {
                color = cv::Scalar(0, 255, 0);
            } else if (speed < 10.0f) {
                color = cv::Scalar(0, 255, 255);
            } else {
                color = cv::Scalar(0, 0, 255);
            }
            
            cv::line(frame, history[i-1], history[i], color, 2);
        }
        
        if (history.size() >= 2) {
            cv::Point2f lastPos = history.back();
            cv::Point2f avgVelocity = calculateAverageVelocity();
            cv::Point2f direction = lastPos + avgVelocity * 20;
            cv::arrowedLine(frame, lastPos, direction, cv::Scalar(255, 0, 0), 2);
        }
    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV Error in drawTrajectory: " << e.what() << std::endl;
    }
}

bool TrackedObject::isLost() const {
    return missedFrames > 10;
}

void TrackedObject::markLost() {
    missedFrames++;
}

void TrackedObject::resetMissedFrames() {
    missedFrames = 0;
}

int TrackedObject::getObjectId() const {
    return objectId;
}

int TrackedObject::getAge() const {
    return age;
}

float TrackedObject::getConfidence() const {
    return confidence;
}
