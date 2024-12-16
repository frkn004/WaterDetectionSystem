#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <deque>
#include <string>
#include <stdexcept>

class TrackedObject {
public:
    struct KalmanConfig {
        static constexpr float DEFAULT_PROCESS_NOISE = 1e-4f;
        static constexpr float DEFAULT_MEASUREMENT_NOISE = 1e-1f;
        static constexpr float DEFAULT_ERROR_COVARIANCE = 1.0f;
        
        float processNoise;
        float measurementNoise;
        float errorCovariance;

        KalmanConfig() : 
            processNoise(DEFAULT_PROCESS_NOISE),
            measurementNoise(DEFAULT_MEASUREMENT_NOISE),
            errorCovariance(DEFAULT_ERROR_COVARIANCE) {}

        void validate() const {
            if (processNoise <= 0 || measurementNoise <= 0 || errorCovariance <= 0)
                throw std::invalid_argument("Invalid Kalman filter parameters");
        }
    };

private:
    static constexpr size_t MAX_HISTORY_SIZE = 30;
    static constexpr int MAX_MISSED_FRAMES = 10;
    static constexpr float MIN_CONFIDENCE = 0.1f;

    cv::KalmanFilter kf;
    cv::Point2f lastPosition;
    cv::Point2f velocity;
    bool initialized;
    std::deque<cv::Point2f> history;  // Using deque for efficient front removal
    const int objectId;
    int age;
    int missedFrames;
    float confidence;
    KalmanConfig kalmanConfig;

    void initializeKalmanFilter();
    cv::Point2f calculateAverageVelocity() const;
    std::string getDirectionFromAngle(float angle) const;
    void updateHistory(const cv::Point2f& pos);

public:
    class TrackingError : public std::runtime_error {
        using std::runtime_error::runtime_error;
    };

    explicit TrackedObject(int id, const KalmanConfig& config = KalmanConfig());
    ~TrackedObject() = default;

    // Delete copy and move constructors
    TrackedObject(const TrackedObject&) = delete;
    TrackedObject& operator=(const TrackedObject&) = delete;
    TrackedObject(TrackedObject&&) = delete;
    TrackedObject& operator=(TrackedObject&&) = delete;

    // Core tracking methods
    void update(const cv::Point2f& pos);
    void predict();
    void correct(const cv::Point2f& measurement);

    // Getters
    cv::Point2f getPosition() const { return lastPosition; }
    cv::Point2f getVelocity() const { return velocity; }
    std::string getPredictedDirection() const;
    bool isLost() const { return missedFrames > MAX_MISSED_FRAMES; }
    int getObjectId() const { return objectId; }
    int getAge() const { return age; }
    float getConfidence() const { return confidence; }

    // State management
    void markLost() { missedFrames++; }
    void resetMissedFrames() { missedFrames = 0; }

    // Visualization
    void drawTrajectory(cv::Mat& frame) const;
    void setKalmanConfig(const KalmanConfig& config);
};