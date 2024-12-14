#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

class TrackedObject {
private:
    cv::KalmanFilter kf;
    cv::Point2f lastPosition;
    cv::Point2f velocity;
    bool initialized;
    std::vector<cv::Point2f> history;
    static const int HISTORY_LENGTH = 10;
    int objectId;
    int age;
    int missedFrames;
    float confidence;
    int id;

    cv::Point2f calculateAverageVelocity() const;
    std::string getDirectionFromAngle(float angle) const;

public:
    TrackedObject() = delete;  // Varsayılan yapıcıyı devre dışı bırak
    explicit TrackedObject(int id);  // Sadece deklarasyon, implementasyon yok!

    void update(const cv::Point2f& pos);
    cv::Point2f getPosition() const;
    cv::Point2f getVelocity() const;
    std::string getPredictedDirection() const;
    void drawTrajectory(cv::Mat& frame);
    bool isLost() const;
    void markLost();
    void resetMissedFrames();
    int getObjectId() const;
    int getAge() const;
    float getConfidence() const;
};
