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
    const int HISTORY_LENGTH = 10;

public:
    // Constructor
    TrackedObject();

    // Update and tracking functions
    void update(const cv::Point2f& pos);
    cv::Point2f getPosition() const;
    cv::Point2f getVelocity() const;
    std::string getPredictedDirection() const;

    // Visualization
    void drawTrajectory(cv::Mat& frame);
};
