#pragma once
#include <opencv2/opencv.hpp>
#include <unordered_map>
#include <string>

class FeatureControl {
public:
    enum class Feature {
        SHOW_FPS,
        SHOW_TRAJECTORY,
        SHOW_DISTANCE,
        SHOW_DIRECTION,
        SHOW_WATER_LEVEL,
        SHOW_SKIN_DETECTION,
        SHOW_SEA_DETECTION,
        SHOW_HUMAN_DETECTION,
        SHOW_BOAT_DETECTION,
        ENABLE_NOTIFICATIONS,
        SHOW_OBJECT_ID,
        SHOW_CONFIDENCE,
        SHOW_SPEED,
        SHOW_ACCELERATION,
        SHOW_HEAT_MAP,
        ENABLE_NIGHT_MODE,
        SHOW_GRID,
        SHOW_COORDINATES,
        ENABLE_RECORDING,
        SHOW_TIMESTAMP
    };

private:
    std::unordered_map<Feature, bool> features;
    std::unordered_map<Feature, char> featureKeys;
    std::unordered_map<Feature, std::string> featureNames;

public:
    FeatureControl();
    
    // Feature kontrolü için metodlar
    void toggleFeature(Feature feature);
    void toggleFeature(char key);
    bool isEnabled(Feature feature) const;
    
    // Görselleştirme metodları
    void displayHelp(cv::Mat& frame);
    void displayFeatureStatus(cv::Mat& frame);
    std::string getFeatureName(Feature feature) const;
};