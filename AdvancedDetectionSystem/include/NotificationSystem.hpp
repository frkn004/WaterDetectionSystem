#pragma once
#include <string>
#include <chrono>
#include <queue>
#include <mutex>
#include <vector>
#include <memory>

class NotificationSystem {
public:
    struct NotificationConfig {
        bool enableSound{true};
        bool enableVisual{true};
        unsigned int maxQueueSize{10};
        std::chrono::milliseconds cooldown{3000};
    };

private:
    std::string logFile;
    bool isAudioEnabled;
    std::chrono::steady_clock::time_point lastNotificationTime;
    const std::chrono::milliseconds NOTIFICATION_COOLDOWN{3000};
    
    struct Notification {
        std::string message;
        bool isUrgent;
        std::chrono::steady_clock::time_point timestamp;
    };
    
    std::queue<Notification> notificationQueue;
    std::mutex queueMutex;
    const size_t MAX_QUEUE_SIZE = 10;
    NotificationConfig config;

    void logNotification(const std::string& message);
    void playNotificationSound(bool isUrgent);
    void manageNotificationQueue(const Notification& notification);

public:
    explicit NotificationSystem(const std::string& logPath = "notifications.log",
                              bool enableAudio = true);
    
    void notify(const std::string& message, bool isUrgent = false);
    void toggleAudio();
    bool getAudioStatus() const { return isAudioEnabled; }
    void clearNotifications();
    std::vector<std::string> getRecentNotifications(int count = 5);
    void setConfig(const NotificationConfig& newConfig);
};