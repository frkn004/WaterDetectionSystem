#pragma once
#include <string>
#include <chrono>
#include <queue>
#include <mutex>
#include <vector>

class NotificationSystem {
private:
    std::string logFile;
    bool isAudioEnabled;
    std::chrono::steady_clock::time_point lastNotificationTime;
    const int NOTIFICATION_COOLDOWN_MS = 3000;
    
    struct Notification {
        std::string message;
        bool isUrgent;
        std::chrono::steady_clock::time_point timestamp;
    };
    
    std::queue<Notification> notificationQueue;
    std::mutex queueMutex;
    const size_t MAX_QUEUE_SIZE = 10;

    // Private helper functions
    void logNotification(const std::string& message,
                        const std::chrono::system_clock::time_point& time);
    void playNotificationSound(bool isUrgent);
    void manageNotificationQueue(const Notification& notification);

public:
    // Constructor
    NotificationSystem(const std::string& logPath = "notifications.log",
                      bool enableAudio = true);

    // Main notification functions
    void notify(const std::string& message, bool isUrgent = false);
    void toggleAudio();
    bool getAudioStatus() const { return isAudioEnabled; }
    
    // Queue management
    void clearNotifications();
    std::vector<std::string> getRecentNotifications(int count = 5);
};
