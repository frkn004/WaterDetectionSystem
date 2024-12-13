#include "NotificationSystem.hpp"
#include <fstream>
#include <iostream>
#include <iomanip>

NotificationSystem::NotificationSystem(const std::string& logPath, bool enableAudio)
    : logFile(logPath), isAudioEnabled(enableAudio) {
    lastNotificationTime = std::chrono::steady_clock::now();
}

void NotificationSystem::notify(const std::string& message, bool isUrgent) {
    auto currentTime = std::chrono::steady_clock::now();
    auto timeSinceLastNotification =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            currentTime - lastNotificationTime).count();

    if (timeSinceLastNotification >= NOTIFICATION_COOLDOWN_MS) {
        // Yeni bildirimi oluştur
        Notification newNotification{
            message,
            isUrgent,
            currentTime
        };
        
        // Bildirim kuyruğunu yönet
        manageNotificationQueue(newNotification);
        
        // Sistem saatine göre zamanı al
        auto systemTime = std::chrono::system_clock::now();
        
        // Bildirim logla
        logNotification(message, systemTime);
        
        // Ses çal (eğer aktifse ve acilse)
        if (isUrgent && isAudioEnabled) {
            playNotificationSound(isUrgent);
        }
        
        lastNotificationTime = currentTime;
    }
}

void NotificationSystem::logNotification(const std::string& message,
                                       const std::chrono::system_clock::time_point& time) {
    try {
        std::ofstream log(logFile, std::ios::app);
        if (log.is_open()) {
            auto time_t = std::chrono::system_clock::to_time_t(time);
            auto tm = *std::localtime(&time_t);
            
            log << std::put_time(&tm, "%Y-%m-%d %H:%M:%S") << " - "
                << message << "\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "Error writing to log file: " << e.what() << std::endl;
    }
}

void NotificationSystem::playNotificationSound(bool isUrgent) {
    try {
        #ifdef _WIN32
            Beep(isUrgent ? 1000 : 500, isUrgent ? 200 : 100);
        #else
            if (isUrgent) {
                system("afplay /System/Library/Sounds/Ping.aiff &");
            } else {
                system("afplay /System/Library/Sounds/Tink.aiff &");
            }
        #endif
    } catch (const std::exception& e) {
        std::cerr << "Error playing notification sound: " << e.what() << std::endl;
    }
}

void NotificationSystem::toggleAudio() {
    isAudioEnabled = !isAudioEnabled;
}

void NotificationSystem::manageNotificationQueue(const Notification& notification) {
    std::lock_guard<std::mutex> lock(queueMutex);
    
    // Kuyruk maksimum boyuta ulaştıysa en eski bildirimi çıkar
    if (notificationQueue.size() >= MAX_QUEUE_SIZE) {
        notificationQueue.pop();
    }
    
    // Yeni bildirimi ekle
    notificationQueue.push(notification);
}

void NotificationSystem::clearNotifications() {
    std::lock_guard<std::mutex> lock(queueMutex);
    while (!notificationQueue.empty()) {
        notificationQueue.pop();
    }
}

std::vector<std::string> NotificationSystem::getRecentNotifications(int count) {
    std::lock_guard<std::mutex> lock(queueMutex);
    std::vector<std::string> notifications;
    
    std::queue<Notification> tempQueue = notificationQueue;
    while (!tempQueue.empty() && notifications.size() < static_cast<size_t>(count)) {
        notifications.push_back(tempQueue.front().message);
        tempQueue.pop();
    }
    
    return notifications;
}
