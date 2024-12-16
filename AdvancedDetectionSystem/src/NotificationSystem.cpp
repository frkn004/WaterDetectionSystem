#include "NotificationSystem.hpp"
#include <fstream>
#include <iostream>
#include <iomanip>

NotificationSystem::NotificationSystem(const std::string& logPath, bool enableAudio) 
    : logFile(logPath),
      isAudioEnabled(enableAudio),
      lastNotificationTime(std::chrono::steady_clock::now()) {
    
    // Varsayılan konfigürasyon
    config = NotificationConfig{};
}

void NotificationSystem::setConfig(const NotificationConfig& newConfig) {
    std::lock_guard<std::mutex> lock(queueMutex);
    config = newConfig;
}

void NotificationSystem::notify(const std::string& message, bool isUrgent) {
    auto currentTime = std::chrono::steady_clock::now();
    auto timeSinceLastNotification =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            currentTime - lastNotificationTime).count();

    // Bildirimler arası minimum süre kontrolü
    if (timeSinceLastNotification >= config.cooldown.count()) {
        try {
            // Yeni bildirim oluştur
            Notification newNotification{
                message,
                isUrgent,
                currentTime
            };
            
            // Bildirimi kuyruğa ekle ve yönet
            manageNotificationQueue(newNotification);
            
            // Log dosyasına kaydet
            logNotification(message);
            
            // Acil durumda ve ses aktifse uyarı sesi çal
            if (isUrgent && isAudioEnabled) {
                playNotificationSound(isUrgent);
            }
            
            lastNotificationTime = currentTime;
            
        } catch (const std::exception& e) {
            std::cerr << "Bildirim işleme hatası: " << e.what() << std::endl;
        }
    }
}

void NotificationSystem::logNotification(const std::string& message) {
    try {
        std::ofstream log(logFile, std::ios::app);
        if (log.is_open()) {
            auto now = std::chrono::system_clock::now();
            auto time_t = std::chrono::system_clock::to_time_t(now);
            auto tm = *std::localtime(&time_t);
            
            // ISO 8601 formatında zaman damgası ile log
            log << std::put_time(&tm, "%Y-%m-%d %H:%M:%S") 
                << " - " << message << "\n";
        } else {
            throw std::runtime_error("Log dosyası açılamadı: " + logFile);
        }
    } catch (const std::exception& e) {
        std::cerr << "Log yazma hatası: " << e.what() << std::endl;
    }
}

void NotificationSystem::playNotificationSound(bool isUrgent) {
    try {
        #ifdef _WIN32
            // Windows için farklı tonlarda uyarı sesi
            Beep(isUrgent ? 1000 : 500, isUrgent ? 200 : 100);
        #else
            // macOS veya Linux için sistem sesleri
            if (isUrgent) {
                system("afplay /System/Library/Sounds/Ping.aiff &");
            } else {
                system("afplay /System/Library/Sounds/Tink.aiff &");
            }
        #endif
    } catch (const std::exception& e) {
        std::cerr << "Ses çalma hatası: " << e.what() << std::endl;
    }
}

void NotificationSystem::manageNotificationQueue(const Notification& notification) {
    std::lock_guard<std::mutex> lock(queueMutex);
    
    // Kuyruk boyut kontrolü ve eski bildirimleri temizleme
    if (notificationQueue.size() >= config.maxQueueSize) {
        notificationQueue.pop();
    }
    
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
    
    // İstenen sayıda son bildirimi kopyala
    std::queue<Notification> tempQueue = notificationQueue;
    while (!tempQueue.empty() && notifications.size() < static_cast<size_t>(count)) {
        notifications.push_back(tempQueue.front().message);
        tempQueue.pop();
    }
    
    return notifications;
}