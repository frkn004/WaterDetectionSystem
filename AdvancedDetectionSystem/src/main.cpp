#include <iostream>
#include <string>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "ObjectDetector.hpp"
#include "HumanDetector.hpp"
#include "SeaDetector.hpp"
#include "NotificationSystem.hpp"

void displayMenu() {
    std::cout << "\n=== Deniz Güvenlik Sistemi ===\n";
    std::cout << "1. Sistemi Başlat\n";
    std::cout << "2. Ayarlar\n";
    std::cout << "3. Çıkış\n";
    std::cout << "Seçiminiz (1-3): ";
}

void displaySettings() {
    std::cout << "\n=== Ayarlar ===\n";
    std::cout << "1. İnsan Tespiti (Açık/Kapalı)\n";
    std::cout << "2. Yüz Tespiti (Açık/Kapalı)\n";
    std::cout << "3. Deniz Analizi (Açık/Kapalı)\n";
    std::cout << "4. Ana Menüye Dön\n";
    std::cout << "Seçiminiz (1-4): ";
}

class SystemSettings {
public:
    bool humanDetectionEnabled = true;
    bool faceDetectionEnabled = true;
    bool seaAnalysisEnabled = true;
};

void runDetectionSystem(SystemSettings& settings) {
    HumanDetector humanDetector;
    SeaDetector seaDetector;
    NotificationSystem notifier;
    
    // Yüz tespiti için cascade sınıflandırıcı
    cv::CascadeClassifier faceCascade;
    if (settings.faceDetectionEnabled) {
        if (!faceCascade.load("haarcascade_frontalface_default.xml")) {
            std::cerr << "Yüz cascade dosyası yüklenemedi!" << std::endl;
            return;
        }
    }
    
    cv::VideoCapture camera;
    camera.open(0);
    if (!camera.isOpened()) {
        std::cerr << "Kamera açılamadı!" << std::endl;
        return;
    }
    
    while (true) {
        cv::Mat frame;
        camera >> frame;
        if (frame.empty()) break;
        
        // İnsan tespiti
        if (settings.humanDetectionEnabled) {
            auto detections = humanDetector.detect(frame);
            for (const auto& detection : detections) {
                cv::rectangle(frame, detection.boundingBox, cv::Scalar(0, 255, 0), 2);
            }
        }
        
        // Yüz tespiti
        if (settings.faceDetectionEnabled) {
            cv::Mat gray;
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
            std::vector<cv::Rect> faces;
            faceCascade.detectMultiScale(gray, faces, 1.1, 4);
            
            for (const auto& face : faces) {
                cv::rectangle(frame, face, cv::Scalar(255, 0, 0), 2);
                cv::putText(frame, "Yuz", 
                           cv::Point(face.x, face.y - 5),
                           cv::FONT_HERSHEY_SIMPLEX, 0.5, 
                           cv::Scalar(255, 0, 0), 2);
            }
        }
        
        // Deniz analizi
        if (settings.seaAnalysisEnabled) {
            auto sceneAnalysis = seaDetector.analyzeScene(frame);
            seaDetector.visualizeResults(frame, sceneAnalysis);
            
            // Tehlike durumu kontrolü
            if (sceneAnalysis.isDangerous) {
                notifier.sendAlert("Tehlikeli dalga seviyesi tespit edildi!");
            }
        }
        
        // Kontrol tuşları bilgisi
        cv::putText(frame, "ESC: Cikis | S: Ayarlar", 
                   cv::Point(10, frame.rows - 20),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, 
                   cv::Scalar(255, 255, 255), 1);
        
        cv::imshow("Deniz Guvenlik Sistemi", frame);
        
        char key = cv::waitKey(1);
        if (key == 27) break;        // ESC
        else if (key == 's' || key == 'S') {
            // Ayarlar menüsünü göster
            camera.release();
            cv::destroyAllWindows();
            return;
        }
    }
    
    camera.release();
    cv::destroyAllWindows();
}

int main() {
    SystemSettings settings;
    
    while (true) {
        displayMenu();
        
        std::string choice;
        std::getline(std::cin, choice);
        
        if (choice == "1") {
            runDetectionSystem(settings);
        }
        else if (choice == "2") {
            while (true) {
                displaySettings();
                std::string settingChoice;
                std::getline(std::cin, settingChoice);
                
                if (settingChoice == "1") {
                    settings.humanDetectionEnabled = !settings.humanDetectionEnabled;
                    std::cout << "İnsan Tespiti: " 
                             << (settings.humanDetectionEnabled ? "Açık" : "Kapalı") 
                             << std::endl;
                }
                else if (settingChoice == "2") {
                    settings.faceDetectionEnabled = !settings.faceDetectionEnabled;
                    std::cout << "Yüz Tespiti: " 
                             << (settings.faceDetectionEnabled ? "Açık" : "Kapalı") 
                             << std::endl;
                }
                else if (settingChoice == "3") {
                    settings.seaAnalysisEnabled = !settings.seaAnalysisEnabled;
                    std::cout << "Deniz Analizi: " 
                             << (settings.seaAnalysisEnabled ? "Açık" : "Kapalı") 
                             << std::endl;
                }
                else if (settingChoice == "4") {
                    break;
                }
            }
        }
        else if (choice == "3") {
            std::cout << "Sistemden çıkılıyor...\n";
            break;
        }
    }
    
    return 0;
}
