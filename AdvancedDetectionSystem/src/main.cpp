#include <iostream>
#include <string>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "ObjectDetector.hpp"

void displaySystemInfo() {
    std::cout << "\n=== Gelişmiş Tespit Sistemi - Özellikler ===\n";
    std::cout << "Aktif Özellikler:\n";
    std::cout << "1. FPS Gösterimi (1 tuşu)\n";
    std::cout << "2. Hareket İzleme (2 tuşu)\n";
    std::cout << "3. Mesafe Ölçümü (3 tuşu)\n";
    std::cout << "4. Yön Tespiti (4 tuşu)\n";
    std::cout << "5. Su Seviyesi (5 tuşu)\n";
    std::cout << "6. Cilt Tespiti (6 tuşu)\n";
    std::cout << "7. Deniz Tespiti (7 tuşu)\n";
    std::cout << "8. İnsan Tespiti (8 tuşu)\n\n";
    std::cout << "Kontroller:\n";
    std::cout << "H: Yardım Menüsü\n";
    std::cout << "Q: Çıkış\n";
    std::cout << "========================================\n\n";
}

void detectFacesAndObjects(const std::string& videoSource) {
    try {
        // MacOS için cascade dosyası konumları
        std::vector<std::string> possiblePaths = {
            "/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
            "/opt/homebrew/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
            "haarcascade_frontalface_default.xml",
            "../resources/haarcascade_frontalface_default.xml",
            "./resources/haarcascade_frontalface_default.xml"
        };

        ObjectDetector* detector = nullptr;
        for (const auto& path : possiblePaths) {
            try {
                detector = new ObjectDetector(path);
                std::cout << "Cascade sınıflandırıcı başarıyla yüklendi: " << path << std::endl;
                break;
            } catch (const std::exception& e) {
                continue;
            }
        }

        if (!detector) {
            throw std::runtime_error("Cascade sınıflandırıcı dosyası bulunamadı!");
        }

        cv::VideoCapture cap;
        if (videoSource == "0") {
            cap.open(0);
            if (!cap.isOpened()) {
                throw std::runtime_error("Webcam açılamadı!");
            }
            std::cout << "Webcam başarıyla açıldı.\n";
            cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
            cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
            cap.set(cv::CAP_PROP_FPS, 30);
            
        } else {
            // Video dosyası kontrol et
            std::ifstream fileCheck(videoSource.c_str());
            if (!fileCheck.good()) {
                throw std::runtime_error("Video dosyası bulunamadı: " + videoSource);
            }
            fileCheck.close();
            
            cap.open(videoSource);
            if (!cap.isOpened()) {
                throw std::runtime_error("Video dosyası açılamadı: " + videoSource);
            }
            
            int frameWidth = cap.get(cv::CAP_PROP_FRAME_WIDTH);
            int frameHeight = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
            double fps = cap.get(cv::CAP_PROP_FPS);
            
            std::cout << "Video başarıyla açıldı: " << videoSource << "\n";
            std::cout << "Video özellikleri:\n";
                        std::cout << "Genişlik: " << frameWidth << "\n";
                        std::cout << "Yükseklik: " << frameHeight << "\n";
                        std::cout << "FPS: " << fps << "\n";
                    }

                    cv::namedWindow("Detection System", cv::WINDOW_NORMAL);
                    cv::resizeWindow("Detection System", 1280, 720);

                    displaySystemInfo();

                    cv::Mat frame;
                    char key = 0;
                    
                    while (key != 'q' && key != 'Q') {
                        cap >> frame;
                        if (frame.empty()) {
                            std::cout << "Video bitti veya frame alınamadı.\n";
                            break;
                        }

                        try {
                            detector->detectAndTrack(frame);
                            cv::imshow("Detection System", frame);
                        } catch (const std::exception& e) {
                            std::cerr << "Frame işlenirken hata oluştu: " << e.what() << std::endl;
                            continue;
                        }

                        key = cv::waitKey(1) & 0xFF;
                        detector->handleKeyPress(key);
                    }

                    delete detector;
                    cap.release();
                    cv::destroyAllWindows();

                } catch (const std::exception& e) {
                    std::cerr << "Hata: " << e.what() << std::endl;
                }
            }

            void mainMenu() {
                while (true) {
                    std::cout << "\n=== Gelişmiş Tespit Sistemi ===\n";
                    std::cout << "1. Tespit sistemini başlat\n";
                    std::cout << "2. Sistem bilgilerini göster\n";
                    std::cout << "3. Çıkış\n";
                    std::cout << "Seçiminiz (1-3): ";

                    std::string choice;
                    std::getline(std::cin, choice);

                    if (choice == "1") {
                        std::cout << "\nVideo kaynağını seçin:\n";
                        std::cout << "1. Webcam\n";
                        std::cout << "2. Video dosyası\n";
                        std::cout << "Seçiminiz (1-2): ";
                        
                        std::string sourceChoice;
                        std::getline(std::cin, sourceChoice);
                        
                        std::string source;
                        if (sourceChoice == "1") {
                            source = "0";
                            std::cout << "Webcam başlatılıyor...\n";
                        } else if (sourceChoice == "2") {
                            std::cout << "Video dosyası yolunu girin: ";
                            std::getline(std::cin, source);
                            // Yoldaki gereksiz boşlukları temizle
                            source.erase(0, source.find_first_not_of(" \t\n\r\f\v"));
                            source.erase(source.find_last_not_of(" \t\n\r\f\v") + 1);
                            std::cout << "Video yükleniyor: " << source << "\n";
                        } else {
                            std::cout << "Geçersiz seçim!\n";
                            continue;
                        }

                        detectFacesAndObjects(source);
                    }
                    else if (choice == "2") {
                        displaySystemInfo();
                    }
                    else if (choice == "3") {
                        std::cout << "Sistemden çıkılıyor...\n";
                        break;
                    }
                    else {
                        std::cout << "Geçersiz seçim! Lütfen 1-3 arası bir sayı girin.\n";
                    }
                }
            }

            int main() {
                try {
                    std::cout << "Gelişmiş Tespit Sistemi Başlatılıyor...\n";
                    mainMenu();
                } catch (const std::exception& e) {
                    std::cerr << "Kritik hata: " << e.what() << std::endl;
                    return 1;
                }
                return 0;
            }
