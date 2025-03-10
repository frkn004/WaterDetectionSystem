#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <thread>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "ObjectDetector.hpp"

class VideoSource {
public:
    enum class Type { CAMERA, VIDEO_FILE };

    struct Config {
        Type type;
        int cameraId;
        std::string videoPath;
        bool loop;
        int width;
        int height;
        double fps;

        Config() 
            : type(Type::CAMERA)
            , cameraId(0)
            , videoPath("")
            , loop(false)
            , width(640)
            , height(480)
            , fps(30.0)
        {}

        void validate() const {
            if (type == Type::VIDEO_FILE && videoPath.empty()) {
                throw std::invalid_argument("Video path is empty");
            }
            if (width <= 0 || height <= 0) {
                throw std::invalid_argument("Invalid dimensions");
            }
            if (fps <= 0) {
                throw std::invalid_argument("Invalid FPS");
            }
        }
    };

private:
    cv::VideoCapture capture;
    Config config;
    bool isInitialized;
    std::chrono::steady_clock::time_point lastFrameTime;
    double frameInterval;

public:
    explicit VideoSource(const Config& cfg = Config()) 
        : config(cfg)
        , isInitialized(false)
        , lastFrameTime(std::chrono::steady_clock::now())
        , frameInterval(1000.0 / cfg.fps)
    {
    }

    ~VideoSource() {
        release();
    }

    // Copy prevention
    VideoSource(const VideoSource&) = delete;
    VideoSource& operator=(const VideoSource&) = delete;
    
    // Move operations
    VideoSource(VideoSource&&) noexcept = default;
    VideoSource& operator=(VideoSource&&) noexcept = delete;

    bool initialize(const Config& cfg) {
        try {
            config = cfg;
            config.validate();
            isInitialized = false;

            if (config.type == Type::CAMERA) {
                capture.open(config.cameraId);
                if (capture.isOpened()) {
                    capture.set(cv::CAP_PROP_FRAME_WIDTH, config.width);
                    capture.set(cv::CAP_PROP_FRAME_HEIGHT, config.height);
                    capture.set(cv::CAP_PROP_FPS, config.fps);
                }
            } else {
                capture.open(config.videoPath);
            }

            if (!capture.isOpened()) {
                throw std::runtime_error("Failed to open video source!");
            }

            lastFrameTime = std::chrono::steady_clock::now();
            isInitialized = true;
            return true;

        } catch (const std::exception& e) {
            std::cerr << "Error initializing video source: " << e.what() << std::endl;
            return false;
        }
    }

    bool read(cv::Mat& frame) {
        if (!isInitialized) return false;

        try {
            // FPS kontrolü
            auto currentTime = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>
                         (currentTime - lastFrameTime).count();
            
            if (elapsed < frameInterval) {
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(
                        static_cast<int>(frameInterval - elapsed)
                    )
                );
            }

            if (!capture.read(frame)) {
                if (config.type == Type::VIDEO_FILE && config.loop) {
                    capture.set(cv::CAP_PROP_POS_FRAMES, 0);
                    return capture.read(frame);
                }
                return false;
            }

            lastFrameTime = std::chrono::steady_clock::now();
            return true;

        } catch (const cv::Exception& e) {
            std::cerr << "OpenCV error in read: " << e.what() << std::endl;
            return false;
        }
    }

    void release() {
        if (isInitialized) {
            capture.release();
            isInitialized = false;
        }
    }
};

class Application {
private:
    std::unique_ptr<ObjectDetector> detector;
    std::unique_ptr<VideoSource> videoSource;
    bool isRunning;

    void processFrame(cv::Mat& frame) {
        try {
            detector->detectAndTrack(frame);
            cv::imshow("Deniz Güvenlik Sistemi", frame);
        } catch (const std::exception& e) {
            std::cerr << "Error processing frame: " << e.what() << std::endl;
        }
    }

public:
    Application() : isRunning(false) {}

    bool initialize() {
        try {
            std::vector<std::string> cascadePaths = {
                "haarcascade_frontalface_default.xml",
                "./resources/haarcascade_frontalface_default.xml",
                "../resources/haarcascade_frontalface_default.xml",
                "/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
                "/opt/homebrew/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
            };

            bool modelLoaded = false;
            for (const auto& path : cascadePaths) {
                try {
                    detector = std::make_unique<ObjectDetector>(path);
                    std::cout << "Model başarıyla yüklendi: " << path << std::endl;
                    modelLoaded = true;
                    break;
                } catch (const std::exception&) {
                    continue;
                }
            }

            if (!modelLoaded) {
                throw std::runtime_error("Yüz tanıma modeli yüklenemedi.");
            }

            // Sea Detector konfigürasyonunu kontrol et
            try {
                auto& seaDetector = detector->getSeaDetector();
                auto config = seaDetector.getConfig();
                std::cout << "\nSea Detector konfigürasyonu:\n";
                std::cout << "Max dalga yüksekliği: " << config.maxWaveHeight << "m\n";
                std::cout << "Tehlikeli dalga yüksekliği: " << config.dangerousWaveHeight << "m\n";
                std::cout << "Örnekleme hızı: " << config.samplingRate << " Hz\n";
            } catch (const std::exception& e) {
                std::cerr << "Sea Detector konfigürasyon hatası: " << e.what() << std::endl;
                return false;
            }
            
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Initialization error: " << e.what() << std::endl;
            return false;
        }
    }

    void run() {
        while (true) {
            std::cout << "\n=== Deniz Güvenlik Sistemi ===\n"
                      << "1. Kamera ile Başlat\n"
                      << "2. Video Dosyasından Başlat\n"
                      << "3. Ayarlar\n"
                      << "4. Çıkış\n"
                      << "Seçiminiz (1-4): ";

            std::string choice;
            std::getline(std::cin, choice);

            if (choice == "1") {
                VideoSource::Config config;
                config.type = VideoSource::Type::CAMERA;
                
                std::cout << "Kamera ID (varsayılan=0): ";
                std::string cameraId;
                std::getline(std::cin, cameraId);
                if (!cameraId.empty()) {
                    config.cameraId = std::stoi(cameraId);
                }

                runDetection(config);
            }
            else if (choice == "2") {
                std::cout << "Video dosyası yolunu girin: ";
                std::string videoPath;
                std::getline(std::cin, videoPath);
                
                VideoSource::Config config;
                config.type = VideoSource::Type::VIDEO_FILE;
                config.videoPath = videoPath;
                config.loop = true;
                
                runDetection(config);
            }
            else if (choice == "3") {
                showSettings();
            }
            else if (choice == "4") {
                std::cout << "Programdan çıkılıyor...\n";
                break;
            }
            else {
                std::cout << "Geçersiz seçim!\n";
            }
        }
    }
    private:
    void runDetection(const VideoSource::Config& config) {
        try {
            videoSource = std::make_unique<VideoSource>();
            if (!videoSource->initialize(config)) {
                throw std::runtime_error("Video kaynağı başlatılamadı!");
            }

            std::cout << "\nKontroller:\n"
                      << "- ESC: Çıkış\n"
                      << "- H: Yardım menüsü\n"
                      << "- 1-9: Farklı görüntüleme modları\n"
                      << "- Diğer tuşlar için yardım menüsüne bakın\n\n";

            isRunning = true;
            cv::Mat frame;
            
            while (isRunning && videoSource->read(frame)) {
                processFrame(frame);
                
                char key = cv::waitKey(1);
                if (key == 27) break;  // ESC
                if (key >= 0) {
                    detector->handleKeyPress(key);
                }
            }

            videoSource->release();
            cv::destroyAllWindows();
            
        } catch (const std::exception& e) {
            std::cerr << "Error in detection: " << e.what() << std::endl;
        }
    }

    void showSettings() {
        while (true) {
            std::cout << "\n=== Ayarlar ===\n"
                      << "1. Deniz Analiz Ayarları\n"
                      << "2. Video Akış Hızı\n"
                      << "3. Tespit Hassasiyeti\n"
                      << "4. Ana Menüye Dön\n"
                      << "Seçiminiz (1-4): ";

            std::string choice;
            std::getline(std::cin, choice);

            if (choice == "1") {
                configureSeaDetector();
            }
            else if (choice == "2") {
                std::cout << "Yeni FPS değeri girin (10-60): ";
                std::string fpsStr;
                std::getline(std::cin, fpsStr);
                try {
                    double fps = std::stod(fpsStr);
                    if (fps >= 10 && fps <= 60) {
                        if (videoSource) {
                            VideoSource::Config newConfig;
                            newConfig.fps = fps;
                            videoSource->initialize(newConfig);
                        }
                        std::cout << "FPS güncellendi: " << fps << std::endl;
                    } else {
                        std::cout << "Geçersiz FPS değeri!\n";
                    }
                } catch (...) {
                    std::cout << "Geçersiz giriş!\n";
                }
            }
            else if (choice == "3") {
                std::cout << "Tespit hassasiyeti (0.0-1.0): ";
                std::string sensStr;
                std::getline(std::cin, sensStr);
                try {
                    float sensitivity = std::stof(sensStr);
                    if (sensitivity >= 0.0f && sensitivity <= 1.0f) {
                        // Burada detektör hassasiyetini güncelle
                        std::cout << "Hassasiyet güncellendi: " << sensitivity << std::endl;
                    } else {
                        std::cout << "Geçersiz hassasiyet değeri!\n";
                    }
                } catch (...) {
                    std::cout << "Geçersiz giriş!\n";
                }
            }
            else if (choice == "4") {
                break;
            }
            else {
                std::cout << "Geçersiz seçim!\n";
            }
        }
    }

    void configureSeaDetector() {
        try {
            auto& seaDetector = detector->getSeaDetector();
            auto currentConfig = seaDetector.getConfig();

            while (true) {
                std::cout << "\n=== Deniz Analiz Ayarları ===\n"
                          << "1. Dalga Yüksekliği Limitleri\n"
                          << "2. Su Seviyesi Eşikleri\n"
                          << "3. Sıcaklık Ayarları\n"
                          << "4. Görüntü İşleme Parametreleri\n"
                          << "5. Ana Menüye Dön\n"
                          << "Seçiminiz (1-5): ";

                std::string choice;
                std::getline(std::cin, choice);

                if (choice == "1") {
                    std::cout << "Maksimum dalga yüksekliği (m): ";
                    std::string heightStr;
                    std::getline(std::cin, heightStr);
                    try {
                        float height = std::stof(heightStr);
                        if (height > 0) {
                            currentConfig.maxWaveHeight = height;
                            currentConfig.dangerousWaveHeight = height * 0.75f;
                            seaDetector.setConfig(currentConfig);
                            std::cout << "Dalga yüksekliği limitleri güncellendi.\n";
                        }
                    } catch (...) {
                        std::cout << "Geçersiz giriş!\n";
                    }
                }
                else if (choice == "2") {
                    std::cout << "Su seviyesi değişim eşiği (m/h): ";
                    std::string threshStr;
                    std::getline(std::cin, threshStr);
                    try {
                        float thresh = std::stof(threshStr);
                        if (thresh > 0) {
                            currentConfig.waterLevelThreshold = thresh;
                            seaDetector.setConfig(currentConfig);
                            std::cout << "Su seviyesi eşiği güncellendi.\n";
                        }
                    } catch (...) {
                        std::cout << "Geçersiz giriş!\n";
                    }
                }
                else if (choice == "3") {
                    std::cout << "Sıcaklık değişim eşiği (°C): ";
                    std::string tempStr;
                    std::getline(std::cin, tempStr);
                    try {
                        float temp = std::stof(tempStr);
                        if (temp > 0) {
                            currentConfig.temperatureThreshold = temp;
                            seaDetector.setConfig(currentConfig);
                            std::cout << "Sıcaklık eşiği güncellendi.\n";
                        }
                    } catch (...) {
                        std::cout << "Geçersiz giriş!\n";
                    }
                }
                else if (choice == "4") {
                    std::cout << "Bulanıklaştırma boyutu (tek sayı): ";
                    std::string blurStr;
                    std::getline(std::cin, blurStr);
                    try {
                        int blur = std::stoi(blurStr);
                        if (blur > 0 && blur % 2 == 1) {
                            currentConfig.blurSize = blur;
                            seaDetector.setConfig(currentConfig);
                            std::cout << "Görüntü işleme parametreleri güncellendi.\n";
                        }
                    } catch (...) {
                        std::cout << "Geçersiz giriş!\n";
                    }
                }
                else if (choice == "5") {
                    break;
                }
                else {
                    std::cout << "Geçersiz seçim!\n";
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "Ayar güncelleme hatası: " << e.what() << std::endl;
        }
    }
};

int main() {
    try {
        Application app;
        if (!app.initialize()) {
            return -1;
        }
        app.run();
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return -1;
    }
}