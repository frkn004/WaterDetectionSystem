#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <deque>
#include <memory>
#include <mutex>
#include <chrono>

class SeaDetector {
public:
    struct TemperatureData {
        float value{0.0f};
        float confidence{0.0f};
        std::chrono::system_clock::time_point timestamp;
        
        bool isValid() const {
            return value > -273.15f && confidence >= 0.0f && confidence <= 1.0f;
        }
    };

    struct WaveMetrics {
        float height{0.0f};        // metre cinsinden yükseklik
        float period{0.0f};        // saniye cinsinden periyot
        float direction{0.0f};     // derece cinsinden yön
        float energy{0.0f};        // dalga enerjisi (J/m²)
        float frequency{0.0f};     // Hz cinsinden frekans
        float velocity{0.0f};      // m/s cinsinden hız
        float turbulence{0.0f};    // türbülans seviyesi (0-1)
        bool isDangerous{false};   // tehlike durumu

        bool isValid() const {
            return height >= 0 && period > 0 && 
                   direction >= 0 && direction <= 360 &&
                   frequency >= 0 && velocity >= 0 &&
                   turbulence >= 0 && turbulence <= 1;
        }
    };

    struct WaterLevel {
        float currentLevel{0.0f};    // metre cinsinden mevcut seviye
        float changeRate{0.0f};      // metre/saat değişim hızı
        float referenceLevel{0.0f};  // referans seviye
        bool isRising{false};        // yükseliyor mu?
        std::chrono::system_clock::time_point lastUpdate;

        bool isValid() const {
            return currentLevel >= 0 && std::abs(changeRate) < 10.0f;
        }
    };

    struct SeaInfo {
        bool isDetected{false};
        float waveIntensity{0.0f};           // dalga yoğunluğu (0-1)
        cv::Point2f centerPoint{0.0f, 0.0f}; // deniz merkez noktası
        std::vector<cv::Point> contour;       // deniz kontur noktaları
        float area{0.0f};                    // deniz alanı (piksel²)
        WaveMetrics waveMetrics;             // dalga metrikleri
        WaterLevel waterLevel;               // su seviyesi bilgisi
        float visibility{0.0f};              // görünürlük (metre)
        TemperatureData temperature;         // sıcaklık bilgisi
        std::chrono::system_clock::time_point timestamp;

        bool isValid() const {
            return area >= 0 && visibility >= 0 && 
                   temperature.isValid() && waveMetrics.isValid() &&
                   waterLevel.isValid();
        }
    };

    struct Config {
        // Su tespiti için renk aralıkları
        cv::Scalar waterLowerBound{90, 40, 40};   // HSV
        cv::Scalar waterUpperBound{130, 255, 255}; // HSV
        
        // Temel parametreler
        float minArea{1000.0f};              // minimum su alanı
        float maxWaveHeight{2.0f};           // maksimum dalga yüksekliği
        float dangerousWaveHeight{1.5f};     // tehlikeli dalga yüksekliği
        float dangerousWavePeriod{5.0f};     // tehlikeli dalga periyodu
        float waterLevelThreshold{0.5f};     // su seviyesi değişim eşiği
        float temperatureThreshold{1.0f};    // sıcaklık değişim eşiği
        
        // Görüntü işleme parametreleri
        int morphSize{3};                    // morfolojik işlem boyutu
        bool useAdaptiveThreshold{true};     // adaptif eşikleme kullan
        int blurSize{5};                     // bulanıklaştırma boyutu
        float contrastAlpha{1.2f};           // kontrast artırma katsayısı
        
        // Dalga analizi parametreleri
        int fftSize{512};                    // FFT boyutu
        float samplingRate{30.0f};           // örnekleme hızı (FPS)
        float minWaveFreq{0.1f};            // minimum dalga frekansı
        float maxWaveFreq{2.0f};            // maksimum dalga frekansı
        
        void validate() const {
            if (minArea <= 0) 
                throw std::invalid_argument("Invalid minArea");
            if (maxWaveHeight <= 0)
                throw std::invalid_argument("Invalid maxWaveHeight");
            if (morphSize <= 0 || morphSize % 2 == 0)
                throw std::invalid_argument("Invalid morphSize");
            if (blurSize <= 0 || blurSize % 2 == 0)
                throw std::invalid_argument("Invalid blurSize");
            if (fftSize <= 0 || (fftSize & (fftSize - 1)) != 0)
                throw std::invalid_argument("FFT size must be power of 2");
        }
    };

    struct CalibrationData {
        float knownTemperature{20.0f};      // bilinen sıcaklık değeri
        float knownWaterLevel{0.0f};        // bilinen su seviyesi
        cv::Point2f referencePoints[4];      // referans noktaları
        float pixelToMeterRatio{1.0f};      // piksel/metre oranı
        std::chrono::system_clock::time_point timestamp;
    };

    class SeaDetectionError : public std::runtime_error {
        using std::runtime_error::runtime_error;
    };

    // Constructors
    SeaDetector();
    explicit SeaDetector(const Config& config);
    ~SeaDetector() = default;

    // Copy/Move prevention
    SeaDetector(const SeaDetector&) = delete;
    SeaDetector& operator=(const SeaDetector&) = delete;
    SeaDetector(SeaDetector&&) = delete;
    SeaDetector& operator=(SeaDetector&&) = delete;

    // Main functions
    SeaInfo detectSea(const cv::Mat& frame);
    void calibrate(const CalibrationData& calibData);
    void setConfig(const Config& newConfig);
    Config getConfig() const;
    void visualizeResults(cv::Mat& frame, const SeaInfo& info) const;
    std::vector<WaveMetrics> getWaveHistory() const;
    std::vector<TemperatureData> getTemperatureHistory() const;

private:
    mutable std::mutex mtx;
    Config config;
    CalibrationData calibration;
    
    // Görüntü işleme bufferları
    cv::Mat kernel;
    cv::Mat prevFrame;
    cv::Mat prevGray;
    cv::Mat preprocessBuffer;
    cv::Mat waterMaskBuffer;
    cv::Mat fftBuffer;
    std::vector<std::vector<cv::Point>> contoursBuffer;

    // Veri depolama
    std::deque<WaveMetrics> waveHistory;
    std::deque<TemperatureData> temperatureHistory;
    std::deque<WaterLevel> waterLevelHistory;
    static constexpr size_t MAX_HISTORY_SIZE = 300; // 10 saniye @ 30fps

    // Private helper functions
    cv::Mat preprocessFrame(const cv::Mat& frame);
    cv::Mat detectWaterRegion(const cv::Mat& frame);
    WaveMetrics analyzeWaves(const cv::Mat& frame, const cv::Mat& waterMask);
    WaterLevel detectWaterLevel(const cv::Mat& frame, const cv::Mat& waterMask);
    TemperatureData measureTemperature(const cv::Mat& frame, const cv::Mat& waterMask);
    cv::Mat computeOpticalFlow(const cv::Mat& current);
    void performFFTAnalysis(const cv::Mat& data);
    float calculateWaveFrequency(const cv::Mat& flow);
    cv::Point2f calculateWaveVelocity(const cv::Mat& flow);
    float calculateWaveEnergy(float height, float period);
    float calculateTurbulence(const cv::Mat& frame, const cv::Mat& waterMask);
    float estimateVisibility(const cv::Mat& frame, const cv::Mat& waterMask);
    std::vector<cv::Point> findLargestContour(const cv::Mat& mask);
    cv::Point2f calculateCenterPoint(const std::vector<cv::Point>& contour) const;
    bool evaluateWaveConditions(const WaveMetrics& metrics) const;
    void updateHistories(const WaveMetrics& wave, const TemperatureData& temp, 
                        const WaterLevel& level);
    float pixelsToMeters(float pixels) const;
    float metersToPixels(float meters) const;
};