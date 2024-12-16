#include "FeatureControl.hpp"

FeatureControl::FeatureControl() {
    // Özellikler için başlangıç durumlarının ayarlanması
    features = {
        {Feature::SHOW_FPS, true},
        {Feature::SHOW_TRAJECTORY, true},
        {Feature::SHOW_DISTANCE, true},
        {Feature::SHOW_DIRECTION, true},
        {Feature::SHOW_WATER_LEVEL, true},
        {Feature::SHOW_SKIN_DETECTION, true},
        {Feature::SHOW_SEA_DETECTION, true},
        {Feature::SHOW_HUMAN_DETECTION, true},
        {Feature::SHOW_BOAT_DETECTION, true},
        {Feature::ENABLE_NOTIFICATIONS, true},
        {Feature::SHOW_OBJECT_ID, true},
        {Feature::SHOW_CONFIDENCE, true},
        {Feature::SHOW_SPEED, true},
        {Feature::SHOW_ACCELERATION, true},
        {Feature::SHOW_HEAT_MAP, false},
        {Feature::ENABLE_NIGHT_MODE, false},
        {Feature::SHOW_GRID, false},
        {Feature::SHOW_COORDINATES, false},
        {Feature::ENABLE_RECORDING, false},
        {Feature::SHOW_TIMESTAMP, true}
    };

    // Tuş atamaları - Her özellik için kısayol tuşu tanımlama
    featureKeys = {
        {Feature::SHOW_FPS, '1'},
        {Feature::SHOW_TRAJECTORY, '2'},
        {Feature::SHOW_DISTANCE, '3'},
        {Feature::SHOW_DIRECTION, '4'},
        {Feature::SHOW_WATER_LEVEL, '5'},
        {Feature::SHOW_SKIN_DETECTION, '6'},
        {Feature::SHOW_SEA_DETECTION, '7'},
        {Feature::SHOW_HUMAN_DETECTION, '8'},
        {Feature::SHOW_BOAT_DETECTION, '9'},
        {Feature::ENABLE_NOTIFICATIONS, '0'},
        {Feature::SHOW_OBJECT_ID, 'i'},
        {Feature::SHOW_CONFIDENCE, 'c'},
        {Feature::SHOW_SPEED, 's'},
        {Feature::SHOW_ACCELERATION, 'a'},
        {Feature::SHOW_HEAT_MAP, 'h'},
        {Feature::ENABLE_NIGHT_MODE, 'n'},
        {Feature::SHOW_GRID, 'g'},
        {Feature::SHOW_COORDINATES, 'x'},
        {Feature::ENABLE_RECORDING, 'r'},
        {Feature::SHOW_TIMESTAMP, 't'}
    };

    // Görüntülenecek özellik isimleri
    featureNames = {
        {Feature::SHOW_FPS, "FPS Gösterimi"},
        {Feature::SHOW_TRAJECTORY, "Yörünge Takibi"},
        {Feature::SHOW_DISTANCE, "Mesafe Gösterimi"},
        {Feature::SHOW_DIRECTION, "Yön Gösterimi"},
        {Feature::SHOW_WATER_LEVEL, "Su Seviyesi"},
        {Feature::SHOW_SKIN_DETECTION, "Cilt Tespiti"},
        {Feature::SHOW_SEA_DETECTION, "Deniz Tespiti"},
        {Feature::SHOW_HUMAN_DETECTION, "İnsan Tespiti"},
        {Feature::SHOW_BOAT_DETECTION, "Tekne Tespiti"},
        {Feature::ENABLE_NOTIFICATIONS, "Bildirimler"},
        {Feature::SHOW_OBJECT_ID, "Nesne ID"},
        {Feature::SHOW_CONFIDENCE, "Güven Skoru"},
        {Feature::SHOW_SPEED, "Hız Gösterimi"},
        {Feature::SHOW_ACCELERATION, "İvme Gösterimi"},
        {Feature::SHOW_HEAT_MAP, "Isı Haritası"},
        {Feature::ENABLE_NIGHT_MODE, "Gece Modu"},
        {Feature::SHOW_GRID, "Izgara Gösterimi"},
        {Feature::SHOW_COORDINATES, "Koordinat Gösterimi"},
        {Feature::ENABLE_RECORDING, "Kayıt"},
        {Feature::SHOW_TIMESTAMP, "Zaman Damgası"}
    };
}

void FeatureControl::toggleFeature(Feature feature) {
    // O(1) karmaşıklıkta özellik durumu değiştirme
    if (features.find(feature) != features.end()) {
        features[feature] = !features[feature];
    }
}

void FeatureControl::toggleFeature(char key) {
    // Tuşa basılan özelliği bul ve durumunu değiştir
    for (const auto& [feature, featureKey] : featureKeys) {
        if (featureKey == key) {
            toggleFeature(feature);
            break;
        }
    }
}

bool FeatureControl::isEnabled(Feature feature) const {
    // Thread-safe özellik durumu kontrolü
    auto it = features.find(feature);
    return it != features.end() && it->second;
}

void FeatureControl::displayHelp(cv::Mat& frame) {
    // Görsel ayarlar
    const int lineHeight = 20;
    int startY = frame.rows - 400; // Tüm özellikleri gösterebilmek için yüksek başlangıç
    cv::Point startPoint(10, startY);
    const cv::Scalar textColor(255, 255, 255);
    const cv::Scalar activeColor(0, 255, 0);
    const cv::Scalar inactiveColor(0, 0, 255);
    const double fontScale = 0.5;
    const int thickness = 1;

    // Yarı saydam siyah arka plan için overlay
    cv::Mat overlay = frame.clone();
    cv::rectangle(overlay, 
                 cv::Point(5, startY - 25),
                 cv::Point(250, startY + 420),
                 cv::Scalar(0, 0, 0), 
                 -1);
    cv::addWeighted(overlay, 0.7, frame, 0.3, 0, frame);

    // Başlık
    cv::putText(frame, "Kontroller:", startPoint,
               cv::FONT_HERSHEY_SIMPLEX, fontScale, textColor, thickness);
    startPoint.y += lineHeight;

    // Tüm özellikleri listele
    for (const auto& [feature, key] : featureKeys) {
        // Özellik durumuna göre renk seç
        bool isActive = isEnabled(feature);
        cv::Scalar color = isActive ? activeColor : inactiveColor;
        
        // Özellik adı ve durum metni
        std::string status = isActive ? "(AÇIK)" : "(KAPALI)";
        std::string text = std::string(1, key) + ": " + 
                          getFeatureName(feature) + " " + status;
        
        // Metni ekrana yaz
        cv::putText(frame, text, startPoint, 
                   cv::FONT_HERSHEY_SIMPLEX, fontScale,
                   color, thickness);
        
        startPoint.y += lineHeight;
    }
}

void FeatureControl::displayFeatureStatus(cv::Mat& frame) {
    // Aktif özellikleri ekranın sağ üst köşesinde göster
    int startY = 30;
    const int lineHeight = 20;
    cv::Scalar activeColor(0, 255, 0);
    
    for (const auto& [feature, enabled] : features) {
        if (enabled) {
            cv::putText(frame, 
                       getFeatureName(feature) + ": AÇIK",
                       cv::Point(frame.cols - 200, startY),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5,
                       activeColor, 1);
            startY += lineHeight;
        }
    }
}

std::string FeatureControl::getFeatureName(Feature feature) const {
    // Özellik ismini bul ve döndür, bulunamazsa varsayılan metin döndür
    auto it = featureNames.find(feature);
    return it != featureNames.end() ? it->second : "Bilinmeyen Özellik";
}