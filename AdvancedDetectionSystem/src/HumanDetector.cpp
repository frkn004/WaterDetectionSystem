#include "HumanDetector.hpp"
#include <fstream>
#include <iostream>

HumanDetector::HumanDetector() :
    CONFIDENCE_THRESHOLD(0.5f),
    NMS_THRESHOLD(0.4f),
    CONF_THRESHOLD(0.5f) {
    
    // HOG dedektörünü başlat
    hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());
    
    // Varsayılan kamera parametreleri
    params.focalLength = 615.0f;
    params.realHeight = 1700.0f;  // mm cinsinden ortalama insan boyu
    params.verticalFOV = 58.0f;   // derece cinsinden
    
    // Kamera matrisi başlangıç değerleri
    params.cameraMatrix = (cv::Mat_<double>(3,3) <<
        615.0, 0, 320.0,
        0, 615.0, 240.0,
        0, 0, 1);
    params.distCoeffs = cv::Mat::zeros(5, 1, CV_64F);
}

void HumanDetector::loadDeepLearningModel(const std::string& modelPath,
                                        const std::string& configPath,
                                        const std::string& classesPath) {
    try {
        net = cv::dnn::readNet(modelPath, configPath);
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        
        // Sınıf isimlerini yükle
        std::ifstream ifs(classesPath);
        std::string line;
        while (std::getline(ifs, line)) {
            classes.push_back(line);
        }
    } catch (const cv::Exception& e) {
        std::cerr << "Could not load deep learning model: " << e.what() << std::endl;
    }
}

void HumanDetector::setCameraParameters(float focalLength, float realHeight,
                                      float verticalFOV) {
    params.focalLength = focalLength;
    params.realHeight = realHeight;
    params.verticalFOV = verticalFOV;
    
    // Kamera matrisini güncelle
    params.cameraMatrix.at<double>(0,0) = focalLength;
    params.cameraMatrix.at<double>(1,1) = focalLength;
}

std::vector<HumanDetector::DetectionInfo> HumanDetector::detectHumans(const cv::Mat& frame) {
    std::vector<DetectionInfo> detections;
    
    try {
        // Görüntüyü küçült ve gri tonlamaya çevir
        cv::Mat resized, gray;
        cv::resize(frame, resized, cv::Size(), 0.5, 0.5);
        cv::cvtColor(resized, gray, cv::COLOR_BGR2GRAY);
        cv::equalizeHist(gray, gray); // Kontrast iyileştirme
        
        // HOG tespiti
        std::vector<cv::Rect> found;
        std::vector<double> weights;
        hog.detectMultiScale(gray, found, weights, 0,
                            cv::Size(8,8), cv::Size(32,32), 1.05, 2);
        
        static cv::Rect previousBox;
        
        // HOG tespitlerini ekle
        for (size_t i = 0; i < found.size(); i++) {
            if (weights[i] > CONFIDENCE_THRESHOLD) {
                DetectionInfo info;
                // Orijinal boyuta geri dönüştür
                info.box = cv::Rect(found[i].x * 2, found[i].y * 2,
                                  found[i].width * 2, found[i].height * 2);
                info.confidence = weights[i];
                info.label = "Person (HOG)";
                info.distance = calculateDistance(info.box);
                info.direction = calculateDirection(info.box, previousBox);
                info.isMoving = isPersonMoving(info.box, previousBox);
                detections.push_back(info);
                previousBox = info.box;
            }
        }
        
        // DNN tespiti (eğer model yüklüyse)
        if (!net.empty()) {
            cv::Mat blob;
            cv::dnn::blobFromImage(frame, blob, 1/255.0, cv::Size(416,416));
            net.setInput(blob);
            
            std::vector<cv::Mat> outs;
            net.forward(outs, getOutputsNames(net));
            
            postprocess(frame, outs, detections);
        }
    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV Error in detectHumans: " << e.what() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error in detectHumans: " << e.what() << std::endl;
    }
    
    return detections;
}

float HumanDetector::calculateDistance(const cv::Rect& box) {
    if (box.height <= 0) return -1.0f;
    
    // Gelişmiş mesafe hesaplama
    float distance = (params.realHeight * params.focalLength) / (box.height * 1000.0f);
    
    // Perspektif düzeltmesi
    float verticalPosition = box.y + box.height/2.0f;
    float imageHeight = params.cameraMatrix.at<double>(1,2) * 2;
    float angleCorrection = 1.0f + (verticalPosition - imageHeight/2) * 0.001f;
    
    // FOV bazlı düzeltme
    float verticalAngle = (verticalPosition / imageHeight - 0.5f) * params.verticalFOV;
    float cosCorrection = std::cos(verticalAngle * CV_PI / 180.0f);
    
    return distance * angleCorrection * cosCorrection;
}

cv::Point2f HumanDetector::calculateDirection(const cv::Rect& currentBox,
                                            const cv::Rect& previousBox) {
    cv::Point2f direction(0, 0);
    if (previousBox.width > 0 && previousBox.height > 0) {
        direction.x = (currentBox.x + currentBox.width/2.0f) -
                     (previousBox.x + previousBox.width/2.0f);
        direction.y = (currentBox.y + currentBox.height/2.0f) -
                     (previousBox.y + previousBox.height/2.0f);
        
        // Normalize
        float norm = std::sqrt(direction.x*direction.x + direction.y*direction.y);
        if (norm > 0) {
            direction.x /= norm;
            direction.y /= norm;
        }
    }
    return direction;
}

bool HumanDetector::isPersonMoving(const cv::Rect& currentBox,
                                 const cv::Rect& previousBox,
                                 float threshold) {
    if (previousBox.width > 0 && previousBox.height > 0) {
        float dx = (currentBox.x + currentBox.width/2.0f) -
                  (previousBox.x + previousBox.width/2.0f);
        float dy = (currentBox.y + currentBox.height/2.0f) -
                  (previousBox.y + previousBox.height/2.0f);
        
        float movement = std::sqrt(dx*dx + dy*dy);
        return movement > threshold;
    }
    return false;
}

std::vector<std::string> HumanDetector::getOutputsNames(const cv::dnn::Net& net) {
    std::vector<std::string> names;
    std::vector<int> outLayers = net.getUnconnectedOutLayers();
    std::vector<std::string> layersNames = net.getLayerNames();
    names.resize(outLayers.size());
    for (size_t i = 0; i < outLayers.size(); ++i) {
        names[i] = layersNames[outLayers[i] - 1];
    }
    return names;
}

void HumanDetector::postprocess(const cv::Mat& frame,
                              const std::vector<cv::Mat>& outs,
                              std::vector<DetectionInfo>& detections) {
    static cv::Rect previousDnnBox;
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    
    for (const auto& out : outs) {
        for (int i = 0; i < out.rows; ++i) {
            const float* data = out.ptr<float>(i);
            float confidence = data[4];
            
            if (confidence > CONF_THRESHOLD) {
                int centerX = static_cast<int>(data[0] * frame.cols);
                int centerY = static_cast<int>(data[1] * frame.rows);
                int width = static_cast<int>(data[2] * frame.cols);
                int height = static_cast<int>(data[3] * frame.rows);
                int left = centerX - width/2;
                int top = centerY - height/2;
                
                classIds.push_back(static_cast<int>(data[5]));
                confidences.push_back(confidence);
                boxes.push_back(cv::Rect(left, top, width, height));
            }
        }
    }
    
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD, indices);
    
    for (size_t i = 0; i < indices.size(); ++i) {
        int idx = indices[i];
        if (classIds[idx] == 0) {  // COCO dataset'inde 0 person sınıfıdır
            DetectionInfo info;
            info.box = boxes[idx];
            info.confidence = confidences[idx];
            info.label = "Person (DNN)";
            info.distance = calculateDistance(info.box);
            info.direction = calculateDirection(info.box, previousDnnBox);
            info.isMoving = isPersonMoving(info.box, previousDnnBox);
            detections.push_back(info);
            previousDnnBox = info.box;
        }
    }
}

bool HumanDetector::detectHumanPose(const cv::Mat& frame, std::vector<cv::Point>& keypoints) {
    // OpenPose veya MediaPipe kullanarak vücut pozisyonu tespiti
    // Bu, yüzme hareketlerini daha iyi tespit etmemizi sağlar
    return false; // Şimdilik boş implementasyon
}

float HumanDetector::calculateConfidenceScore(const cv::Rect& detection) {
    // Tespit güvenilirlik skorunu hesapla
    // Dalga ve ışık yansımalarını göz önünde bulundur
    return 0.0f; // Şimdilik boş implementasyon
}

std::vector<TrackedObject> HumanDetector::detect(const cv::Mat& frame) {
    std::vector<TrackedObject> detections;
    
    try {
        // Görüntüyü ön işle
        cv::Mat resized, gray;
        cv::resize(frame, resized, cv::Size(), 0.5, 0.5);
        cv::cvtColor(resized, gray, cv::COLOR_BGR2GRAY);
        cv::equalizeHist(gray, gray); // Kontrast iyileştirme
        
        // HOG tespiti
        std::vector<cv::Rect> found;
        std::vector<double> weights;
        hog.detectMultiScale(gray, found, weights, 0,
                            cv::Size(8,8), cv::Size(32,32), 1.05, 2);
        
        // Tespitleri işle
        for (size_t i = 0; i < found.size(); i++) {
            if (weights[i] > CONFIDENCE_THRESHOLD) {
                TrackedObject obj;
                // Orijinal boyuta geri dönüştür
                obj.boundingBox = cv::Rect(found[i].x * 2, found[i].y * 2,
                                         found[i].width * 2, found[i].height * 2);
                obj.confidence = weights[i];
                obj.distance = calculateDistance(obj.boundingBox);
                obj.direction = calculateDirection(obj.boundingBox, lastDetectedBox);
                obj.isMoving = isPersonMoving(obj.boundingBox, lastDetectedBox);
                
                detections.push_back(obj);
                lastDetectedBox = obj.boundingBox;
            }
        }
        
    } catch (const cv::Exception& e) {
        std::cerr << "İnsan tespiti sırasında hata: " << e.what() << std::endl;
    }
    
    return detections;
}

void HumanDetector::loadYOLOModel(const std::string& modelPath, const std::string& configPath) {
    try {
        net = cv::dnn::readNetFromDarknet(configPath, modelPath);
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        
        std::cout << "YOLO modeli başarıyla yüklendi.\n";
    } catch (const cv::Exception& e) {
        std::cerr << "YOLO modeli yüklenirken hata: " << e.what() << std::endl;
        throw;
    }
}

std::vector<TrackedObject> HumanDetector::detectWithYOLO(const cv::Mat& frame) {
    std::vector<TrackedObject> detections;
    
    try {
        // Görüntüyü blob formatına dönüştür
        cv::Mat blob;
        cv::dnn::blobFromImage(frame, blob, 1/255.0, 
                              cv::Size(INPUT_WIDTH, INPUT_HEIGHT),
                              cv::Scalar(0,0,0), true, false);
        
        net.setInput(blob);
        
        // Forward pass
        std::vector<cv::Mat> outs;
        net.forward(outs, getOutputLayerNames());
        
        // Tespitleri işle
        std::vector<int> classIds;
        std::vector<float> confidences;
        std::vector<cv::Rect> boxes;
        
        for (const auto& out : outs) {
            for (int i = 0; i < out.rows; ++i) {
                const float* data = out.ptr<float>(i);
                float confidence = data[4];
                
                if (confidence > CONF_THRESHOLD) {
                    // COCO veri setinde insan sınıfı ID'si 0'dır
                    if (static_cast<int>(data[5]) == 0) { // Sadece insan sınıfı
                        float centerX = data[0] * frame.cols;
                        float centerY = data[1] * frame.rows;
                        float width = data[2] * frame.cols;
                        float height = data[3] * frame.rows;
                        
                        cv::Rect box;
                        box.x = centerX - width/2;
                        box.y = centerY - height/2;
                        box.width = width;
                        box.height = height;
                        
                        boxes.push_back(box);
                        confidences.push_back(confidence);
                        classIds.push_back(0);
                    }
                }
            }
        }
        
        // Non-maximum suppression uygula
        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, CONF_THRESHOLD, 
                         NMS_THRESHOLD, indices);
        
        // Tespit edilen nesneleri TrackedObject'e dönüştür
        for (size_t i = 0; i < indices.size(); ++i) {
            int idx = indices[i];
            TrackedObject obj;
            obj.boundingBox = boxes[idx];
            obj.confidence = confidences[idx];
            obj.distance = calculateDistance(boxes[idx]);
            obj.direction = calculateDirection(boxes[idx], lastDetectedBox);
            obj.isMoving = isPersonMoving(boxes[idx], lastDetectedBox);
            
            detections.push_back(obj);
            lastDetectedBox = boxes[idx];
        }
        
    } catch (const cv::Exception& e) {
        std::cerr << "YOLO tespiti sırasında hata: " << e.what() << std::endl;
    }
    
    return detections;
}

std::vector<std::string> HumanDetector::getOutputLayerNames() {
    std::vector<std::string> names;
    std::vector<int> outLayers = net.getUnconnectedOutLayers();
    std::vector<std::string> layersNames = net.getLayerNames();
    names.resize(outLayers.size());
    for (size_t i = 0; i < outLayers.size(); ++i) {
        names[i] = layersNames[outLayers[i] - 1];
    }
    return names;
}
