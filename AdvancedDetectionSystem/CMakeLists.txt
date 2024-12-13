cmake_minimum_required(VERSION 3.10)
project(AdvancedDetectionSystem VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenCV bulma
find_package(OpenCV REQUIRED)

# Include dizinleri
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(include)

# Kaynak dosyaları
file(GLOB SOURCES
    src/*.cpp
)

# Header dosyaları
file(GLOB HEADERS
    include/*.hpp
)

# Executable oluştur
add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

# OpenCV kütüphanelerini bağla
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

# Resources klasörünü kopyala
if(EXISTS "${CMAKE_SOURCE_DIR}/resources")
    file(COPY ${CMAKE_SOURCE_DIR}/resources
         DESTINATION ${CMAKE_BINARY_DIR})
endif()
