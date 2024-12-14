cmake_minimum_required(VERSION 3.10)
project(AdvancedDetectionSystem VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenCV'yi bul
find_package(OpenCV REQUIRED)
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV bulunamadı!")
endif()

# OpenCV bilgilerini göster
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

# Include dizinleri
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/include)

# Kaynak dosyaları
file(GLOB SOURCES
    "${CMAKE_SOURCE_DIR}/src/*.cpp"
)

# Header dosyaları
file(GLOB HEADERS
    "${CMAKE_SOURCE_DIR}/include/*.hpp"
)

# Executable oluştur
add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

# OpenCV kütüphanelerini bağla
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS})

# Resources klasörünü kopyala
if(EXISTS "${CMAKE_SOURCE_DIR}/resources")
    file(COPY ${CMAKE_SOURCE_DIR}/resources
         DESTINATION ${CMAKE_BINARY_DIR})
endif()
