cmake_minimum_required(VERSION 3.14)

if(APPLE)
    # MacOS specific settings
    set(CMAKE_OSX_SYSROOT /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
    
    # Include paths for standard libraries
    include_directories(SYSTEM
        /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include
        /Library/Developer/CommandLineTools/usr/include/c++/v1
        /opt/homebrew/include
    )
    
    # Linker flags
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L/opt/homebrew/lib")
endif()

project(AdvancedDetectionSystem VERSION 0.1.0 LANGUAGES CXX)

# Compiler settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Build type configuration
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Platform-specific optimizations
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
elseif(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2")
endif()

# Find OpenCV
find_package(OpenCV REQUIRED)
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found!")
endif()

# Display OpenCV information
message(STATUS "OpenCV Configuration:")
message(STATUS "    Version: ${OpenCV_VERSION}")
message(STATUS "    Libraries: ${OpenCV_LIBS}")
message(STATUS "    Include paths: ${OpenCV_INCLUDE_DIRS}")

# Include directories
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/include
)

# Source files
file(GLOB_RECURSE SOURCES 
    "${CMAKE_SOURCE_DIR}/src/*.cpp"
)

# Header files
file(GLOB_RECURSE HEADERS 
    "${CMAKE_SOURCE_DIR}/include/*.hpp"
)

# Main executable
add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS})

# Resource management
set(RESOURCE_DIR "${CMAKE_SOURCE_DIR}/resources")
set(BINARY_RESOURCE_DIR "${CMAKE_BINARY_DIR}/resources")

if(EXISTS "${RESOURCE_DIR}")
    add_custom_command(
        TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${RESOURCE_DIR} ${BINARY_RESOURCE_DIR}
    )
else()
    message(WARNING "Resources directory not found!")
    file(MAKE_DIRECTORY ${RESOURCE_DIR})
endif()

# Cascade file handling
if(APPLE)
    set(CASCADE_SEARCH_PATHS
        ${OpenCV_INCLUDE_DIRS}/../share/opencv4/haarcascades
        /opt/homebrew/share/opencv4/haarcascades
        /usr/local/share/opencv4/haarcascades
    )
else()
    set(CASCADE_SEARCH_PATHS
        ${OpenCV_INCLUDE_DIRS}/../share/opencv4/haarcascades
    )
endif()

find_file(FACE_CASCADE haarcascade_frontalface_default.xml
    PATHS ${CASCADE_SEARCH_PATHS}
)

if(FACE_CASCADE)
    add_custom_command(
        TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
        ${FACE_CASCADE} ${CMAKE_BINARY_DIR}
    )
else()
    message(WARNING "Cascade file not found! Searching in: ${CASCADE_SEARCH_PATHS}")
endif()

# Installation
install(TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION bin
)

if(EXISTS "${RESOURCE_DIR}")
    install(DIRECTORY ${RESOURCE_DIR}
        DESTINATION share/${PROJECT_NAME}
    )
endif()