# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.31

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /opt/homebrew/bin/cmake

# The command to remove a file.
RM = /opt/homebrew/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/furkansevinc/Desktop/AdvancedDetectionSystem/AdvancedDetectionSystem

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/furkansevinc/Desktop/AdvancedDetectionSystem/AdvancedDetectionSystem/build

# Include any dependencies generated for this target.
include CMakeFiles/AdvancedDetectionSystem.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/AdvancedDetectionSystem.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/AdvancedDetectionSystem.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/AdvancedDetectionSystem.dir/flags.make

CMakeFiles/AdvancedDetectionSystem.dir/codegen:
.PHONY : CMakeFiles/AdvancedDetectionSystem.dir/codegen

CMakeFiles/AdvancedDetectionSystem.dir/src/BoatDetector.cpp.o: CMakeFiles/AdvancedDetectionSystem.dir/flags.make
CMakeFiles/AdvancedDetectionSystem.dir/src/BoatDetector.cpp.o: /Users/furkansevinc/Desktop/AdvancedDetectionSystem/AdvancedDetectionSystem/src/BoatDetector.cpp
CMakeFiles/AdvancedDetectionSystem.dir/src/BoatDetector.cpp.o: CMakeFiles/AdvancedDetectionSystem.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/furkansevinc/Desktop/AdvancedDetectionSystem/AdvancedDetectionSystem/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/AdvancedDetectionSystem.dir/src/BoatDetector.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/AdvancedDetectionSystem.dir/src/BoatDetector.cpp.o -MF CMakeFiles/AdvancedDetectionSystem.dir/src/BoatDetector.cpp.o.d -o CMakeFiles/AdvancedDetectionSystem.dir/src/BoatDetector.cpp.o -c /Users/furkansevinc/Desktop/AdvancedDetectionSystem/AdvancedDetectionSystem/src/BoatDetector.cpp

CMakeFiles/AdvancedDetectionSystem.dir/src/BoatDetector.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/AdvancedDetectionSystem.dir/src/BoatDetector.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/furkansevinc/Desktop/AdvancedDetectionSystem/AdvancedDetectionSystem/src/BoatDetector.cpp > CMakeFiles/AdvancedDetectionSystem.dir/src/BoatDetector.cpp.i

CMakeFiles/AdvancedDetectionSystem.dir/src/BoatDetector.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/AdvancedDetectionSystem.dir/src/BoatDetector.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/furkansevinc/Desktop/AdvancedDetectionSystem/AdvancedDetectionSystem/src/BoatDetector.cpp -o CMakeFiles/AdvancedDetectionSystem.dir/src/BoatDetector.cpp.s

CMakeFiles/AdvancedDetectionSystem.dir/src/FeatureControl.cpp.o: CMakeFiles/AdvancedDetectionSystem.dir/flags.make
CMakeFiles/AdvancedDetectionSystem.dir/src/FeatureControl.cpp.o: /Users/furkansevinc/Desktop/AdvancedDetectionSystem/AdvancedDetectionSystem/src/FeatureControl.cpp
CMakeFiles/AdvancedDetectionSystem.dir/src/FeatureControl.cpp.o: CMakeFiles/AdvancedDetectionSystem.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/furkansevinc/Desktop/AdvancedDetectionSystem/AdvancedDetectionSystem/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/AdvancedDetectionSystem.dir/src/FeatureControl.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/AdvancedDetectionSystem.dir/src/FeatureControl.cpp.o -MF CMakeFiles/AdvancedDetectionSystem.dir/src/FeatureControl.cpp.o.d -o CMakeFiles/AdvancedDetectionSystem.dir/src/FeatureControl.cpp.o -c /Users/furkansevinc/Desktop/AdvancedDetectionSystem/AdvancedDetectionSystem/src/FeatureControl.cpp

CMakeFiles/AdvancedDetectionSystem.dir/src/FeatureControl.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/AdvancedDetectionSystem.dir/src/FeatureControl.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/furkansevinc/Desktop/AdvancedDetectionSystem/AdvancedDetectionSystem/src/FeatureControl.cpp > CMakeFiles/AdvancedDetectionSystem.dir/src/FeatureControl.cpp.i

CMakeFiles/AdvancedDetectionSystem.dir/src/FeatureControl.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/AdvancedDetectionSystem.dir/src/FeatureControl.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/furkansevinc/Desktop/AdvancedDetectionSystem/AdvancedDetectionSystem/src/FeatureControl.cpp -o CMakeFiles/AdvancedDetectionSystem.dir/src/FeatureControl.cpp.s

CMakeFiles/AdvancedDetectionSystem.dir/src/HumanDetector.cpp.o: CMakeFiles/AdvancedDetectionSystem.dir/flags.make
CMakeFiles/AdvancedDetectionSystem.dir/src/HumanDetector.cpp.o: /Users/furkansevinc/Desktop/AdvancedDetectionSystem/AdvancedDetectionSystem/src/HumanDetector.cpp
CMakeFiles/AdvancedDetectionSystem.dir/src/HumanDetector.cpp.o: CMakeFiles/AdvancedDetectionSystem.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/furkansevinc/Desktop/AdvancedDetectionSystem/AdvancedDetectionSystem/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/AdvancedDetectionSystem.dir/src/HumanDetector.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/AdvancedDetectionSystem.dir/src/HumanDetector.cpp.o -MF CMakeFiles/AdvancedDetectionSystem.dir/src/HumanDetector.cpp.o.d -o CMakeFiles/AdvancedDetectionSystem.dir/src/HumanDetector.cpp.o -c /Users/furkansevinc/Desktop/AdvancedDetectionSystem/AdvancedDetectionSystem/src/HumanDetector.cpp

CMakeFiles/AdvancedDetectionSystem.dir/src/HumanDetector.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/AdvancedDetectionSystem.dir/src/HumanDetector.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/furkansevinc/Desktop/AdvancedDetectionSystem/AdvancedDetectionSystem/src/HumanDetector.cpp > CMakeFiles/AdvancedDetectionSystem.dir/src/HumanDetector.cpp.i

CMakeFiles/AdvancedDetectionSystem.dir/src/HumanDetector.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/AdvancedDetectionSystem.dir/src/HumanDetector.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/furkansevinc/Desktop/AdvancedDetectionSystem/AdvancedDetectionSystem/src/HumanDetector.cpp -o CMakeFiles/AdvancedDetectionSystem.dir/src/HumanDetector.cpp.s

CMakeFiles/AdvancedDetectionSystem.dir/src/NotificationSystem.cpp.o: CMakeFiles/AdvancedDetectionSystem.dir/flags.make
CMakeFiles/AdvancedDetectionSystem.dir/src/NotificationSystem.cpp.o: /Users/furkansevinc/Desktop/AdvancedDetectionSystem/AdvancedDetectionSystem/src/NotificationSystem.cpp
CMakeFiles/AdvancedDetectionSystem.dir/src/NotificationSystem.cpp.o: CMakeFiles/AdvancedDetectionSystem.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/furkansevinc/Desktop/AdvancedDetectionSystem/AdvancedDetectionSystem/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/AdvancedDetectionSystem.dir/src/NotificationSystem.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/AdvancedDetectionSystem.dir/src/NotificationSystem.cpp.o -MF CMakeFiles/AdvancedDetectionSystem.dir/src/NotificationSystem.cpp.o.d -o CMakeFiles/AdvancedDetectionSystem.dir/src/NotificationSystem.cpp.o -c /Users/furkansevinc/Desktop/AdvancedDetectionSystem/AdvancedDetectionSystem/src/NotificationSystem.cpp

CMakeFiles/AdvancedDetectionSystem.dir/src/NotificationSystem.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/AdvancedDetectionSystem.dir/src/NotificationSystem.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/furkansevinc/Desktop/AdvancedDetectionSystem/AdvancedDetectionSystem/src/NotificationSystem.cpp > CMakeFiles/AdvancedDetectionSystem.dir/src/NotificationSystem.cpp.i

CMakeFiles/AdvancedDetectionSystem.dir/src/NotificationSystem.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/AdvancedDetectionSystem.dir/src/NotificationSystem.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/furkansevinc/Desktop/AdvancedDetectionSystem/AdvancedDetectionSystem/src/NotificationSystem.cpp -o CMakeFiles/AdvancedDetectionSystem.dir/src/NotificationSystem.cpp.s

CMakeFiles/AdvancedDetectionSystem.dir/src/ObjectDetector.cpp.o: CMakeFiles/AdvancedDetectionSystem.dir/flags.make
CMakeFiles/AdvancedDetectionSystem.dir/src/ObjectDetector.cpp.o: /Users/furkansevinc/Desktop/AdvancedDetectionSystem/AdvancedDetectionSystem/src/ObjectDetector.cpp
CMakeFiles/AdvancedDetectionSystem.dir/src/ObjectDetector.cpp.o: CMakeFiles/AdvancedDetectionSystem.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/furkansevinc/Desktop/AdvancedDetectionSystem/AdvancedDetectionSystem/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/AdvancedDetectionSystem.dir/src/ObjectDetector.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/AdvancedDetectionSystem.dir/src/ObjectDetector.cpp.o -MF CMakeFiles/AdvancedDetectionSystem.dir/src/ObjectDetector.cpp.o.d -o CMakeFiles/AdvancedDetectionSystem.dir/src/ObjectDetector.cpp.o -c /Users/furkansevinc/Desktop/AdvancedDetectionSystem/AdvancedDetectionSystem/src/ObjectDetector.cpp

CMakeFiles/AdvancedDetectionSystem.dir/src/ObjectDetector.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/AdvancedDetectionSystem.dir/src/ObjectDetector.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/furkansevinc/Desktop/AdvancedDetectionSystem/AdvancedDetectionSystem/src/ObjectDetector.cpp > CMakeFiles/AdvancedDetectionSystem.dir/src/ObjectDetector.cpp.i

CMakeFiles/AdvancedDetectionSystem.dir/src/ObjectDetector.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/AdvancedDetectionSystem.dir/src/ObjectDetector.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/furkansevinc/Desktop/AdvancedDetectionSystem/AdvancedDetectionSystem/src/ObjectDetector.cpp -o CMakeFiles/AdvancedDetectionSystem.dir/src/ObjectDetector.cpp.s

CMakeFiles/AdvancedDetectionSystem.dir/src/SeaDetector.cpp.o: CMakeFiles/AdvancedDetectionSystem.dir/flags.make
CMakeFiles/AdvancedDetectionSystem.dir/src/SeaDetector.cpp.o: /Users/furkansevinc/Desktop/AdvancedDetectionSystem/AdvancedDetectionSystem/src/SeaDetector.cpp
CMakeFiles/AdvancedDetectionSystem.dir/src/SeaDetector.cpp.o: CMakeFiles/AdvancedDetectionSystem.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/furkansevinc/Desktop/AdvancedDetectionSystem/AdvancedDetectionSystem/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/AdvancedDetectionSystem.dir/src/SeaDetector.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/AdvancedDetectionSystem.dir/src/SeaDetector.cpp.o -MF CMakeFiles/AdvancedDetectionSystem.dir/src/SeaDetector.cpp.o.d -o CMakeFiles/AdvancedDetectionSystem.dir/src/SeaDetector.cpp.o -c /Users/furkansevinc/Desktop/AdvancedDetectionSystem/AdvancedDetectionSystem/src/SeaDetector.cpp

CMakeFiles/AdvancedDetectionSystem.dir/src/SeaDetector.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/AdvancedDetectionSystem.dir/src/SeaDetector.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/furkansevinc/Desktop/AdvancedDetectionSystem/AdvancedDetectionSystem/src/SeaDetector.cpp > CMakeFiles/AdvancedDetectionSystem.dir/src/SeaDetector.cpp.i

CMakeFiles/AdvancedDetectionSystem.dir/src/SeaDetector.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/AdvancedDetectionSystem.dir/src/SeaDetector.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/furkansevinc/Desktop/AdvancedDetectionSystem/AdvancedDetectionSystem/src/SeaDetector.cpp -o CMakeFiles/AdvancedDetectionSystem.dir/src/SeaDetector.cpp.s

CMakeFiles/AdvancedDetectionSystem.dir/src/SkinDetector.cpp.o: CMakeFiles/AdvancedDetectionSystem.dir/flags.make
CMakeFiles/AdvancedDetectionSystem.dir/src/SkinDetector.cpp.o: /Users/furkansevinc/Desktop/AdvancedDetectionSystem/AdvancedDetectionSystem/src/SkinDetector.cpp
CMakeFiles/AdvancedDetectionSystem.dir/src/SkinDetector.cpp.o: CMakeFiles/AdvancedDetectionSystem.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/furkansevinc/Desktop/AdvancedDetectionSystem/AdvancedDetectionSystem/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object CMakeFiles/AdvancedDetectionSystem.dir/src/SkinDetector.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/AdvancedDetectionSystem.dir/src/SkinDetector.cpp.o -MF CMakeFiles/AdvancedDetectionSystem.dir/src/SkinDetector.cpp.o.d -o CMakeFiles/AdvancedDetectionSystem.dir/src/SkinDetector.cpp.o -c /Users/furkansevinc/Desktop/AdvancedDetectionSystem/AdvancedDetectionSystem/src/SkinDetector.cpp

CMakeFiles/AdvancedDetectionSystem.dir/src/SkinDetector.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/AdvancedDetectionSystem.dir/src/SkinDetector.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/furkansevinc/Desktop/AdvancedDetectionSystem/AdvancedDetectionSystem/src/SkinDetector.cpp > CMakeFiles/AdvancedDetectionSystem.dir/src/SkinDetector.cpp.i

CMakeFiles/AdvancedDetectionSystem.dir/src/SkinDetector.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/AdvancedDetectionSystem.dir/src/SkinDetector.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/furkansevinc/Desktop/AdvancedDetectionSystem/AdvancedDetectionSystem/src/SkinDetector.cpp -o CMakeFiles/AdvancedDetectionSystem.dir/src/SkinDetector.cpp.s

CMakeFiles/AdvancedDetectionSystem.dir/src/TrackedObject.cpp.o: CMakeFiles/AdvancedDetectionSystem.dir/flags.make
CMakeFiles/AdvancedDetectionSystem.dir/src/TrackedObject.cpp.o: /Users/furkansevinc/Desktop/AdvancedDetectionSystem/AdvancedDetectionSystem/src/TrackedObject.cpp
CMakeFiles/AdvancedDetectionSystem.dir/src/TrackedObject.cpp.o: CMakeFiles/AdvancedDetectionSystem.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/furkansevinc/Desktop/AdvancedDetectionSystem/AdvancedDetectionSystem/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CXX object CMakeFiles/AdvancedDetectionSystem.dir/src/TrackedObject.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/AdvancedDetectionSystem.dir/src/TrackedObject.cpp.o -MF CMakeFiles/AdvancedDetectionSystem.dir/src/TrackedObject.cpp.o.d -o CMakeFiles/AdvancedDetectionSystem.dir/src/TrackedObject.cpp.o -c /Users/furkansevinc/Desktop/AdvancedDetectionSystem/AdvancedDetectionSystem/src/TrackedObject.cpp

CMakeFiles/AdvancedDetectionSystem.dir/src/TrackedObject.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/AdvancedDetectionSystem.dir/src/TrackedObject.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/furkansevinc/Desktop/AdvancedDetectionSystem/AdvancedDetectionSystem/src/TrackedObject.cpp > CMakeFiles/AdvancedDetectionSystem.dir/src/TrackedObject.cpp.i

CMakeFiles/AdvancedDetectionSystem.dir/src/TrackedObject.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/AdvancedDetectionSystem.dir/src/TrackedObject.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/furkansevinc/Desktop/AdvancedDetectionSystem/AdvancedDetectionSystem/src/TrackedObject.cpp -o CMakeFiles/AdvancedDetectionSystem.dir/src/TrackedObject.cpp.s

CMakeFiles/AdvancedDetectionSystem.dir/src/main.cpp.o: CMakeFiles/AdvancedDetectionSystem.dir/flags.make
CMakeFiles/AdvancedDetectionSystem.dir/src/main.cpp.o: /Users/furkansevinc/Desktop/AdvancedDetectionSystem/AdvancedDetectionSystem/src/main.cpp
CMakeFiles/AdvancedDetectionSystem.dir/src/main.cpp.o: CMakeFiles/AdvancedDetectionSystem.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/furkansevinc/Desktop/AdvancedDetectionSystem/AdvancedDetectionSystem/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Building CXX object CMakeFiles/AdvancedDetectionSystem.dir/src/main.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/AdvancedDetectionSystem.dir/src/main.cpp.o -MF CMakeFiles/AdvancedDetectionSystem.dir/src/main.cpp.o.d -o CMakeFiles/AdvancedDetectionSystem.dir/src/main.cpp.o -c /Users/furkansevinc/Desktop/AdvancedDetectionSystem/AdvancedDetectionSystem/src/main.cpp

CMakeFiles/AdvancedDetectionSystem.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/AdvancedDetectionSystem.dir/src/main.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/furkansevinc/Desktop/AdvancedDetectionSystem/AdvancedDetectionSystem/src/main.cpp > CMakeFiles/AdvancedDetectionSystem.dir/src/main.cpp.i

CMakeFiles/AdvancedDetectionSystem.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/AdvancedDetectionSystem.dir/src/main.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/furkansevinc/Desktop/AdvancedDetectionSystem/AdvancedDetectionSystem/src/main.cpp -o CMakeFiles/AdvancedDetectionSystem.dir/src/main.cpp.s

# Object files for target AdvancedDetectionSystem
AdvancedDetectionSystem_OBJECTS = \
"CMakeFiles/AdvancedDetectionSystem.dir/src/BoatDetector.cpp.o" \
"CMakeFiles/AdvancedDetectionSystem.dir/src/FeatureControl.cpp.o" \
"CMakeFiles/AdvancedDetectionSystem.dir/src/HumanDetector.cpp.o" \
"CMakeFiles/AdvancedDetectionSystem.dir/src/NotificationSystem.cpp.o" \
"CMakeFiles/AdvancedDetectionSystem.dir/src/ObjectDetector.cpp.o" \
"CMakeFiles/AdvancedDetectionSystem.dir/src/SeaDetector.cpp.o" \
"CMakeFiles/AdvancedDetectionSystem.dir/src/SkinDetector.cpp.o" \
"CMakeFiles/AdvancedDetectionSystem.dir/src/TrackedObject.cpp.o" \
"CMakeFiles/AdvancedDetectionSystem.dir/src/main.cpp.o"

# External object files for target AdvancedDetectionSystem
AdvancedDetectionSystem_EXTERNAL_OBJECTS =

AdvancedDetectionSystem: CMakeFiles/AdvancedDetectionSystem.dir/src/BoatDetector.cpp.o
AdvancedDetectionSystem: CMakeFiles/AdvancedDetectionSystem.dir/src/FeatureControl.cpp.o
AdvancedDetectionSystem: CMakeFiles/AdvancedDetectionSystem.dir/src/HumanDetector.cpp.o
AdvancedDetectionSystem: CMakeFiles/AdvancedDetectionSystem.dir/src/NotificationSystem.cpp.o
AdvancedDetectionSystem: CMakeFiles/AdvancedDetectionSystem.dir/src/ObjectDetector.cpp.o
AdvancedDetectionSystem: CMakeFiles/AdvancedDetectionSystem.dir/src/SeaDetector.cpp.o
AdvancedDetectionSystem: CMakeFiles/AdvancedDetectionSystem.dir/src/SkinDetector.cpp.o
AdvancedDetectionSystem: CMakeFiles/AdvancedDetectionSystem.dir/src/TrackedObject.cpp.o
AdvancedDetectionSystem: CMakeFiles/AdvancedDetectionSystem.dir/src/main.cpp.o
AdvancedDetectionSystem: CMakeFiles/AdvancedDetectionSystem.dir/build.make
AdvancedDetectionSystem: /opt/homebrew/opt/opencv@4/lib/libopencv_gapi.4.10.0.dylib
AdvancedDetectionSystem: /opt/homebrew/opt/opencv@4/lib/libopencv_stitching.4.10.0.dylib
AdvancedDetectionSystem: /opt/homebrew/opt/opencv@4/lib/libopencv_alphamat.4.10.0.dylib
AdvancedDetectionSystem: /opt/homebrew/opt/opencv@4/lib/libopencv_aruco.4.10.0.dylib
AdvancedDetectionSystem: /opt/homebrew/opt/opencv@4/lib/libopencv_bgsegm.4.10.0.dylib
AdvancedDetectionSystem: /opt/homebrew/opt/opencv@4/lib/libopencv_bioinspired.4.10.0.dylib
AdvancedDetectionSystem: /opt/homebrew/opt/opencv@4/lib/libopencv_ccalib.4.10.0.dylib
AdvancedDetectionSystem: /opt/homebrew/opt/opencv@4/lib/libopencv_dnn_objdetect.4.10.0.dylib
AdvancedDetectionSystem: /opt/homebrew/opt/opencv@4/lib/libopencv_dnn_superres.4.10.0.dylib
AdvancedDetectionSystem: /opt/homebrew/opt/opencv@4/lib/libopencv_dpm.4.10.0.dylib
AdvancedDetectionSystem: /opt/homebrew/opt/opencv@4/lib/libopencv_face.4.10.0.dylib
AdvancedDetectionSystem: /opt/homebrew/opt/opencv@4/lib/libopencv_freetype.4.10.0.dylib
AdvancedDetectionSystem: /opt/homebrew/opt/opencv@4/lib/libopencv_fuzzy.4.10.0.dylib
AdvancedDetectionSystem: /opt/homebrew/opt/opencv@4/lib/libopencv_hfs.4.10.0.dylib
AdvancedDetectionSystem: /opt/homebrew/opt/opencv@4/lib/libopencv_img_hash.4.10.0.dylib
AdvancedDetectionSystem: /opt/homebrew/opt/opencv@4/lib/libopencv_intensity_transform.4.10.0.dylib
AdvancedDetectionSystem: /opt/homebrew/opt/opencv@4/lib/libopencv_line_descriptor.4.10.0.dylib
AdvancedDetectionSystem: /opt/homebrew/opt/opencv@4/lib/libopencv_mcc.4.10.0.dylib
AdvancedDetectionSystem: /opt/homebrew/opt/opencv@4/lib/libopencv_quality.4.10.0.dylib
AdvancedDetectionSystem: /opt/homebrew/opt/opencv@4/lib/libopencv_rapid.4.10.0.dylib
AdvancedDetectionSystem: /opt/homebrew/opt/opencv@4/lib/libopencv_reg.4.10.0.dylib
AdvancedDetectionSystem: /opt/homebrew/opt/opencv@4/lib/libopencv_rgbd.4.10.0.dylib
AdvancedDetectionSystem: /opt/homebrew/opt/opencv@4/lib/libopencv_saliency.4.10.0.dylib
AdvancedDetectionSystem: /opt/homebrew/opt/opencv@4/lib/libopencv_sfm.4.10.0.dylib
AdvancedDetectionSystem: /opt/homebrew/opt/opencv@4/lib/libopencv_signal.4.10.0.dylib
AdvancedDetectionSystem: /opt/homebrew/opt/opencv@4/lib/libopencv_stereo.4.10.0.dylib
AdvancedDetectionSystem: /opt/homebrew/opt/opencv@4/lib/libopencv_structured_light.4.10.0.dylib
AdvancedDetectionSystem: /opt/homebrew/opt/opencv@4/lib/libopencv_superres.4.10.0.dylib
AdvancedDetectionSystem: /opt/homebrew/opt/opencv@4/lib/libopencv_surface_matching.4.10.0.dylib
AdvancedDetectionSystem: /opt/homebrew/opt/opencv@4/lib/libopencv_tracking.4.10.0.dylib
AdvancedDetectionSystem: /opt/homebrew/opt/opencv@4/lib/libopencv_videostab.4.10.0.dylib
AdvancedDetectionSystem: /opt/homebrew/opt/opencv@4/lib/libopencv_viz.4.10.0.dylib
AdvancedDetectionSystem: /opt/homebrew/opt/opencv@4/lib/libopencv_wechat_qrcode.4.10.0.dylib
AdvancedDetectionSystem: /opt/homebrew/opt/opencv@4/lib/libopencv_xfeatures2d.4.10.0.dylib
AdvancedDetectionSystem: /opt/homebrew/opt/opencv@4/lib/libopencv_xobjdetect.4.10.0.dylib
AdvancedDetectionSystem: /opt/homebrew/opt/opencv@4/lib/libopencv_xphoto.4.10.0.dylib
AdvancedDetectionSystem: /opt/homebrew/opt/opencv@4/lib/libopencv_shape.4.10.0.dylib
AdvancedDetectionSystem: /opt/homebrew/opt/opencv@4/lib/libopencv_highgui.4.10.0.dylib
AdvancedDetectionSystem: /opt/homebrew/opt/opencv@4/lib/libopencv_datasets.4.10.0.dylib
AdvancedDetectionSystem: /opt/homebrew/opt/opencv@4/lib/libopencv_plot.4.10.0.dylib
AdvancedDetectionSystem: /opt/homebrew/opt/opencv@4/lib/libopencv_text.4.10.0.dylib
AdvancedDetectionSystem: /opt/homebrew/opt/opencv@4/lib/libopencv_ml.4.10.0.dylib
AdvancedDetectionSystem: /opt/homebrew/opt/opencv@4/lib/libopencv_phase_unwrapping.4.10.0.dylib
AdvancedDetectionSystem: /opt/homebrew/opt/opencv@4/lib/libopencv_optflow.4.10.0.dylib
AdvancedDetectionSystem: /opt/homebrew/opt/opencv@4/lib/libopencv_ximgproc.4.10.0.dylib
AdvancedDetectionSystem: /opt/homebrew/opt/opencv@4/lib/libopencv_video.4.10.0.dylib
AdvancedDetectionSystem: /opt/homebrew/opt/opencv@4/lib/libopencv_videoio.4.10.0.dylib
AdvancedDetectionSystem: /opt/homebrew/opt/opencv@4/lib/libopencv_imgcodecs.4.10.0.dylib
AdvancedDetectionSystem: /opt/homebrew/opt/opencv@4/lib/libopencv_objdetect.4.10.0.dylib
AdvancedDetectionSystem: /opt/homebrew/opt/opencv@4/lib/libopencv_calib3d.4.10.0.dylib
AdvancedDetectionSystem: /opt/homebrew/opt/opencv@4/lib/libopencv_dnn.4.10.0.dylib
AdvancedDetectionSystem: /opt/homebrew/opt/opencv@4/lib/libopencv_features2d.4.10.0.dylib
AdvancedDetectionSystem: /opt/homebrew/opt/opencv@4/lib/libopencv_flann.4.10.0.dylib
AdvancedDetectionSystem: /opt/homebrew/opt/opencv@4/lib/libopencv_photo.4.10.0.dylib
AdvancedDetectionSystem: /opt/homebrew/opt/opencv@4/lib/libopencv_imgproc.4.10.0.dylib
AdvancedDetectionSystem: /opt/homebrew/opt/opencv@4/lib/libopencv_core.4.10.0.dylib
AdvancedDetectionSystem: CMakeFiles/AdvancedDetectionSystem.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/furkansevinc/Desktop/AdvancedDetectionSystem/AdvancedDetectionSystem/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "Linking CXX executable AdvancedDetectionSystem"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/AdvancedDetectionSystem.dir/link.txt --verbose=$(VERBOSE)
	/opt/homebrew/bin/cmake -E copy_directory /Users/furkansevinc/Desktop/AdvancedDetectionSystem/AdvancedDetectionSystem/resources /Users/furkansevinc/Desktop/AdvancedDetectionSystem/AdvancedDetectionSystem/build/resources
	/opt/homebrew/bin/cmake -E copy /opt/homebrew/share/opencv4/haarcascades/haarcascade_frontalface_default.xml /Users/furkansevinc/Desktop/AdvancedDetectionSystem/AdvancedDetectionSystem/build

# Rule to build all files generated by this target.
CMakeFiles/AdvancedDetectionSystem.dir/build: AdvancedDetectionSystem
.PHONY : CMakeFiles/AdvancedDetectionSystem.dir/build

CMakeFiles/AdvancedDetectionSystem.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/AdvancedDetectionSystem.dir/cmake_clean.cmake
.PHONY : CMakeFiles/AdvancedDetectionSystem.dir/clean

CMakeFiles/AdvancedDetectionSystem.dir/depend:
	cd /Users/furkansevinc/Desktop/AdvancedDetectionSystem/AdvancedDetectionSystem/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/furkansevinc/Desktop/AdvancedDetectionSystem/AdvancedDetectionSystem /Users/furkansevinc/Desktop/AdvancedDetectionSystem/AdvancedDetectionSystem /Users/furkansevinc/Desktop/AdvancedDetectionSystem/AdvancedDetectionSystem/build /Users/furkansevinc/Desktop/AdvancedDetectionSystem/AdvancedDetectionSystem/build /Users/furkansevinc/Desktop/AdvancedDetectionSystem/AdvancedDetectionSystem/build/CMakeFiles/AdvancedDetectionSystem.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/AdvancedDetectionSystem.dir/depend

