# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.13

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /workspace/gofmm

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /workspace/gofmm/build

# Include any dependencies generated for this target.
include CMakeFiles/distributed_matrix.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/distributed_matrix.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/distributed_matrix.dir/flags.make

CMakeFiles/distributed_matrix.dir/example/distributed_matrix.cpp.o: CMakeFiles/distributed_matrix.dir/flags.make
CMakeFiles/distributed_matrix.dir/example/distributed_matrix.cpp.o: ../example/distributed_matrix.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/workspace/gofmm/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/distributed_matrix.dir/example/distributed_matrix.cpp.o"
	/usr/local/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/distributed_matrix.dir/example/distributed_matrix.cpp.o -c /workspace/gofmm/example/distributed_matrix.cpp

CMakeFiles/distributed_matrix.dir/example/distributed_matrix.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/distributed_matrix.dir/example/distributed_matrix.cpp.i"
	/usr/local/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /workspace/gofmm/example/distributed_matrix.cpp > CMakeFiles/distributed_matrix.dir/example/distributed_matrix.cpp.i

CMakeFiles/distributed_matrix.dir/example/distributed_matrix.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/distributed_matrix.dir/example/distributed_matrix.cpp.s"
	/usr/local/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /workspace/gofmm/example/distributed_matrix.cpp -o CMakeFiles/distributed_matrix.dir/example/distributed_matrix.cpp.s

# Object files for target distributed_matrix
distributed_matrix_OBJECTS = \
"CMakeFiles/distributed_matrix.dir/example/distributed_matrix.cpp.o"

# External object files for target distributed_matrix
distributed_matrix_EXTERNAL_OBJECTS =

distributed_matrix: CMakeFiles/distributed_matrix.dir/example/distributed_matrix.cpp.o
distributed_matrix: CMakeFiles/distributed_matrix.dir/build.make
distributed_matrix: /usr/lib/x86_64-linux-gnu/libopenblas.so
distributed_matrix: libhmlp.so
distributed_matrix: /usr/lib/x86_64-linux-gnu/libopenblas.so
distributed_matrix: /usr/lib/x86_64-linux-gnu/libopenblas.so
distributed_matrix: CMakeFiles/distributed_matrix.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/workspace/gofmm/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable distributed_matrix"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/distributed_matrix.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/distributed_matrix.dir/build: distributed_matrix

.PHONY : CMakeFiles/distributed_matrix.dir/build

CMakeFiles/distributed_matrix.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/distributed_matrix.dir/cmake_clean.cmake
.PHONY : CMakeFiles/distributed_matrix.dir/clean

CMakeFiles/distributed_matrix.dir/depend:
	cd /workspace/gofmm/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /workspace/gofmm /workspace/gofmm /workspace/gofmm/build /workspace/gofmm/build /workspace/gofmm/build/CMakeFiles/distributed_matrix.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/distributed_matrix.dir/depend

