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
include CMakeFiles/test_mpigofmm.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/test_mpigofmm.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/test_mpigofmm.dir/flags.make

CMakeFiles/test_mpigofmm.dir/example/test_mpigofmm.cpp.o: CMakeFiles/test_mpigofmm.dir/flags.make
CMakeFiles/test_mpigofmm.dir/example/test_mpigofmm.cpp.o: ../example/test_mpigofmm.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/workspace/gofmm/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/test_mpigofmm.dir/example/test_mpigofmm.cpp.o"
	/usr/local/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test_mpigofmm.dir/example/test_mpigofmm.cpp.o -c /workspace/gofmm/example/test_mpigofmm.cpp

CMakeFiles/test_mpigofmm.dir/example/test_mpigofmm.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_mpigofmm.dir/example/test_mpigofmm.cpp.i"
	/usr/local/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /workspace/gofmm/example/test_mpigofmm.cpp > CMakeFiles/test_mpigofmm.dir/example/test_mpigofmm.cpp.i

CMakeFiles/test_mpigofmm.dir/example/test_mpigofmm.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_mpigofmm.dir/example/test_mpigofmm.cpp.s"
	/usr/local/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /workspace/gofmm/example/test_mpigofmm.cpp -o CMakeFiles/test_mpigofmm.dir/example/test_mpigofmm.cpp.s

# Object files for target test_mpigofmm
test_mpigofmm_OBJECTS = \
"CMakeFiles/test_mpigofmm.dir/example/test_mpigofmm.cpp.o"

# External object files for target test_mpigofmm
test_mpigofmm_EXTERNAL_OBJECTS =

test_mpigofmm: CMakeFiles/test_mpigofmm.dir/example/test_mpigofmm.cpp.o
test_mpigofmm: CMakeFiles/test_mpigofmm.dir/build.make
test_mpigofmm: /usr/lib/x86_64-linux-gnu/libopenblas.so
test_mpigofmm: libhmlp.so
test_mpigofmm: /usr/lib/x86_64-linux-gnu/libopenblas.so
test_mpigofmm: /usr/lib/x86_64-linux-gnu/libopenblas.so
test_mpigofmm: CMakeFiles/test_mpigofmm.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/workspace/gofmm/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable test_mpigofmm"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_mpigofmm.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/test_mpigofmm.dir/build: test_mpigofmm

.PHONY : CMakeFiles/test_mpigofmm.dir/build

CMakeFiles/test_mpigofmm.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/test_mpigofmm.dir/cmake_clean.cmake
.PHONY : CMakeFiles/test_mpigofmm.dir/clean

CMakeFiles/test_mpigofmm.dir/depend:
	cd /workspace/gofmm/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /workspace/gofmm /workspace/gofmm /workspace/gofmm/build /workspace/gofmm/build /workspace/gofmm/build/CMakeFiles/test_mpigofmm.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/test_mpigofmm.dir/depend

