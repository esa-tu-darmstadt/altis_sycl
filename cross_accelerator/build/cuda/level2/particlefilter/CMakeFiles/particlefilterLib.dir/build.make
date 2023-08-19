# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.26

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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/chr1s603/Documents/workspace/git/altis_fpga/unoptimized

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/chr1s603/Documents/workspace/git/altis_fpga/unoptimized/build

# Include any dependencies generated for this target.
include cuda/level2/particlefilter/CMakeFiles/particlefilterLib.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include cuda/level2/particlefilter/CMakeFiles/particlefilterLib.dir/compiler_depend.make

# Include the progress variables for this target.
include cuda/level2/particlefilter/CMakeFiles/particlefilterLib.dir/progress.make

# Include the compile flags for this target's objects.
include cuda/level2/particlefilter/CMakeFiles/particlefilterLib.dir/flags.make

cuda/level2/particlefilter/CMakeFiles/particlefilterLib.dir/ex_particle_CUDA_naive_seq.dp.cpp.o: cuda/level2/particlefilter/CMakeFiles/particlefilterLib.dir/flags.make
cuda/level2/particlefilter/CMakeFiles/particlefilterLib.dir/ex_particle_CUDA_naive_seq.dp.cpp.o: /home/chr1s603/Documents/workspace/git/altis_fpga/unoptimized/cuda/level2/particlefilter/ex_particle_CUDA_naive_seq.dp.cpp
cuda/level2/particlefilter/CMakeFiles/particlefilterLib.dir/ex_particle_CUDA_naive_seq.dp.cpp.o: cuda/level2/particlefilter/CMakeFiles/particlefilterLib.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/chr1s603/Documents/workspace/git/altis_fpga/unoptimized/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object cuda/level2/particlefilter/CMakeFiles/particlefilterLib.dir/ex_particle_CUDA_naive_seq.dp.cpp.o"
	cd /home/chr1s603/Documents/workspace/git/altis_fpga/unoptimized/build/cuda/level2/particlefilter && icpx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT cuda/level2/particlefilter/CMakeFiles/particlefilterLib.dir/ex_particle_CUDA_naive_seq.dp.cpp.o -MF CMakeFiles/particlefilterLib.dir/ex_particle_CUDA_naive_seq.dp.cpp.o.d -o CMakeFiles/particlefilterLib.dir/ex_particle_CUDA_naive_seq.dp.cpp.o -c /home/chr1s603/Documents/workspace/git/altis_fpga/unoptimized/cuda/level2/particlefilter/ex_particle_CUDA_naive_seq.dp.cpp

cuda/level2/particlefilter/CMakeFiles/particlefilterLib.dir/ex_particle_CUDA_naive_seq.dp.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/particlefilterLib.dir/ex_particle_CUDA_naive_seq.dp.cpp.i"
	cd /home/chr1s603/Documents/workspace/git/altis_fpga/unoptimized/build/cuda/level2/particlefilter && icpx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/chr1s603/Documents/workspace/git/altis_fpga/unoptimized/cuda/level2/particlefilter/ex_particle_CUDA_naive_seq.dp.cpp > CMakeFiles/particlefilterLib.dir/ex_particle_CUDA_naive_seq.dp.cpp.i

cuda/level2/particlefilter/CMakeFiles/particlefilterLib.dir/ex_particle_CUDA_naive_seq.dp.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/particlefilterLib.dir/ex_particle_CUDA_naive_seq.dp.cpp.s"
	cd /home/chr1s603/Documents/workspace/git/altis_fpga/unoptimized/build/cuda/level2/particlefilter && icpx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/chr1s603/Documents/workspace/git/altis_fpga/unoptimized/cuda/level2/particlefilter/ex_particle_CUDA_naive_seq.dp.cpp -o CMakeFiles/particlefilterLib.dir/ex_particle_CUDA_naive_seq.dp.cpp.s

# Object files for target particlefilterLib
particlefilterLib_OBJECTS = \
"CMakeFiles/particlefilterLib.dir/ex_particle_CUDA_naive_seq.dp.cpp.o"

# External object files for target particlefilterLib
particlefilterLib_EXTERNAL_OBJECTS =

cuda/level2/particlefilter/libparticlefilterLib.a: cuda/level2/particlefilter/CMakeFiles/particlefilterLib.dir/ex_particle_CUDA_naive_seq.dp.cpp.o
cuda/level2/particlefilter/libparticlefilterLib.a: cuda/level2/particlefilter/CMakeFiles/particlefilterLib.dir/build.make
cuda/level2/particlefilter/libparticlefilterLib.a: cuda/level2/particlefilter/CMakeFiles/particlefilterLib.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/chr1s603/Documents/workspace/git/altis_fpga/unoptimized/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library libparticlefilterLib.a"
	cd /home/chr1s603/Documents/workspace/git/altis_fpga/unoptimized/build/cuda/level2/particlefilter && $(CMAKE_COMMAND) -P CMakeFiles/particlefilterLib.dir/cmake_clean_target.cmake
	cd /home/chr1s603/Documents/workspace/git/altis_fpga/unoptimized/build/cuda/level2/particlefilter && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/particlefilterLib.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
cuda/level2/particlefilter/CMakeFiles/particlefilterLib.dir/build: cuda/level2/particlefilter/libparticlefilterLib.a
.PHONY : cuda/level2/particlefilter/CMakeFiles/particlefilterLib.dir/build

cuda/level2/particlefilter/CMakeFiles/particlefilterLib.dir/clean:
	cd /home/chr1s603/Documents/workspace/git/altis_fpga/unoptimized/build/cuda/level2/particlefilter && $(CMAKE_COMMAND) -P CMakeFiles/particlefilterLib.dir/cmake_clean.cmake
.PHONY : cuda/level2/particlefilter/CMakeFiles/particlefilterLib.dir/clean

cuda/level2/particlefilter/CMakeFiles/particlefilterLib.dir/depend:
	cd /home/chr1s603/Documents/workspace/git/altis_fpga/unoptimized/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/chr1s603/Documents/workspace/git/altis_fpga/unoptimized /home/chr1s603/Documents/workspace/git/altis_fpga/unoptimized/cuda/level2/particlefilter /home/chr1s603/Documents/workspace/git/altis_fpga/unoptimized/build /home/chr1s603/Documents/workspace/git/altis_fpga/unoptimized/build/cuda/level2/particlefilter /home/chr1s603/Documents/workspace/git/altis_fpga/unoptimized/build/cuda/level2/particlefilter/CMakeFiles/particlefilterLib.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : cuda/level2/particlefilter/CMakeFiles/particlefilterLib.dir/depend

