# Altis-SYCL

Altis-SYCL is a SYCL-based implementation of the [Altis GPGPU benchmark suite](https://github.com/utcs-scea/altis) (originally written in CUDA) for CPUs, GPUs, and FPGAs.

Altis-SYCL has been migrated from CUDA using the [DPC++ Compatibility Tool](https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compatibility-tool.html) (DPCT) of oneAPI 2022.1. Our main focus has been to evaluate the performance of these GPU-tailored SYCL kernels and to investigate their optimization potential for FPGAs. For some cases, minor changes were made to speedup the FPGA port as our interest lies in the achievable performance without major rework of the kernels.

The [`benchmarks.md`](benchmarks.md) file contains a portion of our benchmark results with various accelerators.

# Directory Structure

## unmodified_migration
Contains the unmodified output code migrated by DPCT. Does not compile.

## cross_accelerator
Running and validated _level 2_ benchmarks. 

All DPCT-inserted warnings were addressed. Non-required features were removed, e.g., support for USM in benchmarks, or CUDA Graphs. Removed all DPCT library usages to support event-based timing measurements. 

For comparing the performance between CUDA and SYCL, _level 1_ benchmarks are also included. However, these were only tested on NVIDIA GPUs using the CUDA backend of DPCT.

The benchmarks were adapted to close the performance gap between CUDA and SYCL on an RTX 2080 GPU. These adaptations encompass e.g., removal of loop-unrolling and altering inlining behaviour of functions due to differences in NVCC and DPC++ compilers.

Runs correctly on the following hardware:
* Intel and AMD x64 CPUs (Ryzen, Epyc, Core i and Xeon) - by default
* Intel GPUs - by default
* NVIDIA GPUs using the DPC++ CUDA backend - when `USE_CUDA` CMake variable is set
* Intel FPGAs (Stratix 10, Agilex) - when `USE_FPGA & (USE_STRATIX | USE_AGILEX)` CMake variables are set

## fpga_optimized
Contains optimized FPGA versions of _level 2_ benchmarks. 

Due to added FPGA optimization attributes, it can no longer be executed on CPU or GPUs! However, via the Intel FPGA Emulator and Simulator, it is possible to execute them on regular CPUs. 

The optimization attributes were validated under oneAPI 22.3.0. The more recent 23.0.0 version failed to achieve the same initiation interval (II) in some benchmarks. Note that we currently have no optimized version for the **DWT2D** benchmark due to congestion on shared memory.

The optimized code is tailored for the BittWare 520N card featuring the Stratix 10 FPGA. The support for Agilex FPGAs encompasses _only_ slight code modifications for a more efficient FPGA resource utilization. Examples:
- **CFD64**: for Stratix 10, the kernel could be vectorized 2-times. For Agilex, vectorization was disabled to fit the design on device.
- **CFD32**: could be replicated more on Agilex than on Stratix 10.

_Note_:
Currently, the **Mandelbrot** benchmark requires separate builds for each problem size. See [`mandelbrot.dp.cpp`](fpga_optimized/cuda/level2/mandelbrot/mandelbrot.dp.cpp#L42).

## kmeans_inputs
See section [Benchmark Parameters](#benchmark-parameters).

# Build Process
1. Move into any of the above directories: `cross_accelerator` or `fpga_optimized`. 

2. In the `CMakeLists.txt`, use the appropriate CMake environment variables to choose the target device:
- x86 CPUs and Intel GPUs (by default, the `USE_CUDA` and `USE_FPGA` variables are commented out)
- NVIDIA GPUs (uncomment `USE_CUDA`, comment out `USE_FPGA`) 
- Intel FPGAs (uncomment `USE_FPGA`, comment out `USE_CUDA`)

Currently, only one target can be active at once. When building for FPGAs, be sure that the BSP locations and exact part numbers in the CMake file are correct. 

3. Create a `build` folder and move into it, and then run:

```
cmake ..
```

Then you can build specific benchmarks. Examples for CPUs and GPUs:

```
make cfd
```

```
make fdtd2d
```

```
make lavamd
```

When targeting FPGAs, you can build for alternative modes: 
- Separate HLS report (takes minutes, and thus, it is recommended before starting a hardware build)
- Intel FPGA emulator (takes minutes)
- FPGA bitstream (takes between 4h - 24h for Stratix 10)

The FPGA targets have the following structure:

```
make cfdLib_fpga_report
```

```
make cfdLib_fpga_emu
```

```
make cfdLib_fpga
```

In the `CMakeLists.txt` for each benchmark, you can pass some options to the compiler by appending them to `COMPILE_FLAGS` or `LINK_FLAGS`:
- `-Xsseed=42`: use this when you get timing errors
- `-Xsprofile`: use this to build a profiling build. Careful: These will be larger as normal ones, large designs might no longer fit.
- `-Xsclock=42MHz`: playing around with this might let difficult designs build, use this when using `-Xsseed` does not fix possible timing failures. If this does not help, be sure the design is "routable" by looking in the report's kernel memory section. Congestion on a variable might be the issue here.

## Benchmark Parameters
Pass `--size`/`-s` to select between input sizes: 1, 2, 3.
- Default is 1
- FPGA kernels are only optimized for sizes <= 3

Pass `-n` to change how often a benchmark should be run. 
- Default is 10

_Note_:
- **kMeans** and **FDTD2D** do not support the `-n` argument.
- **kMeans** does not support the `--size` argument. Use instead `--inputFile`, which points for instance to any of the files in the [`kmeans_inputs`](kmeans_inputs/) directory.

## Run Benchmarks on CPU or GPU

For CPUs and GPUs, the binaries are under the `build/bin/levelX` folders.
Pass `--gpu` to use a GPU present on the host (if `--gpu` is omitted, the host CPU is used):

```
./bin/level2/lavamd
```

```
./bin/level2/lavamd --gpu
```

```
./bin/level2/lavamd --size 2
```

_Note_:

When building with CUDA support, the benchmarks currently do not run on other accelerators than the specified CUDA architecture in the CMake file. To support older NVIDIA GPUs (feature level < 75) or to also compile kernels for CPUs, change the compiler arguments.

## Run Benchmarks on FPGA or FPGA emulator
For FPGAs, the binaries are located under the `build/cuda/levelX/benchmarkX` folders.
Pass `--fpga` to use the FPGA or `--fpga_emu` to use the emulator. If you try to run e.g., a FPGA bitstream on an emulator or the CPU, you will get a INVALID_BINARY error message.

```
./cuda/level2/srad/sradLib.fpga --fpga -s 1
```

```
./cuda/level2/srad/sradLib.fpga --fpga -s 2
```

```
./cuda/level2/nw/nwLib.fpga_emu --fpga_emu -s 3
```

The reports are placed under the `build/cuda/levelX/benchmarkX/benchmarkXLib_report.prj/reports` folder (report-build) or the `build/cuda/levelX/benchmarkX/benchmarkXLib.prj/reports` folder (hw-build).

For profiling, use:

```
aocl profile ./cuda/level2/srad/sradLib.fpga --fpga -s 1 -n 1
```
