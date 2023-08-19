cmake_minimum_required(VERSION 3.8)

add_library(kmeansLib
    kmmain.dp.cpp
    kmeansraw.dp.cpp
)
target_link_libraries(kmeansLib -lm)

if (DEFINED USE_FPGA)
    set(EMULATOR_TARGET_KMEANS kmeansLib.fpga_emu)
    set(FPGA_TARGET_KMEANS kmeansLib.fpga)

    ###############################################################################
    ### FPGA Emulator
    ###############################################################################
    add_executable(${EMULATOR_TARGET_KMEANS} ${exe_files} kmmain.dp.cpp kmeansraw.dp.cpp)
    set_target_properties(${EMULATOR_TARGET_KMEANS} PROPERTIES COMPILE_FLAGS "${EMULATOR_COMPILE_FLAGS} -qactypes")
    set_target_properties(${EMULATOR_TARGET_KMEANS} PROPERTIES LINK_FLAGS "${EMULATOR_LINK_FLAGS} -qactypes")
    add_custom_target(kmeansLib_fpga_emu DEPENDS ${EMULATOR_TARGET_KMEANS})

    target_link_libraries(${EMULATOR_TARGET_KMEANS} PUBLIC
        AltisCommon 
    )

    ###############################################################################
    ### Generate Report
    ###############################################################################
    set(FPGA_EARLY_IMAGE_KMEANS kmeansLib_report.a)
    # The compile output is not an executable, but an intermediate compilation result unique to DPC++.
    add_executable(${FPGA_EARLY_IMAGE_KMEANS} ${exe_files} kmmain.dp.cpp kmeansraw.dp.cpp)
    add_custom_target(kmeansLib_fpga_report DEPENDS ${FPGA_EARLY_IMAGE_KMEANS})
    set_target_properties(${FPGA_EARLY_IMAGE_KMEANS} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS} -qactypes")
    set_target_properties(${FPGA_EARLY_IMAGE_KMEANS} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -fsycl-link=early -qactypes")
    # fsycl-link=early stops the compiler after RTL generation, before invoking Quartus®

    target_link_libraries(${FPGA_EARLY_IMAGE_KMEANS} PUBLIC
        AltisCommon 
    )

    ###############################################################################
    ### FPGA Hardware
    ###############################################################################
    add_executable(${FPGA_TARGET_KMEANS} EXCLUDE_FROM_ALL ${exe_files} kmmain.dp.cpp kmeansraw.dp.cpp)
    add_custom_target(kmeansLib_fpga DEPENDS ${FPGA_TARGET_KMEANS})
    set_target_properties(${FPGA_TARGET_KMEANS} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS} -Xsseed=42 -reuse-exe=${CMAKE_CURRENT_SOURCE_DIR}/../../../build/cuda/level2/kmeans/kmeansLib.fpga")
    set_target_properties(${FPGA_TARGET_KMEANS} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -Xsseed=42 -reuse-exe=${CMAKE_CURRENT_SOURCE_DIR}/../../../build/cuda/level2/kmeans/kmeansLib.fpga")

    target_link_libraries(${FPGA_TARGET_KMEANS} PUBLIC
        AltisCommon 
    )
endif()
