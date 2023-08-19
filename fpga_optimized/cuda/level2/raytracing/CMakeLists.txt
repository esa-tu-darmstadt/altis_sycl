cmake_minimum_required(VERSION 3.8)

add_subdirectory(raytracing_cuda)

if (DEFINED USE_FPGA)
    set(EMULATOR_TARGET_RT raytracingLib.fpga_emu)
    set(FPGA_TARGET_RT raytracingLib.fpga)

    ###############################################################################
    ### FPGA Emulator
    ###############################################################################
    add_executable(${EMULATOR_TARGET_RT} ${exe_files} raytracing_cuda/raytracing.dp.cpp)
    set_target_properties(${EMULATOR_TARGET_RT} PROPERTIES COMPILE_FLAGS "${EMULATOR_COMPILE_FLAGS}")
    set_target_properties(${EMULATOR_TARGET_RT} PROPERTIES LINK_FLAGS "${EMULATOR_LINK_FLAGS}")
    add_custom_target(raytracingLib_fpga_emu DEPENDS ${EMULATOR_TARGET_RT})

    target_link_libraries(${EMULATOR_TARGET_RT} PUBLIC
        AltisCommon 
    )

    ###############################################################################
    ### Generate Report
    ###############################################################################
    set(FPGA_EARLY_IMAGE_RT raytracingLib_report.a)
    add_executable(${FPGA_EARLY_IMAGE_RT} ${exe_files} raytracing_cuda/raytracing.dp.cpp)
    add_custom_target(raytracingLib_fpga_report DEPENDS ${FPGA_EARLY_IMAGE_RT})
    set_target_properties(${FPGA_EARLY_IMAGE_RT} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS}")
    set_target_properties(${FPGA_EARLY_IMAGE_RT} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -fsycl-link=early")

    target_link_libraries(${FPGA_EARLY_IMAGE_RT} PUBLIC
        AltisCommon 
    )

    ###############################################################################
    ### FPGA Hardware
    ###############################################################################
    add_executable(${FPGA_TARGET_RT} EXCLUDE_FROM_ALL ${exe_files} raytracing_cuda/raytracing.dp.cpp)
    add_custom_target(raytracingLib_fpga DEPENDS ${FPGA_TARGET_RT})
    set_target_properties(${FPGA_TARGET_RT} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS} -Xsseed=42")
    set_target_properties(${FPGA_TARGET_RT} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -Xsseed=42 -reuse-exe=${CMAKE_CURRENT_SOURCE_DIR}/../../../build/cuda/level2/raytracing/raytracingLib.fpga")

    target_link_libraries(${FPGA_TARGET_RT} PUBLIC
        AltisCommon 
    )
endif()
