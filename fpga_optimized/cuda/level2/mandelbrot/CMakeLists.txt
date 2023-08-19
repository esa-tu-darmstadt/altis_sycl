cmake_minimum_required(VERSION 3.8)

add_library(mandelbrotLib mandelbrot.dp.cpp)
target_link_libraries(mandelbrotLib -lm)

if (DEFINED USE_FPGA)
    set(EMULATOR_TARGET_MBROT mandelbrotLib.fpga_emu)
    set(FPGA_TARGET_MBROT mandelbrotLib.fpga)

    ###############################################################################
    ### FPGA Emulator
    ###############################################################################
    add_executable(${EMULATOR_TARGET_MBROT} ${exe_files} mandelbrot.dp.cpp)
    set_target_properties(${EMULATOR_TARGET_MBROT} PROPERTIES COMPILE_FLAGS "${EMULATOR_COMPILE_FLAGS}")
    set_target_properties(${EMULATOR_TARGET_MBROT} PROPERTIES LINK_FLAGS "${EMULATOR_LINK_FLAGS}")
    add_custom_target(mandelbrotLib_fpga_emu DEPENDS ${EMULATOR_TARGET_MBROT})

    target_link_libraries(${EMULATOR_TARGET_MBROT} PUBLIC
        AltisCommon 
    )

    ###############################################################################
    ### Generate Report
    ###############################################################################
    set(FPGA_EARLY_IMAGE_MBROT mandelbrotLib_report.a)
    add_executable(${FPGA_EARLY_IMAGE_MBROT} ${exe_files} mandelbrot.dp.cpp)
    add_custom_target(mandelbrotLib_fpga_report DEPENDS ${FPGA_EARLY_IMAGE_MBROT})
    set_target_properties(${FPGA_EARLY_IMAGE_MBROT} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS} -Xsseed=42 -Xsnum-reorder=1")
    set_target_properties(${FPGA_EARLY_IMAGE_MBROT} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -Xsseed=42 -fsycl-link=early -Xsnum-reorder=1")

    target_link_libraries(${FPGA_EARLY_IMAGE_MBROT} PUBLIC
        AltisCommon
    )

    ###############################################################################
    ### FPGA Hardware
    ###############################################################################
    add_executable(${FPGA_TARGET_MBROT} EXCLUDE_FROM_ALL ${exe_files} mandelbrot.dp.cpp)
    add_custom_target(mandelbrotLib_fpga DEPENDS ${FPGA_TARGET_MBROT})
    set_target_properties(${FPGA_TARGET_MBROT} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS} -Xsseed=1 -Xsnum-reorder=1")
    set_target_properties(${FPGA_TARGET_MBROT} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -Xsseed=1 -Xsnum-reorder=1 -reuse-exe=${CMAKE_CURRENT_SOURCE_DIR}/../../../build/cuda/level2/mandelbrot/mandelbrotLib.fpga")

    target_link_libraries(${FPGA_TARGET_MBROT} PUBLIC
        AltisCommon
    )
endif()
