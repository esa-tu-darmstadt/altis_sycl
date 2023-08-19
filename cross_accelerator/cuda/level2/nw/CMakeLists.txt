cmake_minimum_required(VERSION 3.8)

add_library(nwLib needle.dp.cpp)
target_link_libraries(nwLib -lm)

if (DEFINED USE_FPGA)
    set(EMULATOR_TARGET_NW nwLib.fpga_emu)
    set(FPGA_TARGET_NW nwLib.fpga)

    ###############################################################################
    ### FPGA Emulator
    ###############################################################################
    add_executable(${EMULATOR_TARGET_NW} ${exe_files} needle.dp.cpp)
    set_target_properties(${EMULATOR_TARGET_NW} PROPERTIES COMPILE_FLAGS "${EMULATOR_COMPILE_FLAGS}")
    set_target_properties(${EMULATOR_TARGET_NW} PROPERTIES LINK_FLAGS "${EMULATOR_LINK_FLAGS}")
    add_custom_target(nwLib_fpga_emu DEPENDS ${EMULATOR_TARGET_NW})

    target_link_libraries(${EMULATOR_TARGET_NW} PUBLIC
        AltisCommon 
        nwLib
        mkl_sycl
        mkl_core
        mkl_intel_thread
        mkl_intel_ilp64
        iomp5
    )

    ###############################################################################
    ### Generate Report
    ###############################################################################
    set(FPGA_EARLY_IMAGE_NW nwLib_report.a)
    add_executable(${FPGA_EARLY_IMAGE_NW} ${exe_files} needle.dp.cpp)
    add_custom_target(nwLib_fpga_report DEPENDS ${FPGA_EARLY_IMAGE_NW})
    set_target_properties(${FPGA_EARLY_IMAGE_NW} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS}")
    set_target_properties(${FPGA_EARLY_IMAGE_NW} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -fsycl-link=early")

    target_link_libraries(${FPGA_EARLY_IMAGE_NW} PUBLIC
        AltisCommon 
        nwLib
        mkl_sycl
        mkl_core
        mkl_intel_thread
        mkl_intel_ilp64
        iomp5
    )

    ###############################################################################
    ### FPGA Hardware
    ###############################################################################
    add_executable(${FPGA_TARGET_NW} EXCLUDE_FROM_ALL ${exe_files} needle.dp.cpp)
    add_custom_target(nwLib_fpga DEPENDS ${FPGA_TARGET_NW})
    set_target_properties(${FPGA_TARGET_NW} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS}  -reuse-exe=${CMAKE_CURRENT_SOURCE_DIR}/../../../build/cuda/level2/nw/nwLib.fpga")
    set_target_properties(${FPGA_TARGET_NW} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -reuse-exe=${CMAKE_CURRENT_SOURCE_DIR}/../../../build/cuda/level2/nw/nwLib.fpga")

    target_link_libraries(${FPGA_TARGET_NW} PUBLIC
        AltisCommon 
        nwLib
        mkl_sycl
        mkl_core
        mkl_intel_thread
        mkl_intel_ilp64
        iomp5
    )
endif()
