cmake_minimum_required(VERSION 3.8)

add_library(fdtd2dLib fdtd2d.dp.cpp)
target_link_libraries(fdtd2dLib)

if (DEFINED USE_FPGA)
    set(EMULATOR_TARGET_FDTD fdtd2dLib.fpga_emu)
    set(FPGA_TARGET_FDTD fdtd2dLib.fpga)

    ###############################################################################
    ### FPGA Emulator
    ###############################################################################
    add_executable(${EMULATOR_TARGET_FDTD} ${exe_files} fdtd2d.dp.cpp)
    set_target_properties(${EMULATOR_TARGET_FDTD} PROPERTIES COMPILE_FLAGS "${EMULATOR_COMPILE_FLAGS}")
    set_target_properties(${EMULATOR_TARGET_FDTD} PROPERTIES LINK_FLAGS "${EMULATOR_LINK_FLAGS}")
    add_custom_target(fdtd2dLib_fpga_emu DEPENDS ${EMULATOR_TARGET_FDTD})

    target_link_libraries(${EMULATOR_TARGET_FDTD} PUBLIC
        AltisCommon 
    )

    ###############################################################################
    ### Generate Report
    ###############################################################################
    set(FPGA_EARLY_IMAGE_FDTD fdtd2dLib_report.a)
    add_executable(${FPGA_EARLY_IMAGE_FDTD} ${exe_files} fdtd2d.dp.cpp)
    add_custom_target(fdtd2dLib_fpga_report DEPENDS ${FPGA_EARLY_IMAGE_FDTD})
    set_target_properties(${FPGA_EARLY_IMAGE_FDTD} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS}")
    set_target_properties(${FPGA_EARLY_IMAGE_FDTD} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -fsycl-link=early")

    target_link_libraries(${FPGA_EARLY_IMAGE_FDTD} PUBLIC
        AltisCommon 
    )

    ###############################################################################
    ### FPGA Hardware
    ###############################################################################
    add_executable(${FPGA_TARGET_FDTD} EXCLUDE_FROM_ALL ${exe_files} fdtd2d.dp.cpp)
    add_custom_target(fdtd2dLib_fpga DEPENDS ${FPGA_TARGET_FDTD})
    set_target_properties(${FPGA_TARGET_FDTD} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS} -reuse-exe=${CMAKE_CURRENT_SOURCE_DIR}/build/cuda/level2/fdtd2d/fdtd2dLib.fpga")
    set_target_properties(${FPGA_TARGET_FDTD} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -reuse-exe=${CMAKE_CURRENT_SOURCE_DIR}/build/cuda/level2/fdtd2d/fdtd2dLib.fpga")

    target_link_libraries(${FPGA_TARGET_FDTD} PUBLIC
        AltisCommon 
    )
endif()
