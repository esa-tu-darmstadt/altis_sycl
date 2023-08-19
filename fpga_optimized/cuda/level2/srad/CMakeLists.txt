cmake_minimum_required(VERSION 3.8)

add_library(sradLib
    srad.dp.cpp
)
target_link_libraries(sradLib -lm)

if (DEFINED USE_FPGA)
    set(EMULATOR_TARGET_SRAD sradLib.fpga_emu)
    set(FPGA_TARGET_SRAD sradLib.fpga)

    ###############################################################################
    ### FPGA Emulator
    ###############################################################################
    add_executable(${EMULATOR_TARGET_SRAD} ${exe_files} srad.dp.cpp)
    set_target_properties(${EMULATOR_TARGET_SRAD} PROPERTIES COMPILE_FLAGS "${EMULATOR_COMPILE_FLAGS}")
    set_target_properties(${EMULATOR_TARGET_SRAD} PROPERTIES LINK_FLAGS "${EMULATOR_LINK_FLAGS}")
    add_custom_target(sradLib_fpga_emu DEPENDS ${EMULATOR_TARGET_SRAD})

    target_link_libraries(${EMULATOR_TARGET_SRAD} PUBLIC
        AltisCommon
    )

    ###############################################################################
    ### Generate Report
    ###############################################################################
    set(FPGA_EARLY_IMAGE_SRAD sradLib_report.a)
    add_executable(${FPGA_EARLY_IMAGE_SRAD} ${exe_files} srad.dp.cpp)
    add_custom_target(sradLib_fpga_report DEPENDS ${FPGA_EARLY_IMAGE_SRAD})
    set_target_properties(${FPGA_EARLY_IMAGE_SRAD} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS}")
    set_target_properties(${FPGA_EARLY_IMAGE_SRAD} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -fsycl-link=early")

    target_link_libraries(${FPGA_EARLY_IMAGE_SRAD} PUBLIC
        AltisCommon
    )

    ###############################################################################
    ### FPGA Hardware
    ###############################################################################
    add_executable(${FPGA_TARGET_SRAD} EXCLUDE_FROM_ALL ${exe_files} srad.dp.cpp)
    add_custom_target(sradLib_fpga DEPENDS ${FPGA_TARGET_SRAD})
    set_target_properties(${FPGA_TARGET_SRAD} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS} -Xsseed=1 -reuse-exe=${CMAKE_CURRENT_SOURCE_DIR}/../../../build/cuda/level2/srad/sradLib.fpga")
    set_target_properties(${FPGA_TARGET_SRAD} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -Xsseed=1 -reuse-exe=${CMAKE_CURRENT_SOURCE_DIR}/../../../build/cuda/level2/srad/sradLib.fpga")

    target_link_libraries(${FPGA_TARGET_SRAD} PUBLIC
        AltisCommon
    )
endif()