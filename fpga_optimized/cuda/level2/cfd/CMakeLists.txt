cmake_minimum_required(VERSION 3.8)

add_library(cfdLib euler3d.dp.cpp)
target_link_libraries(cfdLib -lm)

add_library(cfddoubleLib euler3d_double.dp.cpp)
target_link_libraries(cfddoubleLib -lm)

if (DEFINED USE_FPGA)
    set(EMULATOR_TARGET_CFD cfdLib.fpga_emu)
    set(FPGA_TARGET_CFD cfdLib.fpga)
    set(EMULATOR_TARGET_CFDD cfddoubleLib.fpga_emu)
    set(FPGA_TARGET_CFDD cfddoubleLib.fpga)

    ###############################################################################
    ### FPGA Emulator
    ###############################################################################
    add_executable(${EMULATOR_TARGET_CFD} ${exe_files} euler3d.dp.cpp)
    add_executable(${EMULATOR_TARGET_CFDD} ${exe_files} euler3d_double.dp.cpp)
    set_target_properties(${EMULATOR_TARGET_CFD} PROPERTIES COMPILE_FLAGS "${EMULATOR_COMPILE_FLAGS}")
    set_target_properties(${EMULATOR_TARGET_CFD} PROPERTIES LINK_FLAGS "${EMULATOR_LINK_FLAGS}")
    set_target_properties(${EMULATOR_TARGET_CFDD} PROPERTIES COMPILE_FLAGS "${EMULATOR_COMPILE_FLAGS}")
    set_target_properties(${EMULATOR_TARGET_CFDD} PROPERTIES LINK_FLAGS "${EMULATOR_LINK_FLAGS}")
    add_custom_target(cfdLib_fpga_emu DEPENDS ${EMULATOR_TARGET_CFD})
    add_custom_target(cfddoubleLib_fpga_emu DEPENDS ${EMULATOR_TARGET_CFDD})

    target_link_libraries(${EMULATOR_TARGET_CFD} PUBLIC
        AltisCommon 
    )
    target_link_libraries(${EMULATOR_TARGET_CFDD} PUBLIC
        AltisCommon 
    )

    ###############################################################################
    ### Generate Report
    ###############################################################################
    set(FPGA_EARLY_IMAGE_CFD cfdLib_report.a)
    set(FPGA_EARLY_IMAGE_CFDD cfddoubleLib_report.a)
    add_executable(${FPGA_EARLY_IMAGE_CFD} ${exe_files} euler3d.dp.cpp)
    add_executable(${FPGA_EARLY_IMAGE_CFDD} ${exe_files} euler3d_double.dp.cpp)
    add_custom_target(cfdLib_fpga_report DEPENDS ${FPGA_EARLY_IMAGE_CFD})
    add_custom_target(cfddoubleLib_fpga_report DEPENDS ${FPGA_EARLY_IMAGE_CFDD})
    set_target_properties(${FPGA_EARLY_IMAGE_CFD} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS}")
    set_target_properties(${FPGA_EARLY_IMAGE_CFD} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -fsycl-link=early")
    set_target_properties(${FPGA_EARLY_IMAGE_CFDD} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS} -Xsdsp-mode=prefer-dsp")
    set_target_properties(${FPGA_EARLY_IMAGE_CFDD} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -Xsdsp-mode=prefer-dsp -fsycl-link=early")

    target_link_libraries(${FPGA_EARLY_IMAGE_CFD} PUBLIC
        AltisCommon 
    )
    target_link_libraries(${FPGA_EARLY_IMAGE_CFDD} PUBLIC
        AltisCommon 
    )

    ###############################################################################
    ### FPGA Hardware
    ###############################################################################
    add_executable(${FPGA_TARGET_CFD} EXCLUDE_FROM_ALL ${exe_files} euler3d.dp.cpp)
    add_executable(${FPGA_TARGET_CFDD} EXCLUDE_FROM_ALL ${exe_files} euler3d_double.dp.cpp)
    add_custom_target(cfdLib_fpga DEPENDS ${FPGA_TARGET_CFD})
    add_custom_target(cfddoubleLib_fpga DEPENDS ${FPGA_TARGET_CFDD})
    set_target_properties(${FPGA_TARGET_CFD} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS} -qactypes -reuse-exe=${CMAKE_CURRENT_SOURCE_DIR}/../../../build/cuda/level2/cfd/cfdLib.fpga")
    set_target_properties(${FPGA_TARGET_CFD} PROPERTIES LINK_FLAGS " ${HARDWARE_LINK_FLAGS} -qactypes -reuse-exe=${CMAKE_CURRENT_SOURCE_DIR}/../../../build/cuda/level2/cfd/cfdLib.fpga")
    set_target_properties(${FPGA_TARGET_CFDD} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS} -qactypes -reuse-exe=${CMAKE_CURRENT_SOURCE_DIR}/../../../build/cuda/level2/cfd/cfddoubleLib.fpga")
    set_target_properties(${FPGA_TARGET_CFDD} PROPERTIES LINK_FLAGS " ${HARDWARE_LINK_FLAGS} -qactypes -reuse-exe=${CMAKE_CURRENT_SOURCE_DIR}/../../../build/cuda/level2/cfd/cfddoubleLib.fpga")

    target_link_libraries(${FPGA_TARGET_CFD} PUBLIC
        AltisCommon 
    )
    target_link_libraries(${FPGA_TARGET_CFDD} PUBLIC
        AltisCommon  
    )
endif()
