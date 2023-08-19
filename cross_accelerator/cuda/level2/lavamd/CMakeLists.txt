cmake_minimum_required(VERSION 3.8)

add_library(lavamdLib
    lavaMD.dp.cpp
    kernel/kernel_gpu_cuda_wrapper.dp.cpp
)
target_link_libraries(lavamdLib -lm -pthread)
target_include_directories(lavamdLib PRIVATE kernel)

if (DEFINED USE_FPGA)
    set(EMULATOR_TARGET_LMD lavamdLib.fpga_emu)
    set(FPGA_TARGET_LMD lavamdLib.fpga)

    ###############################################################################
    ### FPGA Emulator
    ###############################################################################
    add_executable(${EMULATOR_TARGET_LMD} ${exe_files} lavaMD.dp.cpp kernel/kernel_gpu_cuda_wrapper.dp.cpp)
    set_target_properties(${EMULATOR_TARGET_LMD} PROPERTIES COMPILE_FLAGS "${EMULATOR_COMPILE_FLAGS}")
    set_target_properties(${EMULATOR_TARGET_LMD} PROPERTIES LINK_FLAGS "${EMULATOR_LINK_FLAGS}")
    add_custom_target(lavamdLib_fpga_emu DEPENDS ${EMULATOR_TARGET_LMD})

    target_link_libraries(${EMULATOR_TARGET_LMD} PUBLIC
        AltisCommon
    )

    ###############################################################################
    ### Generate Report
    ###############################################################################
    set(FPGA_EARLY_IMAGE_LMD lavamdLib_report.a)
    add_executable(${FPGA_EARLY_IMAGE_LMD} ${exe_files} lavaMD.dp.cpp kernel/kernel_gpu_cuda_wrapper.dp.cpp)
    add_custom_target(lavamdLib_fpga_report DEPENDS ${FPGA_EARLY_IMAGE_LMD})
    set_target_properties(${FPGA_EARLY_IMAGE_LMD} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS}")
    set_target_properties(${FPGA_EARLY_IMAGE_LMD} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -fsycl-link=early")

    target_link_libraries(${FPGA_EARLY_IMAGE_LMD} PUBLIC
        AltisCommon
    )

    ###############################################################################
    ### FPGA Hardware
    ###############################################################################
    add_executable(${FPGA_TARGET_LMD} EXCLUDE_FROM_ALL ${exe_files} lavaMD.dp.cpp kernel/kernel_gpu_cuda_wrapper.dp.cpp)
    add_custom_target(lavamdLib_fpga DEPENDS ${FPGA_TARGET_LMD})
    set_target_properties(${FPGA_TARGET_LMD} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS} -reuse-exe=${CMAKE_CURRENT_SOURCE_DIR}/../../../build/cuda/level2/lavamd/lavamdLib.fpga")
    set_target_properties(${FPGA_TARGET_LMD} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS}  -reuse-exe=${CMAKE_CURRENT_SOURCE_DIR}/../../../build/cuda/level2/lavamd/lavamdLib.fpga")

    target_link_libraries(${FPGA_TARGET_LMD} PUBLIC
        AltisCommon
    )
endif()
