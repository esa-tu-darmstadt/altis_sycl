cmake_minimum_required(VERSION 3.8)

add_library(particlefilterLib ex_particle_CUDA_naive_seq.dp.cpp)
target_link_libraries(particlefilterLib -lm)

add_library(particlefilterfloatLib ex_particle_CUDA_float_seq.dp.cpp)
target_link_libraries(particlefilterfloatLib -lm)

if (DEFINED USE_FPGA)
    set(EMULATOR_TARGET_PARTF particlefilterfloatLib.fpga_emu)
    set(FPGA_TARGET_PARTF particlefilterfloatLib.fpga)
    set(EMULATOR_TARGET_PARTN particlefilterLib.fpga_emu)
    set(FPGA_TARGET_PARTN particlefilterLib.fpga)

    ###############################################################################
    ### FPGA Emulator
    ###############################################################################
    add_executable(${EMULATOR_TARGET_PARTF} ${exe_files} ex_particle_CUDA_float_seq.dp.cpp)
    set_target_properties(${EMULATOR_TARGET_PARTF} PROPERTIES COMPILE_FLAGS "${EMULATOR_COMPILE_FLAGS}")
    set_target_properties(${EMULATOR_TARGET_PARTF} PROPERTIES LINK_FLAGS "${EMULATOR_LINK_FLAGS}")
    add_custom_target(particlefilterfloatLib_fpga_emu DEPENDS ${EMULATOR_TARGET_PARTF})

    target_link_libraries(${EMULATOR_TARGET_PARTF} PUBLIC
        AltisCommon 
    )
    add_executable(${EMULATOR_TARGET_PARTN} ${exe_files} ex_particle_CUDA_naive_seq.dp.cpp)
    set_target_properties(${EMULATOR_TARGET_PARTN} PROPERTIES COMPILE_FLAGS "${EMULATOR_COMPILE_FLAGS}")
    set_target_properties(${EMULATOR_TARGET_PARTN} PROPERTIES LINK_FLAGS "${EMULATOR_LINK_FLAGS}")
    add_custom_target(particlefilterLib_fpga_emu DEPENDS ${EMULATOR_TARGET_PARTN})

    target_link_libraries(${EMULATOR_TARGET_PARTN} PUBLIC
        AltisCommon 
    )

    ###############################################################################
    ### Generate Report
    ###############################################################################
    set(FPGA_EARLY_IMAGE_PARTF particlefilterfloatLib_report.a)
    add_executable(${FPGA_EARLY_IMAGE_PARTF} ${exe_files} ex_particle_CUDA_float_seq.dp.cpp)
    add_custom_target(particlefilterfloatLib_fpga_report DEPENDS ${FPGA_EARLY_IMAGE_PARTF})
    set_target_properties(${FPGA_EARLY_IMAGE_PARTF} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS}")
    set_target_properties(${FPGA_EARLY_IMAGE_PARTF} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -fsycl-link=early")

    target_link_libraries(${FPGA_EARLY_IMAGE_PARTF} PUBLIC
        AltisCommon 
    )
    set(FPGA_EARLY_IMAGE_PARTN particlefilterLib_report.a)
    add_executable(${FPGA_EARLY_IMAGE_PARTN} ${exe_files} ex_particle_CUDA_naive_seq.dp.cpp)
    add_custom_target(particlefilterLib_fpga_report DEPENDS ${FPGA_EARLY_IMAGE_PARTN})
    set_target_properties(${FPGA_EARLY_IMAGE_PARTN} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS}")
    set_target_properties(${FPGA_EARLY_IMAGE_PARTN} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -fsycl-link=early")

    target_link_libraries(${FPGA_EARLY_IMAGE_PARTN} PUBLIC
        AltisCommon 
    )

    ###############################################################################
    ### FPGA Hardware
    ###############################################################################
    add_executable(${FPGA_TARGET_PARTF} EXCLUDE_FROM_ALL ${exe_files} ex_particle_CUDA_float_seq.dp.cpp)
    add_custom_target(particlefilterfloatLib_fpga DEPENDS ${FPGA_TARGET_PARTF})
    set_target_properties(${FPGA_TARGET_PARTF} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS} -reuse-exe=${CMAKE_CURRENT_SOURCE_DIR}/../../../build/cuda/level2/particlefilter/particlefilterfloatLib.fpga")
    set_target_properties(${FPGA_TARGET_PARTF} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS}  -reuse-exe=${CMAKE_CURRENT_SOURCE_DIR}/../../../build/cuda/level2/particlefilter/particlefilterfloatLib.fpga")

    target_link_libraries(${FPGA_TARGET_PARTF} PUBLIC
        AltisCommon 
    )
    add_executable(${FPGA_TARGET_PARTN} EXCLUDE_FROM_ALL ${exe_files} ex_particle_CUDA_naive_seq.dp.cpp)
    add_custom_target(particlefilterLib_fpga DEPENDS ${FPGA_TARGET_PARTN})
    set_target_properties(${FPGA_TARGET_PARTN} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS} -reuse-exe=${CMAKE_CURRENT_SOURCE_DIR}/../../../build/cuda/level2/particlefilter/particlefilterLib.fpga")
    set_target_properties(${FPGA_TARGET_PARTN} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -reuse-exe=${CMAKE_CURRENT_SOURCE_DIR}/../../../build/cuda/level2/particlefilter/particlefilterLib.fpga")

    target_link_libraries(${FPGA_TARGET_PARTN} PUBLIC
        AltisCommon 
    )
endif()
