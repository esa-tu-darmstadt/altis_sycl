cmake_minimum_required(VERSION 3.8)

set(DWT_CUDA_DIR ${CMAKE_CURRENT_LIST_DIR}/dwt_cuda)
set(DWT_DATA_DIR ${CMAKE_CURRENT_LIST_DIR}/data)

set(sources ${sources} 
    dwt_main.dp.cpp
    dwt.dp.cpp
    components.dp.cpp
    ${DWT_CUDA_DIR}/fdwt53.dp.cpp
    #${DWT_CUDA_DIR}/rdwt53.dp.cpp
    ${DWT_CUDA_DIR}/fdwt97.dp.cpp
    ${DWT_CUDA_DIR}/rdwt97.dp.cpp   
    ${DWT_DATA_DIR}/bmp.cpp
)

add_library(dwt2dLib ${sources})
target_include_directories(dwt2dLib PRIVATE dwt_cuda)

if (DEFINED USE_FPGA)
    set(EMULATOR_TARGET_DWT dwt2dLib.fpga_emu)
    set(FPGA_TARGET_DWT dwt2dLib.fpga)

    ###############################################################################
    ### FPGA Emulator
    ###############################################################################
    add_executable(${EMULATOR_TARGET_DWT} ${exe_files} ${sources})
    set_target_properties(${EMULATOR_TARGET_DWT} PROPERTIES COMPILE_FLAGS "${EMULATOR_COMPILE_FLAGS}")
    set_target_properties(${EMULATOR_TARGET_DWT} PROPERTIES LINK_FLAGS "${EMULATOR_LINK_FLAGS}")
    add_custom_target(dwt2dLib_fpga_emu DEPENDS ${EMULATOR_TARGET_DWT})

    target_link_libraries(${EMULATOR_TARGET_DWT} PUBLIC
        AltisCommon
    )

    ###############################################################################
    ### Generate Report
    ###############################################################################
    set(FPGA_EARLY_IMAGE_DWT dwt2dLib_report.a)
    add_executable(${FPGA_EARLY_IMAGE_DWT} ${exe_files} ${sources})
    add_custom_target(dwt2dLib_fpga_report DEPENDS ${FPGA_EARLY_IMAGE_DWT})
    set_target_properties(${FPGA_EARLY_IMAGE_DWT} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS} -Xsnum-reorder=1 -Xsforce-single-store-ring -Xshyper-optimized-handshaking=off -Xsno-hardware-kernel-invocation-queue")
    set_target_properties(${FPGA_EARLY_IMAGE_DWT} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -Xsnum-reorder=1 -Xsforce-single-store-ring -Xshyper-optimized-handshaking=off -Xsno-hardware-kernel-invocation-queue -fsycl-link=early")

    target_link_libraries(${FPGA_EARLY_IMAGE_DWT} PUBLIC
        AltisCommon
    )

    ###############################################################################
    ### FPGA Hardware
    ###############################################################################
    add_executable(${FPGA_TARGET_DWT} EXCLUDE_FROM_ALL ${exe_files} ${sources})
    add_custom_target(dwt2dLib_fpga DEPENDS ${FPGA_TARGET_DWT})
    set_target_properties(${FPGA_TARGET_DWT} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS} -Xsnum-reorder=1 -Xsforce-single-store-ring -Xshyper-optimized-handshaking=off -Xsno-hardware-kernel-invocation-queue -Xsv -Xsseed=1 -reuse-exe=${CMAKE_CURRENT_SOURCE_DIR}/../../../build/cuda/level2/dwt2d/dwt2dLib.fpga")
    set_target_properties(${FPGA_TARGET_DWT} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -Xsnum-reorder=1 -Xsforce-single-store-ring -Xshyper-optimized-handshaking=off -Xsno-hardware-kernel-invocation-queue -Xsv -Xsseed=1 -reuse-exe=${CMAKE_CURRENT_SOURCE_DIR}/../../../build/cuda/level2/dwt2d/dwt2dLib.fpga")

    target_link_libraries(${FPGA_TARGET_DWT} PUBLIC
        AltisCommon
    )
endif()
