nvbio_module(richard-test)

addsources(
richard_test.cu
)

cuda_add_executable(richard-test ${richard-test_srcs})
target_link_libraries(richard-test nvbio zlibstatic crcstatic lz4 ${SYSTEM_LINK_LIBRARIES})

