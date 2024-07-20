add_executable(SPH md.cu)

target_link_libraries(SPH PRIVATE Thrust)