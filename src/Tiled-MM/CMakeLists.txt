add_library(Tiled-MM STATIC gpu_context.cpp
                            mm_handle.cpp
                            tile_dim.cpp
                            tiled_matrix.cpp
                            tiled_mm.cpp
                            tile_coord.cpp)

target_link_libraries(Tiled-MM PUBLIC CUDA::cublas)

target_include_directories(Tiled-MM PUBLIC 
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
    )

target_compile_features(Tiled-MM PUBLIC cxx_std_14)

if(TILEDMM_WITH_INSTALL)
    install(TARGETS Tiled-MM
            EXPORT Tiled-MM_targets
            LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
            ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
            INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
endif()
