cmake_minimum_required (VERSION 3.0)
project (Raytracer)
set (Base_VERSION_MAJOR 0)
set (Base_VERSION_MINOR 1)
set (Base_VERSION 0.1)

#include (settings.cmake)

set(INSTALL_LIB_DIR lib CACHE PATH "Installation directory for libraries.")
set(INSTALL_BIN_DIR bin CACHE PATH "Installation directory for binaries.")
set(INSTALL_INCLUDE_DIR include CACHE PATH "Installation directory for includes.")
set(INSTALL_CMAKE_DIR cmake CACHE PATH "Installation directory for cmake files.")

set (SOURCE_FILES 
	src/main.cpp
)

add_executable (Raytracer
	${SOURCE_FILES}
)

target_compile_features(Raytracer
	PRIVATE cxx_range_for)
