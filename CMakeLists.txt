cmake_minimum_required(VERSION 3.15...3.30)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

# Headers and libmujoco come from the installed mujoco wheel.
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c
    "import mujoco, pathlib; print(pathlib.Path(mujoco.__file__).parent)"
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE MUJOCO_PKG_DIR
  COMMAND_ERROR_IS_FATAL ANY)
file(GLOB MUJOCO_LIBRARY "${MUJOCO_PKG_DIR}/libmujoco*${CMAKE_SHARED_LIBRARY_SUFFIX}*")
if(NOT MUJOCO_LIBRARY)
  message(FATAL_ERROR "Could not find libmujoco in ${MUJOCO_PKG_DIR}")
endif()
list(GET MUJOCO_LIBRARY 0 MUJOCO_LIBRARY)

nanobind_add_module(_bindings FREE_THREADED NOMINSIZE src/mjbatch/csrc/bindings.cpp)
target_include_directories(_bindings PRIVATE "${MUJOCO_PKG_DIR}/include")
target_link_libraries(_bindings PRIVATE "${MUJOCO_LIBRARY}")

# Find libmujoco next to the installed extension, <site-packages>/mujoco. The
# macOS wheel's install name is a framework path that does not exist, so point
# our load command at the plain file name instead.
if(APPLE)
  get_filename_component(MUJOCO_LIB_NAME "${MUJOCO_LIBRARY}" NAME)
  execute_process(COMMAND otool -D "${MUJOCO_LIBRARY}" OUTPUT_VARIABLE _otool)
  string(REGEX MATCH "@rpath[^\n]*" MUJOCO_INSTALL_NAME "${_otool}")
  add_custom_command(TARGET _bindings POST_BUILD COMMAND install_name_tool -change
    "${MUJOCO_INSTALL_NAME}" "@rpath/${MUJOCO_LIB_NAME}" $<TARGET_FILE:_bindings>)
  set_target_properties(_bindings PROPERTIES INSTALL_RPATH "@loader_path/../mujoco")
else()
  set_target_properties(_bindings PROPERTIES INSTALL_RPATH "$ORIGIN/../mujoco")
endif()

install(TARGETS _bindings DESTINATION mjbatch)
