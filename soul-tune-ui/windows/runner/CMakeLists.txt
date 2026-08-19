cmake_minimum_required(VERSION 3.14)
project(runner LANGUAGES CXX)

# Define the application target. To change its name, change BINARY_NAME in the
# top-level CMakeLists.txt, not the value here, or `flutter run` will no longer
# work.
#
# Any new source files that you add to the application should be added here.
add_executable(${BINARY_NAME} WIN32
  "flutter_window.cpp"
  "main.cpp"
  "utils.cpp"
  "win32_window.cpp"
  "${FLUTTER_MANAGED_DIR}/generated_plugin_registrant.cc"
  "Runner.rc"
  "runner.exe.manifest"
)

# Apply the standard set of build settings. This can be removed for applications
# that need different build settings.
apply_standard_settings(${BINARY_NAME})

# Add preprocessor definitions for the build version.
target_compile_definitions(${BINARY_NAME} PRIVATE "FLUTTER_VERSION=\"${FLUTTER_VERSION}\"")
target_compile_definitions(${BINARY_NAME} PRIVATE "FLUTTER_VERSION_MAJOR=${FLUTTER_VERSION_MAJOR}")
target_compile_definitions(${BINARY_NAME} PRIVATE "FLUTTER_VERSION_MINOR=${FLUTTER_VERSION_MINOR}")
target_compile_definitions(${BINARY_NAME} PRIVATE "FLUTTER_VERSION_PATCH=${FLUTTER_VERSION_PATCH}")
target_compile_definitions(${BINARY_NAME} PRIVATE "FLUTTER_VERSION_BUILD=${FLUTTER_VERSION_BUILD}")

# Disable Windows macros that collide with C++ standard library functions.
target_compile_definitions(${BINARY_NAME} PRIVATE "NOMINMAX")

# Add dependency libraries and include directories. Add any application-specific
# dependencies here.
target_link_libraries(${BINARY_NAME} PRIVATE flutter flutter_wrapper_app)
target_link_libraries(${BINARY_NAME} PRIVATE "dwmapi.lib")
target_include_directories(${BINARY_NAME} PRIVATE "${CMAKE_SOURCE_DIR}")

# Run the Flutter tool portions of the build. This must not be removed.
add_dependencies(${BINARY_NAME} flutter_assemble)

# === Rust native library (flutter_rust_bridge) ===
# Build the cdylib with cargo and copy it next to the runner executable,
# otherwise the app fails to start ("window can't display" — RustLib.init()
# cannot load soul_tune_api.dll).
# 仓库根 = windows/runner 的上一级再上一级（soul-tune-ui/..）
set(SOUL_TUNE_CARGO_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../..")
# Debug/Profile -> debug（省略 --profile，debug 是保留名）；Release -> --release
add_custom_command(TARGET ${BINARY_NAME} POST_BUILD
  COMMAND cargo build -p soul-tune-api
          --manifest-path "${SOUL_TUNE_CARGO_DIR}/Cargo.toml"
          $<$<CONFIG:Release>:--release>
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${SOUL_TUNE_CARGO_DIR}/target/$<IF:$<CONFIG:Release>,release,debug>/soul_tune_api.dll"
          "$<TARGET_FILE_DIR:${BINARY_NAME}>"
  COMMENT "Building & copying soul_tune_api.dll (Rust FRB bridge)"
)
