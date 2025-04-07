cmake_minimum_required(VERSION 3.10)
project(Arc_Cuda CUDA CXX)


# Set standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_ARCHITECTURES "all")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -g -fno-omit-frame-pointer -O1")
# set(CMAKE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS} -fsanitize=address")# Link CUDA
set(CMAKE_BUILD_TYPE Debug)

# Policies
if(POLICY CMP0072)
  cmake_policy(SET CMP0072 NEW)  # Prefer GLVND OpenGL
endif()
if(POLICY CMP0104)
  cmake_policy(SET CMP0104 NEW)  # For CUDA architecture detection
endif()

# Set CUDA language for .cu files
set_source_files_properties(src/kernel.cu PROPERTIES LANGUAGE CUDA)

# Find packages
find_package(CUDA REQUIRED)
find_package(LibWebSockets REQUIRED)
find_package(LibDataChannel REQUIRED)

# Include directories
include_directories(${CUDA_INCLUDE_DIRS})

# Source files
set(SOURCES src/main.cpp src/simulation.cpp src/server.cpp src/kernel.cu)

# Define executable
add_executable(${PROJECT_NAME} ${SOURCES})


target_link_libraries(${PROJECT_NAME} PRIVATE ${CUDA_LIBRARIES})

# Debug and link LibWebSockets
message(STATUS "LibWebSockets_DIR: ${LibWebSockets_DIR}")
message(STATUS "LIBWEBSOCKETS_LIBRARIES: ${LIBWEBSOCKETS_LIBRARIES}")
if(TARGET websockets)
  message(STATUS "Using target websockets")
  target_link_libraries(${PROJECT_NAME} PRIVATE websockets)
elseif(TARGET Libwebsockets::websockets)
  message(STATUS "Using target Libwebsockets::websockets")
  target_link_libraries(${PROJECT_NAME} PRIVATE Libwebsockets::websockets)
elseif(LIBWEBSOCKETS_LIBRARIES)
  message(STATUS "Falling back to LIBWEBSOCKETS_LIBRARIES: ${LIBWEBSOCKETS_LIBRARIES}")
  target_include_directories(${PROJECT_NAME} PRIVATE ${LIBWEBSOCKETS_INCLUDE_DIRS})
  target_link_libraries(${PROJECT_NAME} PRIVATE ${LIBWEBSOCKETS_LIBRARIES})
else()
  message(FATAL_ERROR "LibWebSockets found but no usable targets or libraries defined")
endif()

# Debug and link LibDataChannel
message(STATUS "LibDataChannel_DIR: ${LibDataChannel_DIR}")
if(TARGET LibDataChannel::LibDataChannel)
  message(STATUS "Using target LibDataChannel::LibDataChannel")
  target_link_libraries(${PROJECT_NAME} PRIVATE LibDataChannel::LibDataChannel)
elseif(LIBDATACHANNEL_LIBRARIES)
  message(STATUS "Falling back to LIBDATACHANNEL_LIBRARIES: ${LIBDATACHANNEL_LIBRARIES}")
  target_include_directories(${PROJECT_NAME} PRIVATE ${LIBDATACHANNEL_INCLUDE_DIRS})
  target_link_libraries(${PROJECT_NAME} PRIVATE ${LIBDATACHANNEL_LIBRARIES})
else()
  message(FATAL_ERROR "LibDataChannel found but no targets or libraries defined")
endif()

# Uncomment for OpenGL/GLFW/GLEW if needed later
# find_package(OpenGL REQUIRED)
# find_package(glfw3 REQUIRED)
# find_package(GLEW REQUIRED)
# target_link_libraries(${PROJECT_NAME} PRIVATE ${OPENGL_LIBRARIES} GLEW::GLEW glfw)
