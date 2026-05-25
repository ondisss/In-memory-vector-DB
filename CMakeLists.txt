cmake_minimum_required(VERSION 3.16)
project(VectorDatabase VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type (Debug, Release, RelWithDebInfo)" FORCE)
endif()

option(ENABLE_ASAN "Enable Address Sanitizer" ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG")

if(ENABLE_ASAN AND CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")
endif()

add_compile_options(-Wall -Wextra -Wpedantic)

# nlohmann/json se stáhne automaticky při prvním cmake ..
include(FetchContent)
FetchContent_Declare(
    nlohmann_json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
    URL_HASH SHA256=d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d
)
FetchContent_MakeAvailable(nlohmann_json)

add_executable(vecdb
    src/main.cpp
    src/ui.cpp
    src/db_commands.cpp
    src/embed_commands.cpp
    src/VecMath.cpp
    src/VectorDatabase.cpp
    src/HNSWIndex.cpp
    src/EmbedClient.cpp
)
target_include_directories(vecdb PRIVATE include src)
target_link_libraries(vecdb PRIVATE nlohmann_json::nlohmann_json)

# GNU readline – tab completion v CLI (volitelné)
find_library(READLINE_LIB readline)
if(READLINE_LIB)
    message(STATUS "readline nalezena: ${READLINE_LIB}")
    target_link_libraries(vecdb PRIVATE ${READLINE_LIB})
    target_compile_definitions(vecdb PRIVATE HAVE_READLINE)
else()
    message(STATUS "readline nenalezena – tab completion nebude fungovat")
endif()

# Unit testy (assert-based, bez externího framework)
add_executable(vecdb_tests
    src/tests.cpp
    src/VecMath.cpp
    src/VectorDatabase.cpp
    src/HNSWIndex.cpp
)
target_include_directories(vecdb_tests PRIVATE include)
target_link_libraries(vecdb_tests PRIVATE nlohmann_json::nlohmann_json)
