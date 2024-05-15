file(GLOB_RECURSE SRC
    "${CMAKE_CURRENT_LIST_DIR}/*.cpp"
)

file(GLOB_RECURSE CSRC
    "${CMAKE_CURRENT_LIST_DIR}/*.c"
)

add_library(src ${SRC} ${CSRC})
target_link_libraries(src
    ${lite.ai.toolkit_LIBS}
)