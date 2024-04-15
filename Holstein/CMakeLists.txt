# Normalizing flows executables
add_subdirectory(ML)

# Diagrammatic executable
add_executable(Holstein Holstein.cpp)
target_link_libraries(Holstein PRIVATE NFDMC MLHol)
target_link_options(Holstein PRIVATE -pthread)
