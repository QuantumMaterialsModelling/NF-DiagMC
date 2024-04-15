add_library(MLHol SHARED
  Model/src/Option.cpp
  Model/src/Model.cpp)
target_link_libraries(MLHol PUBLIC "${TORCH_LIBRARIES}" NF)
target_include_directories(MLHol PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/Model/)
set_property(TARGET MLHol PROPERTY CXX_STANDARD 17)

add_executable(Htrain train.cpp)
target_link_libraries(Htrain MLHol)
set_property(TARGET Htrain PROPERTY CXX_STANDARD 17)

add_executable(Heval eval.cpp)
target_link_libraries(Heval MLHol)
set_property(TARGET Heval PROPERTY CXX_STANDARD 17)

