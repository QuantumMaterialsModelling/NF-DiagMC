add_library(DMC SHARED
  Axis.cpp
  Histogram.cpp
  LinearFunApprox.cpp
  NewtonFunApprox.cpp
  CDFInverse.cpp
  SimOptions.cpp
  Bijection.cpp
  )

target_include_directories(DMC PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(DMC PRIVATE ${Boost_LIBRARIES})
target_compile_options(DMC PUBLIC -O3)


if(${TORCH_FOUND})
  add_library(NF SHARED
    fmanager.cpp
    MLP.cpp
    base.cpp
    target.cpp
    histogram.cpp
    norm.cpp
    MLinear.cpp
    RQS.cpp
    Std.cpp
    )

  target_include_directories(NF PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include/)
  target_link_libraries(NF PUBLIC "${TORCH_LIBRARIES}")
  set_property(TARGET NF PROPERTY CXX_STANDARD 17)
  target_compile_options(NF PUBLIC -O3)


  add_library(NFDMC SHARED
    fmanager.cpp
    MLP.cpp
    base.cpp
    target.cpp
    histogram.cpp
    norm.cpp
    MLinear.cpp
    RQS.cpp
    Std.cpp
    Axis.cpp
    Histogram.cpp
    LinearFunApprox.cpp
    NewtonFunApprox.cpp
    CDFInverse.cpp
    SimOptions.cpp
    Bijection.cpp
    )

  target_include_directories(NFDMC PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include/)
  target_link_libraries(NFDMC PUBLIC "${TORCH_LIBRARIES}" ${Boost_LIBRARIES})
  set_property(TARGET NFDMC PROPERTY CXX_STANDARD 17)
  target_compile_options(NFDMC PUBLIC -O3)
endif()

