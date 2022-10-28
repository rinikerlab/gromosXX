# set default build type to release for single generators
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING
            "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif()

# set options
option(OMP "enable OMP" ON)
option(MPI "enable MPI" OFF)
option(CUDAKERNEL "enable CUDA" OFF)
option(XTB "enable XTB" OFF) # -DXTB={XTB_ROOT}
option(FORCEGROUPS "enable forcegroups" OFF)
option(HEAVISIDE "enable heaviside" OFF)

set(CMAKE_CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

if(OMP AND MPI)
    message(FATAL_ERROR "OMP and MPI must NOT be enabled at the same time")
endif()

if(CUDAKERNEL)
    enable_language(CUDA)
endif()

# find option dependent packages
if(OMP)
    find_package(OpenMP REQUIRED)
    add_definitions(-DOMP)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

if(MPI)
    find_package(MPI REQUIRED)
    add_definitions(-DXXMPI)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MPI_CXX_COMPILE_FLAGS}")
    set(EXTERNAL_LIBRARIES ${EXTERNAL_LIBRARIES} ${MPI_CXX_LIBRARIES})
    set(EXTERNAL_INCLUDES ${EXTERNAL_INCLUDES} ${MPI_CXX_INCLUDE_PATH})
endif()