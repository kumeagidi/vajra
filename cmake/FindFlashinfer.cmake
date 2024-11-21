# Function to find Python site-packages directory
function(find_python_site_packages_dir result)
    execute_process(
        COMMAND python3 -c "import site; print(site.getsitepackages()[0])"
        OUTPUT_VARIABLE SITE_PACKAGES_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    set(${result} ${SITE_PACKAGES_DIR} PARENT_SCOPE)
endfunction()

# Try to find the Flashinfer base path
find_python_site_packages_dir(PYTHON_SITE_PACKAGES_DIR)
if(PYTHON_SITE_PACKAGES_DIR)
    set(FLASHINFER_BASE_PATH "${PYTHON_SITE_PACKAGES_DIR}/flashinfer")
else()
    message(WARNING "Unable to detect Python site-packages directory. Set FLASHINFER_BASE_PATH manually if needed.")
endif()

# Allow manual override of base path
if(DEFINED ENV{FLASHINFER_BASE_PATH})
    set(FLASHINFER_BASE_PATH $ENV{FLASHINFER_BASE_PATH})
endif()

# Find Flashinfer libraries
file(GLOB FLASHINFER_LIBRARIES "${FLASHINFER_BASE_PATH}/*.so")

# Handle the REQUIRED and QUIET arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Flashinfer
    REQUIRED_VARS FLASHINFER_LIBRARIES
    FAIL_MESSAGE "Failed to find Flashinfer libraries"
)

# Create an interface library target with full path linking
if(Flashinfer_FOUND AND NOT TARGET Flashinfer::Flashinfer)
    add_library(Flashinfer::Flashinfer INTERFACE IMPORTED)
    
    # Add each library as a separate imported target with absolute path
    foreach(lib ${FLASHINFER_LIBRARIES})
        get_filename_component(lib_name ${lib} NAME_WE)
        get_filename_component(lib_path ${lib} ABSOLUTE)
        add_library(Flashinfer::${lib_name} SHARED IMPORTED)
        set_target_properties(Flashinfer::${lib_name} PROPERTIES
            IMPORTED_LOCATION "${lib_path}"
        )
        target_link_libraries(Flashinfer::Flashinfer INTERFACE "-Wl,--no-as-needed,${lib_path}")
    endforeach()
endif()

# Mark variables as advanced
mark_as_advanced(FLASHINFER_LIBRARIES)

# Print found libraries for debugging
message(STATUS "Flashinfer libraries found: ${FLASHINFER_LIBRARIES}")
