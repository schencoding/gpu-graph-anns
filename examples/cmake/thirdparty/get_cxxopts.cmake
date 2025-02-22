
function(find_and_configure_cxxopts)
    set(oneValueArgs VERSION)
    cmake_parse_arguments(PKG "${options}" "${oneValueArgs}"
            "${multiValueArgs}" ${ARGN} )
    #-----------------------------------------------------
    # Invoke CPM find_package()
    #-----------------------------------------------------
    rapids_cpm_find(cxxopts ${PKG_VERSION}
            CPM_ARGS
            GIT_REPOSITORY https://github.com/jarro2783/cxxopts
            GIT_TAGS v3.2.0
            )
endfunction()
find_and_configure_cxxopts(VERSION 3.2.0)
