FetchContent_Declare(
    indicators
    GIT_REPOSITORY https://github.com/p-ranav/indicators.git
    GIT_TAG 222382c
    # '|| true': tolerate re-running this patch against an already-patched
    # checkout (e.g. on reconfigure) - patch already skips the hunk in that
    # case, but still exits non-zero, which would otherwise fail the build.
    PATCH_COMMAND patch -p1 -i ${CMAKE_SOURCE_DIR}/cmake/patches/Indicators-CMakeMinVer.patch || true
    EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(indicators)
