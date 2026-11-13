# ----------------------------------------------------------------------------
# Shared plugin-support libraries consumed by multiple standard plugins.
#
# This directory is registered FIRST in the parent CMakeLists so the targets
# exist before any consumer's target_link_libraries resolves (qSIBR links
# video_base and appears earlier than qFreeSplatter in the plugin list).
# Each subdirectory owns its own CMakeLists and documents its contract.
# ----------------------------------------------------------------------------

add_subdirectory( video_base )
add_subdirectory( spirula_base )
