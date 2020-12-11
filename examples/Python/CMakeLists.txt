file(GLOB PY_benchmark_TUTORIAL_FILES "benchmark/*.py")
source_group("benchmark" FILES ${PY_benchmark_TUTORIAL_FILES})
file(GLOB PY_core_TUTORIAL_FILES "core/*.py")
source_group("core" FILES ${PY_core_TUTORIAL_FILES})
file(GLOB PY_geometry_TUTORIAL_FILES "geometry/*.py")
source_group("geometry" FILES ${PY_geometry_TUTORIAL_FILES})
file(GLOB PY_gui_TUTORIAL_FILES "gui/*.py")
source_group("gui" FILES ${PY_gui_TUTORIAL_FILES})
file(GLOB PY_misc_TUTORIAL_FILES "misc/*.py")
source_group("misc" FILES ${PY_misc_TUTORIAL_FILES})
file(GLOB PY_pipelines_TUTORIAL_FILES "pipelines/*.py")
source_group("pipelines" FILES ${PY_pipelines_TUTORIAL_FILES})
file(GLOB PY_reconstruction_system_TUTORIAL_FILES "reconstruction_system/*.py")
source_group("reconstruction_system" FILES ${PY_reconstruction_system_TUTORIAL_FILES})
file(GLOB PY_utility_TUTORIAL_FILES "utility/*.py")
source_group("utility" FILES ${PY_utility_TUTORIAL_FILES})
file(GLOB PY_visualization_TUTORIAL_FILES "visualization/*.py")
source_group("visualization" FILES ${PY_visualization_TUTORIAL_FILES})
add_custom_target(Python ALL SOURCES 
${PY_benchmark_TUTORIAL_FILES} 
${PY_core_TUTORIAL_FILES} 
${PY_geometry_TUTORIAL_FILES} 
${PY_gui_TUTORIAL_FILES} 
${PY_misc_TUTORIAL_FILES}
${PY_pipelines_TUTORIAL_FILES}
${PY_reconstruction_system_TUTORIAL_FILES}
${PY_utility_TUTORIAL_FILES}
${PY_visualization_TUTORIAL_FILES})
set_target_properties(Python PROPERTIES FOLDER "examples/")
