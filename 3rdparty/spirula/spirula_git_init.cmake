# Give the extracted Spirula tag archive its own git identity.
#
# Upstream's source lints (cmake/SsChecks.cmake) locate the repo root with
# `git rev-parse --show-toplevel` and inspect uncommitted changes. A tag
# archive carries no .git, so inside the ACloudViewer checkout the lint
# would crawl up and inspect the WRONG tree. Initializing a local repository
# in the extracted source reproduces exactly the environment the upstream CI
# builds in (a git checkout of the tag). This is an environment repair for
# the archive distribution form — NOT a source patch; no tracked file is
# modified.

if(NOT SS_SOURCE_DIR OR NOT EXISTS "${SS_SOURCE_DIR}")
    return()
endif()

if(EXISTS "${SS_SOURCE_DIR}/.git")
    return()  # idempotent across re-configures
endif()

find_package(Git QUIET)
if(NOT Git_FOUND)
    # Upstream itself skips the lints without git ("a checkout with no
    # python and no git still builds").
    return()
endif()

execute_process(COMMAND "${GIT_EXECUTABLE}" init -q
        WORKING_DIRECTORY "${SS_SOURCE_DIR}"
        RESULT_VARIABLE _r OUTPUT_QUIET ERROR_QUIET)
if(NOT _r EQUAL 0)
    return()
endif()

execute_process(COMMAND "${GIT_EXECUTABLE}" add -A
        WORKING_DIRECTORY "${SS_SOURCE_DIR}" OUTPUT_QUIET ERROR_QUIET)
execute_process(
        COMMAND "${GIT_EXECUTABLE}" -c user.email=build@local
                -c user.name=spirula-archive
                commit -q -m "spirula tag archive snapshot"
        WORKING_DIRECTORY "${SS_SOURCE_DIR}"
        OUTPUT_QUIET ERROR_QUIET)
