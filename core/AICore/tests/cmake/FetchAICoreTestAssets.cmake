# Download the small, fixed AICore functional fixture set on developer and
# self-hosted GPU machines. GitHub-hosted CI disables this option by default.

function(_aicore_fetch_asset root relative url minimum_bytes)
    set(destination "${root}/${relative}")
    if(EXISTS "${destination}")
        file(SIZE "${destination}" existing_size)
        if(existing_size GREATER_EQUAL minimum_bytes)
            return()
        endif()
        file(REMOVE "${destination}")
    endif()
    get_filename_component(destination_dir "${destination}" DIRECTORY)
    file(MAKE_DIRECTORY "${destination_dir}")
    set(partial "${destination}.part")
    message(STATUS "AICore tests: downloading ${relative}")
    file(DOWNLOAD "${url}" "${partial}" STATUS status SHOW_PROGRESS
         TLS_VERIFY ON)
    list(GET status 0 result)
    list(GET status 1 detail)
    if(NOT result EQUAL 0)
        file(REMOVE "${partial}")
        message(WARNING "AICore tests: could not download ${relative}: ${detail}")
        return()
    endif()
    file(SIZE "${partial}" downloaded_size)
    if(downloaded_size LESS minimum_bytes)
        file(REMOVE "${partial}")
        message(WARNING "AICore tests: ${relative} is unexpectedly small")
        return()
    endif()
    file(RENAME "${partial}" "${destination}")
endfunction()

function(aicore_fetch_test_assets root)
    if(NOT root)
        message(WARNING "AICore tests: no asset root; model tests will skip")
        return()
    endif()
    set(base "https://github.com/Asher-1/cloudViewer_downloads/releases/download")
    _aicore_fetch_asset("${root}" "da3_models/depth-anything-base-q4_k.gguf"
        "${base}/DA3/depth-anything-base-q4_k.gguf" 90000000)
    _aicore_fetch_asset("${root}" "freesplatter_models/freesplatter-scene-q8_0.gguf"
        "${base}/3dgs/freesplatter-scene-q8_0.gguf" 300000000)
    _aicore_fetch_asset("${root}" "lightglue_models/aliked-n16rot-f32.gguf"
        "${base}/LightGlue/aliked-n16rot-f32.gguf" 2000000)
    _aicore_fetch_asset("${root}" "lightglue_models/aliked-lightglue-q8_0.gguf"
        "${base}/LightGlue/aliked-lightglue-q8_0.gguf" 10000000)
    _aicore_fetch_asset("${root}" "deeplsd_models/deeplsd_md-q8_0.gguf"
        "${base}/DeepLSD/deeplsd_md-q8_0.gguf" 8000000)
    _aicore_fetch_asset("${root}" "facedetect_models/buffalo_l.gguf"
        "${base}/qFaceDetect/buffalo_l.gguf" 150000000)

    set(lightglue_assets
        "sacre_coeur1.jpg"
        "sacre_coeur2.jpg")
    foreach(asset IN LISTS lightglue_assets)
        _aicore_fetch_asset("${root}" "lightglue_test_images/${asset}"
            "https://raw.githubusercontent.com/cvg/LightGlue/main/assets/${asset}"
            100000)
    endforeach()

    set(faces_zip "${root}/friends_faces.zip")
    if(NOT EXISTS "${root}/friends_faces/query/friends1.jpg")
        _aicore_fetch_asset("${root}" "friends_faces.zip"
            "${base}/qFaceDetect/friends_faces.zip" 100000)
        if(EXISTS "${faces_zip}")
            execute_process(
                # Do not request gzip decompression: this fixture is a ZIP
                # archive and CMake's archive reader handles it on all hosts.
                COMMAND "${CMAKE_COMMAND}" -E tar xvf "${faces_zip}"
                WORKING_DIRECTORY "${root}"
                RESULT_VARIABLE extract_result
                ERROR_VARIABLE extract_error)
            if(NOT extract_result EQUAL 0)
                message(WARNING "AICore tests: could not extract friends_faces.zip: ${extract_error}")
            endif()
        endif()
    endif()
endfunction()

if(DEFINED AICORE_TEST_ASSET_ROOT AND NOT AICORE_TEST_ASSET_ROOT STREQUAL "")
    aicore_fetch_test_assets("${AICORE_TEST_ASSET_ROOT}")
endif()
