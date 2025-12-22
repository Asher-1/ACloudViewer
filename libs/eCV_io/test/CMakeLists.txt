# Qt5/Qt6 compatibility - Test module
# Note: Qt6 does not have a separate Test module, it's part of Core
# This test is only available when using Qt5
if(NOT USE_QT6)
    find_package( Qt5Test REQUIRED )

    if ( OPTION_USE_SHAPE_LIB )
        add_executable( TestShpFilter )

        target_sources( TestShpFilter
            PRIVATE
                ${CMAKE_CURRENT_LIST_DIR}/TestShpFilter.cpp
                ${CMAKE_CURRENT_LIST_DIR}/TestShpFilter.h
                ${CMAKE_CURRENT_LIST_DIR}/TestData.qrc
        )

        # Note: Qt5::Test is Qt5-only, no unified alias needed
        target_link_libraries( TestShpFilter
            ECV_IO_LIB
            Qt5::Test
        )

        if ( WIN32 )
            set_target_properties( TestShpFilter PROPERTIES
                WIN32_EXECUTABLE False
            )
        endif()

        add_test( NAME TestShpFilter COMMAND TestShpFilter )
    endif()
endif()
