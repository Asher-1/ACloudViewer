set(header_list
   ${header_list}
   ${CMAKE_CURRENT_SOURCE_DIR}/include/Updater.h
   ${CMAKE_CURRENT_SOURCE_DIR}/include/Downloader.h
   ${CMAKE_CURRENT_SOURCE_DIR}/include/QSimpleUpdater.h
   PARENT_SCOPE
)

set(source_list
   ${source_list}
   ${CMAKE_CURRENT_SOURCE_DIR}/src/Updater.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/src/Downloader.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/src/QSimpleUpdater.cpp
   PARENT_SCOPE
)

set( ui_list
        ${ui_list}
        ${CMAKE_CURRENT_SOURCE_DIR}/ui/Downloader.ui
        PARENT_SCOPE
)

set( qrc_list
	${qrc_list}
	${CMAKE_CURRENT_SOURCE_DIR}/etc/resources/qsimpleupdater.qrc
	PARENT_SCOPE
)
