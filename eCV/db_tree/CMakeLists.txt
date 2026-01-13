
set(header_list
   ${header_list}
   ${CMAKE_CURRENT_SOURCE_DIR}/ecvDBRoot.h
   ${CMAKE_CURRENT_SOURCE_DIR}/ecvPropertiesTreeDelegate.h
   ${CMAKE_CURRENT_SOURCE_DIR}/ecvAxesGridDialog.h
   ${CMAKE_CURRENT_SOURCE_DIR}/ecvCustomLabelsWidget.h
   ${CMAKE_CURRENT_SOURCE_DIR}/matrixDisplayDlg.h
   ${CMAKE_CURRENT_SOURCE_DIR}/sfEditDlg.h
   PARENT_SCOPE
)

set(source_list
   ${source_list}
   ${CMAKE_CURRENT_SOURCE_DIR}/ecvDBRoot.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/ecvPropertiesTreeDelegate.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/ecvAxesGridDialog.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/ecvCustomLabelsWidget.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/matrixDisplayDlg.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/sfEditDlg.cpp
   PARENT_SCOPE
)
