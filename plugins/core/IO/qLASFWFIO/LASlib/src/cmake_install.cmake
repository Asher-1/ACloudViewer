# Install script for directory: D:/develop/thirdParties/LAStools/LASlib/src

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "C:/Program Files/LAStools")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/LASlib" TYPE FILE FILES
    "D:/develop/thirdParties/LAStools/LASlib/inc/lasdefinitions.hpp"
    "D:/develop/thirdParties/LAStools/LASlib/inc/lasfilter.hpp"
    "D:/develop/thirdParties/LAStools/LASlib/inc/lasignore.hpp"
    "D:/develop/thirdParties/LAStools/LASlib/inc/laskdtree.hpp"
    "D:/develop/thirdParties/LAStools/LASlib/inc/lasreader.hpp"
    "D:/develop/thirdParties/LAStools/LASlib/inc/lasreader_asc.hpp"
    "D:/develop/thirdParties/LAStools/LASlib/inc/lasreader_bil.hpp"
    "D:/develop/thirdParties/LAStools/LASlib/inc/lasreader_bin.hpp"
    "D:/develop/thirdParties/LAStools/LASlib/inc/lasreader_dtm.hpp"
    "D:/develop/thirdParties/LAStools/LASlib/inc/lasreader_las.hpp"
    "D:/develop/thirdParties/LAStools/LASlib/inc/lasreader_ply.hpp"
    "D:/develop/thirdParties/LAStools/LASlib/inc/lasreader_qfit.hpp"
    "D:/develop/thirdParties/LAStools/LASlib/inc/lasreader_shp.hpp"
    "D:/develop/thirdParties/LAStools/LASlib/inc/lasreader_txt.hpp"
    "D:/develop/thirdParties/LAStools/LASlib/inc/lasreaderbuffered.hpp"
    "D:/develop/thirdParties/LAStools/LASlib/inc/lasreadermerged.hpp"
    "D:/develop/thirdParties/LAStools/LASlib/inc/lasreaderpipeon.hpp"
    "D:/develop/thirdParties/LAStools/LASlib/inc/lasreaderstored.hpp"
    "D:/develop/thirdParties/LAStools/LASlib/inc/lastransform.hpp"
    "D:/develop/thirdParties/LAStools/LASlib/inc/lasutility.hpp"
    "D:/develop/thirdParties/LAStools/LASlib/inc/lasvlrpayload.hpp"
    "D:/develop/thirdParties/LAStools/LASlib/inc/laswaveform13reader.hpp"
    "D:/develop/thirdParties/LAStools/LASlib/inc/laswaveform13writer.hpp"
    "D:/develop/thirdParties/LAStools/LASlib/inc/laswriter.hpp"
    "D:/develop/thirdParties/LAStools/LASlib/inc/laswriter_bin.hpp"
    "D:/develop/thirdParties/LAStools/LASlib/inc/laswriter_las.hpp"
    "D:/develop/thirdParties/LAStools/LASlib/inc/laswriter_qfit.hpp"
    "D:/develop/thirdParties/LAStools/LASlib/inc/laswriter_txt.hpp"
    "D:/develop/thirdParties/LAStools/LASlib/inc/laswriter_wrl.hpp"
    "D:/develop/thirdParties/LAStools/LASlib/inc/laswritercompatible.hpp"
    "D:/develop/thirdParties/LAStools/LASzip/src/arithmeticdecoder.hpp"
    "D:/develop/thirdParties/LAStools/LASzip/src/arithmeticencoder.hpp"
    "D:/develop/thirdParties/LAStools/LASzip/src/arithmeticmodel.hpp"
    "D:/develop/thirdParties/LAStools/LASzip/src/bytestreamin.hpp"
    "D:/develop/thirdParties/LAStools/LASzip/src/bytestreamin_array.hpp"
    "D:/develop/thirdParties/LAStools/LASzip/src/bytestreamin_file.hpp"
    "D:/develop/thirdParties/LAStools/LASzip/src/bytestreamin_istream.hpp"
    "D:/develop/thirdParties/LAStools/LASzip/src/bytestreaminout.hpp"
    "D:/develop/thirdParties/LAStools/LASzip/src/bytestreaminout_file.hpp"
    "D:/develop/thirdParties/LAStools/LASzip/src/bytestreamout.hpp"
    "D:/develop/thirdParties/LAStools/LASzip/src/bytestreamout_array.hpp"
    "D:/develop/thirdParties/LAStools/LASzip/src/bytestreamout_file.hpp"
    "D:/develop/thirdParties/LAStools/LASzip/src/bytestreamout_nil.hpp"
    "D:/develop/thirdParties/LAStools/LASzip/src/bytestreamout_ostream.hpp"
    "D:/develop/thirdParties/LAStools/LASzip/src/integercompressor.hpp"
    "D:/develop/thirdParties/LAStools/LASzip/src/lasattributer.hpp"
    "D:/develop/thirdParties/LAStools/LASzip/src/lasindex.hpp"
    "D:/develop/thirdParties/LAStools/LASzip/src/lasinterval.hpp"
    "D:/develop/thirdParties/LAStools/LASzip/src/laspoint.hpp"
    "D:/develop/thirdParties/LAStools/LASzip/src/lasquadtree.hpp"
    "D:/develop/thirdParties/LAStools/LASzip/src/lasquantizer.hpp"
    "D:/develop/thirdParties/LAStools/LASzip/src/lasreaditem.hpp"
    "D:/develop/thirdParties/LAStools/LASzip/src/lasreaditemcompressed_v1.hpp"
    "D:/develop/thirdParties/LAStools/LASzip/src/lasreaditemcompressed_v2.hpp"
    "D:/develop/thirdParties/LAStools/LASzip/src/lasreaditemcompressed_v3.hpp"
    "D:/develop/thirdParties/LAStools/LASzip/src/lasreaditemcompressed_v4.hpp"
    "D:/develop/thirdParties/LAStools/LASzip/src/lasreaditemraw.hpp"
    "D:/develop/thirdParties/LAStools/LASzip/src/lasreadpoint.hpp"
    "D:/develop/thirdParties/LAStools/LASzip/src/lasunzipper.hpp"
    "D:/develop/thirdParties/LAStools/LASzip/src/laswriteitem.hpp"
    "D:/develop/thirdParties/LAStools/LASzip/src/laswriteitemcompressed_v1.hpp"
    "D:/develop/thirdParties/LAStools/LASzip/src/laswriteitemcompressed_v2.hpp"
    "D:/develop/thirdParties/LAStools/LASzip/src/laswriteitemcompressed_v3.hpp"
    "D:/develop/thirdParties/LAStools/LASzip/src/laswriteitemcompressed_v4.hpp"
    "D:/develop/thirdParties/LAStools/LASzip/src/laswriteitemraw.hpp"
    "D:/develop/thirdParties/LAStools/LASzip/src/laswritepoint.hpp"
    "D:/develop/thirdParties/LAStools/LASzip/src/laszip.hpp"
    "D:/develop/thirdParties/LAStools/LASzip/src/laszip_common_v1.hpp"
    "D:/develop/thirdParties/LAStools/LASzip/src/laszip_common_v2.hpp"
    "D:/develop/thirdParties/LAStools/LASzip/src/laszip_common_v3.hpp"
    "D:/develop/thirdParties/LAStools/LASzip/src/laszip_decompress_selective_v3.hpp"
    "D:/develop/thirdParties/LAStools/LASzip/src/laszipper.hpp"
    "D:/develop/thirdParties/LAStools/LASzip/src/mydefs.hpp"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/LASlib" TYPE DIRECTORY FILES "D:/develop/thirdParties/LAStools/LASlib/src/../lib/Debug")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/LASlib" TYPE DIRECTORY FILES "D:/develop/thirdParties/LAStools/LASlib/src/../lib/Release")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/LASlib" TYPE DIRECTORY FILES "D:/develop/thirdParties/LAStools/LASlib/src/../lib/MinSizeRel")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/LASlib" TYPE DIRECTORY FILES "D:/develop/thirdParties/LAStools/LASlib/src/../lib/RelWithDebInfo")
endif()

