include(ExternalProject)

ExternalProject_add(libvpx
      GIT_REPOSITORY https://chromium.googlesource.com/webm/libvpx.git
      GIT_TAG v1.9.0
      GIT_PROGRESS ON
      PREFIX libvpx
      BUILD_IN_SOURCE 0
      BUILD_ALWAYS 0
      UPDATE_COMMAND ""
      CONFIGURE_COMMAND <SOURCE_DIR>/configure
            --prefix=<INSTALL_DIR>
            --enable-static
            --disable-shared
            --disable-examples
      BUILD_COMMAND $(MAKE)
)

ExternalProject_Get_Property(libvpx INSTALL_DIR)
set(libvpx_INSTALL_DIR ${INSTALL_DIR})

set_local_or_remote_url(
    DOWNLOAD_URL_PRIMARY
    LOCAL_URL   "${THIRD_PARTY_DOWNLOAD_DIR}/ffmpeg-4.3.1.tar.bz2"
    REMOTE_URLS "http://ffmpeg.org/releases/ffmpeg-4.3.1.tar.bz2"
)

ExternalProject_add(
      ext_ffmpeg
      PREFIX ffmpeg
      URL ${DOWNLOAD_URL_PRIMARY} ${DOWNLOAD_URL_FALLBACK}
      URL_HASH MD5=804707549590e90880e8ecd4e5244fd8
      BUILD_IN_SOURCE ON
      BUILD_ALWAYS 0
      UPDATE_COMMAND ""
      CONFIGURE_COMMAND <SOURCE_DIR>/configure
            --prefix=<INSTALL_DIR>
            --extra-cflags="-I${libvpx_INSTALL_DIR}/include"
            --extra-ldflags="-L${libvpx_INSTALL_DIR}/lib"
            --enable-static
            --disable-shared
#            --enable-shared
#            --disable-static
            --disable-gpl
            --enable-nonfree
            # --enable-libfreetype
            # --enable-libfdk-aac # audio
            # --enable-libmp3lame # audio
            # --enable-libopus # audio
            # --enable-libvorbis # audio
            --enable-libvpx
            # --enable-libx264 # gpl
            # --enable-libx265 # gpl
      BUILD_COMMAND $(MAKE)
      DEPENDS libvpx
)

ExternalProject_Get_Property(ext_ffmpeg INSTALL_DIR)

set(FFMPEG_INCLUDE_DIRS ${INSTALL_DIR}/include/)
set(FFMPEG_LIB_DIR ${INSTALL_DIR}/lib)
set(FFMPEG_LIBRARIES ffmpeg)
