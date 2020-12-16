[Package binary in Ubuntu and macOS](https://zhuanlan.zhihu.com/p/95820992)
=====================

[reference](https://doc.qt.io/qt-5/linux-deployment.html)

1. Copy copylib.sh to build directory

		copy util/copylib.sh BUILD_DIR/bin/
    

2. Find dependeces for ErowCloudViewer and you will get lib directory with dependences such as *.so in it
		
		cd BUILD_DIR/bin/
		./copylib.sh ErowCloudViewer
		cp ErowCloudViewer lib/

3. Make translations

		cp plugins lib/
		cd lib/
		mkdir translations
		cp TRANSLATIONS_DIR/ErowCloudViewer_zh.qm translations/

4. Copy QT_DIR/gcc_64/plugins/* and go into platforms folder
(Note: need copy libQOpenGL.so.5 to ErowCloudViewer directory)
		
		cp QT_DIR/gcc_64/plugins/* ./
		cd platforms
		./copylib.sh libqxcb.so
		mv lib/* ../ && rm -rf lib


5. Find dependences for plugins(libQPDAL_IO_PLUGIN.so and libQMANUAL_SEG_PLUGIN.so)
 
		cd ../plugins/
		./copylib.sh libQPDAL_IO_PLUGIN.so
		mv lib/* ../ && rm -rf lib
		./copylib.sh libQMANUAL_SEG_PLUGIN.so
		mv lib/* ../ && rm -rf lib

6. Copy ErowCloudviewer.desktop, ErowCloudviewer.png, ErowCloudviewer.svg and ErowCloudViewer.sh from source code
		
		cd ..
		cp SOURECE_CODE/ErowCloudviewer* ./

7. Run ErowCloudViewer.sh instead of ErowCloudViewer

		./ErowCloudViewer.sh

8. Structure should like this
lib(文件夹) -- platforms（folder） -- libqxcb.so
         |       |                    \__ *.so
         |       |
         |       \__ ErowCloudViewer
         |       \__ ErowCloudViewer.sh
         |       \__ ErowCloudViewer.png
         |       \__ ErowCloudViewer.svg
         |       \__ ErowCloudViewer.desktop
         |       \__ *.so
		 |
		 | -- plugins (folder) -- libQ__*.so
		 | -- translations (folder) -- ErowCloudViewer_zh.qm
		 | -- xcbglintegrations (folder) -- __*.so
		 | -- sqldrivers (folder) -- __*.so
		 | -- printsupport (folder) -- __*.so
		 | -- platforminputcontexts (folder) -- __*.so
		 | -- imageformats (folder) -- __*.so
		 | -- iconengines (folder) -- __*.so
		 | -- bearer (folder) -- __*.so