[Package binary in Ubuntu and macOS](https://zhuanlan.zhihu.com/p/95820992)
=====================

[reference](https://doc.qt.io/qt-5/linux-deployment.html)

# package for ACloudViewer

[Follows are deprecated and please ref new package strategy](../docker/README.md)

1. Copy copylib_ubuntu.sh to build directory

		cp scripts/copylib_ubuntu.sh BUILD_DIR/bin/
   
2. Find dependeces for ACloudViewer and you will get lib directory with dependences such as *.so in it
		
		cd BUILD_DIR/bin/
		./copylib_ubuntu.sh ACloudViewer
		cp ACloudViewer lib/

3. Make translations

		cp -r plugins lib/
		cd lib/
		mkdir translations
		cp TRANSLATIONS_DIR/ACloudViewer_zh.qm translations/

4. Copy QT_DIR/gcc_64/plugins/* and go into platforms folder
	(Note: need copy xcbglintegrations folder to ACloudViewer directory)
		
		cp -r QT_DIR/gcc_64/plugins/* ./
		cd platforms
		cp ../../copylib_ubuntu.sh ./
		./copylib_ubuntu.sh libqxcb.so
		mv lib/* ../ && rm -rf lib


5. Find dependences for plugins(libQPDAL_IO_PLUGIN.so and libQMANUAL_SEG_PLUGIN.so)

		cd ../plugins/
		cp ../../copylib_ubuntu.sh ./
		./copylib_ubuntu.sh libQPDAL_IO_PLUGIN.so
		mv lib/* ../ && rm -rf lib
		./copylib_ubuntu.sh libQMANUAL_SEG_PLUGIN.so
		mv lib/* ../ && rm -rf lib

6. Copy ACloudViewer.desktop, ACloudViewer.png, ACloudViewer.svg and ACloudViewer.sh from source code
		
		cd ..
		cp SOURECE_CODE/util/* ./

7. Run ACloudViewer.sh instead of ACloudViewer

		./ACloudViewer.sh

8. Structure should like this
	lib(文件夹) -- platforms（folder） -- libqxcb.so
         |       |                    \__ *.so
         |       |
         |       \__ ACloudViewer
         |       \__ ACloudViewer.sh
         |       \__ ACloudViewer.png
         |       \__ ACloudViewer.svg
         |       \__ ACloudViewer.desktop
         |       \__ *.so
		 |
		 | -- plugins (folder) -- libQ__*.so
		 | -- translations (folder) -- ACloudViewer_zh.qm
		 | -- xcbglintegrations (folder) -- __*.so
		 | -- sqldrivers (folder) -- __*.so
		 | -- printsupport (folder) -- __*.so
		 | -- platforminputcontexts (folder) -- __*.so
		 | -- imageformats (folder) -- __*.so
		 | -- iconengines (folder) -- __*.so
		 | -- bearer (folder) -- __*.so

# package for CloudViewer
1. Copy copylib_ubuntu.sh to build directory
		copy scripts/copylib_ubuntu.sh BUILD_DIR/bin/
2. Find dependeces for ACloudViewer and you will get lib directory with dependences such as *.so in it
		cd BUILD_DIR/bin/
		./copylib_ubuntu.sh CloudViewer
		mv lib/* ./ && rm -rf lib
3. Copy Cloudviewer.desktop, Cloudviewer.png, Cloudviewer.svg and CloudViewer.sh from source code
		
		cd ..
		cp SOURECE_CODE/scripts/CloudViewer* ./

4. Run CloudViewer.sh instead of CloudViewer

		./CloudViewer.sh



# Using QtInstallerFramework package For Windows and Linux:
[please refer this](../scripts/QtInstallerFramework/README.md)

