[Package binary in Ubuntu and macOS](https://zhuanlan.zhihu.com/p/95820992)
=====================

[reference](https://doc.qt.io/qt-5/linux-deployment.html)

# package for ErowCloudViewer

1. Copy copylib.sh to build directory

		copy util/copylib.sh BUILD_DIR/bin/
   
2. Find dependeces for ErowCloudViewer and you will get lib directory with dependences such as *.so in it
		
		cd BUILD_DIR/bin/
		./copylib.sh ErowCloudViewer
		cp ErowCloudViewer lib/

3. Make translations

		cp -r plugins lib/
		cd lib/
		mkdir translations
		cp TRANSLATIONS_DIR/ErowCloudViewer_zh.qm translations/

4. Copy QT_DIR/gcc_64/plugins/* and go into platforms folder
	(Note: need copy xcbglintegrations folder to ErowCloudViewer directory)
		
		cp -r QT_DIR/gcc_64/plugins/* ./
		cd platforms
		cp ../../copylib.sh ./
		./copylib.sh libqxcb.so
		mv lib/* ../ && rm -rf lib


5. Find dependences for plugins(libQPDAL_IO_PLUGIN.so and libQMANUAL_SEG_PLUGIN.so)

		cd ../plugins/
		cp ../../copylib.sh ./
		./copylib.sh libQPDAL_IO_PLUGIN.so
		mv lib/* ../ && rm -rf lib
		./copylib.sh libQMANUAL_SEG_PLUGIN.so
		mv lib/* ../ && rm -rf lib

6. Copy ErowCloudviewer.desktop, ErowCloudviewer.png, ErowCloudviewer.svg and ErowCloudViewer.sh from source code
		
		cd ..
		cp SOURECE_CODE/util/ErowCloudViewer* ./

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

# package for CloudViewer
1. Copy copylib.sh to build directory
		copy util/copylib.sh BUILD_DIR/bin/
2. Find dependeces for ErowCloudViewer and you will get lib directory with dependences such as *.so in it
		cd BUILD_DIR/bin/
		./copylib.sh CloudViewer
		mv lib/* ./ && rm -rf lib
3. Copy Cloudviewer.desktop, Cloudviewer.png, Cloudviewer.svg and CloudViewer.sh from source code
		
		cd ..
		cp SOURECE_CODE/util/CloudViewer* ./

4. Run CloudViewer.sh instead of CloudViewer

		./CloudViewer.sh



# Using QtInstallerFramework package For Windows and Linux:
[please refer this](../scripts/QtInstallerFramework/README.md)

