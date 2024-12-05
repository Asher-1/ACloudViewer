# Windows

1，put application data in :  [data for ACloudViewer](./windows/ACloudViewer/packages/com.vendor.product/data) or [data for CloudViewer](./windows/CloudViewer/packages/com.vendor.product/data)

2, modify [config.xml for ACloudViewer](./windows/ACloudViewer/config/config.xml) or  [config.xml for CloudViewer](./windows/CloudViewer/config/config.xml)) and [package.xml for ACloudViewer](./windows/ACloudViewer/packages/com.vendor.product/meta/package.xml)  or [package.xml for CloudViewer](./windows/CloudViewer/packages/com.vendor.product/meta/package.xml) 

3, cd [WORKSPACE for CloudViewer](./windows/CloudViewer) && binarycreator.exe -c config/config.xml -p packages CloudViewer-3.8.0-2021-11-12-win-amd64.exe

4, cd [WORKSPACE for ACloudViewer](./windows/ACloudViewer) && binarycreator.exe -c config/config.xml -p packages ACloudViewer-3.8.0-2021-11-12-win-amd64.exe


# Linux
1，put application data in: [data for ACloudViewer](./linux/ACloudViewer/packages/com.vendor.product/data) or [data for CloudViewer](./linux/CloudViewer/packages/com.vendor.product/data)

2, modify [config.xml for ACloudViewer](./linux/ACloudViewer/config/config.xml) or  [config.xml for CloudViewer](./linux/CloudViewer/config/config.xml) and [package.xml for ACloudViewer](./linux/ACloudViewer/packages/com.vendor.product/meta/package.xml) or [package.xml for CloudViewer](./linux/CloudViewer/packages/com.vendor.product/meta/package.xml) 

3, cd [WORKSPACE for CloudViewer](./linux/CloudViewer) && binarycreator -c config/config.xml -p packages CloudViewer-3.8.0-2021-10.10-ubuntu1804-amd64.run

4, cd [WORKSPACE for ACloudViewer](./linux/ACloudViewer) && binarycreator -c config/config.xml -p packages ACloudViewer-3.8.0-2021-10-10-ubuntu1804-amd64.run


# MacOS
1，put ACloudViewer.app in: [data](./deploy/packages/ACloudViewer/data)

2，put CloudViewer.app data in: [data](./deploy/packages/CloudViewer/data)

3，put colmap.app data in: [data](./deploy/packages/colmap/data)

4, modify [config.xml](./deploy/config/config_mac.xml) and [package.xml](./deploy/packages/ACloudViewer/meta/package.xml)

5, cd [WORKSPACE](./deploy) && binarycreator -c config/config_mac.xml -p packages ACloudViewer-3.9.1-2024-10-24-ARM64.dmg


# MacOS some commands
```
# apply code signer on macos:
codesign --deep --force -s - --timestamp colmap.app
codesign --deep --force -s - --timestamp ACloudViewer.app
codesign --deep --force -s - --timestamp CloudViewer.app
codesign --deep --force -s - --timestamp /Users/asher/cloudViewer_install/deploy/packages/colmap/data/colmap.app
codesign --deep --force -s - --timestamp /Users/asher/cloudViewer_install/deploy/packages/CloudViewer/data/CloudViewer.app
codesign --deep --force -s - --timestamp /Users/asher/cloudViewer_install/deploy/packages/ACloudViewer/data/ACloudViewer.app
codesign --deep --force -s - --timestamp /Users/asher/cloudViewer_install/bin/colmap/colmap.app
codesign --deep --force -s - --timestamp /Users/asher/cloudViewer_install/bin/CloudViewer/CloudViewer.app
codesign --deep --force -s - --timestamp /Users/asher/cloudViewer_install/ACloudViewer/ACloudViewer.app
codesign --deep --force -s - --timestamp --entitlements /Users/asher/develop/code/github/ACloudViewer/eCV/Mac/ACloudViewer.entitlements /Users/asher/cloudViewer_install/deploy/packages/ACloudViewer/data/ACloudViewer.app

# for libtiff.*dylib
/Users/asher/develop/code/github/ACloudViewer/scripts/platforms/mac/reset_libs_rpath.sh /Users/asher/cloudViewer_install/ACloudViewer/ACloudViewer.app/Contents/Frameworks/libtiff.6.dylib
/Users/asher/develop/code/github/ACloudViewer/scripts/platforms/mac/copy_macos_libs.sh /Users/asher/cloudViewer_install/ACloudViewer/ACloudViewer.app/Contents/Frameworks/libtiff.6.dylib
/Users/asher/develop/code/github/ACloudViewer/scripts/platforms/mac/fixup_macosx_libs.sh /Users/asher/cloudViewer_install/ACloudViewer/ACloudViewer.app/Contents/Frameworks/libtiff.6.dylib

# lib deploy
otool -L /Users/asher/cloudViewer_install/ACloudViewer/ACloudViewer.app/Contents/MacOS/ACloudViewer
otool -l /Users/asher/cloudViewer_install/ACloudViewer/ACloudViewer.app/Contents/MacOS/ACloudViewer | grep "path " | awk '{print $2}'
python /Users/asher/develop/code/github/ACloudViewer/scripts/platforms/mac/bundle/lib_bundle_app.py ACloudViewer /Users/asher/cloudViewer_install/ACloudViewer

# sign apps
python /Users/asher/develop/code/github/ACloudViewer/scripts/platforms/mac/bundle/signature_app.py ACloudViewer /Users/asher/cloudViewer_install/ACloudViewer
python /Users/asher/develop/code/github/ACloudViewer/scripts/platforms/mac/bundle/signature_app.py ACloudViewer /Users/asher/cloudViewer_install/deploy/packages/ACloudViewer/data
python /Users/asher/develop/code/github/ACloudViewer/scripts/platforms/mac/bundle/signature_app.py CloudViewer /Users/asher/cloudViewer_install/deploy/packages/CloudViewer/data
python /Users/asher/develop/code/github/ACloudViewer/scripts/platforms/mac/bundle/signature_app.py colmap /Users/asher/cloudViewer_install/deploy/packages/colmap/data

# validation
codesign -dvv --strict /Users/asher/cloudViewer_install/deploy/packages/ACloudViewer/data/ACloudViewer.app
# if resource fork, Finder information, or similar detritus not allowed then xattr -rc . and try again

brew uninstall --ignore-dependencies gflags ; if still crash
```