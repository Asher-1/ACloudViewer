# Windows

1，put application data in :  [data for ErowCloudViewer](./windows/ErowCloudViewer/packages/com.vendor.product/data) or [data for CloudViewer](./windows/CloudViewer/packages/com.vendor.product/data)

2, modify [config.xml for ErowCloudViewer](./windows/ErowCloudViewer/config/config.xml) or  [config.xml for CloudViewer](./windows/CloudViewer/config/config.xml)) and [package.xml for ErowCloudViewer](./windows/ErowCloudViewer/packages/com.vendor.product/meta/package.xml)  or [package.xml for CloudViewer](./windows/CloudViewer/packages/com.vendor.product/meta/package.xml) 

3, cd [WORKSPACE for CloudViewer](./windows/CloudViewer) && binarycreator.exe -c config/config.xml -p packages CloudViewer-3.8.0-2021-11-12-win-amd64.exe

4, cd [WORKSPACE for ErowCloudViewer](./windows/ErowCloudViewer) && binarycreator.exe -c config/config.xml -p packages ErowCloudViewer-3.8.0-2021-11-12-win-amd64.exe


# Linux
1，put application data in: [data for ErowCloudViewer](./linux/ErowCloudViewer/packages/com.vendor.product/data) or [data for CloudViewer](./linux/CloudViewer/packages/com.vendor.product/data)

2, modify [config.xml for ErowCloudViewer](./linux/ErowCloudViewer/config/config.xml) or  [config.xml for CloudViewer](./linux/CloudViewer/config/config.xml) and [package.xml for ErowCloudViewer](./linux/ErowCloudViewer/packages/com.vendor.product/meta/package.xml) or [package.xml for CloudViewer](./linux/CloudViewer/packages/com.vendor.product/meta/package.xml) 

3, cd [WORKSPACE for CloudViewer](./linux/CloudViewer) && binarycreator -c config/config.xml -p packages CloudViewer-3.8.0-2021-10.10-ubuntu1804-amd64.run

4, cd [WORKSPACE for ErowCloudViewer](./linux/ErowCloudViewer) && binarycreator -c config/config.xml -p packages ErowCloudViewer-3.8.0-2021-10-10-ubuntu1804-amd64.run


# MacOS
```
# apply code signer on macos:
sudo codesign -f -s - --deep ErowCloudViewer.app
sudo codesign -f -s - --deep /Users/asher/develop/code/github/macos_install/ErowCloudViewer/ErowCloudViewer.app
sudo codesign -f -s - --deep colmap.app
sudo codesign -f -s - --deep /Users/asher/develop/code/github/macos_install/bin/colmap/colmap.app
# if resource fork, Finder information, or similar detritus not allowed then xattr -rc . and try again

sudo chmod +w /Users/asher/develop/code/github/macos_install/ErowCloudViewer/ErowCloudViewer.app/Contents/Frameworks/*
sudo chown -R asher:staff /Users/asher/develop/code/github/macos_install/ErowCloudViewer/ErowCloudViewer.app/Contents/Frameworks
# for symbol missing
cp /opt/homebrew/Cellar/gcc/12.2.0/lib/gcc/current/libgcc_s.1.1.dylib /Users/asher/develop/code/github/macos_install/bin/colmap/colmap.app/Contents/Frameworks
cp /opt/homebrew/Cellar/suite-sparse/7.0.1/lib/libspqr_cuda.3.dylib /Users/asher/develop/code/github/macos_install/bin/colmap/colmap.app/Contents/Frameworks
cp /opt/homebrew/Cellar/gcc/12.2.0/lib/gcc/current/libgcc_s.1.1.dylib /Users/asher/develop/code/github/macos_install/ErowCloudViewer/ErowCloudViewer.app/Contents/Frameworks
cp /opt/homebrew/Cellar/suite-sparse/7.0.1/lib/libspqr_cuda.3.dylib /Users/asher/develop/code/github/macos_install/ErowCloudViewer/ErowCloudViewer.app/Contents/Frameworks
/Users/asher/develop/code/github/ErowCloudViewer/libs/CVViewer/apps/fixup_macosx_bundle.sh /Users/asher/develop/code/github/macos_install/ErowCloudViewer/ErowCloudViewer.app
/Users/asher/develop/code/github/ErowCloudViewer/libs/CVViewer/apps/fixup_macosx_plugins.sh /Users/asher/develop/code/github/macos_install/ErowCloudViewer/ErowCloudViewer.app
/Users/asher/develop/code/github/ErowCloudViewer/libs/CVViewer/apps/fixup_macosx_frameworks.sh /Users/asher/develop/code/github/macos_install/ErowCloudViewer/ErowCloudViewer.app
/Users/asher/develop/code/github/ErowCloudViewer/libs/CVViewer/apps/fixup_macosx_libs.sh /Users/asher/develop/code/github/macos_install/ErowCloudViewer/ErowCloudViewer.app/Contents/Frameworks/libspqr_cuda.3.dylib
/Users/asher/develop/code/github/ErowCloudViewer/libs/CVViewer/apps/fixup_macosx_libs.sh /Users/asher/develop/code/github/macos_install/ErowCloudViewer/ErowCloudViewer.app/Contents/Frameworks/libgthread-2.0.0.dylib
/Users/asher/develop/code/github/ErowCloudViewer/libs/CVViewer/apps/fixup_macosx_libs.sh /Users/asher/develop/code/github/macos_install/ErowCloudViewer/ErowCloudViewer.app/Contents/Frameworks/libboost_regex-mt.dylib
/Users/asher/develop/code/github/ErowCloudViewer/libs/CVViewer/apps/fixup_macosx_plugins.sh /Users/asher/develop/code/github/macos_install/ErowCloudViewer-3.9.0-arm64/ErowCloudViewer.app
/Users/asher/develop/code/github/ErowCloudViewer/libs/CVViewer/apps/fixup_macosx_frameworks.sh /Users/asher/develop/code/github/macos_install/ErowCloudViewer-3.9.0-arm64/ErowCloudViewer.app
/Users/asher/develop/code/github/ErowCloudViewer/libs/CVViewer/apps/fixup_macosx_libs.sh /Users/asher/develop/code/github/macos_install/ErowCloudViewer-3.9.0-arm64/ErowCloudViewer.app/Contents/Frameworks/libspqr_cuda.3.dylib
/Users/asher/develop/code/github/ErowCloudViewer/libs/CVViewer/apps/fixup_macosx_libs.sh /Users/asher/develop/code/github/macos_install/ErowCloudViewer-3.9.0-arm64/ErowCloudViewer.app/Contents/Frameworks/libgthread-2.0.0.dylib
/Users/asher/develop/code/github/ErowCloudViewer/libs/CVViewer/apps/fixup_macosx_libs.sh /Users/asher/develop/code/github/macos_install/ErowCloudViewer-3.9.0-arm64/ErowCloudViewer.app/Contents/Frameworks/libboost_regex-mt.dylib
/Users/asher/develop/code/github/ErowCloudViewer/libs/CVViewer/apps/fixup_macosx_libs.sh /Users/asher/develop/code/github/macos_install/ErowCloudViewer-3.9.0-arm64/ErowCloudViewer.app/Contents/Frameworks/libmpicxx.12.dylib
/Users/asher/develop/code/github/ErowCloudViewer/libs/CVViewer/apps/fixup_macosx_libs.sh /Users/asher/develop/code/github/macos_install/ErowCloudViewer-3.9.0-arm64/ErowCloudViewer.app/Contents/Frameworks/libmpi.12.dylib
brew uninstall --ignore-dependencies gflags if still crash
```