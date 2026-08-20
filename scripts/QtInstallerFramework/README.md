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

> **NOTE (ARM64 static build):** The official pre-built QtIFW binaries are
> x86_64 only and link Qt **statically** (~38 MB installerbase, zero @rpath deps).
> For Apple Silicon CI we must build a native ARM64 IFW ourselves — and it MUST
> also be statically linked. The previous ARM64 IFW used conda's *shared* Qt:
> the generated `maintenancetool` then died at launch with
> `dyld: Library not loaded: @rpath/libQt5Widgets.5.dylib` (SIGABRT) whenever the
> installer auto-uninstalled an existing installation — see
> [installscript.qs](deploy/packages/ACloudViewer/meta/installscript.qs).
>
> Use the static package `QtIFW-4.6.1-darwin-ARM64-static.zip` (uploaded to
> `cloudViewer_downloads/releases/tag/qt-ifw`), built by:
>
> **Build recipe** (one-time, local macOS ARM64):
> ```bash
> bash scripts/platforms/mac/build_static_qt_ifw.sh
> # Outputs:
> #   ~/opt/Qt/QtIFW-4.6.1-darwin-ARM64-static/bin/{binarycreator,installerbase,...}
> #   ~/opt/Qt/QtIFW-4.6.1-darwin-ARM64-static.zip
> gh release upload qt-ifw --repo Asher-1/cloudViewer_downloads \
>     ~/opt/Qt/QtIFW-4.6.1-darwin-ARM64-static.zip --clobber
> ```
>
> The script builds **static Qt 5.15.14** (qtbase + qtdeclarative + qttools only,
> all 3rdparty libs bundled: zlib/png/jpeg/pcre2/harfbuzz/freetype/sqlite), then
> builds **IFW 4.6.1** against it, and finally verifies with `otool -L` that no
> binary carries `@rpath`, `/opt/homebrew` or `/Users/...` dependencies — so the
> installer .app and its `maintenancetool` run on any macOS machine.
>
> Manual recipe (what the script automates):
> ```bash
> # 1. Download + build static Qt 5.15.14 (qtbase, qtdeclarative, qttools)
> #    ./configure -static -release -no-opengl -no-icu -no-dbus \
> #      -qt-libjpeg -qt-libpng -qt-pcre -qt-zlib -qt-freetype -qt-harfbuzz \
> #      -sql-sqlite -nomake examples -nomake tests -prefix ~/opt/Qt/qt-5.15.14-static
> # 2. Download IFW 4.6.1 source
> curl -L https://download.qt.io/official_releases/qt-installer-framework/4.6.1/installer-framework-everywhere-src-4.6.1.tar.xz | tar xJ
> # 3. Patch: comment out requires(!cross_compile) in installerfw.pro
> # 4. export PATH=~/opt/Qt/qt-5.15.14-static/bin:$PATH && qmake -r
> #    sed -i '' 's/-framework AGL //g' $(find . -name Makefile)
> # 5. make -j$(sysctl -n hw.logicalcpu)
> # 6. Verify: otool -L bin/binarycreator must show NO @rpath / /opt/ deps
> # 7. Assemble bin/ (+ translations/), zip -r -y QtIFW-4.6.1-darwin-ARM64-static.zip
> # 8. gh release upload qt-ifw --repo Asher-1/cloudViewer_downloads QtIFW-4.6.1-darwin-ARM64-static.zip
> ```
>
> **Why static?** With a static IFW, `binarycreator` produces a self-contained
> installer .app and, equally important, a self-contained `maintenancetool`
> (IFW generates it by copying `installerbase`, so static installerbase ⇒ static
> maintenancetool). No `Contents/Frameworks` bundling or rpath fixing is needed
> anywhere in the pipeline.

1，put ACloudViewer.app in: [data](./deploy/packages/ACloudViewer/data)

2，put CloudViewer.app data in: [data](./deploy/packages/CloudViewer/data)

3，put colmap.app data in: [data](./deploy/packages/colmap/data)

4, modify [config.xml](./deploy/config/config_mac.xml) and [package.xml](./deploy/packages/ACloudViewer/meta/package.xml)

5, Automated packaging via `make install` (configured in `cmake/PostInstall.cmake`):
   - Runs `binarycreator` to create the installer .app
   - Fixes nested bundle (Qt IFW creates nested .app when installerbase is a .app bundle)
   - Ad-hoc code signs the installer app
   - Uses `dmgbuild` (Python package, requires conda env with `dmgbuild>=1.6.0`) to create
     a polished DMG with custom background image, correct icon position, and matching window size
   - Falls back to plain `hdiutil` if dmgbuild is unavailable (functional DMG, no beautification)

   Manual alternative (if needed):
   ```bash
   cd [WORKSPACE](./deploy) && binarycreator -c config/config_mac.xml -p packages ACloudViewer-3.9.5-ARM64.dmg
   ```


# MacOS some commands
```
# apply code signer on macos:
codesign --deep --force -s - --timestamp colmap.app
codesign --deep --force -s - --timestamp ACloudViewer.app
codesign --deep --force -s - --timestamp CloudViewer.app
codesign --deep --force -s - --timestamp /Users/asher/develop/code/github/ACloudViewer/build_app/bin/CloudViewer.app
codesign --deep --force -s - --timestamp /Users/asher/cloudViewer_install/deploy/packages/colmap/data/colmap.app
codesign --deep --force -s - --timestamp /Users/asher/cloudViewer_install/deploy/packages/CloudViewer/data/CloudViewer.app
codesign --deep --force -s - --timestamp /Users/asher/cloudViewer_install/deploy/packages/ACloudViewer/data/ACloudViewer.app
codesign --deep --force -s - --timestamp /Users/asher/cloudViewer_install/bin/colmap/colmap.app
codesign --deep --force -s - --timestamp /Users/asher/cloudViewer_install/bin/CloudViewer/CloudViewer.app
codesign --deep --force -s - --timestamp /Users/asher/cloudViewer_install/ACloudViewer/ACloudViewer.app
codesign --deep --force -s - --timestamp --entitlements /Users/asher/develop/code/github/ACloudViewer/app/Mac/ACloudViewer.entitlements /Users/asher/cloudViewer_install/deploy/packages/ACloudViewer/data/ACloudViewer.app

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