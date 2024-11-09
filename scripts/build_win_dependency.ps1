# 设置工作目录和版本号
$WorkDir = "C:\dev"
$VTKVersion = "9.2.6"
$PCLVersion = "1.14.0"
$BoostVersion = "1.84.0"
$EigenVersion = "3.4.0"
$FLANNVersion = "1.9.1"

# 创建工作目录
New-Item -ItemType Directory -Force -Path $WorkDir
Set-Location $WorkDir

# 定义构建函数
function Build-Project {
    param (
        [string]$SourceDir,
        [string]$BuildDir,
        [array]$CMakeArgs
    )
    
    New-Item -ItemType Directory -Force -Path $BuildDir
    Set-Location $BuildDir
    cmake $SourceDir $CMakeArgs
    cmake --build . --config Release
    cmake --install .
    Set-Location $WorkDir
}

# 下载并解压源码
function Download-And-Extract {
    param (
        [string]$Url,
        [string]$FileName
    )
    
    Invoke-WebRequest -Uri $Url -OutFile $FileName
    Expand-Archive $FileName -DestinationPath .
    Remove-Item $FileName
}

# 下载并构建 VTK
Download-And-Extract "https://github.com/Kitware/VTK/archive/v$VTKVersion.zip" "vtk.zip"
$VTKInstallPath = "$WorkDir\vtk-install"
$VTKCMakeArgs = @(
    "-DCMAKE_BUILD_TYPE=Release",
    "-DBUILD_SHARED_LIBS=ON",
    "-DVTK_GROUP_ENABLE_Qt=ON",
    "-DVTK_MODULE_ENABLE_VTK_GUISupportQt=ON",
    "-DVTK_MODULE_ENABLE_VTK_RenderingQt=ON",
    "-DVTK_WRAP_PYTHON=OFF",
    "-DVTK_BUILD_TESTING=OFF",
    "-DQt5_DIR=$env:CONDA_PREFIX\lib\cmake\Qt5",
    "-DCMAKE_INSTALL_PREFIX=$VTKInstallPath"
)
Build-Project -SourceDir ".\VTK-$VTKVersion" -BuildDir ".\vtk-build" -CMakeArgs $VTKCMakeArgs

# 下载并构建 Boost
Download-And-Extract "https://boostorg.jfrog.io/artifactory/main/release/$BoostVersion/source/boost_$($BoostVersion.Replace('.','_')).zip" "boost.zip"
Set-Location ".\boost_$($BoostVersion.Replace('.','_'))"
.\bootstrap.bat
.\b2 --with-system --with-filesystem --with-thread --with-date_time --with-iostreams --with-serialization --with-chrono --with-atomic --with-regex --with-timer --with-program_options
Set-Location $WorkDir

# 下载并构建 Eigen
Download-And-Extract "https://gitlab.com/libeigen/eigen/-/archive/$EigenVersion/eigen-$EigenVersion.zip" "eigen.zip"
$EigenCMakeArgs = @(
    "-DCMAKE_BUILD_TYPE=Release",
    "-DCMAKE_INSTALL_PREFIX=$WorkDir\eigen-install"
)
Build-Project -SourceDir ".\eigen-$EigenVersion" -BuildDir ".\eigen-build" -CMakeArgs $EigenCMakeArgs

# 下载并构建 FLANN
Download-And-Extract "https://github.com/flann-lib/flann/archive/$FLANNVersion.zip" "flann.zip"
$FLANNCMakeArgs = @(
    "-DCMAKE_BUILD_TYPE=Release",
    "-DBUILD_SHARED_LIBS=ON",
    "-DBUILD_EXAMPLES=OFF",
    "-DBUILD_TESTS=OFF",
    "-DBUILD_MATLAB_BINDINGS=OFF",
    "-DCMAKE_INSTALL_PREFIX=$WorkDir\flann-install"
)
Build-Project -SourceDir ".\flann-$FLANNVersion" -BuildDir ".\flann-build" -CMakeArgs $FLANNCMakeArgs

# 下载并构建 PCL
Download-And-Extract "https://github.com/PointCloudLibrary/pcl/archive/pcl-$PCLVersion.zip" "pcl.zip"
$PCLCMakeArgs = @(
    "-DCMAKE_BUILD_TYPE=Release",
    "-DBUILD_SHARED_LIBS=ON",
    "-DWITH_OPENNI=OFF",
    "-DWITH_OPENNI2=OFF",
    "-DWITH_QHULL=OFF",
    "-DWITH_CUDA=OFF",
    "-DBUILD_GPU=OFF",
    "-DBUILD_apps=OFF",
    "-DBUILD_examples=OFF",
    "-DBUILD_tools=OFF",
	"-DBUILD_surface_on_nurbs=ON",
    "-DBUILD_visualization=ON",
    "-DEIGEN_INCLUDE_DIR=$WorkDir\eigen-install\include\eigen3",
    "-DBOOST_ROOT=$WorkDir\boost_$($BoostVersion.Replace('.','_'))",
    "-DFLANN_ROOT=$WorkDir\flann-install",
    "-DVTK_DIR=$VTKInstallPath\lib\cmake\vtk-$($VTKVersion.Split('.')[0]).$($VTKVersion.Split('.')[1])",
    "-DCMAKE_INSTALL_PREFIX=$WorkDir\pcl-install"
)
Build-Project -SourceDir ".\pcl-pcl-$PCLVersion" -BuildDir ".\pcl-build" -CMakeArgs $PCLCMakeArgs

Write-Host "构建完成！"