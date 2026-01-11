// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "renders/utils/TextureLoader.h"

#include <CVLog.h>
#include <CVTools.h>
#include <FileSystem.h>
#include <Utils/sm2cc.h>
#include <ecvMaterial.h>

// Qt
#include <QImage>

// VTK
#include <vtkBMPReader.h>
#include <vtkJPEGReader.h>
#include <vtkPNGReader.h>
#include <vtkPNMReader.h>
#include <vtkQImageToImageSource.h>
#include <vtkSmartPointer.h>
#include <vtkTIFFReader.h>
#include <vtkTexture.h>

namespace PclUtils {
namespace renders {

int TextureLoader::LoadFromPCLMaterial(const pcl::TexMaterial& tex_mat,
                                       vtkTexture* vtk_tex) {
    if (!vtk_tex) {
        return -1;
    }

    QImage qimage = ccMaterial::GetTexture(tex_mat.tex_file.c_str());
    if (!qimage.isNull()) {
        vtkSmartPointer<vtkQImageToImageSource> qimageToImageSource =
                vtkSmartPointer<vtkQImageToImageSource>::New();
        qimageToImageSource->SetQImage(&qimage);
        qimageToImageSource->Update();
        vtk_tex->SetInputConnection(qimageToImageSource->GetOutputPort());
        return 1;
    }

    if (tex_mat.tex_file.empty()) {
        return -1;
    }

    std::string full_path = tex_mat.tex_file;
    if (!cloudViewer::utility::filesystem::FileExists(full_path)) {
        std::string parent_dir =
                cloudViewer::utility::filesystem::GetFileParentDirectory(
                        full_path);
        std::string upper_filename =
                cloudViewer::utility::ToUpper(tex_mat.tex_file);
        std::string real_name;

        try {
            if (!cloudViewer::utility::filesystem::DirectoryExists(
                        parent_dir)) {
                CVLog::Warning(
                        "[TextureLoader::LoadFromPCLMaterial] Parent directory "
                        "'%s' doesn't exist!",
                        parent_dir.c_str());
                return -1;
            }

            if (!cloudViewer::utility::filesystem::IsDirectory(parent_dir)) {
                CVLog::Warning(
                        "[TextureLoader::LoadFromPCLMaterial] Parent '%s' is "
                        "not a directory !",
                        parent_dir.c_str());
                return -1;
            }

            std::vector<std::string> paths_vector;
            cloudViewer::utility::filesystem::ListFilesInDirectory(
                    parent_dir, paths_vector);

            for (const auto& path : paths_vector) {
                if (cloudViewer::utility::filesystem::IsFile(path)) {
                    if (cloudViewer::utility::ToUpper(path) == upper_filename) {
                        real_name = path;
                        break;
                    }
                }
            }

            if (real_name.empty()) {
                CVLog::Warning(
                        "[TextureLoader::LoadFromPCLMaterial] Can not find "
                        "texture file %s!",
                        tex_mat.tex_file.c_str());
                return -1;
            }
        } catch (const std::exception& ex) {
            CVLog::Warning(
                    "[TextureLoader::LoadFromPCLMaterial] Error %s when "
                    "looking for file %s!",
                    ex.what(), tex_mat.tex_file.c_str());
            return -1;
        }

        full_path = real_name;
    }

    std::string extension =
            cloudViewer::utility::filesystem::GetFileExtensionInLowerCase(
                    full_path);

    if ((extension == "jpg") || (extension == "jpeg")) {
        vtkSmartPointer<vtkJPEGReader> jpeg_reader =
                vtkSmartPointer<vtkJPEGReader>::New();
        jpeg_reader->SetFileName(full_path.c_str());
        jpeg_reader->Update();
        vtk_tex->SetInputConnection(jpeg_reader->GetOutputPort());
    } else if (extension == "bmp") {
        vtkSmartPointer<vtkBMPReader> bmp_reader =
                vtkSmartPointer<vtkBMPReader>::New();
        bmp_reader->SetFileName(full_path.c_str());
        bmp_reader->Update();
        vtk_tex->SetInputConnection(bmp_reader->GetOutputPort());
    } else if (extension == "pnm") {
        vtkSmartPointer<vtkPNMReader> pnm_reader =
                vtkSmartPointer<vtkPNMReader>::New();
        pnm_reader->SetFileName(full_path.c_str());
        pnm_reader->Update();
        vtk_tex->SetInputConnection(pnm_reader->GetOutputPort());
    } else if (extension == "png") {
        vtkSmartPointer<vtkPNGReader> png_reader =
                vtkSmartPointer<vtkPNGReader>::New();
        png_reader->SetFileName(full_path.c_str());
        png_reader->Update();
        vtk_tex->SetInputConnection(png_reader->GetOutputPort());
    } else if ((extension == "tiff") || (extension == "tif")) {
        vtkSmartPointer<vtkTIFFReader> tiff_reader =
                vtkSmartPointer<vtkTIFFReader>::New();
        tiff_reader->SetFileName(full_path.c_str());
        tiff_reader->Update();
        vtk_tex->SetInputConnection(tiff_reader->GetOutputPort());
    } else {
        CVLog::Warning(
                "[TextureLoader::LoadFromPCLMaterial] Unhandled image %s "
                "(extension: '%s') for material %s!",
                full_path.c_str(), extension.c_str(), tex_mat.tex_name.c_str());
        return -1;
    }

    return 1;
}

int TextureLoader::LoadFromCCMaterial(ccMaterial::CShared material,
                                      vtkTexture* vtk_tex) {
    if (!material || !vtk_tex) {
        return -1;
    }

    // For traditional multi-texture rendering, prioritize DIFFUSE texture
    using TexType = ccMaterial::TextureMapType;
    QString tex_file = material->getTextureFilename(TexType::DIFFUSE);
    if (tex_file.isEmpty()) {
        // Fallback to legacy texture filename (for backward compatibility)
        tex_file = material->getTextureFilename();
    }

    QImage qimage = ccMaterial::GetTexture(tex_file);
    if (!qimage.isNull()) {
        vtkSmartPointer<vtkQImageToImageSource> qimageToImageSource =
                vtkSmartPointer<vtkQImageToImageSource>::New();
        qimageToImageSource->SetQImage(&qimage);
        qimageToImageSource->Update();
        vtk_tex->SetInputConnection(qimageToImageSource->GetOutputPort());
        return 1;
    }

    if (tex_file.isEmpty()) {
        return -1;
    }

    std::string full_path = CVTools::FromQString(tex_file);
    if (!cloudViewer::utility::filesystem::FileExists(full_path)) {
        std::string parent_dir =
                cloudViewer::utility::filesystem::GetFileParentDirectory(
                        full_path);
        std::string upper_filename = cloudViewer::utility::ToUpper(full_path);
        std::string real_name;

        try {
            if (!cloudViewer::utility::filesystem::DirectoryExists(
                        parent_dir)) {
                CVLog::Warning(
                        "[TextureLoader::LoadFromCCMaterial] Parent directory "
                        "'%s' doesn't exist!",
                        parent_dir.c_str());
                return -1;
            }

            if (!cloudViewer::utility::filesystem::IsDirectory(parent_dir)) {
                CVLog::Warning(
                        "[TextureLoader::LoadFromCCMaterial] Parent '%s' is "
                        "not "
                        "a directory !",
                        parent_dir.c_str());
                return -1;
            }

            std::vector<std::string> paths_vector;
            cloudViewer::utility::filesystem::ListFilesInDirectory(
                    parent_dir, paths_vector);

            for (const auto& path : paths_vector) {
                if (cloudViewer::utility::filesystem::IsFile(path)) {
                    if (cloudViewer::utility::ToUpper(path) == upper_filename) {
                        real_name = path;
                        break;
                    }
                }
            }

            if (real_name.empty()) {
                CVLog::Warning(
                        "[TextureLoader::LoadFromCCMaterial] Can not find "
                        "texture file %s!",
                        full_path.c_str());
                return -1;
            }
        } catch (const std::exception& ex) {
            CVLog::Warning(
                    "[TextureLoader::LoadFromCCMaterial] Error %s when "
                    "looking for file %s!",
                    ex.what(), full_path.c_str());
            return -1;
        }

        full_path = real_name;
    }

    std::string extension =
            cloudViewer::utility::filesystem::GetFileExtensionInLowerCase(
                    full_path);

    if ((extension == "jpg") || (extension == "jpeg")) {
        vtkSmartPointer<vtkJPEGReader> jpeg_reader =
                vtkSmartPointer<vtkJPEGReader>::New();
        jpeg_reader->SetFileName(full_path.c_str());
        jpeg_reader->Update();
        vtk_tex->SetInputConnection(jpeg_reader->GetOutputPort());
    } else if (extension == "bmp") {
        vtkSmartPointer<vtkBMPReader> bmp_reader =
                vtkSmartPointer<vtkBMPReader>::New();
        bmp_reader->SetFileName(full_path.c_str());
        bmp_reader->Update();
        vtk_tex->SetInputConnection(bmp_reader->GetOutputPort());
    } else if (extension == "pnm") {
        vtkSmartPointer<vtkPNMReader> pnm_reader =
                vtkSmartPointer<vtkPNMReader>::New();
        pnm_reader->SetFileName(full_path.c_str());
        pnm_reader->Update();
        vtk_tex->SetInputConnection(pnm_reader->GetOutputPort());
    } else if (extension == "png") {
        vtkSmartPointer<vtkPNGReader> png_reader =
                vtkSmartPointer<vtkPNGReader>::New();
        png_reader->SetFileName(full_path.c_str());
        png_reader->Update();
        vtk_tex->SetInputConnection(png_reader->GetOutputPort());
    } else if ((extension == "tiff") || (extension == "tif")) {
        vtkSmartPointer<vtkTIFFReader> tiff_reader =
                vtkSmartPointer<vtkTIFFReader>::New();
        tiff_reader->SetFileName(full_path.c_str());
        tiff_reader->Update();
        vtk_tex->SetInputConnection(tiff_reader->GetOutputPort());
    } else {
        CVLog::Warning(
                "[TextureLoader::LoadFromCCMaterial] Unhandled image %s "
                "(extension: '%s') for material %s!",
                full_path.c_str(), extension.c_str(),
                CVTools::FromQString(material->getName()).c_str());
        return -1;
    }

    return 1;
}

}  // namespace renders
}  // namespace PclUtils
