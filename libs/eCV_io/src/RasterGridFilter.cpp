// ##########################################################################
// #                                                                        #
// #                              CLOUDVIEWER                               #
// #                                                                        #
// #  This program is free software; you can redistribute it and/or modify  #
// #  it under the terms of the GNU General Public License as published by  #
// #  the Free Software Foundation; version 2 or later of the License.      #
// #                                                                        #
// #  This program is distributed in the hope that it will be useful,       #
// #  but WITHOUT ANY WARRANTY; without even the implied warranty of        #
// #  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the          #
// #  GNU General Public License for more details.                          #
// #                                                                        #
// #          COPYRIGHT: EDF R&D / TELECOM ParisTech (ENST-TSI)             #
// #                                                                        #
// ##########################################################################

#ifdef CV_GDAL_SUPPORT

#include "RasterGridFilter.h"

// DB
#include <ecvMesh.h>
#include <ecvPlane.h>
#include <ecvPointCloud.h>
#include <ecvScalarField.h>

// GDAL
#include <cpl_conv.h>  // for CPLMalloc()
#include <gdal_priv.h>

// Qt
#include <QMessageBox>

// System
#include <string.h>  //for memset

RasterGridFilter::RasterGridFilter()
    : FileIOFilter({"_Raster Grid Filter",
                    16.0f,  // priority
                    QStringList{"tif", "tiff", "adf"}, "tif",
                    QStringList{"RASTER grid (*.*)"}, QStringList(),
                    Import | BuiltIn}) {}

CC_FILE_ERROR RasterGridFilter::loadFile(const QString& filename,
                                         ccHObject& container,
                                         LoadParameters& parameters) {
    GDALAllRegister();
    CVLog::PrintDebug("(GDAL drivers: %i)",
                      GetGDALDriverManager()->GetDriverCount());

    try {
        GDALDataset* poDataset = static_cast<GDALDataset*>(
                GDALOpen(qUtf8Printable(filename), GA_ReadOnly));
        if (poDataset != nullptr) {
            CVLog::Print(QString("Raster file: '%1'").arg(filename));
            CVLog::Print(
                    "Driver: %s/%s", poDataset->GetDriver()->GetDescription(),
                    poDataset->GetDriver()->GetMetadataItem(GDAL_DMD_LONGNAME));

            int rasterCount = poDataset->GetRasterCount();
            int rasterX = poDataset->GetRasterXSize();
            int rasterY = poDataset->GetRasterYSize();
            CVLog::Print("Size is %dx%dx%d", rasterX, rasterY, rasterCount);

            if (poDataset->GetProjectionRef() != nullptr) {
                CVLog::Print("Projection is '%s'",
                             poDataset->GetProjectionRef());
            }

            // See https://gdal.org/user/raster_data_model.html
            // Xgeo = adfGeoTransform(0) + Xpixel * adfGeoTransform(1) + Yline *
            // adfGeoTransform(2) Ygeo = adfGeoTransform(3) + Xpixel *
            // adfGeoTransform(4) + Yline * adfGeoTransform(5)
            double adfGeoTransform[6] = {0.0, 1.0, 0.0, 0.0, 0.0, 1.0};

            if (poDataset->GetGeoTransform(adfGeoTransform) == CE_None) {
                CVLog::Print("Origin = (%.6f,%.6f)", adfGeoTransform[0],
                             adfGeoTransform[3]);
                CVLog::Print("Pixel Size = (%.6f,%.6f)", adfGeoTransform[1],
                             adfGeoTransform[5]);
            }

            if (adfGeoTransform[1] == 0 || adfGeoTransform[5] == 0) {
                CVLog::Warning("Invalid pixel size! Forcing it to (1,1)");
                adfGeoTransform[1] = adfGeoTransform[5] = 1.0;
            }

            const char* aeraOrPoint =
                    poDataset->GetMetadataItem("AREA_OR_POINT");
            CVLog::Print(
                    QString("Pixel Type = ") +
                    (aeraOrPoint ? aeraOrPoint : "AREA"));  // Area by default

            // first check if the raster actually has 'color' bands
            int colorBands = 0;
            {
                for (int i = 1; i <= rasterCount; ++i) {
                    GDALRasterBand* poBand = poDataset->GetRasterBand(i);
                    GDALColorInterp colorInterp =
                            poBand->GetColorInterpretation();

                    switch (colorInterp) {
                        case GCI_RedBand:
                        case GCI_GreenBand:
                        case GCI_BlueBand:
                        case GCI_AlphaBand:
                            ++colorBands;
                            break;
                        default:
                            break;
                    }
                }
            }

            bool loadAsTexturedQuad = false;
            if (colorBands >= 3) {
                loadAsTexturedQuad = false;
                if (parameters.parentWidget)  // otherwise it means we are in
                                              // command line mode --> no popup
                {
                    loadAsTexturedQuad =
                            (QMessageBox::question(
                                     parameters.parentWidget, "Result type",
                                     "Import raster as a cloud (yes) or a "
                                     "texture quad? (no)",
                                     QMessageBox::Yes,
                                     QMessageBox::No) == QMessageBox::No);
                }
            }

            ccPointCloud* pc = new ccPointCloud();

            CCVector3d origin(adfGeoTransform[0], adfGeoTransform[3], 0.0);
            CCVector3d Pshift(0, 0, 0);
            // check for 'big' coordinates
            {
                bool preserveCoordinateShift = true;
                if (HandleGlobalShift(origin, Pshift, preserveCoordinateShift,
                                      parameters)) {
                    if (pc && preserveCoordinateShift) {
                        pc->setGlobalShift(Pshift);
                    }
                    CVLog::Warning(
                            "[RasterFilter::loadFile] Raster has been "
                            "recentered! Translation: (%.2f ; %.2f ; %.2f)",
                            Pshift.x, Pshift.y, Pshift.z);
                }
            }

            // create blank raster 'grid'
            ccMesh* quad = 0;
            QImage quadTexture;
            if (loadAsTexturedQuad) {
                quad = new ccMesh(pc);
                quad->addChild(pc);
                pc->setName("vertices");
                pc->setEnabled(false);

                // reserve memory
                quadTexture = QImage(rasterX, rasterY, QImage::Format_RGB32);
                if (!pc->reserve(4) || !quad->reserve(2) ||
                    quadTexture.size() != QSize(rasterX, rasterY)) {
                    delete quad;
                    return CC_FERR_NOT_ENOUGH_MEMORY;
                }

                // B ------ C
                // |        |
                // A ------ D
                CCVector3d B = origin + Pshift;  // origin is 'top left'
                CCVector3d C = B + CCVector3d(rasterX * adfGeoTransform[1],
                                              rasterX * adfGeoTransform[4], 0);
                CCVector3d A = B + CCVector3d(rasterY * adfGeoTransform[2],
                                              rasterY * adfGeoTransform[5], 0);
                CCVector3d D = C + CCVector3d(rasterY * adfGeoTransform[2],
                                              rasterY * adfGeoTransform[5], 0);

                pc->addPoint(A.toPC());
                pc->addPoint(B.toPC());
                pc->addPoint(C.toPC());
                pc->addPoint(D.toPC());

                quad->addTriangle(0, 2, 1);  // A C B
                quad->addTriangle(0, 3, 2);  // A D C
            } else {
                if (!pc->reserve(static_cast<unsigned>(rasterX * rasterY))) {
                    delete pc;
                    return CC_FERR_NOT_ENOUGH_MEMORY;
                }

                double z = 0.0 + Pshift.z;
                for (int j = 0; j < rasterY; ++j) {
                    for (int i = 0; i < rasterX; ++i) {
                        // Xgeo = adfGeoTransform(0) + Xpixel *
                        // adfGeoTransform(1) + Yline * adfGeoTransform(2) Ygeo
                        // = adfGeoTransform(3) + Xpixel * adfGeoTransform(4) +
                        // Yline * adfGeoTransform(5)
                        double x = adfGeoTransform[0] +
                                   (i + 0.5) * adfGeoTransform[1] +
                                   (j + 0.5) * adfGeoTransform[2] + Pshift.x;
                        double y = adfGeoTransform[3] +
                                   (i + 0.5) * adfGeoTransform[4] +
                                   (j + 0.5) * adfGeoTransform[5] + Pshift.y;
                        CCVector3 P(static_cast<PointCoordinateType>(x),
                                    static_cast<PointCoordinateType>(y),
                                    static_cast<PointCoordinateType>(z));
                        pc->addPoint(P);
                    }
                }
                QVariant xVar = QVariant::fromValue<int>(rasterX);
                QVariant yVar = QVariant::fromValue<int>(rasterY);
                pc->setMetaData("raster_width", xVar);
                pc->setMetaData("raster_height", yVar);
            }

            // fetch raster bands
            bool zRasterProcessed = false;
            cloudViewer::ReferenceCloud validPoints(pc);
            double zMinMax[2] = {0, 0};

            for (int i = 1; i <= rasterCount; ++i) {
                CVLog::Print("[GDAL] Reading band #%i", i);
                GDALRasterBand* poBand = poDataset->GetRasterBand(i);

                GDALColorInterp colorInterp = poBand->GetColorInterpretation();

                int nBlockXSize, nBlockYSize;
                poBand->GetBlockSize(&nBlockXSize, &nBlockYSize);
                CVLog::Print("[GDAL] Block=%dx%d, Type=%s, ColorInterp=%s",
                             nBlockXSize, nBlockYSize,
                             GDALGetDataTypeName(poBand->GetRasterDataType()),
                             GDALGetColorInterpretationName(colorInterp));

                // fetching raster scan-line
                int nXSize = poBand->GetXSize();
                int nYSize = poBand->GetYSize();
                assert(nXSize == rasterX);
                assert(nYSize == rasterY);

                int bGotMin, bGotMax;
                double adfMinMax[2] = {0, 0};
                adfMinMax[0] = poBand->GetMinimum(&bGotMin);
                adfMinMax[1] = poBand->GetMaximum(&bGotMax);
                if (!bGotMin || !bGotMax) {
                    // DGM FIXME: if the file is corrupted (e.g. ASCII ArcGrid
                    // with missing rows) this method will enter in a infinite
                    // loop!
                    GDALComputeRasterMinMax((GDALRasterBandH)poBand, TRUE,
                                            adfMinMax);
                }
                CVLog::Print("[GDAL] Min=%.3fd, Max=%.3f", adfMinMax[0],
                             adfMinMax[1]);

                GDALColorTable* colTable = poBand->GetColorTable();
                if (colTable != nullptr)
                    printf("[GDAL] Band has a color table with %d entries",
                           colTable->GetColorEntryCount());

                if (poBand->GetOverviewCount() > 0)
                    printf("[GDAL] Band has %d overviews",
                           poBand->GetOverviewCount());

                if (colorInterp ==
                            GCI_Undefined  // probably heights? DGM: no GDAL is
                                           // lost if the bands are coded with
                                           // 64 bits values :(
                    && !zRasterProcessed &&
                    (colorBands >= 3 || rasterCount < 4 ||
                     i > (rasterCount == 4 ? 3 : 4)) &&
                    !loadAsTexturedQuad
                    /*&& !colTable*/) {
                    zRasterProcessed = true;
                    zMinMax[0] = adfMinMax[0];
                    zMinMax[1] = adfMinMax[1];

                    double* scanline =
                            (double*)CPLMalloc(sizeof(double) * nXSize);
                    // double* scanline = new double[nXSize];
                    memset(scanline, 0, sizeof(double) * nXSize);

                    if (!validPoints.reserve(pc->capacity())) {
                        assert(!quad);
                        delete pc;
                        CPLFree(scanline);
                        GDALClose(poDataset);
                        return CC_FERR_READING;
                    }

                    for (int j = 0; j < nYSize; ++j) {
                        if (poBand->RasterIO(GF_Read,
                                             /*xOffset=*/0,
                                             /*yOffset=*/j,
                                             /*xSize=*/nXSize,
                                             /*ySize=*/1,
                                             /*buffer=*/scanline,
                                             /*bufferSizeX=*/nXSize,
                                             /*bufferSizeY=*/1,
                                             /*bufferType=*/GDT_Float64,
                                             /*x_offset=*/0,
                                             /*y_offset=*/0) != CE_None) {
                            assert(!quad);
                            delete pc;
                            CPLFree(scanline);
                            GDALClose(poDataset);
                            return CC_FERR_READING;
                        }

                        for (int k = 0; k < nXSize; ++k) {
                            double z =
                                    static_cast<double>(scanline[k]) + Pshift.z;
                            unsigned pointIndex =
                                    static_cast<unsigned>(k + j * rasterX);
                            if (pointIndex <= pc->size()) {
                                const_cast<CCVector3*>(pc->getPoint(pointIndex))
                                        ->z =
                                        static_cast<PointCoordinateType>(z);
                                if (z >= zMinMax[0] && z <= zMinMax[1]) {
                                    validPoints.addPointIndex(pointIndex);
                                }
                            }
                        }
                    }

                    // update bounding-box
                    pc->invalidateBoundingBox();

                    if (scanline) CPLFree(scanline);
                    scanline = 0;
                } else  // colors
                {
                    bool isRGB = false;
                    bool isScalar = false;
                    bool isPalette = false;

                    switch (colorInterp) {
                        case GCI_Undefined:
                            isScalar = true;
                            break;
                        case GCI_PaletteIndex:
                            isPalette = true;
                            break;
                        case GCI_RedBand:
                        case GCI_GreenBand:
                        case GCI_BlueBand:
                            isRGB = true;
                            break;
                        case GCI_AlphaBand:
                            if (adfMinMax[0] != adfMinMax[1]) {
                                if (loadAsTexturedQuad)
                                    isRGB = true;
                                else
                                    isScalar = true;  // we can't load the alpha
                                                      // band as a cloud color
                                                      // (transparency is not
                                                      // handled yet)
                            } else {
                                CVLog::Warning(
                                        QString("Alpha band ignored as it has "
                                                "a unique value (%1)")
                                                .arg(adfMinMax[0]));
                            }
                            break;
                        default:
                            isScalar = true;
                            break;
                    }

                    if (isRGB || isPalette) {
                        // first check that a palette exists if the band is a
                        // palette index
                        if (isPalette && !colTable) {
                            CVLog::Warning(
                                    QString("Band is declared as a '%1' but no "
                                            "palette is associated!")
                                            .arg(GDALGetColorInterpretationName(
                                                    colorInterp)));
                        } else {
                            // instantiate memory for RBG colors if necessary
                            if (!loadAsTexturedQuad && !pc->hasColors() &&
                                !pc->setRGBColor(ecvColor::white)) {
                                CVLog::Warning(
                                        QString("Failed to instantiate memory "
                                                "for storing color band '%1'!")
                                                .arg(GDALGetColorInterpretationName(
                                                        colorInterp)));
                            } else {
                                assert(poBand->GetRasterDataType() <=
                                       GDT_Int32);

                                int* colIndexes =
                                        (int*)CPLMalloc(sizeof(int) * nXSize);
                                // double* scanline = new double[nXSize];
                                memset(colIndexes, 0, sizeof(int) * nXSize);

                                for (int j = 0; j < nYSize; ++j) {
                                    if (poBand->RasterIO(
                                                GF_Read, /*xOffset=*/0,
                                                /*yOffset=*/j, /*xSize=*/nXSize,
                                                /*ySize=*/1,
                                                /*buffer=*/colIndexes,
                                                /*bufferSizeX=*/nXSize,
                                                /*bufferSizeY=*/1,
                                                /*bufferType=*/GDT_Int32,
                                                /*x_offset=*/0,
                                                /*y_offset=*/0) != CE_None) {
                                        CPLFree(colIndexes);
                                        if (quad)
                                            delete quad;
                                        else
                                            delete pc;
                                        return CC_FERR_READING;
                                    }

                                    for (int k = 0; k < nXSize; ++k) {
                                        unsigned pointIndex =
                                                static_cast<unsigned>(
                                                        k + j * rasterX);
                                        if (loadAsTexturedQuad ||
                                            pointIndex <= pc->size()) {
                                            ecvColor::Rgba C;
                                            if (loadAsTexturedQuad) {
                                                QRgb origColor =
                                                        quadTexture.pixel(k, j);
                                                C = ecvColor::FromQRgba(
                                                        origColor);
                                            } else {
                                                const ecvColor::Rgb& point_rgb =
                                                        pc->getPointColor(
                                                                pointIndex);
                                                C = ecvColor::Rgba(point_rgb,
                                                                   1.0);
                                            }

                                            switch (colorInterp) {
                                                case GCI_PaletteIndex:
                                                    assert(colTable);
                                                    {
                                                        GDALColorEntry col;
                                                        colTable->GetColorEntryAsRGB(
                                                                colIndexes[k],
                                                                &col);
                                                        C.r = static_cast<
                                                                ColorCompType>(
                                                                col.c1 &
                                                                ecvColor::MAX);
                                                        C.g = static_cast<
                                                                ColorCompType>(
                                                                col.c2 &
                                                                ecvColor::MAX);
                                                        C.b = static_cast<
                                                                ColorCompType>(
                                                                col.c3 &
                                                                ecvColor::MAX);
                                                    }
                                                    break;

                                                case GCI_RedBand:
                                                    C.r = static_cast<
                                                            ColorCompType>(
                                                            colIndexes[k] &
                                                            ecvColor::MAX);
                                                    break;
                                                case GCI_GreenBand:
                                                    C.g = static_cast<
                                                            ColorCompType>(
                                                            colIndexes[k] &
                                                            ecvColor::MAX);
                                                    break;
                                                case GCI_BlueBand:
                                                    C.b = static_cast<
                                                            ColorCompType>(
                                                            colIndexes[k] &
                                                            ecvColor::MAX);
                                                    break;

                                                case GCI_AlphaBand:
                                                    C.a = static_cast<
                                                            ColorCompType>(
                                                            colIndexes[k] &
                                                            ecvColor::MAX);
                                                    break;

                                                default:
                                                    assert(false);
                                                    break;
                                            }

                                            if (loadAsTexturedQuad) {
                                                quadTexture.setPixel(
                                                        k, j,
                                                        qRgba(C.r, C.g, C.b,
                                                              C.a));
                                            } else {
                                                pc->setPointColor(pointIndex,
                                                                  C);
                                            }
                                        }
                                    }
                                }

                                if (colIndexes) CPLFree(colIndexes);
                                colIndexes = nullptr;
                            }
                        }
                    } else if (isScalar && !loadAsTexturedQuad) {
                        QString sfName =
                                QString("band #%1 (%2)")
                                        .arg(i)
                                        .arg(GDALGetColorInterpretationName(
                                                colorInterp));  // SF names
                                                                // really need
                                                                // to be unique!
                        ccScalarField* sf =
                                new ccScalarField(qPrintable(sfName));
                        if (!sf->resizeSafe(pc->size(), true,
                                            NAN_VALUE)) {
                            CVLog::Warning(
                                    QString("Failed to instantiate memory for "
                                            "storing '%1' as a scalar field!")
                                            .arg(sf->getName()));
                            sf->release();
                            sf = nullptr;
                        } else {
                            double* colValues =
                                    (double*)CPLMalloc(sizeof(double) * nXSize);
                            // double* scanline = new double[nXSize];
                            memset(colValues, 0, sizeof(double) * nXSize);

                            for (int j = 0; j < nYSize; ++j) {
                                if (poBand->RasterIO(
                                            GF_Read, /*xOffset=*/0,
                                            /*yOffset=*/j, /*xSize=*/nXSize,
                                            /*ySize=*/1, /*buffer=*/colValues,
                                            /*bufferSizeX=*/nXSize,
                                            /*bufferSizeY=*/1,
                                            /*bufferType=*/GDT_Float64,
                                            /*x_offset=*/0,
                                            /*y_offset=*/0) != CE_None) {
                                    CPLFree(colValues);
                                    delete pc;
                                    return CC_FERR_READING;
                                }

                                for (int k = 0; k < nXSize; ++k) {
                                    unsigned pointIndex = static_cast<unsigned>(
                                            k + j * rasterX);
                                    if (pointIndex <= pc->size()) {
                                        ScalarType s = static_cast<ScalarType>(
                                                colValues[k]);
                                        sf->setValue(pointIndex, s);
                                    }
                                }
                            }

                            if (colValues) CPLFree(colValues);
                            colValues = 0;

                            sf->computeMinAndMax();
                            pc->addScalarField(sf);
                            if (pc->getNumberOfScalarFields() == 1) {
                                pc->setCurrentDisplayedScalarField(0);
                            }
                        }
                    }
                }
            }

            if (quad) {
                ccPlane::SetQuadTexture(quad, quadTexture.mirrored());
                container.addChild(quad);
            } else if (pc) {
                if (!zRasterProcessed) {
                    CVLog::Warning(
                            "Raster has no height (Z) information: you can "
                            "convert one of its scalar fields to Z with 'Edit "
                            "> Scalar Fields > Set SF as coordinate(s)'");
                } else if (validPoints.size() != 0 &&
                           validPoints.size() < pc->size()) {
                    // shall we remove the points with invalid heights?
                    static bool s_alwaysRemoveInvalidHeights = false;
                    int result = QMessageBox::Yes;
                    if (parameters.parentWidget)  // otherwise it means we are
                                                  // in command line mode --> no
                                                  // popup
                    {
                        result = (s_alwaysRemoveInvalidHeights
                                          ? QMessageBox::Yes
                                          : QMessageBox::question(
                                                    0, "Remove invalid points?",
                                                    "This raster has pixels "
                                                    "with invalid heights. "
                                                    "Shall we remove them?",
                                                    QMessageBox::Yes,
                                                    QMessageBox::YesToAll,
                                                    QMessageBox::No));
                    }
                    if (result != QMessageBox::No)  // Yes = let's remove them
                    {
                        if (result == QMessageBox::YesToAll)
                            s_alwaysRemoveInvalidHeights = true;

                        ccPointCloud* newPC = pc->partialClone(&validPoints);
                        if (newPC) {
                            delete pc;
                            pc = newPC;
                        } else {
                            CVLog::Error(
                                    "Not enough memory to remove the points "
                                    "with invalid heights!");
                        }
                    }
                }
                container.addChild(pc);

                // we give the priority to colors!
                if (pc->hasColors()) {
                    pc->showColors(true);
                    pc->showSF(false);
                } else if (pc->hasScalarFields()) {
                    pc->showSF(true);
                }
            }

            GDALClose(poDataset);
        } else {
            return CC_FERR_UNKNOWN_FILE;
        }
    } catch (...) {
        return CC_FERR_THIRD_PARTY_LIB_EXCEPTION;
    }

    return CC_FERR_NO_ERROR;
}

#endif
