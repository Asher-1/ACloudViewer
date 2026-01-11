// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ExtractSIFT.h"

// LOCAL
#include <Utils/cc2sm.h>
#include <Utils/sm2cc.h>

#include "PclUtils/PCLModules.h"
#include "dialogs/SIFTExtractDlg.h"

// ECV_DB_LIB
#include <ecvPointCloud.h>

// ECV_PLUGINS
#include <ecvMainAppInterface.h>

// QT
#include <QMainWindow>

// SYSTEM
#include <iostream>

ExtractSIFT::ExtractSIFT()
    : BasePclModule(
              PclModuleDescription(tr("Extract SIFT"),
                                   tr("Extract SIFT Keypoints"),
                                   tr("Extract SIFT keypoints for clouds with "
                                      "intensity/RGB or any scalar field"),
                                   ":/toolbar/PclAlgorithms/icons/sift.png")),
      m_dialog(nullptr),
      m_nr_octaves(0),
      m_min_scale(0),
      m_nr_scales_per_octave(0),
      m_min_contrast(0),
      m_use_min_contrast(false),
      m_mode(RGB) {}

ExtractSIFT::~ExtractSIFT() {
    // we must delete parent-less dialogs ourselves!
    if (m_dialog && m_dialog->parent() == nullptr) delete m_dialog;
}

int ExtractSIFT::checkSelected() {
    // do we have a selected cloud?
    int have_cloud = isFirstSelectedCcPointCloud();
    if (have_cloud != 1) return -11;

    // do we have at least a scalar field?
    int have_sf = hasSelectedScalarField();
    if (have_sf == 1) return 1;

    // also having rgb data will be enough
    if (hasSelectedRGB() != 0) return 1;

    return -51;
}

int ExtractSIFT::openInputDialog() {
    // do we have scalar fields?
    std::vector<std::string> fields = getSelectedAvailableScalarFields();

    // do we have rgb fields?
    if (hasSelectedRGB() == 1) {
        // add rgb field
        fields.push_back("rgb");
    }

    if (fields.empty())  // fields?
        return -51;

    // initialize the dialog object
    if (!m_dialog)
        m_dialog =
                new SIFTExtractDlg(m_app ? m_app->getActiveWindow() : nullptr);

    // update the combo box
    m_dialog->updateComboBox(fields);

    if (!m_dialog->exec()) return 0;

    return 1;
}

void ExtractSIFT::getParametersFromDialog() {
    if (!m_dialog) return;

    // get the parameters from the dialog
    m_nr_octaves = m_dialog->nrOctaves->value();
    m_min_scale = static_cast<float>(m_dialog->minScale->value());
    m_nr_scales_per_octave = m_dialog->scalesPerOctave->value();
    m_use_min_contrast = m_dialog->useMinContrast->checkState();
    m_min_contrast =
            m_use_min_contrast
                    ? static_cast<float>(m_dialog->minContrast->value())
                    : 0;
    m_field_to_use = m_dialog->intensityCombo->currentText();

    if (m_field_to_use == "rgb") {
        m_mode = RGB;
    } else {
        m_mode = SCALAR_FIELD;
    }

    QString fieldname(m_field_to_use);
    fieldname.replace(' ', '_');
    m_field_to_use_no_space =
            qPrintable(fieldname);  // DGM: warning, toStdString doesn't
                                    // preserve "local" characters
}

int ExtractSIFT::checkParameters() {
    if ((m_nr_octaves > 0) && (m_min_scale > 0) &&
        (m_nr_scales_per_octave > 0)) {
        if (m_use_min_contrast) {
            if (m_min_contrast > 0) {
                return 1;
            } else {
                return -52;
            }
        } else {
            return 1;
        }
    } else {
        return -52;
    }
}

int ExtractSIFT::compute() {
    ccPointCloud* cloud = getSelectedEntityAsCCPointCloud();
    if (!cloud) return -1;

    std::list<std::string> req_fields;
    try {
        req_fields.push_back("xyz");  // always needed
        switch (m_mode) {
            case RGB:
                req_fields.push_back("rgb");
                break;
            case SCALAR_FIELD:
                req_fields.push_back(qPrintable(
                        m_field_to_use));  // DGM: warning, toStdString doesn't
                                           // preserve "local" characters
                break;
            default:
                assert(false);
                break;
        }
    } catch (const std::bad_alloc&) {
        // not enough memory
        return -1;
    }

    PCLCloud::Ptr sm_cloud = cc2smReader(cloud).getAsSM(req_fields);
    if (!sm_cloud) return -1;

    // Now change the name of the field to use to a standard name, only if in
    // OTHER_FIELD mode
    if (m_mode == SCALAR_FIELD) {
        int field_index =
                pcl::getFieldIndex(*sm_cloud, m_field_to_use_no_space);
        sm_cloud->fields.at(field_index).name =
                "intensity";  // we always use intensity as name... even if it
                              // is curvature or another field.
    }

    // initialize all possible clouds
    pcl::PointCloud<pcl::PointXYZ>::Ptr out_cloud(
            new pcl::PointCloud<pcl::PointXYZ>);

    // Now do the actual computation
    if (m_mode == SCALAR_FIELD) {
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_i(
                new pcl::PointCloud<pcl::PointXYZI>);
        FROM_PCL_CLOUD(*sm_cloud, *cloud_i);
        PCLModules::EstimateSIFT<pcl::PointXYZI, pcl::PointXYZ>(
                cloud_i, out_cloud, m_nr_octaves, m_min_scale,
                m_nr_scales_per_octave, m_min_contrast);
    } else if (m_mode == RGB) {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_rgb(
                new pcl::PointCloud<pcl::PointXYZRGB>);
        FROM_PCL_CLOUD(*sm_cloud, *cloud_rgb);
        PCLModules::EstimateSIFT<pcl::PointXYZRGB, pcl::PointXYZ>(
                cloud_rgb, out_cloud, m_nr_octaves, m_min_scale,
                m_nr_scales_per_octave, m_min_contrast);
    }

    PCLCloud out_cloud_sm;
    TO_PCL_CLOUD(*out_cloud, out_cloud_sm);

    if (out_cloud_sm.height * out_cloud_sm.width == 0) {
        // cloud is empty
        return -53;
    }

    ccPointCloud* out_cloud_cc = pcl2cc::Convert(out_cloud_sm);
    if (!out_cloud_cc) {
        // conversion failed (not enough memory?)
        return -1;
    }

    QString name;
    if (m_mode == RGB)
        name = tr("SIFT Keypoints_%1_rgb_%2_%3_%4")
                       .arg(m_nr_octaves)
                       .arg(m_min_scale)
                       .arg(m_nr_scales_per_octave)
                       .arg(m_min_contrast);
    else
        name = tr("SIFT Keypoints_%1_%2_%3_%4_%5")
                       .arg(m_nr_octaves)
                       .arg(m_field_to_use_no_space.c_str())
                       .arg(m_min_scale)
                       .arg(m_nr_scales_per_octave)
                       .arg(m_min_contrast);

    out_cloud_cc->setName(name);
    out_cloud_cc->setRGBColor(ecvColor::red);
    out_cloud_cc->showColors(true);
    out_cloud_cc->showSF(false);
    out_cloud_cc->setPointSize(5);

    // copy global shift & scale
    out_cloud_cc->setGlobalScale(cloud->getGlobalScale());
    out_cloud_cc->setGlobalShift(cloud->getGlobalShift());

    if (cloud->getParent()) cloud->getParent()->addChild(out_cloud_cc);

    emit newEntity(out_cloud_cc);

    return 1;
}

QString ExtractSIFT::getErrorMessage(int errorCode) {
    switch (errorCode) {
            // THESE CASES CAN BE USED TO OVERRIDE OR ADD FILTER-SPECIFIC ERRORS
            // CODES ALSO IN DERIVED CLASSES DEFULAT MUST BE ""

        case -51:
            return "Selected entity does not have any suitable scalar field or "
                   "RGB. Intensity scalar field or RGB are needed for "
                   "computing SIFT";
        case -52:
            return "Wrong Parameters. One or more parameters cannot be "
                   "accepted";
        case -53:
            return "SIFT keypoint extraction does not returned any point. Try "
                   "relaxing your parameters";
        default:
            // see below
            break;
    }

    return BasePclModule::getErrorMessage(errorCode);
}
