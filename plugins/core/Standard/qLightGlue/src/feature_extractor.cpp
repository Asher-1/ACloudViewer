#include "feature_extractor.h"

#include <QFileInfo>
#include <QImageReader>
#include <QtGlobal>

#include <algorithm>
#include <cmath>
#include <cstring>

#ifdef QLIGHTGLUE_HAS_OPENCV
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#endif

namespace lightglue_plugin {

QImage load_oriented_qimage(const QString& path) {
    QImageReader reader(path);
    reader.setAutoTransform(true);
    return reader.read();
}

namespace {

void set_error(std::string* error, const std::string& msg) {
    if (error) *error = msg;
}

#ifdef QLIGHTGLUE_HAS_OPENCV
cv::Mat load_gray_resized(const QString& path, int max_resize, int* out_w,
                          int* out_h) {
    const QImage oriented = load_oriented_qimage(path);
    if (oriented.isNull()) return {};
    const QImage rgb = oriented.convertToFormat(QImage::Format_RGB888);
    cv::Mat rgbMat(rgb.height(), rgb.width(), CV_8UC3,
                   const_cast<uchar*>(rgb.constBits()), rgb.bytesPerLine());
    cv::Mat bgr;
    cv::cvtColor(rgbMat.clone(), bgr, cv::COLOR_RGB2BGR);
    if (bgr.empty()) return {};
    cv::Mat gray;
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
    const int max_dim = std::max(gray.cols, gray.rows);
    if (max_resize > 0 && max_dim > max_resize) {
        const double scale = static_cast<double>(max_resize) / max_dim;
        cv::resize(gray, gray, cv::Size(), scale, scale, cv::INTER_AREA);
    }
    if (out_w) *out_w = gray.cols;
    if (out_h) *out_h = gray.rows;
    return gray;
}

void apply_rootsift(cv::Mat* descriptors) {
    if (!descriptors || descriptors->empty()) return;
    const float eps = 1e-6f;
    for (int r = 0; r < descriptors->rows; ++r) {
        float l1 = 0.f;
        for (int c = 0; c < descriptors->cols; ++c) {
            l1 += std::max(0.f, descriptors->at<float>(r, c));
        }
        if (l1 <= eps) continue;
        float l2 = 0.f;
        for (int c = 0; c < descriptors->cols; ++c) {
            const float v =
                    std::sqrt(std::max(0.f, descriptors->at<float>(r, c)) / l1);
            descriptors->at<float>(r, c) = v;
            l2 += v * v;
        }
        l2 = std::sqrt(std::max(l2, eps));
        for (int c = 0; c < descriptors->cols; ++c) {
            descriptors->at<float>(r, c) /= l2;
        }
    }
}

bool fill_from_cv(const cv::Mat& gray,
                  const std::vector<cv::KeyPoint>& keypoints,
                  const cv::Mat& descriptors,
                  OwnedFeatures* out) {
    if (!out) return false;
    out->keypoints.clear();
    out->descriptors.clear();
    out->keypoints.reserve(keypoints.size());
    for (const cv::KeyPoint& kp : keypoints) {
        aicore_lightglue_keypoint item{};
        item.x = kp.pt.x;
        item.y = kp.pt.y;
        item.scale = kp.size;
        item.orientation = kp.angle * static_cast<float>(M_PI / 180.0);
        out->keypoints.push_back(item);
    }
    out->descriptors.resize(static_cast<size_t>(descriptors.rows) *
                            static_cast<size_t>(descriptors.cols));
    if (!descriptors.isContinuous()) {
        cv::Mat continuous = descriptors.clone();
        std::memcpy(out->descriptors.data(), continuous.ptr<float>(),
                    out->descriptors.size() * sizeof(float));
    } else {
        std::memcpy(out->descriptors.data(), descriptors.ptr<float>(),
                    out->descriptors.size() * sizeof(float));
    }
    out->view.keypoints = out->keypoints.data();
    out->view.n_keypoints = static_cast<int32_t>(out->keypoints.size());
    out->view.descriptors = out->descriptors.data();
    out->view.descriptor_dim = descriptors.cols;
    out->view.image_width = gray.cols;
    out->view.image_height = gray.rows;
    return true;
}
#endif

bool copy_fixture_side(const aicore_lightglue_features& src, OwnedFeatures* out) {
    if (!out || !src.keypoints || src.n_keypoints <= 0 || !src.descriptors) {
        return false;
    }
    out->keypoints.assign(src.keypoints, src.keypoints + src.n_keypoints);
    const size_t n_desc =
            static_cast<size_t>(src.n_keypoints) *
            static_cast<size_t>(std::max(1, src.descriptor_dim));
    out->descriptors.assign(src.descriptors, src.descriptors + n_desc);
    out->view = src;
    out->view.keypoints = out->keypoints.data();
    out->view.descriptors = out->descriptors.data();
    return true;
}

}  // namespace

void release_owned(aicore_lightglue_features* features) {
    if (!features) return;
    features->keypoints = nullptr;
    features->descriptors = nullptr;
    features->n_keypoints = 0;
}

bool extract_sift_opencv(const QString& image_path,
                         int max_keypoints,
                         int max_resize,
                         OwnedFeatures* out,
                         std::string* error) {
#ifndef QLIGHTGLUE_HAS_OPENCV
    set_error(error, "OpenCV not available — rebuild with BUILD_OPENCV=ON");
    return false;
#else
    if (!out) {
        set_error(error, "null output");
        return false;
    }
    int w = 0;
    int h = 0;
    cv::Mat gray = load_gray_resized(image_path, max_resize, &w, &h);
    if (gray.empty()) {
        set_error(error, "failed to read image: " + image_path.toStdString());
        return false;
    }

    const int nfeatures = max_keypoints > 0 ? max_keypoints : 2048;
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create(nfeatures);
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    sift->detectAndCompute(gray, cv::noArray(), keypoints, descriptors);
    if (keypoints.empty() || descriptors.empty()) {
        set_error(error, "no SIFT features detected");
        return false;
    }
    if (descriptors.type() != CV_32F) {
        descriptors.convertTo(descriptors, CV_32F);
    }
    apply_rootsift(&descriptors);
    if (!fill_from_cv(gray, keypoints, descriptors, out)) {
        set_error(error, "failed to pack SIFT features");
        return false;
    }
    return true;
#endif
}

bool load_fixture_pair(const QString& fixture_path,
                       OwnedFeatures* out0,
                       OwnedFeatures* out1,
                       std::string* error) {
    aicore_lightglue_features f0{};
    aicore_lightglue_features f1{};
    if (aicore_lightglue_load_fixture(fixture_path.toUtf8().constData(), &f0,
                                      &f1) != 0) {
        set_error(error, "invalid LGINP01 fixture: " + fixture_path.toStdString());
        return false;
    }
    const bool ok = copy_fixture_side(f0, out0) && copy_fixture_side(f1, out1);
    aicore_lightglue_free_features(&f0);
    aicore_lightglue_free_features(&f1);
    if (!ok) {
        set_error(error, "empty fixture content");
        return false;
    }
    return true;
}

}  // namespace lightglue_plugin
