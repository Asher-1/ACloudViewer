// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: BSD-3-Clause
// ----------------------------------------------------------------------------
// CVImageViewer.hpp – Template implementations for PclUtils::ImageViewer.

#ifndef CV_IMAGE_VIEWER_HPP_
#define CV_IMAGE_VIEWER_HPP_

#include <vtkContextActor.h>
#include <vtkContextScene.h>
#include <vtkImageData.h>
#include <vtkImageFlip.h>
#include <vtkPointData.h>

#include <pcl/search/organized.h>

#include <cstdlib>

namespace PclUtils {
namespace detail {
/** Local replacement for pcl::visualization::getRandomColors. */
inline void getRandomColors(double& r, double& g, double& b,
                            double min = 0.2, double max = 2.8) {
    double sum;
    static unsigned stepRGBA = 100;
    do {
        r = (std::rand() % stepRGBA) / static_cast<double>(stepRGBA);
        while ((g = (std::rand() % stepRGBA) / static_cast<double>(stepRGBA)) == r) {}
        while (((b = (std::rand() % stepRGBA) / static_cast<double>(stepRGBA)) == r) && (b == g)) {}
        sum = r + g + b;
    } while (sum <= min || sum >= max);
}
}  // namespace detail
}  // namespace PclUtils

///////////////////////////////////////////////////////////////////////////////////////////
template <typename T>
void PclUtils::ImageViewer::convertRGBCloudToUChar(
        const pcl::PointCloud<T>& cloud,
        boost::shared_array<unsigned char>& data) {
    int j = 0;
    for (const auto& point : cloud) {
        data[j++] = point.r;
        data[j++] = point.g;
        data[j++] = point.b;
    }
}

///////////////////////////////////////////////////////////////////////////////////////////
template <typename T>
void PclUtils::ImageViewer::addRGBImage(const pcl::PointCloud<T>& cloud,
                                     const std::string& layer_id,
                                     double opacity) {
    if (data_size_ < cloud.width * cloud.height) {
        data_size_ = cloud.width * cloud.height * 3;
        data_.reset(new unsigned char[data_size_]);
    }
    convertRGBCloudToUChar(cloud, data_);
    return addRGBImage(data_.get(), cloud.width, cloud.height, layer_id,
                       opacity);
}

///////////////////////////////////////////////////////////////////////////////////////////
template <typename T>
void PclUtils::ImageViewer::showRGBImage(const pcl::PointCloud<T>& cloud,
                                      const std::string& layer_id,
                                      double opacity) {
    addRGBImage<T>(cloud, layer_id, opacity);
    render();
}

///////////////////////////////////////////////////////////////////////////////////////////
template <typename T>
bool PclUtils::ImageViewer::addMask(
        const typename pcl::PointCloud<T>::ConstPtr& image,
        const pcl::PointCloud<T>& mask, double r, double g, double b,
        const std::string& layer_id, double opacity) {
    if (!image->isOrganized()) return false;

    LayerMap::iterator am_it = std::find_if(
            layer_map_.begin(), layer_map_.end(), LayerComparator(layer_id));
    if (am_it == layer_map_.end())
        am_it = createLayer(layer_id, getSize()[0] - 1, getSize()[1] - 1,
                            opacity, false);

    pcl::search::OrganizedNeighbor<T> search;
    search.setInputCloud(image);
    std::vector<float> xy;
    xy.reserve(mask.size() * 2);
    for (std::size_t i = 0; i < mask.size(); ++i) {
        pcl::PointXY p_projected;
        search.projectPoint(mask[i], p_projected);
        xy.push_back(p_projected.x);
        xy.push_back(static_cast<float>(image->height) - p_projected.y);
    }

    vtkSmartPointer<context_items::Points> points =
            vtkSmartPointer<context_items::Points>::New();
    points->setColors(static_cast<unsigned char>(r * 255.0),
                      static_cast<unsigned char>(g * 255.0),
                      static_cast<unsigned char>(b * 255.0));
    points->setOpacity(opacity);
    points->set(xy);
    am_it->actor->GetScene()->AddItem(points);
    return true;
}

///////////////////////////////////////////////////////////////////////////////////////////
template <typename T>
bool PclUtils::ImageViewer::addMask(
        const typename pcl::PointCloud<T>::ConstPtr& image,
        const pcl::PointCloud<T>& mask, const std::string& layer_id,
        double opacity) {
    return addMask(image, mask, 1.0, 0.0, 0.0, layer_id, opacity);
}

///////////////////////////////////////////////////////////////////////////////////////////
template <typename T>
bool PclUtils::ImageViewer::addPlanarPolygon(
        const typename pcl::PointCloud<T>::ConstPtr& image,
        const pcl::PlanarPolygon<T>& polygon, double r, double g, double b,
        const std::string& layer_id, double opacity) {
    if (!image->isOrganized()) return false;

    LayerMap::iterator am_it = std::find_if(
            layer_map_.begin(), layer_map_.end(), LayerComparator(layer_id));
    if (am_it == layer_map_.end())
        am_it = createLayer(layer_id, getSize()[0] - 1, getSize()[1] - 1,
                            opacity, false);

    pcl::search::OrganizedNeighbor<T> search;
    search.setInputCloud(image);
    std::vector<float> xy;
    xy.reserve((polygon.getContour().size() + 1) * 2);
    for (std::size_t i = 0; i < polygon.getContour().size(); ++i) {
        pcl::PointXY p;
        search.projectPoint(polygon.getContour()[i], p);
        xy.push_back(p.x);
        xy.push_back(p.y);
    }
    // Close the polygon
    xy[xy.size() - 2] = xy[0];
    xy[xy.size() - 1] = xy[1];

    vtkSmartPointer<context_items::Polygon> poly =
            vtkSmartPointer<context_items::Polygon>::New();
    poly->setColors(static_cast<unsigned char>(r * 255.0),
                    static_cast<unsigned char>(g * 255.0),
                    static_cast<unsigned char>(b * 255.0));
    poly->setOpacity(opacity);
    poly->set(xy);
    am_it->actor->GetScene()->AddItem(poly);
    return true;
}

///////////////////////////////////////////////////////////////////////////////////////////
template <typename T>
bool PclUtils::ImageViewer::addPlanarPolygon(
        const typename pcl::PointCloud<T>::ConstPtr& image,
        const pcl::PlanarPolygon<T>& polygon, const std::string& layer_id,
        double opacity) {
    return addPlanarPolygon(image, polygon, 1.0, 0.0, 0.0, layer_id, opacity);
}

///////////////////////////////////////////////////////////////////////////////////////////
template <typename T>
bool PclUtils::ImageViewer::addRectangle(
        const typename pcl::PointCloud<T>::ConstPtr& image, const T& min_pt,
        const T& max_pt, double r, double g, double b,
        const std::string& layer_id, double opacity) {
    if (!image->isOrganized()) return false;

    LayerMap::iterator am_it = std::find_if(
            layer_map_.begin(), layer_map_.end(), LayerComparator(layer_id));
    if (am_it == layer_map_.end())
        am_it = createLayer(layer_id, getSize()[0] - 1, getSize()[1] - 1,
                            opacity, false);

    pcl::search::OrganizedNeighbor<T> search;
    search.setInputCloud(image);
    T p1, p2, p3, p4, p5, p6, p7, p8;
    p1.x = min_pt.x; p1.y = min_pt.y; p1.z = min_pt.z;
    p2.x = min_pt.x; p2.y = min_pt.y; p2.z = max_pt.z;
    p3.x = min_pt.x; p3.y = max_pt.y; p3.z = min_pt.z;
    p4.x = min_pt.x; p4.y = max_pt.y; p4.z = max_pt.z;
    p5.x = max_pt.x; p5.y = min_pt.y; p5.z = min_pt.z;
    p6.x = max_pt.x; p6.y = min_pt.y; p6.z = max_pt.z;
    p7.x = max_pt.x; p7.y = max_pt.y; p7.z = min_pt.z;
    p8.x = max_pt.x; p8.y = max_pt.y; p8.z = max_pt.z;

    std::vector<pcl::PointXY> pp_2d(8);
    search.projectPoint(p1, pp_2d[0]);
    search.projectPoint(p2, pp_2d[1]);
    search.projectPoint(p3, pp_2d[2]);
    search.projectPoint(p4, pp_2d[3]);
    search.projectPoint(p5, pp_2d[4]);
    search.projectPoint(p6, pp_2d[5]);
    search.projectPoint(p7, pp_2d[6]);
    search.projectPoint(p8, pp_2d[7]);

    pcl::PointXY min_pt_2d, max_pt_2d;
    min_pt_2d.x = min_pt_2d.y = std::numeric_limits<float>::max();
    max_pt_2d.x = max_pt_2d.y = -std::numeric_limits<float>::max();
    for (const auto& point : pp_2d) {
        if (point.x < min_pt_2d.x) min_pt_2d.x = point.x;
        if (point.y < min_pt_2d.y) min_pt_2d.y = point.y;
        if (point.x > max_pt_2d.x) max_pt_2d.x = point.x;
        if (point.y > max_pt_2d.y) max_pt_2d.y = point.y;
    }
    min_pt_2d.y = float(image->height) - min_pt_2d.y;
    max_pt_2d.y = float(image->height) - max_pt_2d.y;

    vtkSmartPointer<context_items::Rectangle> rect =
            vtkSmartPointer<context_items::Rectangle>::New();
    rect->setColors(static_cast<unsigned char>(255.0 * r),
                    static_cast<unsigned char>(255.0 * g),
                    static_cast<unsigned char>(255.0 * b));
    rect->setOpacity(opacity);
    rect->set(min_pt_2d.x, min_pt_2d.y, max_pt_2d.x, max_pt_2d.y);
    am_it->actor->GetScene()->AddItem(rect);
    return true;
}

///////////////////////////////////////////////////////////////////////////////////////////
template <typename T>
bool PclUtils::ImageViewer::addRectangle(
        const typename pcl::PointCloud<T>::ConstPtr& image, const T& min_pt,
        const T& max_pt, const std::string& layer_id, double opacity) {
    return addRectangle<T>(image, min_pt, max_pt, 0.0, 1.0, 0.0, layer_id,
                           opacity);
}

///////////////////////////////////////////////////////////////////////////////////////////
template <typename T>
bool PclUtils::ImageViewer::addRectangle(
        const typename pcl::PointCloud<T>::ConstPtr& image,
        const pcl::PointCloud<T>& mask, double r, double g, double b,
        const std::string& layer_id, double opacity) {
    if (!image->isOrganized()) return false;

    LayerMap::iterator am_it = std::find_if(
            layer_map_.begin(), layer_map_.end(), LayerComparator(layer_id));
    if (am_it == layer_map_.end())
        am_it = createLayer(layer_id, getSize()[0] - 1, getSize()[1] - 1,
                            opacity, false);

    pcl::search::OrganizedNeighbor<T> search;
    search.setInputCloud(image);
    std::vector<pcl::PointXY> pp_2d(mask.size());
    for (std::size_t i = 0; i < mask.size(); ++i)
        search.projectPoint(mask[i], pp_2d[i]);

    pcl::PointXY min_pt_2d, max_pt_2d;
    min_pt_2d.x = min_pt_2d.y = std::numeric_limits<float>::max();
    max_pt_2d.x = max_pt_2d.y = -std::numeric_limits<float>::max();
    for (const auto& point : pp_2d) {
        if (point.x < min_pt_2d.x) min_pt_2d.x = point.x;
        if (point.y < min_pt_2d.y) min_pt_2d.y = point.y;
        if (point.x > max_pt_2d.x) max_pt_2d.x = point.x;
        if (point.y > max_pt_2d.y) max_pt_2d.y = point.y;
    }
    min_pt_2d.y = float(image->height) - min_pt_2d.y;
    max_pt_2d.y = float(image->height) - max_pt_2d.y;

    vtkSmartPointer<context_items::Rectangle> rect =
            vtkSmartPointer<context_items::Rectangle>::New();
    rect->setColors(static_cast<unsigned char>(255.0 * r),
                    static_cast<unsigned char>(255.0 * g),
                    static_cast<unsigned char>(255.0 * b));
    rect->setOpacity(opacity);
    rect->set(min_pt_2d.x, min_pt_2d.y, max_pt_2d.x, max_pt_2d.y);
    am_it->actor->GetScene()->AddItem(rect);
    return true;
}

///////////////////////////////////////////////////////////////////////////////////////////
template <typename T>
bool PclUtils::ImageViewer::addRectangle(
        const typename pcl::PointCloud<T>::ConstPtr& image,
        const pcl::PointCloud<T>& mask, const std::string& layer_id,
        double opacity) {
    return addRectangle(image, mask, 0.0, 1.0, 0.0, layer_id, opacity);
}

///////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT>
bool PclUtils::ImageViewer::showCorrespondences(
        const pcl::PointCloud<PointT>& source_img,
        const pcl::PointCloud<PointT>& target_img,
        const pcl::Correspondences& correspondences, int nth,
        const std::string& layer_id) {
    if (correspondences.empty()) return false;

    LayerMap::iterator am_it = std::find_if(
            layer_map_.begin(), layer_map_.end(), LayerComparator(layer_id));
    if (am_it == layer_map_.end())
        am_it = createLayer(layer_id,
                            source_img.width + target_img.width,
                            std::max(source_img.height, target_img.height),
                            1.0, false);

    int src_size = source_img.width * source_img.height * 3;
    int tgt_size = target_img.width * target_img.height * 3;

    setSize(source_img.width + target_img.width,
            std::max(source_img.height, target_img.height));

    if (data_size_ < static_cast<std::size_t>(src_size + tgt_size)) {
        data_size_ = src_size + tgt_size;
        data_.reset(new unsigned char[data_size_]);
    }

    int j = 0;
    for (std::size_t i = 0;
         i < std::max(source_img.height, target_img.height); ++i) {
        if (i < source_img.height) {
            for (std::size_t k = 0; k < source_img.width; ++k) {
                data_[j++] = source_img[i * source_img.width + k].r;
                data_[j++] = source_img[i * source_img.width + k].g;
                data_[j++] = source_img[i * source_img.width + k].b;
            }
        } else {
            memset(&data_[j], 0, source_img.width * 3);
            j += source_img.width * 3;
        }
        if (i < source_img.height) {
            for (std::size_t k = 0; k < target_img.width; ++k) {
                data_[j++] = target_img[i * source_img.width + k].r;
                data_[j++] = target_img[i * source_img.width + k].g;
                data_[j++] = target_img[i * source_img.width + k].b;
            }
        } else {
            memset(&data_[j], 0, target_img.width * 3);
            j += target_img.width * 3;
        }
    }

    void* data =
            const_cast<void*>(reinterpret_cast<const void*>(data_.get()));

    vtkSmartPointer<vtkImageData> image =
            vtkSmartPointer<vtkImageData>::New();
    image->SetDimensions(source_img.width + target_img.width,
                         std::max(source_img.height, target_img.height), 1);
    image->AllocateScalars(VTK_UNSIGNED_CHAR, 3);
    image->GetPointData()->GetScalars()->SetVoidArray(data, data_size_, 1);

    vtkSmartPointer<ContextImageItem> image_item =
            vtkSmartPointer<ContextImageItem>::New();
    image_item->set(0, 0, image);
    interactor_style_->adjustCamera(image, ren_);
    am_it->actor->GetScene()->AddItem(image_item);
    win_->SetSize(image->GetDimensions()[0], image->GetDimensions()[1]);

    for (std::size_t i = 0; i < correspondences.size(); i += nth) {
        double r, g, b;
        detail::getRandomColors(r, g, b);
        unsigned char u_r = static_cast<unsigned char>(255.0 * r);
        unsigned char u_g = static_cast<unsigned char>(255.0 * g);
        unsigned char u_b = static_cast<unsigned char>(255.0 * b);

        vtkSmartPointer<context_items::Circle> query_circle =
                vtkSmartPointer<context_items::Circle>::New();
        query_circle->setColors(u_r, u_g, u_b);
        vtkSmartPointer<context_items::Circle> match_circle =
                vtkSmartPointer<context_items::Circle>::New();
        match_circle->setColors(u_r, u_g, u_b);
        vtkSmartPointer<context_items::Line> line =
                vtkSmartPointer<context_items::Line>::New();
        line->setColors(u_r, u_g, u_b);

        float query_x =
                correspondences[i].index_query % source_img.width;
        float match_x = correspondences[i].index_match % target_img.width +
                         source_img.width;
        float query_y =
                getSize()[1] -
                correspondences[i].index_query / source_img.width;
        float match_y =
                getSize()[1] -
                correspondences[i].index_match / target_img.width;

        query_circle->set(query_x, query_y, 3.0);
        match_circle->set(match_x, match_y, 3.0);
        line->set(query_x, query_y, match_x, match_y);

        am_it->actor->GetScene()->AddItem(query_circle);
        am_it->actor->GetScene()->AddItem(match_circle);
        am_it->actor->GetScene()->AddItem(line);
    }
    return true;
}

#endif  // CV_IMAGE_VIEWER_HPP_

