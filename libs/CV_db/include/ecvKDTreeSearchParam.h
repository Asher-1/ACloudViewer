// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "CV_db.h"

namespace cloudViewer {
namespace geometry {

/// \class KDTreeSearchParam
///
/// \brief Base class for KDTree search parameters.
class CV_DB_LIB_API KDTreeSearchParam {
public:
    /// \enum SearchType
    ///
    /// \brief Specifies the search type for the search.
    enum class SearchType {
        Knn = 0,
        Radius = 1,
        Hybrid = 2,
    };

public:
    virtual ~KDTreeSearchParam() {}

protected:
    KDTreeSearchParam(SearchType type) : search_type_(type) {}

public:
    /// Get the search type (KNN, Radius, Hybrid) for the search parameter.
    SearchType GetSearchType() const { return search_type_; }

private:
    SearchType search_type_;
};

/// \class KDTreeSearchParamKNN
///
/// \brief KDTree search parameters for pure KNN search.
class CV_DB_LIB_API KDTreeSearchParamKNN : public KDTreeSearchParam {
public:
    /// \brief Default Cosntructor.
    ///
    /// \param knn Specifies the knn neighbors that will searched. Default
    /// is 30.
    KDTreeSearchParamKNN(int knn = 30)
        : KDTreeSearchParam(SearchType::Knn), knn_(knn) {}

public:
    /// Number of the neighbors that will be searched.
    int knn_;
};

/// \class KDTreeSearchParamRadius
///
/// \brief KDTree search parameters for pure radius search.
class CV_DB_LIB_API KDTreeSearchParamRadius : public KDTreeSearchParam {
public:
    /// \brief Default Cosntructor.
    ///
    /// \param radius Specifies the radius of the search.
    KDTreeSearchParamRadius(double radius)
        : KDTreeSearchParam(SearchType::Radius), radius_(radius) {}

public:
    /// Search radius.
    double radius_;
};

/// \class KDTreeSearchParamHybrid
///
/// \brief KDTree search parameters for hybrid KNN and radius search.
class CV_DB_LIB_API KDTreeSearchParamHybrid : public KDTreeSearchParam {
public:
    /// \brief Default Cosntructor.
    ///
    /// \param radius Specifies the radius of the search.
    /// \param max_nn Specifies the max neighbors to be searched.
    KDTreeSearchParamHybrid(double radius, int max_nn)
        : KDTreeSearchParam(SearchType::Hybrid),
          radius_(radius),
          max_nn_(max_nn) {}

public:
    /// Search radius.
    double radius_;
    /// At maximum, max_nn neighbors will be searched.
    int max_nn_;
};

}  // namespace geometry
}  // namespace cloudViewer
