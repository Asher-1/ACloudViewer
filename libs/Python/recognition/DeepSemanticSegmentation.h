// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <ClassMap.h>

#include "eCV_python.h"

class ccHObject;
namespace cloudViewer {
namespace utility {

class ECV_PYTHON_LIB_API DeepSemanticSegmentation {
public:
    DeepSemanticSegmentation();
    ~DeepSemanticSegmentation();

public:
    inline void setEnableVotes(bool state) { m_useVotes = state; }
    inline void setEnableSampling(bool state) { m_useGridSampling = state; }
    void setInputCloud(const ccHObject* cloud);
    void compute(std::vector<std::vector<size_t>>& clusters,
                 std::vector<ClassMap::ClusterMap>& cluster_map);

protected:
    void extract(const std::vector<std::vector<size_t>>& preds,
                 std::vector<ClassMap::ClusterMap>& clusters);

private:
    const ccHObject* m_container;
    bool m_batchMode;
    bool m_useGridSampling;
    bool m_useVotes;
};

}  // namespace utility
}  // namespace cloudViewer
