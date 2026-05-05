#pragma once

#include "CV_db.h"
#include "ecvPointCloud.h"
#include "ecvScalarField.h"

#include <QUndoCommand>
#include <functional>
#include <vector>

class CV_DB_LIB_API ecvScalarFieldEditCommand : public QUndoCommand {
public:
    using RefreshFunc = std::function<void()>;

    ecvScalarFieldEditCommand(ccPointCloud* cloud,
                              int sfIndex,
                              unsigned startIndex,
                              const std::vector<ScalarType>& beforeValues,
                              const std::vector<ScalarType>& afterValues,
                              RefreshFunc refreshFunc,
                              const QString& label,
                              QUndoCommand* parent = nullptr)
        : QUndoCommand(label, parent)
        , m_cloud(cloud)
        , m_sfIndex(sfIndex)
        , m_startIndex(startIndex)
        , m_beforeValues(beforeValues)
        , m_afterValues(afterValues)
        , m_refresh(std::move(refreshFunc)) {}

    void undo() override {
        applyValues(m_beforeValues);
        if (m_refresh) m_refresh();
    }

    void redo() override {
        if (m_firstRedo) {
            m_firstRedo = false;
            return;
        }
        applyValues(m_afterValues);
        if (m_refresh) m_refresh();
    }

    int id() const override { return 2004; }

    qint64 estimatedMemoryBytes() const {
        return static_cast<qint64>((m_beforeValues.size() + m_afterValues.size())
                                   * sizeof(ScalarType));
    }

private:
    void applyValues(const std::vector<ScalarType>& values) {
        if (!m_cloud) return;
        ccScalarField* sf = static_cast<ccScalarField*>(
            m_cloud->getScalarField(m_sfIndex));
        if (!sf) return;
        for (size_t i = 0; i < values.size(); ++i) {
            sf->setValue(m_startIndex + static_cast<unsigned>(i), values[i]);
        }
        sf->computeMinAndMax();
    }

    ccPointCloud* m_cloud;
    int m_sfIndex;
    unsigned m_startIndex;
    std::vector<ScalarType> m_beforeValues;
    std::vector<ScalarType> m_afterValues;
    RefreshFunc m_refresh;
    bool m_firstRedo = true;
};
