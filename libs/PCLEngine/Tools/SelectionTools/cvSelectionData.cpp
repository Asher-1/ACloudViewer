// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "cvSelectionData.h"

// VTK
#include <vtkActor.h>
#include <vtkIdTypeArray.h>
#include <vtkPolyData.h>

// STD
#include <algorithm>

//-----------------------------------------------------------------------------
cvSelectionData::cvSelectionData()
    : m_vtkArray(nullptr), m_fieldAssociation(CELLS) {}

//-----------------------------------------------------------------------------
cvSelectionData::cvSelectionData(vtkIdTypeArray* vtkArray, int association)
    : m_vtkArray(nullptr),
      m_fieldAssociation(static_cast<FieldAssociation>(association)) {
    if (vtkArray) {
        // CRITICAL FIX: Validate array before DeepCopy
        try {
            vtkIdType numTuples = vtkArray->GetNumberOfTuples();
            if (numTuples < 0) {
                // Invalid array
                return;
            }

            m_vtkArray = vtkSmartPointer<vtkIdTypeArray>::New();
            m_vtkArray->DeepCopy(vtkArray);
        } catch (...) {
            // DeepCopy failed - leave m_vtkArray as nullptr
            m_vtkArray = nullptr;
        }
    }
}

//-----------------------------------------------------------------------------
cvSelectionData::cvSelectionData(const QVector<qint64>& ids,
                                 FieldAssociation association)
    : m_vtkArray(nullptr), m_fieldAssociation(association) {
    if (!ids.isEmpty()) {
        m_vtkArray = vtkSmartPointer<vtkIdTypeArray>::New();
        m_vtkArray->SetNumberOfTuples(ids.size());
        for (int i = 0; i < ids.size(); ++i) {
            m_vtkArray->SetValue(i, ids[i]);
        }
    }
}

//-----------------------------------------------------------------------------
cvSelectionData::cvSelectionData(const cvSelectionData& other)
    : m_vtkArray(nullptr),
      m_fieldAssociation(other.m_fieldAssociation),
      m_actorInfos(other.m_actorInfos) {
    if (other.m_vtkArray) {
        // CRITICAL FIX: Validate array before DeepCopy
        try {
            vtkIdType numTuples = other.m_vtkArray->GetNumberOfTuples();
            if (numTuples < 0) {
                // Invalid array
                return;
            }

            m_vtkArray = vtkSmartPointer<vtkIdTypeArray>::New();
            m_vtkArray->DeepCopy(other.m_vtkArray);
        } catch (...) {
            // DeepCopy failed - leave m_vtkArray as nullptr
            m_vtkArray = nullptr;
        }
    }
}

//-----------------------------------------------------------------------------
cvSelectionData& cvSelectionData::operator=(const cvSelectionData& other) {
    if (this != &other) {
        // Copy new data (smart pointer handles cleanup automatically)
        m_fieldAssociation = other.m_fieldAssociation;
        m_actorInfos = other.m_actorInfos;

        if (other.m_vtkArray) {
            // CRITICAL FIX: Validate array before DeepCopy
            try {
                vtkIdType numTuples = other.m_vtkArray->GetNumberOfTuples();
                if (numTuples >= 0) {
                    m_vtkArray = vtkSmartPointer<vtkIdTypeArray>::New();
                    m_vtkArray->DeepCopy(other.m_vtkArray);
                } else {
                    m_vtkArray = nullptr;
                }
            } catch (...) {
                // DeepCopy failed - leave m_vtkArray as nullptr
                m_vtkArray = nullptr;
            }
        } else {
            m_vtkArray = nullptr;
        }
    }
    return *this;
}

//-----------------------------------------------------------------------------
cvSelectionData::~cvSelectionData() {
    // Smart pointer handles cleanup automatically
}

//-----------------------------------------------------------------------------
bool cvSelectionData::isEmpty() const {
    return !m_vtkArray || m_vtkArray->GetNumberOfTuples() == 0;
}

//-----------------------------------------------------------------------------
int cvSelectionData::count() const {
    return m_vtkArray ? m_vtkArray->GetNumberOfTuples() : 0;
}

//-----------------------------------------------------------------------------
QVector<qint64> cvSelectionData::ids() const {
    QVector<qint64> result;
    if (m_vtkArray) {
        vtkIdType numTuples = m_vtkArray->GetNumberOfTuples();
        result.reserve(numTuples);
        for (vtkIdType i = 0; i < numTuples; ++i) {
            result.append(m_vtkArray->GetValue(i));
        }
    }
    return result;
}

//-----------------------------------------------------------------------------
void cvSelectionData::clear() {
    m_vtkArray = nullptr;  // Smart pointer handles cleanup automatically
    m_actorInfos.clear();
}

//-----------------------------------------------------------------------------
QString cvSelectionData::fieldTypeString() const {
    return (m_fieldAssociation == CELLS) ? "cells" : "points";
}

//-----------------------------------------------------------------------------
void cvSelectionData::addActorInfo(const cvActorSelectionInfo& info) {
    m_actorInfos.append(info);

    // Sort by Z-value (front to back: smaller Z = closer to camera)
    std::sort(m_actorInfos.begin(), m_actorInfos.end(),
              [](const cvActorSelectionInfo& a, const cvActorSelectionInfo& b) {
                  return a.zValue < b.zValue;
              });
}

//-----------------------------------------------------------------------------
void cvSelectionData::setActorInfo(vtkActor* actor,
                                   vtkPolyData* polyData,
                                   double zValue) {
    m_actorInfos.clear();

    cvActorSelectionInfo info;
    info.actor = actor;
    info.polyData = polyData;
    info.zValue = zValue;

    m_actorInfos.append(info);
}

//-----------------------------------------------------------------------------
cvActorSelectionInfo cvSelectionData::actorInfo(int index) const {
    if (index >= 0 && index < m_actorInfos.size()) {
        return m_actorInfos[index];
    }
    return cvActorSelectionInfo();  // Return empty struct
}

//-----------------------------------------------------------------------------
vtkActor* cvSelectionData::primaryActor() const {
    return m_actorInfos.isEmpty() ? nullptr : m_actorInfos[0].actor;
}

//-----------------------------------------------------------------------------
vtkPolyData* cvSelectionData::primaryPolyData() const {
    return m_actorInfos.isEmpty() ? nullptr : m_actorInfos[0].polyData;
}

//-----------------------------------------------------------------------------
void cvSelectionData::clearActorInfo() { m_actorInfos.clear(); }
