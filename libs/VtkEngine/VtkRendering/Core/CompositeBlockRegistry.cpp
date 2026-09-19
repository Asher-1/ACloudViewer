// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "CompositeBlockRegistry.h"

// VTK
#include <vtkActor.h>
#include <vtkCompositeDataDisplayAttributes.h>
#include <vtkDataObject.h>
#include <vtkMatrix4x4.h>
#include <vtkMultiBlockDataSet.h>
#include <vtkNew.h>
#include <vtkPolyData.h>
#include <vtkProperty.h>
#include <vtkSmartPointer.h>
#include <vtkVersion.h>

// Mapper selection: vtkCompositePolyDataMapper2 is ParaView's production
// composite mapper and renders into vtkHardwareSelector's attribute passes,
// so block picking (COMPOSITE_INDEX) works. The newer batched delegator
// mapper (VTK >= 9.2) does not implement the selection passes: the group
// actor produced no fragments during hardware picks, so every mesh click
// fell through to the surrounding point cloud. Selection correctness beats
// the unproven batching gain; revisit only with a controlled A/B that also
// proves pick parity.
#include <vtkCompositePolyDataMapper2.h>

namespace VtkRendering {

CompositeBlockRegistry::CompositeBlockRegistry() = default;

CompositeBlockRegistry::~CompositeBlockRegistry() = default;

CompositeBlockRegistry::Group* CompositeBlockRegistry::groupOf(
        const std::string& groupId) {
    auto it = m_groups.find(groupId);
    return it != m_groups.end() ? &it->second : nullptr;
}

const CompositeBlockRegistry::Group* CompositeBlockRegistry::groupOf(
        const std::string& groupId) const {
    auto it = m_groups.find(groupId);
    return it != m_groups.end() ? &it->second : nullptr;
}

vtkDataObject* CompositeBlockRegistry::blockObject(const Group& group,
                                                   vtkIdType flatIndex) const {
    if (flatIndex < 0 ||
        flatIndex >= static_cast<vtkIdType>(group.leafOrder.size()) ||
        group.leafOrder[flatIndex].empty()) {
        return nullptr;
    }
    return group.blocks->GetBlock(flatIndex);
}

vtkActor* CompositeBlockRegistry::ensureGroup(const std::string& groupId) {
    if (auto* existing = groupOf(groupId)) {
        return existing->actor;
    }

    Group& group = m_groups[groupId];
    group.blocks = vtkSmartPointer<vtkMultiBlockDataSet>::New();
    group.attributes =
            vtkSmartPointer<vtkCompositeDataDisplayAttributes>::New();

    // vtkNew keeps sole ownership until SetMapper registers it on the actor;
    // an early UnRegister here would destroy the mapper before SetMapper and
    // crash inside vtkActor::SetMapper on a dangling pointer.
    vtkNew<vtkCompositePolyDataMapper2> mapper;
    mapper->SetInputDataObject(group.blocks);
    mapper->SetCompositeDataDisplayAttributes(group.attributes);

    group.actor = vtkSmartPointer<vtkActor>::New();
    group.actor->GetProperty()->SetInterpolationToFlat();
    group.actor->GetProperty()->SetBackfaceCulling(false);
    group.actor->GetProperty()->SetFrontfaceCulling(false);
    group.actor->SetMapper(mapper);
    m_actorToGroup[group.actor] = groupId;

    return group.actor;
}

bool CompositeBlockRegistry::hasGroup(const std::string& groupId) const {
    return m_groups.count(groupId) != 0;
}

void CompositeBlockRegistry::destroyGroup(const std::string& groupId) {
    auto it = m_groups.find(groupId);
    if (it == m_groups.end()) {
        return;
    }

    for (const auto& leafId : it->second.leafOrder) {
        if (!leafId.empty()) {
            m_leafIndex.erase(leafId);
        }
    }
    m_actorToGroup.erase(it->second.actor);
    m_groups.erase(it);
}

vtkActor* CompositeBlockRegistry::groupActor(const std::string& groupId) const {
    const Group* group = groupOf(groupId);
    return group ? group->actor : nullptr;
}

std::vector<std::string> CompositeBlockRegistry::groupIds() const {
    std::vector<std::string> ids;
    ids.reserve(m_groups.size());
    for (const auto& kv : m_groups) {
        ids.push_back(kv.first);
    }
    return ids;
}

vtkIdType CompositeBlockRegistry::setLeafBlock(const std::string& leafId,
                                               const std::string& groupId,
                                               vtkPolyData* data) {
    if (!data) {
        std::string emptied;
        removeLeaf(leafId, &emptied);
        return -1;
    }

    // A leaf belongs to exactly one group; move it if re-parented.
    if (const auto* previous = [&] {
            auto it = m_leafIndex.find(leafId);
            return it != m_leafIndex.end() ? &it->second : nullptr;
        }()) {
        Group* group = groupOf(groupId);
        if (previous->groupId == groupId && group && group->blocks) {
            // Redraws replace the block data object; migrate the per-leaf
            // display attributes so opacity/visibility/color survive.
            if (vtkDataObject* oldBlock =
                        blockObject(*group, previous->flatIndex)) {
                if (group->attributes->HasBlockVisibility(oldBlock)) {
                    group->attributes->SetBlockVisibility(
                            data,
                            group->attributes->GetBlockVisibility(oldBlock));
                }
                if (group->attributes->HasBlockOpacity(oldBlock)) {
                    group->attributes->SetBlockOpacity(
                            data, group->attributes->GetBlockOpacity(oldBlock));
                }
                if (group->attributes->HasBlockColor(oldBlock)) {
                    double color[3];
                    group->attributes->GetBlockColor(oldBlock, color);
                    group->attributes->SetBlockColor(data, color);
                }
                if (group->attributes->HasBlockPickability(oldBlock)) {
                    group->attributes->SetBlockPickability(
                            data,
                            group->attributes->GetBlockPickability(oldBlock));
                }
            }
            group->blocks->SetBlock(previous->flatIndex, data);
            return previous->flatIndex;
        }
        removeLeaf(leafId);
    }

    // Full initialization must happen BEFORE taking the Group reference:
    // m_groups[groupId] alone would create an empty entry whose blocks are
    // null, and ensureGroup's early-return would then skip initialization —
    // the null->SetBlock crash seen in the field.
    ensureGroup(groupId);
    Group& group = *groupOf(groupId);

    vtkIdType slot;
    if (!group.freeSlots.empty()) {
        slot = group.freeSlots.back();
        group.freeSlots.pop_back();
        group.leafOrder[slot] = leafId;
        group.blocks->SetBlock(slot, data);
    } else {
        slot = static_cast<vtkIdType>(group.leafOrder.size());
        group.leafOrder.push_back(leafId);
        group.blocks->SetBlock(slot, data);
    }

    m_leafIndex[leafId] = LeafRef{groupId, slot};
    return slot;
}

bool CompositeBlockRegistry::removeLeaf(const std::string& leafId,
                                        std::string* emptiedGroup) {
    auto it = m_leafIndex.find(leafId);
    if (it == m_leafIndex.end()) {
        return false;
    }

    const LeafRef ref = it->second;
    m_leafIndex.erase(it);

    auto git = m_groups.find(ref.groupId);
    if (git == m_groups.end()) {
        return true;
    }

    Group& group = git->second;
    group.blocks->SetBlock(ref.flatIndex, nullptr);
    group.leafOrder[ref.flatIndex].clear();
    group.freeSlots.push_back(ref.flatIndex);

    // Destroy empty groups eagerly: the actor must leave the renderer as the
    // last leaf disappears (the caller observes this through emptiedGroup).
    const bool empty = group.freeSlots.size() == group.leafOrder.size();
    if (empty) {
        if (emptiedGroup) {
            *emptiedGroup = ref.groupId;
        }
        m_groups.erase(git);
    }
    return true;
}

bool CompositeBlockRegistry::containsLeaf(const std::string& leafId) const {
    return m_leafIndex.count(leafId) != 0;
}

std::string CompositeBlockRegistry::groupOfLeaf(
        const std::string& leafId) const {
    auto it = m_leafIndex.find(leafId);
    return it != m_leafIndex.end() ? it->second.groupId : std::string();
}

std::string CompositeBlockRegistry::leafAtFlatIndex(const std::string& groupId,
                                                    vtkIdType flatIndex) const {
    const Group* group = groupOf(groupId);
    if (!group || flatIndex < 0 ||
        flatIndex >= static_cast<vtkIdType>(group->leafOrder.size())) {
        return std::string();
    }
    return group->leafOrder[flatIndex];
}

std::string CompositeBlockRegistry::groupIdOfActor(vtkActor* actor) const {
    if (!actor) {
        return std::string();
    }
    auto it = m_actorToGroup.find(actor);
    return it != m_actorToGroup.end() ? it->second : std::string();
}

vtkIdType CompositeBlockRegistry::leafSlotCount(
        const std::string& groupId) const {
    const Group* group = groupOf(groupId);
    return group ? static_cast<vtkIdType>(group->leafOrder.size()) : 0;
}

vtkDataObject* CompositeBlockRegistry::blockAtFlatIndex(
        const std::string& groupId, vtkIdType flatIndex) const {
    const Group* group = groupOf(groupId);
    return group ? blockObject(*group, flatIndex) : nullptr;
}

bool CompositeBlockRegistry::blockVisible(const std::string& groupId,
                                          vtkIdType flatIndex) const {
    const Group* group = groupOf(groupId);
    if (!group) {
        return false;
    }
    vtkDataObject* block = blockObject(*group, flatIndex);
    if (!block) {
        return false;
    }
    // Invisible leaves render nothing, so a CPU pick ray must skip them too.
    if (group->attributes->HasBlockVisibility(block)) {
        return group->attributes->GetBlockVisibility(block);
    }
    return true;
}

bool CompositeBlockRegistry::bakeLeafTransform(const std::string& leafId,
                                               const vtkMatrix4x4* matrix) {
    if (!matrix) {
        return false;
    }
    auto it = m_leafIndex.find(leafId);
    if (it == m_leafIndex.end()) {
        return false;
    }
    Group& group = m_groups[it->second.groupId];
    auto* block =
            vtkPolyData::SafeDownCast(blockObject(group, it->second.flatIndex));
    if (!block) {
        return false;
    }

    vtkPoints* points = block->GetPoints();
    if (!points) {
        return false;
    }

    // In-place bake: same data object keeps per-leaf display attributes and
    // avoids a full block re-registration. O(points) per transformed leaf,
    // cheap for the small parts that aggregate into composite groups.
    const double* m = matrix->GetData();  // row-major 4x4
    for (vtkIdType i = 0; i < points->GetNumberOfPoints(); ++i) {
        const double* p = points->GetPoint(i);
        double out[3];
        for (int r = 0; r < 3; ++r) {
            out[r] = m[r * 4 + 0] * p[0] + m[r * 4 + 1] * p[1] +
                     m[r * 4 + 2] * p[2] + m[r * 4 + 3];
        }
        points->SetPoint(i, out);
    }
    points->Modified();
    block->Modified();
    return true;
}

void CompositeBlockRegistry::setLeafVisibility(const std::string& leafId,
                                               bool visible) {
    auto it = m_leafIndex.find(leafId);
    if (it == m_leafIndex.end()) {
        return;
    }
    Group& group = m_groups[it->second.groupId];
    if (vtkDataObject* block = blockObject(group, it->second.flatIndex)) {
        group.attributes->SetBlockVisibility(block, visible);
    }
}

void CompositeBlockRegistry::setLeafOpacity(const std::string& leafId,
                                            double opacity) {
    auto it = m_leafIndex.find(leafId);
    if (it == m_leafIndex.end()) {
        return;
    }
    Group& group = m_groups[it->second.groupId];
    if (vtkDataObject* block = blockObject(group, it->second.flatIndex)) {
        group.attributes->SetBlockOpacity(block, opacity);
    }
}

void CompositeBlockRegistry::setLeafColor(const std::string& leafId,
                                          const double color[3]) {
    auto it = m_leafIndex.find(leafId);
    if (it == m_leafIndex.end()) {
        return;
    }
    Group& group = m_groups[it->second.groupId];
    if (vtkDataObject* block = blockObject(group, it->second.flatIndex)) {
        if (color) {
            group.attributes->SetBlockColor(block, color);
        } else {
            group.attributes->RemoveBlockColor(block);
        }
    }
}

void CompositeBlockRegistry::setLeafPickability(const std::string& leafId,
                                                bool pickable) {
    auto it = m_leafIndex.find(leafId);
    if (it == m_leafIndex.end()) {
        return;
    }
    Group& group = m_groups[it->second.groupId];
    if (vtkDataObject* block = blockObject(group, it->second.flatIndex)) {
        group.attributes->SetBlockPickability(block, pickable);
    }
}

void CompositeBlockRegistry::setGroupTransform(const std::string& groupId,
                                               const vtkMatrix4x4* matrix) {
    Group* group = groupOf(groupId);
    if (!group) {
        return;
    }
    if (matrix) {
        group->actor->SetUserMatrix(const_cast<vtkMatrix4x4*>(matrix));
    } else {
        group->actor->SetUserMatrix(nullptr);
    }
}

}  // namespace VtkRendering
