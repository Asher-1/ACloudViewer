// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/** @file CompositeBlockRegistry.h
 *  @brief Multi-block composite rendering registry for massive entity groups.
 *
 * ParaView renders tens of thousands of blocks through a single composite
 * mapper: one actor per group, one vtkMultiBlockDataSet input, and per-block
 * visibility / opacity / color / pickability carried by
 * vtkCompositeDataDisplayAttributes. This registry adapts that mechanism to
 * the per-entity viewID model used across VtkVis: leaf entities keep their
 * identity (viewID) while their geometry is rendered as blocks of one group
 * actor, so the frame loop stays O(1) per group instead of O(leaves).
 *
 * Design constraints (see docs for the full reasoning):
 * - Data layer independence: blocks only borrow geometry; the ccHObject tree
 *   and per-entity export/select semantics are untouched.
 * - No per-block transform exists in VTK composite mappers; leaf transforms
 *   are applied by re-converting the leaf geometry (cheap for the small
 *   parts that aggregate here), group transforms by the group actor matrix.
 * - Blocks hold plain vtkPolyData without textures; textured entities keep
 *   their dedicated actors (VTK composite display attributes carry no
 *   per-block texture).
 */

#include <vtkSmartPointer.h>
#include <vtkType.h>

#include <string>
#include <unordered_map>
#include <vector>

#include "qVTK.h"

class vtkActor;
class vtkCompositeDataDisplayAttributes;
class vtkDataObject;
class vtkMatrix4x4;
class vtkMultiBlockDataSet;
class vtkPolyData;

namespace VtkRendering {

class QVTK_ENGINE_LIB_API CompositeBlockRegistry {
public:
    CompositeBlockRegistry();
    ~CompositeBlockRegistry();

    CompositeBlockRegistry(const CompositeBlockRegistry&) = delete;
    CompositeBlockRegistry& operator=(const CompositeBlockRegistry&) = delete;

    /// Creates the group actor (composite mapper over a vtkMultiBlockDataSet)
    /// if missing. The caller owns adding/removing the returned actor to the
    /// renderer; the registry only fills it with blocks.
    vtkActor* ensureGroup(const std::string& groupId);
    bool hasGroup(const std::string& groupId) const;
    /// Drops the group and every leaf referencing it.
    void destroyGroup(const std::string& groupId);
    vtkActor* groupActor(const std::string& groupId) const;
    std::vector<std::string> groupIds() const;

    /// Registers (or replaces) a leaf geometry inside @a groupId. Returns the
    /// flat leaf index used later for pick reverse lookup. Null geometry
    /// removes the leaf instead.
    vtkIdType setLeafBlock(const std::string& leafId,
                           const std::string& groupId,
                           vtkPolyData* data);
    /// Removes a leaf (block slot becomes a reusable hole). @returns true if
    /// the leaf existed. When its group runs empty the group is destroyed and
    /// @a emptiedGroup is set to its id.
    bool removeLeaf(const std::string& leafId,
                    std::string* emptiedGroup = nullptr);
    bool containsLeaf(const std::string& leafId) const;
    std::string groupOfLeaf(const std::string& leafId) const;

    /// Reverse lookup for block picking: flat composite leaf index -> leaf id.
    std::string leafAtFlatIndex(const std::string& groupId,
                                vtkIdType flatIndex) const;

    // ---- CPU picking support (composite hw-selection fallback) ----
    /// Total slot count of a group's block array (including reusable holes).
    vtkIdType leafSlotCount(const std::string& groupId) const;
    /// Block data object at a flat slot (nullptr for holes / unknown group).
    vtkDataObject* blockAtFlatIndex(const std::string& groupId,
                                    vtkIdType flatIndex) const;
    /// Effective visibility of a block (composite display attributes +
    /// group default); CPU picking must skip invisible blocks.
    bool blockVisible(const std::string& groupId, vtkIdType flatIndex) const;

    /// Reverse lookup for picking: composite group actor -> group id (empty
    /// when the actor is not a group of this registry).
    std::string groupIdOfActor(vtkActor* actor) const;

    /// Applies a 4x4 transform in place to a leaf's block geometry (VTK
    /// composite mappers have no per-block transform). Keeps the same block
    /// data object so per-leaf display attributes survive the update.
    bool bakeLeafTransform(const std::string& leafId,
                           const vtkMatrix4x4* matrix);

    // ---- per-leaf display control (no-ops for unknown leaves) ----
    void setLeafVisibility(const std::string& leafId, bool visible);
    void setLeafOpacity(const std::string& leafId, double opacity);
    /// Overrides the leaf color; pass @a color == nullptr to clear the
    /// override so the leaf follows the group mapper color again.
    void setLeafColor(const std::string& leafId, const double color[3]);
    void setLeafPickability(const std::string& leafId, bool pickable);

    /// Group-level transform (applies to every leaf at once, O(1)).
    void setGroupTransform(const std::string& groupId,
                           const vtkMatrix4x4* matrix);

    size_t groupCount() const { return m_groups.size(); }
    size_t leafCount() const { return m_leafIndex.size(); }

private:
    struct Group {
        vtkSmartPointer<vtkMultiBlockDataSet> blocks;
        vtkSmartPointer<vtkCompositeDataDisplayAttributes> attributes;
        vtkSmartPointer<vtkActor> actor;
        // Flat slot -> leaf id; empty string marks a reusable hole.
        std::vector<std::string> leafOrder;
        std::vector<vtkIdType> freeSlots;
    };

    struct LeafRef {
        std::string groupId;
        vtkIdType flatIndex = -1;
    };

    Group* groupOf(const std::string& groupId);
    const Group* groupOf(const std::string& groupId) const;
    vtkDataObject* blockObject(const Group& group, vtkIdType flatIndex) const;

    std::unordered_map<std::string, Group> m_groups;
    std::unordered_map<std::string, LeafRef> m_leafIndex;
    std::unordered_map<vtkActor*, std::string> m_actorToGroup;
};

}  // namespace VtkRendering
