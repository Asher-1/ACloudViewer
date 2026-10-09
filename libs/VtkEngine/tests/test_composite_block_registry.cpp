// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

/// Regression gate for the composite aggregation data layer — the structure
/// behind the massive-actor acceleration. Every case pins an invariant that
/// historically broke in the field (attribute revival on replace, hole reuse
/// after delete, group teardown on the last leaf) so the DB-tree
/// show/hide/delete/add pipes cannot silently regress again.

#include <gtest/gtest.h>
#include <vtkActor.h>
#include <vtkCellArray.h>
#include <vtkIdTypeArray.h>
#include <vtkNew.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkSmartPointer.h>

#include <QSet>
#include <string>

#include "CompositeBlockRegistry.h"
#include "Tools/SelectionTools/cvSelectionPipeline.h"

namespace {

vtkSmartPointer<vtkPolyData> MakeSingleVertexPolyData(double offset) {
    vtkNew<vtkPoints> points;
    points->InsertNextPoint(offset, 0.0, 0.0);
    vtkNew<vtkIdTypeArray> ids;
    ids->SetNumberOfValues(1);
    ids->SetValue(0, 0);
    vtkNew<vtkCellArray> cells;
    cells->SetCells(1, ids);
    vtkSmartPointer<vtkPolyData> poly = vtkSmartPointer<vtkPolyData>::New();
    poly->SetPoints(points);
    poly->SetVerts(cells);
    return poly;
}

cvSelectionData MakeSelection(const QVector<qint64>& sourceIds,
                              cvSelectionData::FieldAssociation association,
                              vtkActor* actor,
                              vtkPolyData* polyData,
                              unsigned int blockIndex,
                              const QVector<qint64>& localIds) {
    cvSelectionData selection(sourceIds, association);
    cvActorSelectionInfo info;
    info.actor = actor;
    info.polyData = polyData;
    info.blockIndex = blockIndex;
    info.selectedIds = localIds;
    selection.addActorInfo(info);
    return selection;
}

QSet<qint64> LocalIds(const cvSelectionData& selection,
                      vtkActor* actor,
                      vtkPolyData* polyData,
                      unsigned int blockIndex) {
    for (const cvActorSelectionInfo& info : selection.actorInfos()) {
        if (info.actor == actor && info.polyData == polyData &&
            info.blockIndex == blockIndex) {
            QSet<qint64> ids;
            for (qint64 id : info.selectedIds) ids.insert(id);
            return ids;
        }
    }
    return {};
}

class CompositeBlockRegistryTest : public ::testing::Test {
protected:
    void SetUp() override {
        group_ = "container#lw1";
        for (int i = 0; i < 3; ++i) {
            leaves_.push_back("leaf" + std::to_string(i));
        }
    }

    std::string group_;
    std::vector<std::string> leaves_;
    VtkRendering::CompositeBlockRegistry registry_;
};

TEST(SelectionPipelineRegression, MergesActorLocalIdsForPointsAndCells) {
    vtkNew<vtkActor> actor;
    vtkSmartPointer<vtkPolyData> poly = MakeSingleVertexPolyData(0.0);

    const cvSelectionData pointFirst =
            MakeSelection({100}, cvSelectionData::POINTS, actor, poly, 0, {5});
    const cvSelectionData pointSecond =
            MakeSelection({200}, cvSelectionData::POINTS, actor, poly, 0, {9});
    const cvSelectionData added = cvSelectionPipeline::combineSelections(
            pointFirst, pointSecond, cvSelectionPipeline::OPERATION_ADDITION);
    EXPECT_EQ((QSet<qint64>{5, 9}), LocalIds(added, actor, poly, 0));

    const cvSelectionData cellFirst =
            MakeSelection({30}, cvSelectionData::CELLS, actor, poly, 0, {3});
    const cvSelectionData cellSecond =
            MakeSelection({80}, cvSelectionData::CELLS, actor, poly, 0, {8});
    const cvSelectionData cellsAdded = cvSelectionPipeline::combineSelections(
            cellFirst, cellSecond, cvSelectionPipeline::OPERATION_ADDITION);
    EXPECT_EQ((QSet<qint64>{3, 8}), LocalIds(cellsAdded, actor, poly, 0));
}

TEST(SelectionPipelineRegression, PreservesActorBlockBoundariesAndModifiers) {
    vtkNew<vtkActor> actor;
    vtkNew<vtkActor> otherActor;
    vtkSmartPointer<vtkPolyData> firstPoly = MakeSingleVertexPolyData(0.0);
    vtkSmartPointer<vtkPolyData> secondPoly = MakeSingleVertexPolyData(1.0);

    const cvSelectionData first = MakeSelection({5, 9}, cvSelectionData::POINTS,
                                                actor, firstPoly, 0, {5, 9});
    const cvSelectionData sameBlock = MakeSelection(
            {9, 11}, cvSelectionData::POINTS, actor, firstPoly, 0, {9, 11});
    const cvSelectionData toggled = cvSelectionPipeline::combineSelections(
            first, sameBlock, cvSelectionPipeline::OPERATION_TOGGLE);
    EXPECT_EQ((QSet<qint64>{5, 11}), LocalIds(toggled, actor, firstPoly, 0));

    const cvSelectionData otherBlock = MakeSelection(
            {22}, cvSelectionData::POINTS, actor, secondPoly, 1, {2});
    const cvSelectionData otherActorSelection = MakeSelection(
            {33}, cvSelectionData::POINTS, otherActor, secondPoly, 0, {7});
    const cvSelectionData withDistinctScopes =
            cvSelectionPipeline::combineSelections(
                    cvSelectionPipeline::combineSelections(
                            first, otherBlock,
                            cvSelectionPipeline::OPERATION_ADDITION),
                    otherActorSelection,
                    cvSelectionPipeline::OPERATION_ADDITION);
    EXPECT_EQ((QSet<qint64>{5, 9}),
              LocalIds(withDistinctScopes, actor, firstPoly, 0));
    EXPECT_EQ((QSet<qint64>{2}),
              LocalIds(withDistinctScopes, actor, secondPoly, 1));
    EXPECT_EQ((QSet<qint64>{7}),
              LocalIds(withDistinctScopes, otherActor, secondPoly, 0));

    const cvSelectionData subtracted = cvSelectionPipeline::combineSelections(
            first,
            MakeSelection({9}, cvSelectionData::POINTS, actor, firstPoly, 0,
                          {9}),
            cvSelectionPipeline::OPERATION_SUBTRACTION);
    EXPECT_EQ((QSet<qint64>{5}), LocalIds(subtracted, actor, firstPoly, 0));
}

TEST_F(CompositeBlockRegistryTest, RegisterAndReverseLookup) {
    for (int i = 0; i < 3; ++i) {
        registry_.setLeafBlock(leaves_[i], group_, MakeSingleVertexPolyData(i));
    }
    EXPECT_EQ(3, registry_.leafSlotCount(group_));
    EXPECT_TRUE(registry_.hasGroup(group_));
    for (int i = 0; i < 3; ++i) {
        EXPECT_TRUE(registry_.containsLeaf(leaves_[i]));
        EXPECT_EQ(group_, registry_.groupOfLeaf(leaves_[i]));
        EXPECT_EQ(leaves_[i], registry_.leafAtFlatIndex(group_, i));
    }
}

TEST_F(CompositeBlockRegistryTest, VisibilityTogglePerBlock) {
    for (int i = 0; i < 3; ++i) {
        registry_.setLeafBlock(leaves_[i], group_, MakeSingleVertexPolyData(i));
    }
    registry_.setLeafVisibility(leaves_[1], false);
    EXPECT_FALSE(registry_.blockVisible(group_, 1));
    EXPECT_TRUE(registry_.blockVisible(group_, 0));
    EXPECT_TRUE(registry_.blockVisible(group_, 2));
    registry_.setLeafVisibility(leaves_[1], true);
    EXPECT_TRUE(registry_.blockVisible(group_, 1));
}

// A hidden leaf whose geometry is re-registered (redraw replace) must STAY
// hidden: setLeafBlock migrates per-leaf display attributes. Reviving the
// hidden state here is the exact "ghost geometry" regression class.
TEST_F(CompositeBlockRegistryTest, ReplaceMigratesVisibility) {
    registry_.setLeafBlock(leaves_[0], group_, MakeSingleVertexPolyData(0));
    registry_.setLeafVisibility(leaves_[0], false);
    registry_.setLeafBlock(leaves_[0], group_, MakeSingleVertexPolyData(5));
    EXPECT_TRUE(registry_.containsLeaf(leaves_[0]));
    EXPECT_FALSE(registry_.blockVisible(group_, 0));
}

TEST_F(CompositeBlockRegistryTest, RemoveCreatesReusableHole) {
    for (int i = 0; i < 3; ++i) {
        registry_.setLeafBlock(leaves_[i], group_, MakeSingleVertexPolyData(i));
    }
    std::string emptied;
    EXPECT_TRUE(registry_.removeLeaf(leaves_[1], &emptied));
    EXPECT_TRUE(emptied.empty());  // two leaves remain: group survives
    EXPECT_FALSE(registry_.containsLeaf(leaves_[1]));
    EXPECT_EQ(std::string(), registry_.leafAtFlatIndex(group_, 1));

    // A new leaf reuses the hole (same flat slot), born visible.
    registry_.setLeafBlock("leafNew", group_, MakeSingleVertexPolyData(9));
    EXPECT_EQ(3, registry_.leafSlotCount(group_));
    EXPECT_EQ("leafNew", registry_.leafAtFlatIndex(group_, 1));
    EXPECT_TRUE(registry_.blockVisible(group_, 1));
}

TEST_F(CompositeBlockRegistryTest, LastLeafDestroysGroup) {
    registry_.setLeafBlock(leaves_[0], group_, MakeSingleVertexPolyData(0));
    std::string emptied;
    EXPECT_TRUE(registry_.removeLeaf(leaves_[0], &emptied));
    EXPECT_EQ(group_, emptied);
    EXPECT_FALSE(registry_.hasGroup(group_));
    EXPECT_EQ(0, registry_.leafSlotCount(group_));
}

TEST_F(CompositeBlockRegistryTest, PrefixCleanupRemovesGroupFamilies) {
    registry_.setLeafBlock(leaves_[0], group_, MakeSingleVertexPolyData(0));
    registry_.setLeafBlock(leaves_[1], "container#lw2",
                           MakeSingleVertexPolyData(1));
    EXPECT_TRUE(registry_.hasCompositeGroupWithPrefix("container"));
    EXPECT_EQ(2, registry_.removeCompositeGroupsByPrefix("container"));
    EXPECT_FALSE(registry_.hasGroup(group_));
    EXPECT_FALSE(registry_.hasGroup("container#lw2"));
    EXPECT_FALSE(registry_.containsLeaf(leaves_[0]));
}

// The change-detecting push (steady-pass de-churn) contract: an UNSET block
// reads back as visible, so the first hide is a real change, repeats are
// no-ops, and unknown leaves report no change. Pinning the read-back
// default here — the whole O(changed) steady-pass optimization rests on it.
TEST_F(CompositeBlockRegistryTest, SetVisibilityIfChangedSemantics) {
    registry_.setLeafBlock(leaves_[0], group_, MakeSingleVertexPolyData(0));

    // Fresh leaf: unset reads visible, so hiding IS a change.
    EXPECT_TRUE(registry_.setLeafVisibilityIfChanged(leaves_[0], false));
    EXPECT_FALSE(registry_.blockVisible(group_, 0));

    // Repeating the same state is a no-op (the steady-pass skip).
    EXPECT_FALSE(registry_.setLeafVisibilityIfChanged(leaves_[0], false));
    EXPECT_FALSE(registry_.blockVisible(group_, 0));

    // Flipping back is a change again.
    EXPECT_TRUE(registry_.setLeafVisibilityIfChanged(leaves_[0], true));
    EXPECT_TRUE(registry_.blockVisible(group_, 0));
    EXPECT_FALSE(registry_.setLeafVisibilityIfChanged(leaves_[0], true));

    // Unknown leaves report no change and must not create anything.
    EXPECT_FALSE(registry_.setLeafVisibilityIfChanged("ghost", false));
    EXPECT_FALSE(registry_.containsLeaf("ghost"));
}

// Line-family aggregation contract (#plw group family): the replay
// suspension evicts sensor groups by the "#lw" SUBSTRING, so the
// polyline/lineset family MUST NOT collide with it — a "#plw" group id
// contains no "#lw" substring (static lines keep aggregating through
// replays), while the container-scoped prefix teardown still reaches both
// families.
TEST_F(CompositeBlockRegistryTest, LineFamilyGroupIsolatedFromSensorFamily) {
    registry_.setLeafBlock("lineLeaf", "container#plw2",
                           MakeSingleVertexPolyData(0));
    registry_.setLeafBlock(leaves_[0], "container#lw1",
                           MakeSingleVertexPolyData(1));

    // The replay eviction filter (groupId.find("#lw") == npos → skip) must
    // NOT match the line family. If this ever fires, a rename of the
    // group-family suffix would let replays evict aggregated lines while
    // their dedicated actors no longer exist — lines would vanish.
    EXPECT_EQ(std::string::npos, std::string("container#plw2").find("#lw"));

    // Container-scoped teardown reaches BOTH families (leaf-level removal
    // and folder deletion share this path; the prefix is the bare container
    // viewID — groupKeyMatchesPrefix requires the next char to be '#', so
    // sibling containers that share a name stem stay untouched).
    EXPECT_EQ(2, registry_.groupIdsWithPrefix("container").size());
    EXPECT_EQ(2, registry_.removeCompositeGroupsByPrefix("container"));
    EXPECT_FALSE(registry_.containsLeaf("lineLeaf"));
    EXPECT_FALSE(registry_.containsLeaf(leaves_[0]));
    EXPECT_FALSE(registry_.hasGroup("container#plw2"));
    EXPECT_FALSE(registry_.hasGroup("container#lw1"));
}

}  // namespace
