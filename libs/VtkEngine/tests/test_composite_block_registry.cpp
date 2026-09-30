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
#include <vtkCellArray.h>
#include <vtkIdTypeArray.h>
#include <vtkNew.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkSmartPointer.h>

#include <string>

#include "CompositeBlockRegistry.h"

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

}  // namespace
