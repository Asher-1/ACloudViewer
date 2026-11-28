// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "ccItemTree.h"
#include "ccMPlaneErrors.h"
#include "mocks/ccMainAppInterfaceMock.h"

namespace {

class ItemHierarchyTest : public testing::Test {
protected:
    void SetUp() override {
        // Stack objects must not acquire the default child-delete dependency.
        ASSERT_TRUE(child1.addChild(&child2, ccHObject::DP_NONE));
        ASSERT_TRUE(cloud.addChild(&child1, ccHObject::DP_NONE));
        ASSERT_TRUE(folder.addChild(&cloud, ccHObject::DP_NONE));
        ASSERT_TRUE(folder.addChild(&child3, ccHObject::DP_NONE));
    }

    ccHObject folder{"ICI Acquisition"};
    ccPointCloud cloud{"Parent"};
    ccPointCloud child1{"Child1"};
    ccPointCloud child2{"Child2"};
    ccPointCloud child3{"Child3"};
};

TEST_F(ItemHierarchyTest, InstantiatesClouds) {
    ASSERT_NE(&cloud, nullptr);
    ASSERT_NE(&child1, nullptr);
    ASSERT_NE(&child2, nullptr);
}

TEST_F(ItemHierarchyTest, RejectsNullRoot) {
    ASSERT_THROW(CC_ITEM_TREE::findItemInHierarchy(
                         nullptr, CC_ITEM_TREE::HIERARCHY_OBJECT),
                 MplaneInvalidArgument);
}

TEST_F(ItemHierarchyTest, FindsFirstLevelChildByName) {
    ASSERT_EQ(CC_ITEM_TREE::findItemInHierarchy(
                      &folder, CC_ITEM_TREE::HIERARCHY_OBJECT, "Child1"),
              &child1);
}

TEST_F(ItemHierarchyTest, FindsSecondLevelChildByName) {
    ASSERT_EQ(CC_ITEM_TREE::findItemInHierarchy(
                      &folder, CC_ITEM_TREE::HIERARCHY_OBJECT, "Child2"),
              &child2);
}

TEST_F(ItemHierarchyTest, ReturnsNullForUnknownName) {
    ASSERT_EQ(CC_ITEM_TREE::findItemInHierarchy(
                      &folder, CC_ITEM_TREE::HIERARCHY_OBJECT, "invalid"),
              nullptr);
}

TEST_F(ItemHierarchyTest, ReturnsNullForUnmatchedType) {
    ASSERT_EQ(CC_ITEM_TREE::findItemInHierarchy(&folder, CC_ITEM_TREE::LABEL2D),
              nullptr);
}

TEST_F(ItemHierarchyTest, ReturnsNullForUnmatchedTypeWithKnownName) {
    ASSERT_EQ(CC_ITEM_TREE::findItemInHierarchy(&folder, CC_ITEM_TREE::LABEL2D,
                                                "Child1"),
              nullptr);
}

TEST_F(ItemHierarchyTest, ReturnsNullForUnmatchedTypeAndName) {
    ASSERT_EQ(CC_ITEM_TREE::findItemInHierarchy(&folder, CC_ITEM_TREE::LABEL2D,
                                                "invalid"),
              nullptr);
}

TEST_F(ItemHierarchyTest, ReturnsFirstMatchingObject) {
    ASSERT_EQ(CC_ITEM_TREE::findItemInHierarchy(&folder,
                                                CC_ITEM_TREE::HIERARCHY_OBJECT),
              &cloud);
}

class ItemContainerTest : public testing::Test {
protected:
    void SetUp() override {
        ASSERT_TRUE(folder.addChild(&childFolder, ccHObject::DP_NONE));
    }

    testing::StrictMock<MainAppInterfaceMock> app;
    ccHObject folder{"Parent"};
    ccHObject childFolder{"Child"};
};

TEST_F(ItemContainerTest, RejectsNullParent) {
    ASSERT_THROW(
            CC_ITEM_TREE::findOrCreateContainer(nullptr, "New Container", &app),
            MplaneInvalidArgument);
}

TEST_F(ItemContainerTest, RejectsEmptyName) {
    ASSERT_THROW(CC_ITEM_TREE::findOrCreateContainer(&folder, "", &app),
                 MplaneInvalidArgument);
}

TEST_F(ItemContainerTest, RejectsNullApp) {
    ASSERT_THROW(CC_ITEM_TREE::findOrCreateContainer(&folder, "New Container",
                                                     nullptr),
                 MplaneInvalidArgument);
}

TEST_F(ItemContainerTest, ReusesMatchingChild) {
    ASSERT_EQ(CC_ITEM_TREE::findOrCreateContainer(&folder, "Child", &app),
              &childFolder);
}

TEST_F(ItemContainerTest, CreatesContainerAndAddsItToDatabase) {
    EXPECT_CALL(app, addToDB(testing::_, false, true, false, false)).Times(1);
    ccHObject* container =
            CC_ITEM_TREE::findOrCreateContainer(&folder, "New Container", &app);
    ASSERT_NE(container, nullptr);
    ASSERT_EQ(container->getName(), QString("New Container"));
}

class PointLabelTest : public testing::Test {
protected:
    void SetUp() override {
        ASSERT_TRUE(cloud.reserveThePointsTable(1));
        cloud.addPoint(CCVector3(1, 1, 1));
    }

    ccPointCloud cloud{"cloud"};
};

TEST_F(PointLabelTest, RejectsEmptyText) {
    ASSERT_THROW(CC_ITEM_TREE::createPointLabel2D("", &cloud, QPoint(1, 1), 0),
                 MplaneInvalidArgument);
}

TEST_F(PointLabelTest, RejectsNullCloud) {
    ASSERT_THROW(
            CC_ITEM_TREE::createPointLabel2D("Label", nullptr, QPoint(1, 1), 0),
            MplaneInvalidArgument);
}

TEST_F(PointLabelTest, ReturnsNullWithoutActiveDisplay) {
    ASSERT_EQ(CC_ITEM_TREE::createPointLabel2D("Test label", &cloud,
                                               QPoint(1, 1), 0),
              nullptr);
}

TEST_F(PointLabelTest, CreatesLabelWithActiveDisplay) {
    GTEST_SKIP() << "The original suite left active-display label creation "
                    "unimplemented.";
}

}  // namespace
