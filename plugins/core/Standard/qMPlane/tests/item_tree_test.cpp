//##########################################################################
//#                                                                        #
//#                CLOUDCOMPARE PLUGIN: qMPlane                            #
//#                                                                        #
//#  This program is free software; you can redistribute it and/or modify  #
//#  it under the terms of the GNU General Public License as published by  #
//#  the Free Software Foundation; version 2 of the License.               #
//#                                                                        #
//#  This program is distributed in the hope that it will be useful,       #
//#  but WITHOUT ANY WARRANTY; without even the implied warranty of        #
//#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         #
//#  GNU General Public License for more details.                          #
//#                                                                        #
//#            COPYRIGHT: AIT-Austrian Institute of Technology             #
//#                                                                        #
//##########################################################################

// Testing
#include "catch2/catch.hpp"
#include "trompeloeil\catch2\trompeloeil.hpp"

// Mocks
#include "mocks\ccMainAppInterfaceMock.h"
#include "mocks\ccGenericGLDisplayMock.h"

// CloudCompare
#include "ccPointCloud.h"
#include "ccHObject.h"

// Qt
#include "qmainwindow.h"

// Local dependencies
#include "ccItemTree.h"
#include "ccMPlaneErrors.h"

TEST_CASE ("Test findItemInHierarchy() and getChildrenByType() ", "[item-tree]")
{
	ccHObject folder = ("ICI Acquisition");
	ccPointCloud cloud = ("Parent");
	ccPointCloud child1 = ("Child1");
	ccPointCloud child2 = ("Child2");
	ccPointCloud child3 = ("Child3");

	child1.addChild(&child2);
	cloud.addChild(&child1);
	folder.addChild(&cloud);
	folder.addChild(&child3);

	SECTION("Check instantiation") {
		REQUIRE(&cloud != nullptr);
		REQUIRE(&child1 != nullptr);
		REQUIRE(&child2 != nullptr);
	}

	SECTION("Test findItemInHierarchy() with nullptr root returns nullptr") {
		REQUIRE_THROWS_AS(
			CC_ITEM_TREE::findItemInHierarchy(nullptr, CC_ITEM_TREE::CC_TYPES_MAP::HIERARCHY_OBJECT),
			MplaneInvalidArgument
		);
	}

	SECTION("Test findItemInHierarchy() with valid name returns 1st sub-level child") {
		REQUIRE(CC_ITEM_TREE::findItemInHierarchy(&folder, CC_ITEM_TREE::CC_TYPES_MAP::HIERARCHY_OBJECT, "Child1") == &child1);
	}

	SECTION("Test findItemInHierarchy() with valid name returns 2nd sub-level child") {
		REQUIRE(CC_ITEM_TREE::findItemInHierarchy(&folder, CC_ITEM_TREE::CC_TYPES_MAP::HIERARCHY_OBJECT, "Child2") == &child2);
	}

	SECTION("Test findItemInHierarchy() with invalid name returns nullptr") {
		REQUIRE(CC_ITEM_TREE::findItemInHierarchy(&folder, CC_ITEM_TREE::CC_TYPES_MAP::HIERARCHY_OBJECT, "invalid") == nullptr);
	}

	SECTION("Test findItemInHierarchy() with invalid CC_TYPES_MAP returns nullptr") {
		REQUIRE(CC_ITEM_TREE::findItemInHierarchy(&folder, CC_ITEM_TREE::CC_TYPES_MAP::LABEL2D) == nullptr);
	}

	SECTION("Test findItemInHierarchy() with invalid CC_TYPES_MAP and valid name returns nullptr") {
		REQUIRE(CC_ITEM_TREE::findItemInHierarchy(&folder, CC_ITEM_TREE::CC_TYPES_MAP::LABEL2D, "Child1") == nullptr);
	}

	SECTION("Test findItemInHierarchy() with invalid CC_TYPES_MAP and invalid name returns nullptr") {
		REQUIRE(CC_ITEM_TREE::findItemInHierarchy(&folder, CC_ITEM_TREE::CC_TYPES_MAP::LABEL2D, "invalid") == nullptr);
	}

	SECTION("Test findItemInHierarchy() returns first ccHObject with matching CC_TYPES_MAP") {
		REQUIRE(CC_ITEM_TREE::findItemInHierarchy(&folder, CC_ITEM_TREE::CC_TYPES_MAP::HIERARCHY_OBJECT) == &cloud);
	}
}

TEST_CASE("Test findOrCreateContainer() ", "[item-tree]")
{
	MainAppInterfaceMock app;
	ccHObject folder("Parent");
	ccHObject childFolder("Child");
	folder.addChild(&childFolder);

	SECTION("Test findOrCreateContainer() with nullptr as parent returns nullptr") {
		REQUIRE_THROWS_AS(
			CC_ITEM_TREE::findOrCreateContainer(nullptr, "New Container", &app),
			MplaneInvalidArgument
		);
	}

	SECTION("Test findOrCreateContainer() with empty containerName returns nullptr") {
		REQUIRE_THROWS_AS(
			CC_ITEM_TREE::findOrCreateContainer(&folder, "", &app),
			MplaneInvalidArgument
		);
	}

	SECTION("Test findOrCreateContainer() with nullptr as app returns nullptr") {
		REQUIRE_THROWS_AS(
			CC_ITEM_TREE::findOrCreateContainer(&folder, "New Container", nullptr),
			MplaneInvalidArgument
		);
	}

	SECTION("Test findOrCreateContainer() with existing child and matching containerName returns child") {
		REQUIRE(CC_ITEM_TREE::findOrCreateContainer(&folder, "Child", &app) == &childFolder);
	}

	SECTION("Test findOrCreateContainer() with new containerName returns new container") {
		REQUIRE_CALL(app, addToDB(ANY(ccHObject*), false, true, false, false));
		REQUIRE(CC_ITEM_TREE::findOrCreateContainer(&folder, "New Container", &app)->getName() == "New Container");
	}
}

TEST_CASE("Test createPointLabel2D() ", "[item-tree]")
{
	ccPointCloud cloud("cloud");
	cloud.reserveThePointsTable(1);
	cloud.addPoint(CCVector3(1, 1, 1));

	ccGenericGLDisplayMock display;
	trompeloeil::sequence seq;

	SECTION("TEST createPointLabel2D throws std::invalid_exception when labelText is empty") {
		REQUIRE_THROWS_AS(
			CC_ITEM_TREE::createPointLabel2D("", &cloud, QPoint(1, 1), 0),
			MplaneInvalidArgument
		);
	}

	SECTION("TEST createPointLabel2D throws std::invalid_exception when sourceCloud is nullptr") {
		REQUIRE_THROWS_AS(
			CC_ITEM_TREE::createPointLabel2D("Label", nullptr, QPoint(1, 1), 0),
			MplaneInvalidArgument
		);
	}

	SECTION("TEST createPointLabel2D returns nullptr if sourceCloud's display is nullptr") {
		REQUIRE(CC_ITEM_TREE::createPointLabel2D("Test label", &cloud, QPoint(1, 1), 0) == nullptr);
	}

	// ToDo: call setDisplay() and afterwards createPointLabel2D()
	SECTION("TEST createPointLabel2D returns Label2D when Display is set") {
		//REQUIRE_CALL(display, invalidateViewport());
		//REQUIRE_CALL(display, deprecate3DLayer()).TIMES(1);
		//cloud.setDisplay(&display);
	}
}