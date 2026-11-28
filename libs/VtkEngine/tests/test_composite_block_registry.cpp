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

#include <vtkVersion.h>
#if VTK_VERSION_NUMBER >= VTK_VERSION_CHECK(9, 4, 0)
#include <vtk_glad.h>
#else
#include <vtk_glew.h>
#endif

#include <QVTKOpenGLNativeWidget.h>
#include <ecvMesh.h>
#include <ecvPointCloud.h>
#include <ecvRepresentationManager.h>
#include <ecvViewManager.h>
#include <ecvViewRepresentation.h>
#include <gtest/gtest.h>
#include <vtkActor.h>
#include <vtkActorCollection.h>
#include <vtkCamera.h>
#include <vtkCellArray.h>
#include <vtkCellData.h>
#include <vtkCompositeDataDisplayAttributes.h>
#include <vtkCompositePolyDataMapper.h>
#if VTK_VERSION_NUMBER < VTK_VERSION_CHECK(9, 5, 0)
#include <vtkCompositePolyDataMapper2.h>
#endif
#include <vtkCubeSource.h>
#include <vtkDataObject.h>
#include <vtkDoubleArray.h>
#include <vtkIdTypeArray.h>
#include <vtkInformation.h>
#include <vtkIntArray.h>
#include <vtkLookupTable.h>
#include <vtkMatrix4x4.h>
#include <vtkMultiBlockDataSet.h>
#include <vtkNew.h>
#include <vtkPartitionedDataSet.h>
#include <vtkPartitionedDataSetCollection.h>
#include <vtkPlane.h>
#include <vtkPlaneCollection.h>
#include <vtkPointData.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkSelection.h>
#include <vtkSelectionNode.h>
#include <vtkSmartPointer.h>
#include <vtkTable.h>
#include <vtkTransform.h>
#include <vtkVersion.h>

#include <QApplication>
#include <QMainWindow>
#include <QSet>
#include <QSurfaceFormat>
#include <cmath>
#include <cstdlib>
#include <memory>
#include <set>
#include <string>

#include "CompositeBlockRegistry.h"
#include "Tools/SelectionTools/cvHardwareSelector.h"
#include "Tools/SelectionTools/cvSelectionHighlighter.h"
#include "Tools/SelectionTools/cvSelectionPipeline.h"
#include "Tools/SelectionTools/cvViewSelectionManager.h"
#include "VTKExtensions/InteractionStyle/vtkCustomInteractorStyle.h"
#include "Visualization/VtkDisplayTools.h"
#include "Visualization/VtkVis.h"
#include "Visualization/vtkGLView.h"

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

vtkSmartPointer<vtkActor> MakeSelectionActor(int kind, double z) {
    vtkNew<vtkPoints> points;
    points->InsertNextPoint(-0.5, -0.5, z);
    points->InsertNextPoint(0.5, -0.5, z);
    points->InsertNextPoint(0.0, 0.5, z);
    vtkNew<vtkCellArray> cells;
    const vtkIdType ids[] = {0, 1, 2};
    cells->InsertNextCell(kind == 0 ? 1 : 3, ids);
    vtkNew<vtkPolyData> poly;
    poly->SetPoints(points);
    if (kind == 0) {
        poly->SetVerts(cells);
    } else if (kind == 1) {
        poly->SetLines(cells);
    } else {
        poly->SetPolys(cells);
    }
    vtkNew<vtkPolyDataMapper> mapper;
    mapper->SetInputData(poly);
    auto actor = vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);
    actor->GetProperty()->SetPointSize(20);
    actor->GetProperty()->SetLineWidth(8);
    return actor;
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

// Real hardware-selection regression: run under Xvfb on headless Linux.
TEST(SelectionPipelineRegression, HardwareSelectionExcludesAuxiliaryProps) {
#if defined(__linux__)
    if (!std::getenv("DISPLAY")) {
        GTEST_SKIP() << "Hardware selection requires DISPLAY (use xvfb-run)";
    }
#endif
    vtkNew<vtkRenderer> renderer;
    vtkNew<vtkRenderWindow> window;
    window->SetSize(200, 200);
    window->SetMultiSamples(0);
    window->SetOffScreenRendering(1);
    window->AddRenderer(renderer);

    auto source = MakeSelectionActor(2, 0.0);
    renderer->AddActor(source);
    auto neverPickable = MakeSelectionActor(2, 0.8);
    neverPickable->PickableOff();
    renderer->AddActor(neverPickable);
    auto invisible = MakeSelectionActor(2, 0.9);
    invisible->VisibilityOff();
    renderer->AddActor(invisible);
    auto* camera = renderer->GetActiveCamera();
    camera->SetPosition(0, 0, 5);
    camera->SetFocalPoint(0, 0, 0);
    camera->ParallelProjectionOn();
    camera->SetParallelScale(1.0);
    renderer->ResetCameraClippingRange();
    window->Render();

    vtkNew<cvHardwareSelector> selector;
    selector->SetRenderer(renderer);
    // Inclusion must not make an originally non-pickable prop pickable.
    selector->SetSelectableProps(
            {source.GetPointer(), neverPickable.GetPointer()});
    int region[] = {10, 10, 189, 189};
    int polygon[] = {10, 10, 189, 10, 189, 189, 10, 189};
    auto selectedIds = [&](int association, bool usePolygon) {
        selector->SetFieldAssociation(association);
        selector->InvalidateCachedSelection();
        vtkSmartPointer<vtkSelection> selection;
        selection.TakeReference(usePolygon ? selector->PolygonSelect(polygon, 8)
                                           : selector->Select(region));
        QSet<qint64> ids;
        EXPECT_NE(nullptr, selection.GetPointer());
        if (!selection) return ids;
        for (unsigned int i = 0; i < selection->GetNumberOfNodes(); ++i) {
            auto* node = selection->GetNode(i);
            EXPECT_EQ(source.GetPointer(),
                      node->GetProperties()->Get(vtkSelectionNode::PROP()));
            auto* values =
                    vtkIdTypeArray::SafeDownCast(node->GetSelectionList());
            EXPECT_NE(nullptr, values);
            if (values) {
                for (vtkIdType j = 0; j < values->GetNumberOfTuples(); ++j) {
                    ids.insert(values->GetValue(j));
                }
            }
        }
        EXPECT_EQ(1, source->GetPickable());
        EXPECT_EQ(0, neverPickable->GetPickable());
        EXPECT_EQ(1, source->GetVisibility());
        EXPECT_EQ(1, neverPickable->GetVisibility());
        EXPECT_EQ(0, invisible->GetVisibility());
        return ids;
    };

    for (int association : {vtkDataObject::FIELD_ASSOCIATION_POINTS,
                            vtkDataObject::FIELD_ASSOCIATION_CELLS}) {
        for (bool usePolygon : {false, true}) {
            SCOPED_TRACE(::testing::Message() << "association=" << association
                                              << " polygon=" << usePolygon);
            const auto baseline = selectedIds(association, usePolygon);
            const QSet<qint64> expected =
                    association == vtkDataObject::FIELD_ASSOCIATION_POINTS
                            ? QSet<qint64>{0, 1, 2}
                            : QSet<qint64>{0};
            ASSERT_EQ(expected, baseline);
            for (int kind : {0, 1, 2}) {
                SCOPED_TRACE(::testing::Message() << "helper kind=" << kind);
                auto helper = MakeSelectionActor(kind, 0.5);
                renderer->AddActor(helper);
                EXPECT_EQ(baseline, selectedIds(association, usePolygon));
                EXPECT_EQ(1, helper->GetPickable());
                EXPECT_EQ(1, helper->GetVisibility());
                renderer->RemoveActor(helper);
                EXPECT_EQ(baseline, selectedIds(association, usePolygon));
            }
            selector->SetSelectableProps({});
            EXPECT_TRUE(selectedIds(association, usePolygon).isEmpty());
            selector->SetSelectableProps(
                    {source.GetPointer(), neverPickable.GetPointer()});
        }
    }
}

TEST(SelectionPipelineRegression,
     DbBackedSelectionExcludesRegisteredLabelGeometry) {
#if defined(__linux__)
    if (!std::getenv("DISPLAY")) GTEST_SKIP() << "Use xvfb-run";
#endif
    vtkNew<vtkRenderer> renderer;
    vtkNew<vtkRenderWindow> window;
    window->SetSize(200, 200);
    window->SetMultiSamples(0);
    window->SetOffScreenRendering(1);
    window->AddRenderer(renderer);
    vtkNew<VTKExtensions::vtkCustomInteractorStyle> style;
    Visualization::VtkVis vis(renderer, window, style);
    ccHObject root("selection-test");
    ccPointCloud cloud("source");
    ASSERT_TRUE(cloud.reserve(13));
    for (int i = 0; i < 13; ++i) {
        cloud.addPoint(i == 2   ? CCVector3(-0.5, -0.5, 0)
                       : i == 7 ? CCVector3(0.5, -0.5, 0)
                                : CCVector3(0, 0.5, 0));
    }
    ccMesh mesh(&cloud);
    ASSERT_TRUE(mesh.reserve(5));
    for (int i = 0; i < 5; ++i) mesh.addTriangle(2, 7, 12);
    root.addChild(&cloud, ccHObject::DP_NONE);
    root.addChild(&mesh, ccHObject::DP_NONE);
    struct RestoreRoot {
        ccHObject& root;
        ccHObject* previous = ecvViewManager::instance().globalDBRoot();
        ~RestoreRoot() {
            ecvViewManager::instance().setGlobalDBRoot(previous);
            root.removeAllChildren();  // DP_NONE: do not own stack entities.
        }
    } restore{root};
    ecvViewManager::instance().setGlobalDBRoot(&root);

    auto geometry = MakeSelectionActor(2, 0.0);
    auto* poly = vtkPolyData::SafeDownCast(geometry->GetMapper()->GetInput());
    vtkNew<vtkIdTypeArray> pointIds;
    pointIds->SetName("vtkOriginalPointIds");
    for (vtkIdType id : {2, 7, 12}) pointIds->InsertNextValue(id);
    poly->GetPointData()->AddArray(pointIds);
    vtkNew<vtkIdTypeArray> cellIds;
    cellIds->SetName("vtkOriginalCellIds");
    cellIds->InsertNextValue(4);
    poly->GetCellData()->AddArray(cellIds);
    const auto sourceId = mesh.getViewId().toStdString();
    ASSERT_TRUE(vis.addPointCloud(poly, sourceId));
    vis.setCurrentSourceObject(&mesh, sourceId);
    auto* source = vis.getActorById(sourceId);
    ASSERT_NE(nullptr, source);
    // Non-identity geometry must be tested in world space by through-selection.
    source->SetPosition(0.1, 0.1, 0.0);
    auto* camera = renderer->GetActiveCamera();
    camera->SetPosition(0, 0, 5);
    camera->SetFocalPoint(0, 0, 0);
    camera->ParallelProjectionOn();
    camera->SetParallelScale(1.0);
    camera->SetClippingRange(0.1, 10.0);
    window->Render();

    // Like a label's facet: ordinary registered mesh, but NOT a DB child.
    ccMesh temporaryMesh(&cloud);
    const auto helperId = temporaryMesh.getViewId().toStdString();
    auto helperGeometry = MakeSelectionActor(2, 0.5);
    auto* helperPoly =
            vtkPolyData::SafeDownCast(helperGeometry->GetMapper()->GetInput());
    cvSelectionPipeline pipeline;
    pipeline.setVisualizer(&vis);
    int region[] = {10, 10, 189, 189};
    vtkNew<vtkIntArray> polygon;
    polygon->SetNumberOfComponents(2);
    for (int xy : {10, 10, 189, 10, 189, 189, 10, 189}) {
        polygon->InsertNextValue(xy);
    }
    for (bool attached : {false, true, false}) {
        if (attached) {
            ASSERT_TRUE(vis.addPointCloud(helperPoly, helperId));
            vis.setCurrentSourceObject(&temporaryMesh, helperId);
        } else {
            vis.removePointCloud(helperId);
            vis.removeSourceObject(helperId);
        }
        const auto props = cvSelectionPipeline::dataSelectionProps(&vis);
        ASSERT_EQ(1u, props.size());
        EXPECT_EQ(1u, props.count(source));
        for (bool cells : {false, true}) {
            for (int mode : {0, 1, 2}) {
                SCOPED_TRACE(::testing::Message()
                             << "helper=" << attached << " cells=" << cells
                             << " mode=" << mode);
                cvSelectionData selected;
                if (mode == 0) {
                    selected = cells ? pipeline.selectCellsOnSurface(region)
                                     : pipeline.selectPointsOnSurface(region);
                } else if (mode == 1) {
                    selected = cells ? pipeline.selectCellsInPolygon(polygon)
                                     : pipeline.selectPointsInPolygon(polygon);
                } else {
                    selected = cells ? pipeline.selectCellsInFrustum(region)
                                     : pipeline.selectPointsInFrustum(region);
                }
                ASSERT_EQ(1, selected.actorInfos().size());
                EXPECT_EQ(source, selected.primaryActor());
                EXPECT_EQ((cells ? QSet<qint64>{0} : QSet<qint64>{0, 1, 2}),
                          LocalIds(selected, source, poly, 0));
                QSet<qint64> sourceIds;
                for (qint64 id : selected.ids()) sourceIds.insert(id);
                EXPECT_EQ((cells ? QSet<qint64>{4} : QSet<qint64>{2, 7, 12}),
                          sourceIds);
                if (attached) {
                    EXPECT_EQ(1, vis.getActorById(helperId)->GetVisibility());
                    EXPECT_EQ(1, vis.getActorById(helperId)->GetPickable());
                }
            }
        }
    }
    // Only the translated top vertex is in this small world-space region.
    int translatedVertex[] = {105, 155, 120, 170};
    auto throughPoint = pipeline.selectPointsInFrustum(translatedVertex);
    EXPECT_EQ((QSet<qint64>{2}), LocalIds(throughPoint, source, poly, 0));
    EXPECT_EQ((QVector<qint64>{12}), throughPoint.ids());
    auto throughCell = pipeline.selectCellsInFrustum(translatedVertex);
    EXPECT_EQ((QSet<qint64>{0}), LocalIds(throughCell, source, poly, 0));
    EXPECT_EQ((QVector<qint64>{4}), throughCell.ids());

    // Composite input is a data object, not a dataset. Cover both mapper
    // generations and keep hidden/unpickable blocks out of through-selection.
    vtkNew<vtkMultiBlockDataSet> blocks;
    blocks->SetBlock(0, poly);
    vtkNew<vtkCompositeDataDisplayAttributes> attributes;
#if VTK_VERSION_NUMBER >= VTK_VERSION_CHECK(9, 5, 0)
    vtkNew<vtkCompositePolyDataMapper> composite;
#else
    vtkNew<vtkCompositePolyDataMapper2> composite;
#endif
    composite->SetInputDataObject(blocks);
    composite->SetCompositeDataDisplayAttributes(attributes);
    source->SetMapper(composite);
    for (bool cells : {false, true}) {
        auto select = [&]() {
            return cells ? pipeline.selectCellsInFrustum(region)
                         : pipeline.selectPointsInFrustum(region);
        };
        attributes->SetBlockVisibility(poly, false);
        EXPECT_TRUE(select().isEmpty());
        attributes->SetBlockVisibility(poly, true);
        attributes->SetBlockPickability(poly, false);
        EXPECT_TRUE(select().isEmpty());
        attributes->SetBlockPickability(poly, true);
        const auto selected = select();
        ASSERT_EQ(1, selected.actorInfos().size());
        EXPECT_EQ((cells ? QSet<qint64>{0} : QSet<qint64>{0, 1, 2}),
                  LocalIds(selected, source, poly, 0));
        EXPECT_EQ((cells ? QVector<qint64>{4} : QVector<qint64>{2, 7, 12}),
                  selected.ids());
    }
}

TEST(SelectionPipelineRegression,
     HighlightActorLocalGeometryInWorldCoordinates) {
    vtkNew<vtkRenderer> renderer;
    vtkNew<vtkRenderWindow> window;
    window->AddRenderer(renderer);
    vtkNew<VTKExtensions::vtkCustomInteractorStyle> style;
    Visualization::VtkVis vis(renderer, window, style);
    cvSelectionHighlighter highlighter;
    highlighter.setVisualizer(&vis);
    vtkSmartPointer<vtkPolyData> output;
    QObject::connect(&highlighter,
                     &cvSelectionHighlighter::selectionOverlayUpdated,
                     [&](vtkPolyData* poly, int) {
                         output = vtkSmartPointer<vtkPolyData>::New();
                         output->DeepCopy(poly);
                     });
    auto first = MakeSelectionActor(2, 0.0);
    first->SetPosition(3, 4, 5);
    first->SetScale(2, 3, 1);
    auto second = MakeSelectionActor(2, 1.0);
    vtkNew<vtkTransform> transform;
    transform->Translate(-4, 2, 1);
    transform->RotateZ(90);
    second->SetUserTransform(transform);
    // Identity actor stands for a composite block already baked to world space.
    auto world = MakeSelectionActor(2, 7.0);
    for (auto actor : {first, second, world}) renderer->AddActor(actor);
    for (auto association : {cvSelectionData::POINTS, cvSelectionData::CELLS}) {
        cvSelectionData selected({99}, association);
        for (auto actor : {first, second, world}) {
            cvActorSelectionInfo info;
            info.actor = actor;
            info.polyData =
                    vtkPolyData::SafeDownCast(actor->GetMapper()->GetInput());
            info.selectedIds = association == cvSelectionData::POINTS
                                       ? QVector<qint64>{0, 1, 2}
                                       : QVector<qint64>{0};
            info.sourceIds = info.selectedIds;
            selected.addActorInfo(info);
        }
        output = nullptr;
        ASSERT_TRUE(highlighter.highlightSelection(
                selected, cvSelectionHighlighter::SELECTED));
        ASSERT_NE(nullptr, output.GetPointer());
        ASSERT_EQ(9, output->GetNumberOfPoints());
        // Compare sets rather than depending on vtkExtractSelection point
        // order.
        for (auto actor : {first, second, world}) {
            for (vtkIdType i = 0; i < 3; ++i) {
                double local[4] = {0, 0, 0, 1}, expected[4];
                actor->GetMapper()->GetInput()->GetPoint(i, local);
                actor->GetMatrix()->MultiplyPoint(local, expected);
                bool found = false;
                for (vtkIdType j = 0; j < output->GetNumberOfPoints(); ++j) {
                    double actual[3];
                    output->GetPoint(j, actual);
                    if (std::abs(actual[0] - expected[0]) < 1e-5 &&
                        std::abs(actual[1] - expected[1]) < 1e-5 &&
                        std::abs(actual[2] - expected[2]) < 1e-5)
                        found = true;
                }
                EXPECT_TRUE(found)
                        << "actor=" << actor.GetPointer() << " point=" << i;
            }
        }
    }
}

// Source IDs, not integer equality or actor pointers, define global selection.
TEST(SelectionPipelineRegression, PrimarySourceIdsMatchTheirActor) {
    vtkNew<vtkSelection> selection;
    vtkNew<vtkActor> actors[2];
    vtkNew<vtkPolyDataMapper> mappers[2];
    for (int i = 0; i < 2; ++i) {
        auto poly = MakeSingleVertexPolyData(i);
        vtkNew<vtkIdTypeArray> original;
        original->SetName("vtkOriginalPointIds");
        original->InsertNextValue(100 + i);
        poly->GetPointData()->AddArray(original);
        mappers[i]->SetInputData(poly);
        actors[i]->SetMapper(mappers[i]);
        vtkNew<vtkSelectionNode> node;
        node->SetContentType(vtkSelectionNode::INDICES);
        node->SetFieldType(vtkSelectionNode::POINT);
        node->GetProperties()->Set(vtkSelectionNode::PROP(), actors[i]);
        vtkNew<vtkIdTypeArray> ids;
        ids->InsertNextValue(0);
        node->SetSelectionList(ids);
        selection->AddNode(node);
    }
    auto result = cvSelectionPipeline::convertToCvSelectionData(
            selection, cvSelectionPipeline::FIELD_ASSOCIATION_POINTS);
    ASSERT_EQ(2, result.actorCount());
    const qint64 expected =
            result.primaryActor() == actors[0].GetPointer() ? 100 : 101;
    ASSERT_FALSE(result.ids().isEmpty());
    EXPECT_EQ(expected, result.ids().front());
    EXPECT_EQ(result.ids(), result.actorInfo().sourceIds);
}

cvSelectionData QualifiedSelection(vtkActor* actor,
                                   vtkPolyData* poly,
                                   const QString& source,
                                   qint64 sourceId,
                                   qint64 localId = 0) {
    auto selection = MakeSelection({sourceId}, cvSelectionData::POINTS, actor,
                                   poly, 0, {localId});
    auto info = selection.actorInfo();
    info.sourceId = source;
    info.sourceIds = {sourceId};
    selection.clearActorInfo();
    selection.addActorInfo(info);
    return selection;
}

TEST(SelectionPipelineRegression, EqualIdsOnDifferentSourcesRemainIndependent) {
    vtkNew<vtkActor> a, b;
    auto pa = MakeSingleVertexPolyData(0);
    auto pb = MakeSingleVertexPolyData(1);
    auto first = QualifiedSelection(a, pa, "source-a", 7);
    auto second = QualifiedSelection(b, pb, "source-b", 7);
    auto subtracted = cvSelectionPipeline::combineSelections(
            first, second, cvSelectionPipeline::OPERATION_SUBTRACTION);
    EXPECT_FALSE(subtracted.isEmpty());
    EXPECT_EQ(a.GetPointer(), subtracted.primaryActor());
    EXPECT_EQ((QSet<qint64>{0}), LocalIds(subtracted, a, pa, 0));
    auto toggled = cvSelectionPipeline::combineSelections(
            first, second, cvSelectionPipeline::OPERATION_TOGGLE);
    EXPECT_FALSE(toggled.isEmpty());
    EXPECT_EQ(2, toggled.actorCount());
    EXPECT_EQ((QSet<qint64>{0}), LocalIds(toggled, a, pa, 0));
    EXPECT_EQ((QSet<qint64>{0}), LocalIds(toggled, b, pb, 0));
    // A second view of the SAME source is not a second object.
    auto sameSource = QualifiedSelection(b, pb, "source-a", 7);
    EXPECT_TRUE(
            cvSelectionPipeline::combineSelections(
                    first, sameSource, cvSelectionPipeline::OPERATION_TOGGLE)
                    .isEmpty());
}

TEST(SelectionPipelineRegression, PublishesCompleteIdentityAwareSelection) {
    auto* manager = cvViewSelectionManager::instance();
    manager->clearSelection();
    vtkNew<vtkActor> a, b;
    auto pa = MakeSingleVertexPolyData(0);
    auto pb = MakeSingleVertexPolyData(1);
    QVector<cvSelectionData> published;
    auto connection = QObject::connect(
            manager,
            QOverload<const cvSelectionData&>::of(
                    &cvViewSelectionManager::selectionChanged),
            [&](const cvSelectionData& value) {
                published.append(value);
                EXPECT_EQ(value.actorCount(),
                          manager->currentSelection().actorCount());
            });
    manager->setCurrentSelection(QualifiedSelection(a, pa, "source-a", 17));
    manager->setCurrentSelection(QualifiedSelection(b, pb, "source-b", 17));
    QObject::disconnect(connection);
    ASSERT_EQ(2, published.size());
    ASSERT_EQ(1, published[0].actorCount());
    EXPECT_EQ(a.GetPointer(), published[0].primaryActor());
    EXPECT_EQ(QString("source-a"), published[0].actorInfo().sourceId);
    EXPECT_EQ((QVector<qint64>{17}), published[0].actorInfo().sourceIds);
    EXPECT_EQ(b.GetPointer(), published[1].primaryActor());
    EXPECT_EQ(QString("source-b"), published[1].actorInfo().sourceId);
    manager->clearCurrentSelection();
    EXPECT_TRUE(manager->currentSelection().isEmpty());
    EXPECT_EQ(nullptr, manager->getSourceObject());
}

TEST(SelectionPipelineRegression,
     NonPickableWhitelistedPointDoesNotOccludeSource) {
#if defined(__linux__)
    if (!std::getenv("DISPLAY")) GTEST_SKIP() << "Use xvfb-run";
#endif
    vtkNew<vtkRenderer> renderer;
    vtkNew<vtkRenderWindow> window;
    window->AddRenderer(renderer);
    window->SetSize(200, 200);
    window->SetMultiSamples(0);
    window->SetOffScreenRendering(1);
    auto source = MakeSelectionActor(2, 0);
    auto helper = MakeSelectionActor(0, 0.5);
    renderer->AddActor(source);
    renderer->AddActor(helper);
    helper->PickableOff();
    helper->VisibilityOff();
    auto* camera = renderer->GetActiveCamera();
    camera->SetPosition(0, 0, 5);
    camera->SetFocalPoint(0, 0, 0);
    camera->ParallelProjectionOn();
    camera->SetParallelScale(1);
    camera->SetClippingRange(0.1, 10);
    window->Render();
    vtkNew<cvHardwareSelector> selector;
    selector->SetRenderer(renderer);
    selector->SetFieldAssociation(vtkDataObject::FIELD_ASSOCIATION_POINTS);
    auto pick = [&]() {
        selector->SetSelectableProps(
                {source.GetPointer(), helper.GetPointer()});
        int region[] = {10, 10, 189, 189};
        vtkSmartPointer<vtkSelection> selection;
        selection.TakeReference(selector->Select(region));
        QSet<qint64> result;
        EXPECT_NE(nullptr, selection.GetPointer());
        if (!selection) return result;
        for (unsigned i = 0; i < selection->GetNumberOfNodes(); ++i) {
            auto* node = selection->GetNode(i);
            EXPECT_EQ(source.GetPointer(),
                      node->GetProperties()->Get(vtkSelectionNode::PROP()));
            auto* ids = vtkIdTypeArray::SafeDownCast(node->GetSelectionList());
            if (ids)
                for (vtkIdType j = 0; j < ids->GetNumberOfTuples(); ++j)
                    result.insert(ids->GetValue(j));
        }
        return result;
    };
    const auto baseline = pick();
    ASSERT_EQ((QSet<qint64>{0, 1, 2}), baseline);
    helper->VisibilityOn();
    EXPECT_EQ(baseline, pick());
    EXPECT_EQ(0, helper->GetPickable());
    EXPECT_EQ(1, helper->GetVisibility());
}

TEST(SelectionPipelineRegression, ResolvesSourceIdsInEachViewsRepresentation) {
    vtkNew<vtkRenderer> ra, rb;
    vtkNew<vtkRenderWindow> wa, wb;
    wa->AddRenderer(ra);
    wb->AddRenderer(rb);
    vtkNew<VTKExtensions::vtkCustomInteractorStyle> sa, sb;
    Visualization::VtkVis va(ra, wa, sa), vb(rb, wb, sb);
    auto ga = MakeSelectionActor(2, 0);
    auto gb = MakeSelectionActor(2, 2);
    auto* pa = vtkPolyData::SafeDownCast(ga->GetMapper()->GetInput());
    auto* pb = vtkPolyData::SafeDownCast(gb->GetMapper()->GetInput());
    auto originals = [](vtkPolyData* poly, const QVector<qint64>& values) {
        vtkNew<vtkIdTypeArray> ids;
        ids->SetName("vtkOriginalPointIds");
        for (auto value : values) ids->InsertNextValue(value);
        poly->GetPointData()->AddArray(ids);
    };
    originals(pa, {11, 22, 33});
    originals(pb, {33, 22, 11});
    ASSERT_TRUE(va.addPointCloud(pa, "shared-source"));
    ASSERT_TRUE(vb.addPointCloud(pb, "shared-source"));
    auto* a = va.getActorById("shared-source");
    auto* b = vb.getActorById("shared-source");
    a->SetPosition(-9, -8, -7);
    b->SetPosition(4, 5, 6);
    b->SetScale(2, 3, 4);
    auto selected = QualifiedSelection(a, pa, "shared-source", 11);
    const auto resolved =
            cvSelectionPipeline::resolveSelectionForView(selected, &vb);
    ASSERT_EQ(1, resolved.actorCount());
    EXPECT_EQ(b, resolved.primaryActor());
    EXPECT_EQ((QSet<qint64>{2}), LocalIds(resolved, b, pb, 0));
    auto overlay =
            cvSelectionHighlighter::renderSelectionForView(&vb, selected);
    ASSERT_NE(nullptr, overlay.GetPointer());
    ASSERT_EQ(1, overlay->GetNumberOfPoints());
    double local[4] = {0, 0, 0, 1}, expected[4], actual[3];
    pb->GetPoint(2, local);
    b->GetMatrix()->MultiplyPoint(local, expected);
    overlay->GetPoint(0, actual);
    for (int axis = 0; axis < 3; ++axis)
        EXPECT_NEAR(expected[axis], actual[axis], 1e-6);

    // Faces obey the same source-qualified/world-transform contract.
    for (auto* poly : {pa, pb}) {
        vtkNew<vtkIdTypeArray> cellIds;
        cellIds->SetName("vtkOriginalCellIds");
        cellIds->InsertNextValue(99);
        poly->GetCellData()->AddArray(cellIds);
    }
    auto faces = MakeSelection({99}, cvSelectionData::CELLS, a, pa, 0, {0});
    auto faceInfo = faces.actorInfo();
    faceInfo.sourceId = "shared-source";
    faceInfo.sourceIds = {99};
    faces.clearActorInfo();
    faces.addActorInfo(faceInfo);
    const auto resolvedFaces =
            cvSelectionPipeline::resolveSelectionForView(faces, &vb);
    EXPECT_EQ((QSet<qint64>{0}), LocalIds(resolvedFaces, b, pb, 0));
    auto faceOverlay =
            cvSelectionHighlighter::renderSelectionForView(&vb, faces);
    ASSERT_NE(nullptr, faceOverlay.GetPointer());
    EXPECT_EQ(1, faceOverlay->GetNumberOfCells());
    EXPECT_EQ(3, faceOverlay->GetNumberOfPoints());
    double expectedBounds[6], actualBounds[6];
    b->GetBounds(expectedBounds);
    faceOverlay->GetBounds(actualBounds);
    for (int bound = 0; bound < 6; ++bound)
        EXPECT_NEAR(expectedBounds[bound], actualBounds[bound], 1e-6);

    // HOVER uses a separate overlay and clearing it cannot erase SELECTED.
    auto* selectedActor = vb.getActorById("__highlight_selected_overlay__");
    ASSERT_NE(nullptr, selectedActor);
    auto hovered = QualifiedSelection(a, pa, "shared-source", 22, 1);
    ASSERT_EQ(1, b->GetVisibility());
    ASSERT_EQ(1, b->GetPickable());
    ASSERT_TRUE(rb->HasViewProp(b));
    auto* mapping = pb->GetPointData()->GetArray("vtkOriginalPointIds");
    ASSERT_NE(nullptr, mapping);
    ASSERT_EQ(22, mapping->GetComponent(1, 0));
    const auto resolvedHover =
            cvSelectionPipeline::resolveSelectionForView(hovered, &vb);
    ASSERT_EQ(1, resolvedHover.actorCount());
    EXPECT_EQ((QSet<qint64>{1}), LocalIds(resolvedHover, b, pb, 0));
    ASSERT_NE(nullptr, cvSelectionHighlighter::renderSelectionForView(
                               &vb, hovered, cvSelectionHighlighter::HOVER)
                               .GetPointer());
    EXPECT_EQ(selectedActor, vb.getActorById("__highlight_selected_overlay__"));
    EXPECT_NE(nullptr, vb.getActorById("__highlight_hover_overlay__"));
    auto foreign = QualifiedSelection(a, pa, "absent-source", 11);
    EXPECT_TRUE(cvSelectionPipeline::resolveSelectionForView(foreign, &vb)
                        .isEmpty());
    EXPECT_EQ(nullptr, cvSelectionHighlighter::renderSelectionForView(
                               &vb, foreign, cvSelectionHighlighter::HOVER)
                               .GetPointer());
    EXPECT_EQ(selectedActor, vb.getActorById("__highlight_selected_overlay__"));

    b->VisibilityOff();
    EXPECT_TRUE(cvSelectionPipeline::resolveSelectionForView(selected, &vb)
                        .isEmpty());
    b->VisibilityOn();
    ASSERT_TRUE(vb.removePointCloud("shared-source"));
    EXPECT_TRUE(cvSelectionPipeline::resolveSelectionForView(selected, &vb)
                        .isEmpty());
    // Source-space state survives actor rebuild and a different render order.
    auto rebuilt = MakeSelectionActor(2, 9);
    auto* rebuiltPoly =
            vtkPolyData::SafeDownCast(rebuilt->GetMapper()->GetInput());
    originals(rebuiltPoly, {22, 11, 33});
    ASSERT_TRUE(vb.addPointCloud(rebuiltPoly, "shared-source"));
    const auto rebound =
            cvSelectionPipeline::resolveSelectionForView(selected, &vb);
    ASSERT_EQ(1, rebound.actorCount());
    EXPECT_EQ((QSet<qint64>{1}),
              LocalIds(rebound, vb.getActorById("shared-source"), rebuiltPoly,
                       0));
    EXPECT_EQ((QVector<qint64>{11}), rebound.actorInfo().sourceIds);
}

TEST(SelectionPipelineRegression, ExpiredLegacyActorsFailClosed) {
    vtkNew<vtkRenderer> renderer;
    vtkNew<vtkRenderWindow> window;
    window->AddRenderer(renderer);
    vtkNew<VTKExtensions::vtkCustomInteractorStyle> style;
    Visualization::VtkVis vis(renderer, window, style);
    cvSelectionData legacy;
    {
        auto actor = MakeSelectionActor(2, 0);
        auto* poly = vtkPolyData::SafeDownCast(actor->GetMapper()->GetInput());
        legacy = MakeSelection({0}, cvSelectionData::POINTS, actor, poly, 0,
                               {0});
        renderer->AddActor(actor);
        EXPECT_FALSE(cvSelectionPipeline::resolveSelectionForView(legacy, &vis)
                             .isEmpty());
        renderer->RemoveActor(actor);
    }
    EXPECT_EQ(nullptr, legacy.primaryActor());
    EXPECT_EQ(nullptr, legacy.primaryPolyData());
    EXPECT_TRUE(cvSelectionPipeline::resolveSelectionForView(legacy, &vis)
                        .isEmpty());
    EXPECT_EQ(
            nullptr,
            cvSelectionHighlighter::buildSelectionOverlay(legacy).GetPointer());
}

TEST(SelectionPipelineRegression,
     HardwarePicksStayInTheirSourceAndRenderWindow) {
#if defined(__linux__)
    if (!std::getenv("DISPLAY")) GTEST_SKIP() << "Use xvfb-run";
#endif
    vtkNew<vtkRenderer> ra, rb;
    vtkNew<vtkRenderWindow> wa, wb;
    vtkNew<VTKExtensions::vtkCustomInteractorStyle> sa, sb;
    wa->AddRenderer(ra);
    wb->AddRenderer(rb);
    for (auto* window : {wa.GetPointer(), wb.GetPointer()}) {
        window->SetSize(200, 200);
        window->SetMultiSamples(0);
        window->SetOffScreenRendering(1);
    }
    Visualization::VtkVis va(ra, wa, sa), vb(rb, wb, sb);
    ccHObject root("two-window-selection-test");
    ccPointCloud cloudA("cloud-a"), cloudB("mesh-b-vertices");
    ASSERT_TRUE(cloudA.reserve(3));
    ASSERT_TRUE(cloudB.reserve(3));
    cloudA.addPoint(CCVector3(-0.5, -0.5, 0));
    cloudA.addPoint(CCVector3(0.5, -0.5, 0));
    cloudA.addPoint(CCVector3(0, 0.5, 0));
    cloudB.addPoint(CCVector3(0.5, -0.5, 0));
    cloudB.addPoint(CCVector3(0, 0.5, 0));
    cloudB.addPoint(CCVector3(-0.5, -0.5, 0));
    ccMesh meshB(&cloudB);
    ASSERT_TRUE(meshB.reserve(1));
    meshB.addTriangle(2, 0, 1);
    root.addChild(&cloudA, ccHObject::DP_NONE);
    root.addChild(&cloudB, ccHObject::DP_NONE);
    root.addChild(&meshB, ccHObject::DP_NONE);
    struct RestoreRoot {
        ccHObject& root;
        ccHObject* previous = ecvViewManager::instance().globalDBRoot();
        ~RestoreRoot() {
            ecvViewManager::instance().setGlobalDBRoot(previous);
            root.removeAllChildren();
        }
    } restore{root};
    ecvViewManager::instance().setGlobalDBRoot(&root);

    auto geometryA = MakeSelectionActor(2, 0);
    auto geometryB = MakeSelectionActor(2, 0);
    auto* pa = vtkPolyData::SafeDownCast(geometryA->GetMapper()->GetInput());
    auto* pb = vtkPolyData::SafeDownCast(geometryB->GetMapper()->GetInput());
    // Point clouds need one vertex CELL for every point, not triangle topology.
    vtkNew<vtkCellArray> vertices;
    for (vtkIdType i = 0; i < 3; ++i) vertices->InsertNextCell(1, &i);
    pa->SetPolys(nullptr);
    pa->SetVerts(vertices);
    vtkNew<vtkIdTypeArray> originalA, originalB;
    originalA->SetName("vtkOriginalPointIds");
    originalB->SetName("vtkOriginalPointIds");
    for (vtkIdType i : {0, 1, 2}) originalA->InsertNextValue(i);
    for (vtkIdType i : {2, 0, 1}) originalB->InsertNextValue(i);
    pa->GetPointData()->AddArray(originalA);
    pb->GetPointData()->AddArray(originalB);
    const auto sourceA = cloudA.getViewId().toStdString();
    const auto sourceB = meshB.getViewId().toStdString();
    ASSERT_TRUE(va.addPointCloud(pa, sourceA));
    ASSERT_TRUE(vb.addPointCloud(pb, sourceB));
    va.setCurrentSourceObject(&cloudA, sourceA);
    vb.setCurrentSourceObject(&meshB, sourceB);
    auto* actorA = va.getActorById(sourceA);
    auto* actorB = vb.getActorById(sourceB);
    ASSERT_NE(nullptr, actorA);
    ASSERT_NE(nullptr, actorB);
    actorB->SetPosition(0.1, 0.1, 0);
    for (auto* renderer : {ra.GetPointer(), rb.GetPointer()}) {
        auto* camera = renderer->GetActiveCamera();
        camera->SetPosition(0, 0, 5);
        camera->SetFocalPoint(0, 0, 0);
        camera->ParallelProjectionOn();
        camera->SetParallelScale(1);
        camera->SetClippingRange(0.1, 10);
    }
    // A transient label-like marker directly covers B's selected vertex.
    auto helper = MakeSelectionActor(0, 0.5);
    helper->SetPosition(0.1, 0.1, 0);
    rb->AddActor(helper);
    wa->Render();
    wb->Render();
    cvSelectionPipeline pipelineA, pipelineB;
    pipelineA.setVisualizer(&va);
    pipelineB.setVisualizer(&vb);
    int topA[] = {96, 146, 104, 154};
    int bottomLeftB[] = {56, 56, 64, 64};
    const auto pickedA = pipelineA.selectPointsOnSurface(topA);
    const auto pickedB = pipelineB.selectPointsOnSurface(bottomLeftB);
    ASSERT_EQ(1, pickedA.actorCount());
    ASSERT_EQ(1, pickedB.actorCount());
    EXPECT_EQ(QString::fromStdString(sourceA), pickedA.actorInfo().sourceId);
    EXPECT_EQ(QString::fromStdString(sourceB), pickedB.actorInfo().sourceId);
    EXPECT_EQ((QVector<qint64>{2}), pickedA.actorInfo().sourceIds);
    EXPECT_EQ((QVector<qint64>{2}), pickedB.actorInfo().sourceIds);
    EXPECT_EQ((QSet<qint64>{2}), LocalIds(pickedA, actorA, pa, 0));
    EXPECT_EQ((QSet<qint64>{0}), LocalIds(pickedB, actorB, pb, 0));
    EXPECT_EQ(1, helper->GetVisibility());
    EXPECT_EQ(1, helper->GetPickable());
    auto combined = cvSelectionPipeline::combineSelections(
            pickedA, pickedB, cvSelectionPipeline::OPERATION_ADDITION);
    ASSERT_EQ(2, combined.actorCount());
    EXPECT_EQ(2, combined.count());
    auto overlayA =
            cvSelectionHighlighter::renderSelectionForView(&va, combined);
    auto overlayB =
            cvSelectionHighlighter::renderSelectionForView(&vb, combined);
    ASSERT_NE(nullptr, overlayA.GetPointer());
    ASSERT_NE(nullptr, overlayB.GetPointer());
    ASSERT_EQ(1, overlayA->GetNumberOfPoints());
    ASSERT_EQ(1, overlayB->GetNumberOfPoints());
    double pointA[3], pointB[3];
    overlayA->GetPoint(0, pointA);
    overlayB->GetPoint(0, pointB);
    const double expectedA[] = {0, 0.5, 0};
    const double expectedB[] = {-0.4, -0.4, 0};
    for (int axis = 0; axis < 3; ++axis) {
        EXPECT_NEAR(expectedA[axis], pointA[axis], 1e-6);
        EXPECT_NEAR(expectedB[axis], pointB[axis], 1e-6);
    }
    const auto faces = pipelineB.selectCellsOnSurface(bottomLeftB);
    ASSERT_EQ(1, faces.actorCount());
    EXPECT_EQ(cvSelectionData::CELLS, faces.fieldAssociation());
    EXPECT_EQ(QString::fromStdString(sourceB), faces.actorInfo().sourceId);
    EXPECT_EQ((QSet<qint64>{0}), LocalIds(faces, actorB, pb, 0));
    cvSelectionHighlighter highlighter;
    highlighter.setVisualizer(&vb);
    vtkSmartPointer<vtkPolyData> faceOverlay;
    int overlayKind = cvSelectionHighlighter::SelectionOverlayNone;
    QObject::connect(&highlighter,
                     &cvSelectionHighlighter::selectionOverlayUpdated,
                     [&](vtkPolyData* poly, int kind) {
                         faceOverlay = poly;
                         overlayKind = kind;
                     });
    ASSERT_TRUE(highlighter.highlightSelection(faces));
    EXPECT_EQ(cvSelectionHighlighter::SelectionOverlaySurface, overlayKind);
    ASSERT_NE(nullptr, faceOverlay.GetPointer());
    EXPECT_EQ(1, faceOverlay->GetNumberOfPolys());
    EXPECT_EQ(0, faceOverlay->GetNumberOfVerts());
    EXPECT_EQ(1, helper->GetVisibility());
    EXPECT_EQ(1, helper->GetPickable());
}

TEST(SelectionPipelineRegression, ClosingViewUnbindsAllSelectionConsumers) {
    vtkNew<vtkRenderer> renderer;
    vtkNew<vtkRenderWindow> window;
    window->AddRenderer(renderer);
    vtkNew<VTKExtensions::vtkCustomInteractorStyle> style;
    Visualization::VtkVis vis(renderer, window, style);
    auto* manager = cvViewSelectionManager::instance();
    manager->clearSelection();
    manager->setVisualizer(&vis);
    EXPECT_EQ(&vis, manager->getHighlighter()->getVisualizer());
    manager->setVisualizer(nullptr);
    EXPECT_EQ(nullptr, manager->getVisualizer());
    EXPECT_EQ(nullptr, manager->getHighlighter()->getVisualizer());
    int region[] = {0, 0, 10, 10};
    EXPECT_EQ(nullptr,
              manager->getPipeline()
                      ->executeRectangleSelection(
                              region, cvSelectionPipeline::SURFACE_POINTS)
                      .GetPointer());
}

TEST(SelectionPipelineRegression, RemovedSourceIsNotDereferenced) {
    ccHObject root("source-lifetime-test");
    struct RestoreRoot {
        ccHObject* previous = ecvViewManager::instance().globalDBRoot();
        ~RestoreRoot() { ecvViewManager::instance().setGlobalDBRoot(previous); }
    } restore;
    ecvViewManager::instance().setGlobalDBRoot(&root);
    auto* source = new ccPointCloud("removed-source");
    root.addChild(source, ccHObject::DP_NONE);
    cvSelectionData selected(QVector<qint64>{0}, cvSelectionData::POINTS);
    cvActorSelectionInfo info;
    info.sourceId = source->getViewId();
    info.sourceIds = {0};
    selected.addActorInfo(info);
    auto* manager = cvViewSelectionManager::instance();
    manager->clearSelection();
    manager->setSourceObject(source);
    manager->setCurrentSelection(selected);
    EXPECT_EQ(source, manager->getSourcePointCloud());
    root.detachChild(source);
    delete source;
    EXPECT_FALSE(manager->isSourceObjectValid());
    EXPECT_EQ(nullptr, manager->getSourceObject());
    manager->clearCurrentSelection();
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

TEST_F(CompositeBlockRegistryTest, LeafBlockTracksReplacementAndRemoval) {
    auto first = MakeSingleVertexPolyData(0);
    auto replacement = MakeSingleVertexPolyData(1);
    registry_.setLeafBlock(leaves_[0], group_, first);
    EXPECT_EQ(first.GetPointer(), registry_.leafBlock(leaves_[0]));
    registry_.setLeafBlock(leaves_[0], group_, replacement);
    EXPECT_EQ(replacement.GetPointer(), registry_.leafBlock(leaves_[0]));
    registry_.removeLeaf(leaves_[0]);
    EXPECT_EQ(nullptr, registry_.leafBlock(leaves_[0]));
    EXPECT_EQ(nullptr, registry_.leafBlock("unregistered"));
}

// This is deliberately in the existing gate: real view/source resolution,
// not a second test-only cutter. Run under xvfb-run on Linux.
TEST(CompositeSliceRegression, NestedAndPartitionedRepresentationState) {
#if defined(__linux__)
    if (!std::getenv("DISPLAY")) GTEST_SKIP() << "Use xvfb-run";
#endif
    int argc = 1;
    char name[] = "composite-slice";
    char* argv[] = {name, nullptr};
    std::unique_ptr<QApplication> app;
    if (!QCoreApplication::instance()) {
        QSurfaceFormat::setDefaultFormat(
                QVTKOpenGLNativeWidget::defaultFormat());
        app = std::make_unique<QApplication>(argc, argv);
    }
    ASSERT_NE(nullptr,
              qobject_cast<QApplication*>(QCoreApplication::instance()));
    if (ecvViewManager::instance().displayTools())
        GTEST_SKIP() << "Requires isolated display tools";
    ccPointCloud entity("composite source");
    entity.setVisible(true);
    struct ViewFixture {
        QMainWindow window;
        ccHObject* previousRoot = ecvViewManager::instance().globalDBRoot();
        std::unique_ptr<vtkGLView> view;
        ViewFixture() {
            ecvViewManager::instance().initDisplayTools(
                    new Visualization::VtkDisplayTools(), &window, false);
            view.reset(vtkGLView::Create(&window, false, false));
        }
        ~ViewFixture() {
            view.reset();
            ecvViewManager::instance().releaseDisplayTools();
            ecvViewManager::instance().setGlobalDBRoot(previousRoot);
        }
    } fixture;
    auto* view = fixture.view.get();
    auto* vis = view->getVisualizer3D();
    auto* renderer = vis->getCurrentRenderer();
    const auto id = entity.getViewId().toStdString();
    vtkNew<vtkCubeSource> cube;
    cube->Update();
    vtkNew<vtkPolyData> first, second;
    first->ShallowCopy(cube->GetOutput());
    second->DeepCopy(first);
    for (vtkIdType i = 0; i < second->GetNumberOfPoints(); ++i) {
        double point[3];
        second->GetPoint(i, point);
        point[0] += 3;
        second->GetPoints()->SetPoint(i, point);
    }
    vtkNew<vtkDoubleArray> heat;
    heat->SetName("heat");
    heat->SetNumberOfTuples(first->GetNumberOfPoints());
    heat->FillComponent(0, 15);
    first->GetPointData()->SetScalars(heat);
    second->GetPointData()->SetScalars(heat);
    ASSERT_TRUE(vis->addPointCloud(first, id));
    vis->setCurrentSourceObject(&entity, id);
    auto* source = vis->getActorById(id);
    ASSERT_NE(nullptr, source);
    source->SetPosition(2, 0, 0);
    source->GetProperty()->SetOpacity(0.8);
    vtkNew<vtkPlane> otherPlane;
    otherPlane->SetNormal(0, 1, 0);
    vtkNew<vtkLookupTable> table;
    table->SetRange(10, 20);
    table->Build();

    for (bool partitioned : {false, true}) {
        SCOPED_TRACE(partitioned ? "partitioned" : "nested multiblock");
        vtkSmartPointer<vtkCompositeDataSet> root;
        vtkSmartPointer<vtkDataObject> branch;
        if (partitioned) {
            vtkNew<vtkPartitionedDataSetCollection> collection;
            vtkNew<vtkPartitionedDataSet> partitions;
            partitions->SetPartition(0, first);
            partitions->SetPartition(1, second);
            partitions->SetNumberOfPartitions(3);  // empty leaf
            collection->SetPartitionedDataSet(0, partitions);
            root = collection;
            branch = partitions;
        } else {
            vtkNew<vtkMultiBlockDataSet> blocks, nested;
            nested->SetBlock(0, first);
            nested->SetBlock(1, second);
            vtkNew<vtkTable> nonGeometry;
            nested->SetBlock(2, nonGeometry);
            blocks->SetBlock(0, nested);
            root = blocks;
            branch = nested;
        }
        vtkNew<vtkCompositeDataDisplayAttributes> attrs;
        attrs->SetBlockVisibility(branch, false);
        attrs->SetBlockVisibility(first, true);
        attrs->SetBlockVisibility(second, true);
        attrs->SetBlockOpacity(branch, 0.4);
        attrs->SetBlockOpacity(first, 0.25);
        const double red[3] = {1, 0, 0};
        attrs->SetBlockColor(first, red);
        vtkSmartPointer<vtkMapper> mapper;
#if VTK_VERSION_NUMBER >= VTK_VERSION_CHECK(9, 3, 0)
        if (partitioned) {
            vtkNew<vtkCompositePolyDataMapper> composite;
            composite->SetCompositeDataDisplayAttributes(attrs);
            attrs->SetBlockArrayName(branch, "heat");
            attrs->SetBlockScalarMode(branch,
                                      VTK_SCALAR_MODE_USE_POINT_FIELD_DATA);
            mapper = composite;
        } else
#endif
        {
#if VTK_VERSION_NUMBER >= VTK_VERSION_CHECK(9, 5, 0)
            vtkNew<vtkCompositePolyDataMapper> composite;
#else
            vtkNew<vtkCompositePolyDataMapper2> composite;
#endif
            composite->SetCompositeDataDisplayAttributes(attrs);
            mapper = composite;
        }
        mapper->SetInputDataObject(root);
        mapper->SetLookupTable(table);
        mapper->SetScalarRange(10, 20);
        mapper->AddClippingPlane(otherPlane);
        source->SetMapper(mapper);
        // No manager representation is needed for a canonical source object.
        auto* lod = vtkPVLODActor::SafeDownCast(source);
        vtkNew<vtkPolyDataMapper> lowResolution;
        if (lod) {
            lowResolution->SetInputData(first);  // deliberately omits leaf 2
            lod->SetLODMapper(lowResolution);
            lod->SetEnableLOD(1);
        }
        view->enableSliceMode(true);
        auto local = vtkSmartPointer<vtkMapper>(
                lod ? lod->GetFullResolutionMapper() : source->GetMapper());
        if (lod) {
            EXPECT_EQ(0, lowResolution->GetNumberOfClippingPlanes());
            EXPECT_EQ(1, lod->GetLODMapper()->GetNumberOfClippingPlanes());
        }
        EXPECT_NE(mapper.GetPointer(), local.GetPointer());
        EXPECT_EQ(1, mapper->GetNumberOfClippingPlanes());
        EXPECT_EQ(2, local->GetNumberOfClippingPlanes());
        std::set<vtkActor*> original;
        auto* actors = renderer->GetActors();
        actors->InitTraversal();
        while (auto* actor = actors->GetNextActor()) original.insert(actor);
        auto cuts = [&]() {
            std::vector<vtkActor*> result;
            actors = renderer->GetActors();
            actors->InitTraversal();
            while (auto* actor = actors->GetNextActor())
                if (!original.count(actor)) result.push_back(actor);
            return result;
        };
        view->setMultiSlicePositions(2, {0});  // world Z, not camera XY enum
        ASSERT_EQ(2, cuts().size());
        bool firstSeen = false, secondSeen = false;
        for (auto* cut : cuts()) {
            EXPECT_FALSE(cut->GetPickable());
            EXPECT_EQ(2, cut->GetMapper()->GetNumberOfClippingPlanes());
            EXPECT_NE(table.GetPointer(), cut->GetMapper()->GetLookupTable());
            if (cut->GetBounds()[0] < 3) {
                firstSeen = true;
                EXPECT_NEAR(1.5, cut->GetBounds()[0], 1e-6);
                EXPECT_DOUBLE_EQ(0.25, cut->GetProperty()->GetOpacity());
                EXPECT_DOUBLE_EQ(1, cut->GetProperty()->GetDiffuseColor()[0]);
                if (!partitioned)
                    EXPECT_FALSE(cut->GetMapper()->GetScalarVisibility());
            } else {
                secondSeen = true;
                EXPECT_NEAR(4.5, cut->GetBounds()[0], 1e-6);
                EXPECT_DOUBLE_EQ(0.4, cut->GetProperty()->GetOpacity());
#if VTK_VERSION_NUMBER >= VTK_VERSION_CHECK(9, 3, 0)
                if (partitioned) {
                    EXPECT_STREQ("heat", cut->GetMapper()->GetArrayName());
                    EXPECT_EQ(VTK_SCALAR_MODE_USE_POINT_FIELD_DATA,
                              cut->GetMapper()->GetScalarMode());
                }
#endif
            }
        }
        EXPECT_TRUE(firstSeen && secondSeen);
        attrs->SetBlockVisibility(second, false);
        source->SetPosition(3, 0, 0);
        view->updateScene();
        ASSERT_EQ(1, cuts().size());
        EXPECT_NEAR(2.5, cuts()[0]->GetBounds()[0], 1e-6);
        auto* representation =
                ecvRepresentationManager::instance().ensureRepresentation(
                        &entity, view);
        representation->setVisible(false);
        view->updateScene();
        EXPECT_TRUE(cuts().empty());
        representation->setVisible(true);
        view->updateScene();
        ASSERT_EQ(1, cuts().size());
        source->VisibilityOff();
        view->updateScene();
        EXPECT_TRUE(cuts().empty());
        EXPECT_EQ(1, local->GetNumberOfClippingPlanes());
        view->enableSliceMode(false);
        if (lod) {
            EXPECT_EQ(0, lod->GetLODMapper()->GetNumberOfClippingPlanes());
            lod->SetEnableLOD(0);
            lod->SetLODMapper(nullptr);
        }
        EXPECT_EQ(1, local->GetNumberOfClippingPlanes());
        EXPECT_EQ(otherPlane.GetPointer(),
                  local->GetClippingPlanes()->GetItem(0));
        EXPECT_DOUBLE_EQ(10, table->GetRange()[0]);
        EXPECT_DOUBLE_EQ(20, table->GetRange()[1]);
        view->clearMultiSlicePositions();
        source->VisibilityOn();
        source->SetPosition(2, 0, 0);
    }
}

}  // namespace
