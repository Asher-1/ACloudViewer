#pragma once

//##########################################################################
//#                                                                        #
//#            CLOUDVIEWER PLUGIN: ColorimetricSegmenter                   #
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
//#    COPYRIGHT:	Tri-Thien TRUONG, Ronan COLLIER, Mathieu LETRONE       #
//#                                                                        #
//##########################################################################

#include "ecvStdPluginInterface.h"

//cloudViewer
#include <ReferenceCloud.h>

//Qt
#include <QObject>
#include <QtGui>

class ccPointCloud;

class ColorimetricSegmenter : public QObject, public ccStdPluginInterface
{
	Q_OBJECT
	Q_INTERFACES(ccPluginInterface ccStdPluginInterface)
	Q_PLUGIN_METADATA(IID "cccorp.cloudcompare.plugin.ColorimetricSegmenter" FILE "info.json")

public:
	explicit ColorimetricSegmenter(QObject* parent = nullptr);
	~ColorimetricSegmenter() override = default;

	// inherited from ccStdPluginInterface
	void onNewSelection(const ccHObject::Container& selectedEntities) override;
	QList<QAction*> getActions() override;

private:
	std::vector<ccPointCloud*> getSelectedPointClouds();

	//! Filter a cloud with RGB color
	void filterRgb();

	void filterHSV();

	void filterScalar();

	void HistogramClustering();

	void KmeansClustering();

	bool addPoint(cloudViewer::ReferenceCloud& filteredCloud, unsigned int j);

	template <typename T>
	void createClouds(	T& dlg,
						ccPointCloud* cloud,
						const cloudViewer::ReferenceCloud& filteredCloudInside,
						const cloudViewer::ReferenceCloud& filteredCloudOutside,
						QString name);

	void createCloud(	ccPointCloud* cloud,
						const cloudViewer::ReferenceCloud& referenceCloud,
						QString name);

	//! Segment a cloud with RGB color
	void filterRgbWithSegmentation();

	//! Region (shared)
	typedef QSharedPointer<cloudViewer::ReferenceCloud> Region;
	//! Region set
	typedef std::vector<Region> RegionSet;

	/**
	 * @brief Segmentation method grouping the points into regions of similar colors.
	 * Method described in Qingming Zhan, Yubin Liang, Yinghui Xiao, 2009 "Color-based segmentation of point clouds".
	 * @param regions output regions
	 * @param pointCloud The point cloud to segment.
	 * @param TNN Point-point colorimetrical similarity threshold.
	 * @param TPP Number of neighbours to search using KNN.
	 * @param TD Threshold distance between neighbouring points.
	 * @return success.
	 */
	static bool RegionGrowing(	RegionSet& regions,
								ccPointCloud* pointCloud,
								const unsigned TNN,
								const double TPP,
								const double TD);

	/**
	 * @brief Merge previously created regions in 'regionGrowing' method.
	 * @param mergedRegions refined and merged regions
	 * @param basePointCloud The base segmented point cloud used to create the regions.
	 * @param regions Vector containing the regions.
	 * @param TNN Point-point colorimetrical similarity threshold.
	 * @param TRR Region-region colorimetrical similarity threshold.
	 * @param TD Threshold distance between neighbouring regions. Used to merge close regions.
	 * @param Min Minimal size for a region.
	 * @return Vector containing the resulting merged and refined regions.
	 */
	static bool RegionMergingAndRefinement(	RegionSet& mergedRegions,
											ccPointCloud* basePointCloud,
											const RegionSet& regions,
											const unsigned TNN,
											const double TRR,
											const double TD,
											const unsigned Min);

private: //members

	QAction* m_action_filterRgb;
	//QAction* m_action_filterRgbWithSegmentation;
	QAction* m_action_filterHSV;
	QAction* m_action_filterScalar;
	QAction* m_action_histogramClustering;
	QAction* m_action_kMeansClustering;

	//! Error state after the last call to addPoint
	bool m_addPointError;
};
