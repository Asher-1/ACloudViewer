//##########################################################################
//#                                                                        #
//#                              CLOUDVIEWER                               #
//#                                                                        #
//#  This program is free software; you can redistribute it and/or modify  #
//#  it under the terms of the GNU General Public License as published by  #
//#  the Free Software Foundation; version 2 or later of the License.      #
//#                                                                        #
//#  This program is distributed in the hope that it will be useful,       #
//#  but WITHOUT ANY WARRANTY; without even the implied warranty of        #
//#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the          #
//#  GNU General Public License for more details.                          #
//#                                                                        #
//#                    COPYRIGHT: CLOUDVIEWER  project                     #
//#                                                                        #
//##########################################################################

#include "CurveFitting.h"
#include "PclUtils/vtk2cc.h"

// ECV_DB_LIB
#include <ecvPolyline.h>
#include <ecvPointCloud.h>

// VTK
#include <vtkPolyData.h>
#include <vtkSmartPointer.h>
#include <vtkSCurveSpline.h>
#include <vtkParametricSpline.h>
#include <vtkParametricFunctionSource.h>

using namespace std;
using namespace pcl;
using namespace Eigen;

namespace CurveFittingTool
{

	ccPolyline* BsplineFitting(const ccPointCloud& cloud)
	{
		vtkSmartPointer<vtkPoints> points =
			vtkSmartPointer<vtkPoints>::New();
		for (const CCVector3& p : cloud.getPoints())
		{
			points->InsertNextPoint(p.u);
		}

		vtkSmartPointer<vtkSCurveSpline> xSpline =
			vtkSmartPointer<vtkSCurveSpline>::New();
		vtkSmartPointer<vtkSCurveSpline> ySpline =
			vtkSmartPointer<vtkSCurveSpline>::New();
		vtkSmartPointer<vtkSCurveSpline> zSpline =
			vtkSmartPointer<vtkSCurveSpline>::New();

		vtkSmartPointer<vtkParametricSpline> spline =
			vtkSmartPointer<vtkParametricSpline>::New();
		spline->SetXSpline(xSpline);
		spline->SetYSpline(ySpline);
		spline->SetZSpline(zSpline);
		spline->SetPoints(points);
		spline->ClosedOff();

		vtkSmartPointer<vtkParametricFunctionSource> functionSource =
			vtkSmartPointer<vtkParametricFunctionSource>::New();
		functionSource->SetParametricFunction(spline);
		functionSource->Update();

		vtkPolyData* result = functionSource->GetOutput();
		if (!result) return nullptr;

		return vtk2ccConverter().getPolylineFromPolyData(result);
	}

	CurveFitting::CurveFitting()
	{
	}

	CurveFitting::~CurveFitting()
	{
		cloud->clear();
	}

	void CurveFitting::setInputcloud(PointCloudT::Ptr input_cloud) {
		cloud = input_cloud;
		getMinMax3D(*input_cloud, point_min, point_max);
	}

	void CurveFitting::grid_mean_xyz(
		double x_resolution,
		double y_resolution,
		vector<double>&x_mean,
		vector<double> &y_mean,
		vector<double>&z_mean,
		PointCloudT::Ptr &new_cloud)
	{
		if (y_resolution <= 0)
		{
			y_resolution = point_max.y - point_min.y;
		}
		int raster_rows, raster_cols;
		raster_rows = ceil((point_max.x - point_min.x) / x_resolution);
		raster_cols = ceil((point_max.y - point_min.y) / y_resolution);
		vector<int>idx_point;
		vector<vector<vector<float>>>row_col;
		vector<vector<float>>col_;
		vector<float>vector_4;
		vector_4.resize(4);
		col_.resize(raster_cols, vector_4);
		row_col.resize(raster_rows, col_);
		int point_num = cloud->size();
		for (int i_point = 0; i_point < point_num; i_point++)
		{
			int row_idx = ceil((cloud->points[i_point].x - point_min.x) / x_resolution) - 1;
			int col_idx = ceil((cloud->points[i_point].y - point_min.y) / y_resolution) - 1;
			if (row_idx < 0)row_idx = 0;
			if (col_idx < 0)col_idx = 0;
			row_col[row_idx][col_idx][0] += cloud->points[i_point].x;
			row_col[row_idx][col_idx][1] += cloud->points[i_point].y;
			row_col[row_idx][col_idx][2] += cloud->points[i_point].z;
			row_col[row_idx][col_idx][3] += 1;
		}
		PointT point_mean_tem;
		for (int i_row = 0; i_row < row_col.size(); i_row++)
		{
			for (int i_col = 0; i_col < row_col[i_row].size(); i_col++)
			{
				if (row_col[i_row][i_col][3] != 0)
				{
					double x_mean_tem = row_col[i_row][i_col][0] / row_col[i_row][i_col][3];
					double y_mean_tem = row_col[i_row][i_col][1] / row_col[i_row][i_col][3];
					double z_mean_tem = row_col[i_row][i_col][2] / row_col[i_row][i_col][3];
					x_mean.push_back(x_mean_tem);
					y_mean.push_back(y_mean_tem);
					z_mean.push_back(z_mean_tem);
					point_mean_tem.x = x_mean_tem;
					point_mean_tem.y = y_mean_tem;
					point_mean_tem.z = z_mean_tem;
					new_cloud->push_back(point_mean_tem);
				}
			}
		}
	}

	void CurveFitting::grid_mean_xyz_display(PointCloudT::Ptr new_cloud) {
		visualization::PCLVisualizer::Ptr view(new visualization::PCLVisualizer("�ֶ��������"));
		visualization::PointCloudColorHandlerCustom<PointT>color_1(new_cloud, 255, 0, 0);
		view->addPointCloud(new_cloud, color_1, "11");
		view->setPointCloudRenderingProperties(visualization::PCL_VISUALIZER_POINT_SIZE, 3, "11");
		PointCloudT::Ptr new_cloud_final(new PointCloudT);
		for (int i_point = 0; i_point < cloud->size(); i_point++)
		{
			PointT tem_point;
			tem_point.x = cloud->points[i_point].x;
			tem_point.y = cloud->points[i_point].y;
			tem_point.z = cloud->points[i_point].z;
			new_cloud_final->push_back(tem_point);
		}
		view->addPointCloud(new_cloud_final, "22");
		view->spin();
	}

	void CurveFitting::line_fitting(vector<double> x, vector<double> y, double &k, double &b) {
		MatrixXd A_(2, 2), B_(2, 1), A12(2, 1);
		int num_point = x.size();
		double A01(0.0), A02(0.0), B00(0.0), B10(0.0);
		for (int i_point = 0; i_point < num_point; i_point++)
		{
			A01 += x[i_point] * x[i_point];
			A02 += x[i_point];
			B00 += x[i_point] * y[i_point];
			B10 += y[i_point];
		}
		A_ << A01, A02,
			A02, num_point;
		B_ << B00,
			B10;
		A12 = A_.inverse()*B_;
		k = A12(0, 0);
		b = A12(1, 0);
	}

	void CurveFitting::polynomial2D_fitting(vector<double>x, vector<double>y, double &a, double &b, double &c) {
		MatrixXd A_(3, 3), B_(3, 1), A123(3, 1);
		int num_point = x.size();
		double A01(0.0), A02(0.0), A12(0.0), A22(0.0), B00(0.0), B10(0.0), B12(0.0);
		for (int i_point = 0; i_point < num_point; i_point++)
		{
			A01 += x[i_point];
			A02 += x[i_point] * x[i_point];
			A12 += x[i_point] * x[i_point] * x[i_point];
			A22 += x[i_point] * x[i_point] * x[i_point] * x[i_point];
			B00 += y[i_point];
			B10 += x[i_point] * y[i_point];
			B12 += x[i_point] * x[i_point] * y[i_point];
		}
		A_ << num_point, A01, A02,
			A01, A02, A12,
			A02, A12, A22;
		B_ << B00,
			B10,
			B12;
		A123 = A_.inverse()*B_;
		a = A123(2, 0);
		b = A123(1, 0);
		c = A123(0, 0);
	}

	void CurveFitting::polynomial3D_fitting(vector<double>x, vector<double>y, vector<double>z, double &a, double &b, double &c) {
		int num_point = x.size();
		MatrixXd A_(3, 3), B_(3, 1), A123(3, 1);
		double A01(0.0), A02(0.0), A12(0.0), A22(0.0), B00(0.0), B10(0.0), B12(0.0);
		for (int i_point = 0; i_point < num_point; i_point++)
		{
			double x_y = sqrt(pow(x[i_point], 2) + pow(y[i_point], 2));
			A01 += x_y;
			A02 += pow(x_y, 2);
			A12 += pow(x_y, 3);
			A22 += pow(x_y, 4);
			B00 += z[i_point];
			B10 += x_y * z[i_point];
			B12 += pow(x_y, 2) * z[i_point];
		}
		A_ << num_point, A01, A02,
			A01, A02, A12,
			A02, A12, A22;
		B_ << B00,
			B10,
			B12;
		A123 = A_.inverse()*B_;
		line_fitting(x, y, k_line, b_line);
		a = A123(2, 0);
		b = A123(1, 0);
		c = A123(0, 0);
		c_3d = c;
		b_3d = b;
		a_3d = a;
	}

	void CurveFitting::polynomial3D_fitting_display(double step_) {
		PointT point_min_, point_max_;
		getMinMax3D(*cloud, point_min_, point_max_);
		//������С������xֵ������ϵ�ֱ�������㣬����Ľ��㼴Ϊ��ά���ߵĶ˵�ֵ***********
		int idx_minx, idx_maxy;//xȡ�����ֵ����Сֵ�ĵ������
		for (int i_point = 0; i_point < cloud->size(); i_point++)
		{
			if (cloud->points[i_point].x == point_min_.x) idx_minx = i_point;
			if (cloud->points[i_point].x == point_max_.x) idx_maxy = i_point;
		}
		float m_min = cloud->points[idx_minx].x + k_line * cloud->points[idx_minx].y;
		float m_max = cloud->points[idx_maxy].x + k_line * cloud->points[idx_maxy].y;

		float x_min = (m_min - b_line * k_line) / (1 + k_line * k_line);
		float x_max = (m_max - b_line * k_line) / (1 + k_line * k_line);
		//---------------------------------------------------------------------------------------
		vector<double>xx, yy, zz;
		int step_num = ceil((x_max - x_min) / step_);
		vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
		for (int i_ = 0; i_ < step_num + 1; i_++)
		{
			double tem_value = x_min + i_ * step_;
			if (tem_value > x_max)
			{
				tem_value = x_max;
			}
			xx.push_back(tem_value);
			yy.push_back(k_line*xx[i_] + b_line);
			double xxyy = sqrt(pow(xx[i_], 2) + pow(yy[i_], 2));
			zz.push_back(c_3d + b_3d * xxyy + a_3d * pow(xxyy, 2));
			points->InsertNextPoint(xx[i_], yy[i_], zz[i_]);
		}
		vtkSmartPointer<vtkPolyLine> polyLine = vtkSmartPointer<vtkPolyLine>::New();
		vtkSmartPointer<vtkPolyData> polyData = vtkSmartPointer<vtkPolyData>::New();
		vtkSmartPointer<vtkCellArray> cells = vtkSmartPointer<vtkCellArray>::New();
		polyData->SetPoints(points);
		polyLine->GetPointIds()->SetNumberOfIds(points->GetNumberOfPoints());
		for (unsigned int i = 0; i < points->GetNumberOfPoints(); i++)
			polyLine->GetPointIds()->SetId(i, i);
		cells->InsertNextCell(polyLine);
		polyData->SetLines(cells);
		visualization::PCLVisualizer::Ptr viewer(new visualization::PCLVisualizer("�����ϵĶ���ʽ����"));
		viewer->addModelFromPolyData(polyData, "1");
		//*******************************************
		PointCloudT::Ptr tem_point(new PointCloudT);
		for (int i = 0; i < xx.size(); i++)
		{
			PointT point_;
			point_.x = xx[i];
			point_.y = yy[i];
			point_.z = zz[i];
			tem_point->push_back(point_);
		}
		visualization::PointCloudColorHandlerCustom<PointT>color1(tem_point, 255, 0, 0);
		viewer->addPointCloud(tem_point, color1, "point1");
		viewer->setPointCloudRenderingProperties(visualization::PCL_VISUALIZER_POINT_SIZE, 3, "point1");

		PointCloudT::Ptr tem_point1(new PointCloudT);
		for (int i = 0; i < cloud->size(); i++)
		{
			PointT point_1;
			point_1.x = cloud->points[i].x;
			point_1.y = cloud->points[i].y;
			point_1.z = cloud->points[i].z;
			tem_point1->push_back(point_1);
		}
		viewer->addPointCloud(tem_point1, "orginal");
		viewer->setPointCloudRenderingProperties(visualization::PCL_VISUALIZER_POINT_SIZE, 2, "orginal");
		//��ʾ�˵�
		PointCloudT::Ptr duandian_point(new PointCloudT);
		duandian_point->push_back(tem_point->points[0]);
		duandian_point->push_back(tem_point->points[tem_point->size() - 1]);
		visualization::PointCloudColorHandlerCustom<PointT>color2(duandian_point, 0, 255, 255);
		viewer->addPointCloud(duandian_point, color2, "duandian");
		viewer->setPointCloudRenderingProperties(visualization::PCL_VISUALIZER_POINT_SIZE, 5, "duandian");
		std::cout << "�˵�ֵ1Ϊ��" << "X1= " << duandian_point->points[0].x << ", " << "Y1= " << duandian_point->points[0].y << ", " << "Z1= " << duandian_point->points[0].z << std::endl;
		std::cout << "�˵�ֵ2Ϊ��" << "X2= " << duandian_point->points[1].x << ", " << "Y2= " << duandian_point->points[1].y << ", " << "Z2= " << duandian_point->points[1].z << std::endl;
		std::cout << "�ռ����ʽ���߷���Ϊ�� " << "z=" << a_3d << "*(x^2+y^2)+" << b_3d << "*sqrt(x^2+y^2)+" << c_3d << std::endl;
		viewer->spin();
		//�������+�˵�ֵ+ɢ��ͼ��άƽ��չʾ������Ҫ����ȡ��ע��----------------------------------------------------------
		/*vector<double>vector_1, vector_2, vector_3, vector_4;
		vector_1.push_back(duandian_point->points[0].x);
		vector_1.push_back(duandian_point->points[1].x);
		vector_2.push_back(duandian_point->points[0].y);
		vector_2.push_back(duandian_point->points[1].y);
		for (int i = 0; i < cloud->size();i++)
		{
			vector_3.push_back(cloud->points[i].x);
			vector_4.push_back(cloud->points[i].y);
		}
		std::vector<double> func1(2, 0);
		func1[0] = b_line;
		func1[1] = k_line;
		visualization::PCLPlotter *plot_line1(new visualization::PCLPlotter);
		plot_line1->addPlotData(func1, vector_1[0], vector_1[1]);
		plot_line1->addPlotData(vector_3, vector_4, "display", vtkChart::POINTS);//X,Y��Ϊdouble�͵�����
		plot_line1->addPlotData(vector_1, vector_2, "display", vtkChart::POINTS);//X,Y��Ϊdouble�͵�����
		plot_line1->setShowLegend(false);
		plot_line1->plot();*/
	}

	void CurveFitting::getPolynomial3D(PointCloudT::Ptr outCurve, double step_)
	{
		PointT point_min_, point_max_;
		getMinMax3D(*cloud, point_min_, point_max_);
		//������С������xֵ������ϵ�ֱ�������㣬����Ľ��㼴Ϊ��ά���ߵĶ˵�ֵ***********
		int idx_minx, idx_maxy;//xȡ�����ֵ����Сֵ�ĵ������
		for (int i_point = 0; i_point < cloud->size(); i_point++)
		{
			if (cloud->points[i_point].x == point_min_.x) idx_minx = i_point;
			if (cloud->points[i_point].x == point_max_.x) idx_maxy = i_point;
		}
		float m_min = cloud->points[idx_minx].x + k_line * cloud->points[idx_minx].y;
		float m_max = cloud->points[idx_maxy].x + k_line * cloud->points[idx_maxy].y;

		float x_min = (m_min - b_line * k_line) / (1 + k_line * k_line);
		float x_max = (m_max - b_line * k_line) / (1 + k_line * k_line);

		int step_num = ceil((x_max - x_min) / step_);
		for (int i_ = 0; i_ < step_num + 1; i_++)
		{
			double tem_value = x_min + i_ * step_;
			if (tem_value > x_max)
			{
				tem_value = x_max;
			}
			PointT point_;
			point_.x = tem_value;
			point_.y = k_line * tem_value + b_line;
			double xxyy = sqrt(pow(point_.x, 2) + pow(point_.y, 2));
			point_.z = c_3d + b_3d * xxyy + a_3d * pow(xxyy, 2);
			outCurve->push_back(point_);
		}
	}

	void CurveFitting::display_point(vector<double>vector_1, vector<double>vector_2) {
		visualization::PCLPlotter *plot_line1(new visualization::PCLPlotter);
		plot_line1->addPlotData(vector_1, vector_2, "display", vtkChart::POINTS);//X,Y��Ϊdouble�͵�����
		plot_line1->setShowLegend(false);
		plot_line1->plot();
	}

	void CurveFitting::display_line(vector<double>vector_1, vector<double>vector_2, double c, double b, double a) {
		visualization::PCLPlotter *plot_line1(new visualization::PCLPlotter);
		std::vector<double> func1(3, 0);
		func1[0] = c;
		func1[1] = b;
		func1[2] = a;
		plot_line1->addPlotData(func1, point_min.x, point_max.x);
		plot_line1->addPlotData(vector_1, vector_2, "display", vtkChart::POINTS);//X,Y��Ϊdouble�͵�����
		plot_line1->setShowLegend(false);
		plot_line1->plot();
	}

}