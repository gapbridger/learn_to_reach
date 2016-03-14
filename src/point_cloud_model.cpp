#include "../inc/point_cloud_model.h"

using namespace cv;

PointCloudModel::PointCloudModel(Mat& point_cloud)
{
	point_cloud_ = Mat::zeros(point_cloud.rows, point_cloud.cols, CV_64F);
	point_cloud.copyTo(point_cloud_);
}

int PointCloudModel::GetPointCloudSize()
{
	return point_cloud_.rows;
}

int PointCloudModel::GetPointCloudDim()
{
	return point_cloud_.cols;
}