#include "opencv2/core/core.hpp"

class PointCloudModel
{
private:
	cv::Mat point_cloud_;
public:
	PointCloudModel(cv::Mat& point_cloud);
	int GetPointCloudSize();
	int GetPointCloudDim();
};