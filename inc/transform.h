#ifndef _TRANSFORM_H
#define _TRANSFORM_H


#include "opencv2/core/core.hpp"

class Transform
{
public:
	virtual void CalcTransformation(std::vector<cv::Mat>& feature) = 0;
	virtual void TransformCloud(
		const cv::Mat& input_cloud, 
		std::vector<std::vector<cv::Mat>>& output_cloud) = 0;
	virtual void CalculateGradient(const std::vector<std::vector<cv::Mat>>& matched_target_cloud, const std::vector<std::vector<cv::Mat>>& matched_predicted_cloud, const std::vector<std::vector<cv::Mat>>& matched_query_cloud, const std::vector<std::vector<cv::Mat>>& point_weight, const std::vector<cv::Mat>& feature) = 0;
	virtual void Update() = 0;
	virtual cv::Mat w(int idx) = 0;
 	virtual void set_w(const cv::Mat& w, int idx) = 0;
};

#endif

