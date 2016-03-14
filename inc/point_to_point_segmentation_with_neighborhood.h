#ifndef _POINT_TO_POINT_SEGMENTATION_WITH_WEIGHTS_H
#define _POINT_TO_POINT_SEGMENTATION_WITH_WEIGHTS_H

#include "point_to_point_segmentation.h"

class PointToPointSegmentationWithNeighborhood : public PointToPointSegmentation
{
public:
	PointToPointSegmentationWithNeighborhood(int model_size, int model_dim, int num_segments, int batch_size, int max_num_neighbors, double beta, double sigma);
	static void SetPointWeight(const cv::Mat& original_averaged_dists, const cv::Mat& averaged_dists, std::vector<std::vector<cv::Mat>> segmented_point_weight, int num_segments, int batch_size);
	// static void SetPointWeightWithGaussianMixture(const cv::Mat& original_averaged_dists, const cv::Mat& averaged_dists, std::vector<std::vector<cv::Mat>>& segmented_point_weight, int num_segments, int batch_size);
	static void SetPointWeightWithHeuristic(const cv::Mat& original_averaged_dists, const cv::Mat& averaged_dists, std::vector<std::vector<cv::Mat>>& segmented_point_weight, int num_segments, int batch_size);
	static void ThresholdMat(cv::Mat& normalized_matching_dists);
	// static void SetClusterLabelMatrixByEigenVector(const cv::Mat& eigen_vector, const cv::Mat& em_mean, cv::Mat& cluster_label, int num_segment);
	void Match(int train_iteration, int icm_iterations) override;
	void Segment() override;
};

#endif