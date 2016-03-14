#ifndef _POINT_TO_POINT_SEGMENTATION_H
#define _POINT_TO_POINT_SEGMENTATION_H
#include "../inc/segmentation.h"
#include "../inc/running_stat.h"
#include <vector>

class PointToPointSegmentation : public Segmentation
{

protected:
	int model_size_;
	int model_dim_;
	int num_segments_;
	int batch_size_;
	int max_num_neighbors_;
	double beta_; // continuity constraint
	double sigma_; // continuity constraint
	std::vector<std::vector<cv::Mat>> segmented_home_cloud_;
	std::vector<std::vector<cv::Mat>> segmented_predicted_cloud_;
	std::vector<std::vector<cv::Mat>> segmented_target_cloud_;
	std::vector<std::vector<cv::Mat>> segmented_point_weight_;
	std::vector<cv::Mat> target_cloud_;
	std::vector<std::vector<cv::Mat>> predicted_cloud_;
	cv::Mat home_cloud_;
	cv::Mat home_cloud_label_;
	cv::Mat home_cloud_neighbor_indices_;
	cv::Mat batch_dists_;
	cv::Mat original_batch_dists_;
	std::vector<std::vector<cv::Mat>> batch_transform_;
	std::vector<std::vector<cv::Mat>> segmented_batch_transform_;
	std::vector<cv::Mat> segmented_batch_dists_;
	std::vector<cv::Mat> segmented_original_batch_dists_;
	cv::Mat segmented_avg_curr_transform_;
	cv::Mat dist_likelihood_;
	std::vector<std::vector<cv::Mat>> kd_tree_indices_;
	std::vector<std::vector<cv::Mat>> kd_tree_min_dists_;

	std::vector<std::vector<int>> multi_scale_neighbor_indices_;
	cv::Mat scale_indices_;

	std::vector<RunningStat> predicted_error_stat_;
	std::vector<RunningStat> original_error_stat_;
	std::vector<std::vector<RunningStat>> transform_stat_;

	cv::Mat explained_variance_;
	cv::Mat correlation_;

public:
	// on line mode when batch size = 1
	PointToPointSegmentation(int model_size, int model_dim, int num_segments, int batch_size, int max_num_neighbors);
	void Match(int train_iteration, int icm_iterations) override;
	void Segment() override;
	void UpdateKinematicStructure(std::vector<cv::Mat>& curr_prop) override;
	cv::Mat correlation();
	std::vector<std::vector<cv::Mat>> segmented_home_cloud()
	{
		return segmented_home_cloud_;
	}

	std::vector<std::vector<cv::Mat>> segmented_predicted_cloud()
	{
		return segmented_predicted_cloud_;
	}

	std::vector<std::vector<cv::Mat>> segmented_target_cloud()
	{
		return segmented_target_cloud_;
	}

	cv::Mat home_cloud_label() const
	{
		return home_cloud_label_;
	}

	cv::Mat averaged_dists() const
	{
		return batch_dists_;
	}

	cv::Mat original_averaged_dists() const
	{
		return original_batch_dists_;
	}

	std::vector<std::vector<cv::Mat>> segmented_point_weight() const
	{
		return segmented_point_weight_;
	}
	
	cv::Mat explained_variance() const
	{
		return explained_variance_;
	}

	void set_target_cloud(const std::vector<cv::Mat>& target_cloud)
	{
		target_cloud_ = target_cloud;
	}

	void set_predicted_cloud(const std::vector<std::vector<cv::Mat>>& predicted_cloud)
	{
		predicted_cloud_ = predicted_cloud;
	}

	void set_home_cloud(const cv::Mat& home_cloud)
	{
		home_cloud_ = home_cloud;
	}

	void set_home_cloud_label(const cv::Mat& home_cloud_label)
	{
		home_cloud_label.copyTo(home_cloud_label_);
	}

	void set_kd_tree_indices(const std::vector<std::vector<cv::Mat>>& kd_tree_indices)
	{
		kd_tree_indices_ = kd_tree_indices;
	}

	void set_kd_tree_min_dists(const std::vector<std::vector<cv::Mat>>& kd_tree_min_dists)
	{
		kd_tree_min_dists_ = kd_tree_min_dists;
	}

	void set_home_cloud_neighbor_indices(const cv::Mat& home_cloud_neighbor_indices)
	{
		home_cloud_neighbor_indices_ = home_cloud_neighbor_indices;
	}

	void set_multi_scale_neighbor_indices(const std::vector<std::vector<int>>& multi_scale_neighbor_indices)
	{
		multi_scale_neighbor_indices_ = multi_scale_neighbor_indices;
	}

	void set_scale_indices(const cv::Mat& scale_indices)
	{
		scale_indices_ = scale_indices;
	}

};
#endif