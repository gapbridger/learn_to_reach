#include "../inc/point_to_point_segmentation.h"
#include <iostream>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/types_c.h"
#include "opencv2/nonfree/features2d.hpp"

using namespace cv;
using namespace std;

PointToPointSegmentation::PointToPointSegmentation(int model_size, int model_dim, int num_segments, int batch_size, int max_num_neighbors)
{
	model_size_ = model_size;
	model_dim_ = model_dim;
	num_segments_ = num_segments;
	batch_size_ = batch_size;
	// beta_ = beta;
	// sigma_ = sigma;
	max_num_neighbors_ = max_num_neighbors;
	home_cloud_label_ = Mat::zeros(model_size_, num_segments_, CV_64F);
	batch_dists_ = Mat::zeros(model_size_, num_segments_, CV_64F);
	original_batch_dists_ = Mat::zeros(model_size_, num_segments_, CV_64F);
	dist_likelihood_ = Mat::zeros(model_size_, num_segments_, CV_64F);
	segmented_avg_curr_transform_ = Mat::zeros(num_segments_, model_dim_ - 1, CV_64F);
	segmented_original_batch_dists_ = vector<Mat>(num_segments_);
	segmented_batch_dists_ = vector<Mat>(num_segments_);
	

	segmented_target_cloud_ = vector<vector<Mat>>(batch_size_);
	segmented_home_cloud_ = vector<vector<Mat>>(batch_size_);
	segmented_predicted_cloud_ = vector<vector<Mat>>(batch_size_);
	segmented_point_weight_ = vector<vector<Mat>>(batch_size_);
	batch_transform_ = vector<vector<Mat>>(batch_size);
	segmented_batch_transform_ = vector<vector<Mat>>(batch_size);
	for(int i = 0; i < batch_size_; i++)
	{
		segmented_target_cloud_[i] = vector<Mat>(num_segments_);
		segmented_predicted_cloud_[i] = vector<Mat>(num_segments_);
		segmented_home_cloud_[i] = vector<Mat>(num_segments_);
		segmented_point_weight_[i] = vector<Mat>(num_segments_);
		batch_transform_[i] = vector<Mat>(num_segments_);
		segmented_batch_transform_[i] = vector<Mat>(num_segments_);
	}

	predicted_error_stat_ = vector<RunningStat>(num_segments_);
	original_error_stat_ = vector<RunningStat>(num_segments_);
	transform_stat_ = vector<vector<RunningStat>>(num_segments_);
	for(int i = 0; i < num_segments_; i++)
	{
		predicted_error_stat_[i] = RunningStat(0);
		original_error_stat_[i] = RunningStat(0);
		transform_stat_[i] = vector<RunningStat>(model_dim_ - 1);
		for(int j = 0; j < model_dim_ - 1; j++)
		{
			transform_stat_[i][j] = RunningStat(num_segments_ - 1);
		}
	}
	explained_variance_ = Mat::zeros(num_segments_, 1, CV_64F);
	correlation_ = Mat::zeros(num_segments_, model_dim_ - 1, CV_64F);

	RNG rng(getTickCount());
	for(int i = 0; i < model_size_; i++)
	{
		int idx = rng.uniform(0, num_segments_);
		home_cloud_label_.at<double>(i, idx) = 1;
	}
}

// do set_home_cloud_label every time before calling the Match function
void PointToPointSegmentation::Match(int train_iteration, int icm_iterations)
{
	double min_value = 0;
	double max_value = 0;
	// Mat home_cloud_label_updated = Mat::zeros(model_size_, num_segments_, CV_64F);
	// dist_likelihood_ = Mat::zeros(model_size_, num_segments_, CV_64F);
	batch_dists_ = Mat::zeros(model_size_, num_segments_, CV_64F);
	original_batch_dists_ = Mat::zeros(model_size_, num_segments_, CV_64F);
	Mat averaged_neighborhood_dists = Mat::zeros(model_size_, num_segments_, CV_64F);
	
	Mat potential = Mat::zeros(1, num_segments_, CV_64F);
	Mat home_cloud_label_updated;
	Mat tmp_mat_1, tmp_mat_2, tmp_zeros;
	Point min_location, max_location;
	RNG rng(getTickCount());
	
	for(int segment_idx = 0; segment_idx < num_segments_; segment_idx++)
	{
		for(int batch_idx = 0; batch_idx < batch_size_; batch_idx++)
		{
			batch_transform_[batch_idx][segment_idx] = Mat::zeros(model_size_, model_dim_ - 1, CV_64F);
		}
	}

	// average over the batch
	for(int point_idx = 0; point_idx < model_size_; point_idx++)
	{
		for(int segment_idx = 0; segment_idx < num_segments_; segment_idx++)
		{
			for(int batch_idx = 0; batch_idx < batch_size_; batch_idx++)
			{
				// averaged dists after prediction, equivalent to residual
				batch_dists_.at<double>(point_idx, segment_idx) += sqrt(kd_tree_min_dists_[batch_idx][segment_idx].at<float>(point_idx, 0));
				// original averaged dists before prediction
				int curr_idx = kd_tree_indices_[batch_idx][segment_idx].at<int>(point_idx, 0);
				double curr_original_dist = norm(target_cloud_[batch_idx].rowRange(curr_idx, curr_idx + 1) - home_cloud_.rowRange(point_idx, point_idx + 1), NORM_L2);
				original_batch_dists_.at<double>(point_idx, segment_idx) += curr_original_dist;
				// get the averaged_transform
				Mat curr_diff = (predicted_cloud_[batch_idx][segment_idx].rowRange(point_idx, point_idx + 1) - home_cloud_.rowRange(point_idx, point_idx + 1));
				curr_diff = curr_diff.colRange(0, model_dim_ - 1);
				curr_diff.copyTo(batch_transform_[batch_idx][segment_idx].rowRange(point_idx, point_idx + 1));
			}
			batch_dists_.at<double>(point_idx, segment_idx) = batch_dists_.at<double>(point_idx, segment_idx) / batch_size_;
			original_batch_dists_.at<double>(point_idx, segment_idx) = original_batch_dists_.at<double>(point_idx, segment_idx) / batch_size_;
		}
	}
	// update home cloud label according to the averaged matching dist within neighborhood
	// home_cloud_label_updated = Mat::zeros(model_size_, num_segments_, CV_64F);
	Mat tmp_dists = Mat::zeros(batch_dists_.rows, batch_dists_.cols, CV_64F);
	batch_dists_.copyTo(tmp_dists);
	for(int icm_idx = 0; icm_idx < icm_iterations; icm_idx++)
	{
		for(int point_idx = 0; point_idx < model_size_; point_idx++)
		{
			int neighborhood_point_count = 0;
			for(int neighbor_idx = 0; neighbor_idx < max_num_neighbors_; neighbor_idx++)
			{
				int neighbor_point_idx = home_cloud_neighbor_indices_.at<int>(point_idx, neighbor_idx);
				if(neighbor_point_idx != -1)
				{
					for(int segment_idx = 0; segment_idx < num_segments_; segment_idx++)
					{
						averaged_neighborhood_dists.at<double>(point_idx, segment_idx) = averaged_neighborhood_dists.at<double>(point_idx, segment_idx) + tmp_dists.at<double>(neighbor_point_idx, segment_idx);
					}
					neighborhood_point_count++;
				}
				else
				{
					for(int segment_idx = 0; segment_idx < num_segments_; segment_idx++)
					{
						averaged_neighborhood_dists.at<double>(point_idx, segment_idx) = averaged_neighborhood_dists.at<double>(point_idx, segment_idx) / neighborhood_point_count;
					}
					break;
				}
			}	
		}
		averaged_neighborhood_dists.copyTo(tmp_dists);
		if(icm_idx != icm_iterations - 1)
		{
			averaged_neighborhood_dists = Mat::zeros(model_size_, num_segments_, CV_64F);
		}
	}

	home_cloud_label_updated = Mat::zeros(model_size_, num_segments_, CV_64F);
	for(int point_idx = 0; point_idx < model_size_; point_idx++)
	{
		averaged_neighborhood_dists.rowRange(point_idx, point_idx + 1).copyTo(potential);
		tmp_zeros = Mat::zeros(1, num_segments_, CV_64F);
		tmp_zeros.copyTo(home_cloud_label_updated.rowRange(point_idx, point_idx + 1));
		minMaxLoc(potential, &min_value, &max_value, &min_location, &max_location);
		home_cloud_label_updated.at<double>(point_idx, min_location.x) = 1;
	}
    home_cloud_label_updated.copyTo(home_cloud_label_);
}

void PointToPointSegmentation::Segment()
{
	double min_value = 0;
	double max_value = 0;
	Point min_location, max_location;
	Mat count = Mat::zeros(num_segments_, batch_size_, CV_32S);
	for(int segment_idx = 0; segment_idx < num_segments_; segment_idx++)
	{
		segmented_batch_dists_[segment_idx] = Mat::zeros(model_size_, 1, CV_64F);
		segmented_original_batch_dists_[segment_idx] = Mat::zeros(model_size_, 1, CV_64F);
		for(int batch_idx = 0; batch_idx < batch_size_; batch_idx++)
		{
			segmented_target_cloud_[batch_idx][segment_idx] = Mat::zeros(model_size_, model_dim_, CV_64F);
			segmented_home_cloud_[batch_idx][segment_idx] = Mat::zeros(model_size_, model_dim_, CV_64F);
			segmented_predicted_cloud_[batch_idx][segment_idx] = Mat::zeros(model_size_, model_dim_, CV_64F);
			segmented_batch_transform_[batch_idx][segment_idx] = Mat::zeros(model_size_, model_dim_ - 1, CV_64F);
		}
	}
	for(int point_idx = 0; point_idx < model_size_; point_idx++)
	{
		minMaxLoc(home_cloud_label_.rowRange(point_idx, point_idx + 1), &min_value, &max_value, &min_location, &max_location);
		int label_idx = max_location.x;
		for(int batch_idx = 0; batch_idx < batch_size_; batch_idx++)
		{
			int curr_count = count.at<int>(label_idx, batch_idx);
			int curr_idx = kd_tree_indices_[batch_idx][label_idx].at<int>(point_idx, 0);
			target_cloud_[batch_idx].rowRange(curr_idx, curr_idx + 1).copyTo(segmented_target_cloud_[batch_idx][label_idx].rowRange(curr_count, curr_count + 1));
			predicted_cloud_[batch_idx][label_idx].rowRange(point_idx, point_idx + 1).copyTo(segmented_predicted_cloud_[batch_idx][label_idx].rowRange(curr_count, curr_count + 1));
			home_cloud_.rowRange(point_idx, point_idx + 1).copyTo(segmented_home_cloud_[batch_idx][label_idx].rowRange(curr_count, curr_count + 1));	
			batch_transform_[batch_idx][label_idx].rowRange(point_idx, point_idx + 1).copyTo(segmented_batch_transform_[batch_idx][label_idx].rowRange(curr_count, curr_count + 1));
			if(batch_idx == 0)
			{
				segmented_batch_dists_[label_idx].at<double>(curr_count, 0) = batch_dists_.at<double>(point_idx, label_idx);
				segmented_original_batch_dists_[label_idx].at<double>(curr_count, 0) = original_batch_dists_.at<double>(point_idx, label_idx);
			}
			count.at<int>(label_idx, batch_idx) = count.at<int>(label_idx, batch_idx) + 1;
		}
	}
	for(int segment_idx = 0; segment_idx < num_segments_; segment_idx++)
	{
		segmented_batch_dists_[segment_idx] = segmented_batch_dists_[segment_idx].rowRange(0, count.at<int>(segment_idx, 0));
		segmented_original_batch_dists_[segment_idx] = segmented_original_batch_dists_[segment_idx].rowRange(0, count.at<int>(segment_idx, 0));
		for(int batch_idx = 0; batch_idx < batch_size_; batch_idx++)
		{
			segmented_target_cloud_[batch_idx][segment_idx] = segmented_target_cloud_[batch_idx][segment_idx].rowRange(0, count.at<int>(segment_idx, batch_idx));
			segmented_home_cloud_[batch_idx][segment_idx] = segmented_home_cloud_[batch_idx][segment_idx].rowRange(0, count.at<int>(segment_idx, batch_idx));
			segmented_predicted_cloud_[batch_idx][segment_idx] = segmented_predicted_cloud_[batch_idx][segment_idx].rowRange(0, count.at<int>(segment_idx, batch_idx));
			segmented_batch_transform_[batch_idx][segment_idx] = segmented_batch_transform_[batch_idx][segment_idx].rowRange(0, count.at<int>(segment_idx, batch_idx));
			segmented_point_weight_[batch_idx][segment_idx] = Mat::ones(count.at<int>(segment_idx, batch_idx), 1, CV_64F);
		}
	}
}


void PointToPointSegmentation::UpdateKinematicStructure(vector<Mat>& curr_prop)
{
	Mat curr_original_dist = Mat::zeros(1, 1, CV_64F);
	Mat curr_predicted_dist = Mat::zeros(1, 1, CV_64F);
	for(int segment_idx = 0; segment_idx < num_segments_; segment_idx++)
	{
		// explained variance
		reduce(segmented_batch_dists_[segment_idx], curr_predicted_dist, 0, CV_REDUCE_AVG);
		reduce(segmented_original_batch_dists_[segment_idx], curr_original_dist, 0, CV_REDUCE_AVG);
		predicted_error_stat_[segment_idx].Push(curr_predicted_dist.at<double>(0, 0));
		original_error_stat_[segment_idx].Push(curr_original_dist.at<double>(0, 0));
		double predicted_variance = predicted_error_stat_[segment_idx].VarianceX();
		double original_variance = original_error_stat_[segment_idx].VarianceX();
		double curr_explained_variance =  original_variance == 0 ? 0 : 1 - predicted_variance / original_variance;
		curr_explained_variance = curr_explained_variance < 0 ? 0 : curr_explained_variance;
		explained_variance_.at<double>(segment_idx, 0) = curr_explained_variance;
	}
	// accumulating correlation stats...
	for(int batch_idx = 0; batch_idx < batch_size_; batch_idx++)
	{
		for(int segment_idx = 0; segment_idx < num_segments_; segment_idx++)
		{
			reduce(segmented_batch_transform_[batch_idx][segment_idx], segmented_avg_curr_transform_.rowRange(segment_idx, segment_idx + 1), 0, CV_REDUCE_AVG);
			for(int dim_idx = 0; dim_idx < model_dim_ - 1; dim_idx++)
			{
				transform_stat_[segment_idx][dim_idx].Push(segmented_avg_curr_transform_.at<double>(segment_idx, dim_idx), curr_prop[batch_idx]);
			}
		}
	}
}

Mat PointToPointSegmentation::correlation()
{
	// sort the explained variance...
	// Mat sorted_explained_variance_idx;
	// sortIdx(explained_variance_, sorted_explained_variance_idx, CV_SORT_EVERY_COLUMN + CV_SORT_DESCENDING);
	// top three correlation...
	for(int segment_idx = 0; segment_idx < num_segments_; segment_idx++)
	{
		Mat segment_corr = Mat::zeros(model_dim_ - 1, num_segments_ - 1, CV_64F);
		Mat segment_corr_avg = Mat::zeros(1, num_segments_ - 1, CV_64F);
		// int curr_segment_idx = sorted_explained_variance_idx.at<int>(segment_idx, 0);
		for(int dim_idx = 0; dim_idx < model_dim_ - 1; dim_idx++)
		{
			Mat corr = transform_stat_[segment_idx][dim_idx].Correlation();
			corr.copyTo(segment_corr.rowRange(dim_idx, dim_idx + 1));
		}
		// take the absolute value of correlations
		segment_corr = abs(segment_corr);
		reduce(segment_corr, segment_corr_avg, 0, CV_REDUCE_AVG);
		segment_corr_avg.copyTo(correlation_.rowRange(segment_idx, segment_idx + 1));
	}
	return correlation_;
}
