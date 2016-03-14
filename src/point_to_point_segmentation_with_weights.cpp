#include "../inc/point_to_point_segmentation_with_weights.h"
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/ml.hpp"
#include "../inc/fio.h"

using namespace cv;
using namespace std;

PointToPointSegmentationWithWeights::PointToPointSegmentationWithWeights(int model_size, int model_dim, int num_segments, int batch_size, int max_num_neighbors, double beta, double sigma)
	:PointToPointSegmentation(model_size, model_dim, num_segments, batch_size, max_num_neighbors)
{
	
}

void PointToPointSegmentationWithWeights::Segment()
{
	original_batch_dists_ = Mat::zeros(model_size_, num_segments_, CV_64F);

	for(int batch_idx = 0; batch_idx < batch_size_; batch_idx++)
	{
		for(int segment_idx = 0; segment_idx < num_segments_; segment_idx++)
		{
			segmented_target_cloud_[batch_idx][segment_idx] = Mat::zeros(model_size_, model_dim_, CV_64F);
			segmented_home_cloud_[batch_idx][segment_idx] = Mat::zeros(model_size_, model_dim_, CV_64F);
			segmented_predicted_cloud_[batch_idx][segment_idx] = Mat::zeros(model_size_, model_dim_, CV_64F);
		}
	}

	for(int point_idx = 0; point_idx < model_size_; point_idx++)
	{
		for(int segment_idx = 0; segment_idx < num_segments_; segment_idx++)
		{
			for(int batch_idx = 0; batch_idx < batch_size_; batch_idx++)
			{
				int curr_idx = kd_tree_indices_[batch_idx][segment_idx].at<int>(point_idx, 0);
				target_cloud_[batch_idx].rowRange(curr_idx, curr_idx + 1).copyTo(segmented_target_cloud_[batch_idx][segment_idx].rowRange(point_idx, point_idx + 1));
				predicted_cloud_[batch_idx][segment_idx].rowRange(point_idx, point_idx + 1).copyTo(segmented_predicted_cloud_[batch_idx][segment_idx].rowRange(point_idx, point_idx + 1));
				home_cloud_.rowRange(point_idx, point_idx + 1).copyTo(segmented_home_cloud_[batch_idx][segment_idx].rowRange(point_idx, point_idx + 1));	
				original_batch_dists_.at<double>(point_idx, segment_idx) = original_batch_dists_.at<double>(point_idx, segment_idx) + norm(home_cloud_.rowRange(point_idx, point_idx + 1) - target_cloud_[batch_idx].rowRange(curr_idx, curr_idx + 1), NORM_L2);
			}
			original_batch_dists_.at<double>(point_idx, segment_idx) = original_batch_dists_.at<double>(point_idx, segment_idx) / batch_size_;
		}
	}

	// SetPointWeight(original_batch_dists_, batch_dists_, segmented_point_weight_, num_segments_, batch_size_);
	// SetPointWeightWithGaussianMixture(original_batch_dists_, batch_dists_, segmented_point_weight_, num_segments_, batch_size_);
	SetPointWeightWithHeuristic(original_batch_dists_, batch_dists_, segmented_point_weight_, num_segments_, batch_size_);
}

void PointToPointSegmentationWithWeights::ThresholdMat(Mat& data)
{
	for(int i = 0; i < data.rows; i++)
	{
		for(int j = 0; j < data.cols; j++)
		{
			data.at<double>(i, j) = data.at<double>(i, j) > 0 ? data.at<double>(i, j) : 0;
		}
	}
}

void PointToPointSegmentationWithWeights::SetPointWeight(const Mat& original_averaged_dists, const Mat& averaged_dists, vector<vector<Mat>> segmented_point_weight, int num_segments, int batch_size)
{
	int num_rows = original_averaged_dists.rows;
	int num_cols = original_averaged_dists.rows;
	Mat normalized_matching_dists = Mat::zeros(num_rows, num_cols, CV_64F);
	Mat ratio = Mat::zeros(num_rows, num_cols, CV_64F);
	for(int point_idx = 0; point_idx < num_rows; point_idx++)
	{
		normalize(averaged_dists.rowRange(point_idx, point_idx + 1), normalized_matching_dists.rowRange(point_idx, point_idx + 1), 1, 0, NORM_L1);	
	}
	normalized_matching_dists = normalized_matching_dists - 0.5;

	ThresholdMat(normalized_matching_dists);
	normalized_matching_dists = normalized_matching_dists * 2;

	Mat diff = original_averaged_dists - averaged_dists;
	divide(diff, original_averaged_dists, ratio);
	ThresholdMat(ratio);

	Mat point_weight_mat = 10.0 * normalized_matching_dists.mul(ratio);

	// normalize the weights... to help learning...

	for(int segment_idx = 0; segment_idx < num_segments; segment_idx++)
	{
		Mat sum, sum_replicated;
		reduce(point_weight_mat.colRange(segment_idx, segment_idx + 1), sum, 0, REDUCE_SUM);
		sum.at<double>(0, 0) = sum.at<double>(0, 0) == 0 ? 1.0 : sum.at<double>(0, 0);
		repeat(sum, point_weight_mat.rows, 1, sum_replicated);
		divide(point_weight_mat.colRange(segment_idx, segment_idx + 1), sum_replicated, point_weight_mat.colRange(segment_idx, segment_idx + 1), point_weight_mat.rows);
	}

	for(int batch_idx = 0; batch_idx < batch_size; batch_idx++)
	{
		for(int segment_idx = 0; segment_idx < num_segments; segment_idx++)
		{
			point_weight_mat.colRange(segment_idx, segment_idx + 1).copyTo(segmented_point_weight[batch_idx][segment_idx]);
		}
	}
}

void PointToPointSegmentationWithWeights::SetPointWeightWithHeuristic(const Mat& original_averaged_dists, const Mat& averaged_dists, vector<vector<Mat>>& segmented_point_weight, int num_segments, int batch_size)
{
	int num_rows = original_averaged_dists.rows;
	int num_cols = original_averaged_dists.cols;
	double saturation_factor = 40.0;
	double middle_point = 0.5;
	Mat normalized_matching_dists = Mat::zeros(num_rows, num_cols, CV_64F);
	Mat weight = Mat::zeros(num_rows, num_cols, CV_64F);
	Mat tmp, weight_0, weight_1;
	// heuristic equation...
	// prob_0 = (exp((0.5 - dist) * saturation_factor) - exp(-saturation_factor * 0.5)) / (exp(saturation_factor * 0.5) - exp(-saturation_factor * 0.5));
	// prob_1 = (exp((dist - 0.5) * saturation_factor) - exp(-saturation_factor * 0.5)) / (exp(saturation_factor * 0.5) - exp(-saturation_factor * 0.5));
	// normalize matching dists
	for(int point_idx = 0; point_idx < num_rows; point_idx++)
	{
		normalize(averaged_dists.rowRange(point_idx, point_idx + 1), normalized_matching_dists.rowRange(point_idx, point_idx + 1), 1, 0, NORM_L1);	
	}
	tmp = (middle_point - normalized_matching_dists.colRange(0, 1)) * saturation_factor;
	exp(tmp, tmp);
	// weight_0 = (tmp - exp(-saturation_factor * 0.5)) / (exp(saturation_factor * 0.5) - exp(-saturation_factor * 0.5)) ;
	weight_0 = tmp / (1 + tmp);
	weight_1 = 1 - weight_0;
	/*tmp = (normalized_matching_dists.colRange(0, 1) - middle_point) * saturation_factor;
	exp(tmp, tmp);
	// weight_1 = (tmp - exp(-saturation_factor * 0.5)) / (exp(saturation_factor * 0.5) - exp(-saturation_factor * 0.5));
	weight_1 = tmp / exp(saturation_factor * 0.5);*/
	weight_0.copyTo(weight.colRange(0, 1));
	weight_1.copyTo(weight.colRange(1, 2));

	/*char output_dir[400];
	sprintf(output_dir, "D:/Document/HKUST/Year 5/Research/Solutions/unified_framework/output/para_%d/weight_mat.bin", 61);
	FileIO::WriteMatDouble(weight, weight.rows, weight.cols, output_dir);*/

	for(int batch_idx = 0; batch_idx < batch_size; batch_idx++)
	{
		for(int segment_idx = 0; segment_idx < num_segments; segment_idx++)
		{
			weight.colRange(segment_idx, segment_idx + 1).copyTo(segmented_point_weight[batch_idx][segment_idx]);
		}
	}
}

void PointToPointSegmentationWithWeights::SetPointWeightWithGaussianMixture(const Mat& original_averaged_dists, const Mat& averaged_dists, vector<vector<Mat>>& segmented_point_weight, int num_segments, int batch_size)
{
	int num_rows = original_averaged_dists.rows;
	int num_cols = original_averaged_dists.cols;
	Mat normalized_matching_dists = Mat::zeros(num_rows, num_cols, CV_64F);
	Mat ratio = Mat::zeros(num_rows, num_cols, CV_64F);
	Mat cluster_label;
	// normalize matching dists
	for(int point_idx = 0; point_idx < num_rows; point_idx++)
	{
		normalize(averaged_dists.rowRange(point_idx, point_idx + 1), normalized_matching_dists.rowRange(point_idx, point_idx + 1), 1, 0, NORM_L1);	
	}
	// do pca to project matching dists...
	PCA pca(normalized_matching_dists, Mat(), PCA::DATA_AS_ROW);
	Mat projection = pca.project(normalized_matching_dists);
	Mat em_data = Mat::zeros(projection.rows, 1, CV_64F);
	projection.colRange(0, 1).copyTo(em_data);
	Mat pca_eigen_vector = pca.eigenvectors;
	// do em to the rotated matching dists...
	Mat em_probability = Mat::zeros(em_data.rows, num_segments + 1, CV_64F);
	TermCriteria term_criteria = TermCriteria(TermCriteria::COUNT, 10, 1e-8);
	EM expectation_maximization(num_segments + 1, EM::COV_MAT_DIAGONAL, term_criteria);
	expectation_maximization.train(em_data, noArray(), noArray(), em_probability);
	Mat em_means = expectation_maximization.get<Mat>("means");
	// not sure whether this way would work for background...
	// get the weights by posterior...
	SetClusterLabelMatrixByEigenVector(pca_eigen_vector.rowRange(0, 1), em_means, cluster_label, num_segments);
	Mat weight_mat = em_probability * cluster_label;
	for(int batch_idx = 0; batch_idx < batch_size; batch_idx++)
	{
		for(int segment_idx = 0; segment_idx < num_segments; segment_idx++)
		{
			weight_mat.colRange(segment_idx, segment_idx + 1).copyTo(segmented_point_weight[batch_idx][segment_idx]);
		}
	}
	
}

// eigen vector is dim by 1 vector
void PointToPointSegmentationWithWeights::SetClusterLabelMatrixByEigenVector(const Mat& eigen_vector, const Mat& em_mean, Mat& cluster_label, int num_segment)
{
	if(num_segment == 2)
	{
		Mat reference_vector = Mat::zeros(num_segment, 1, CV_64F);
		reference_vector.at<double>(0, 0) = -1; reference_vector.at<double>(1, 0) = 1;
		Mat result = eigen_vector * reference_vector;
		Mat em_mean_idx;
		sortIdx(em_mean, em_mean_idx, CV_SORT_EVERY_COLUMN + CV_SORT_DESCENDING);
		cluster_label = Mat::zeros(3, 2, CV_64F);
		if(result.at<double>(0, 0) > 0)
		{
			cluster_label.at<double>(em_mean_idx.at<int>(0, 0), 0) = 1;
			cluster_label.at<double>(em_mean_idx.at<int>(2, 0), 1) = 1;
		}
		else
		{
			cluster_label.at<double>(em_mean_idx.at<int>(0, 0), 1) = 1;
			cluster_label.at<double>(em_mean_idx.at<int>(2, 0), 0) = 1;
		}
	}
	else
	{
		cout << "not supported dimension number" << endl;
		exit(0);
	}
}

/*
char output_dir[400];
sprintf(output_dir, "D:/Document/HKUST/Year 5/Research/Solutions/unified_framework/output/para_%d/em_probability.bin", 61);
FileIO::WriteMatDouble(em_probability, em_probability.rows, em_probability.cols, output_dir);

sprintf(output_dir, "D:/Document/HKUST/Year 5/Research/Solutions/unified_framework/output/para_%d/cluster_label.bin", 61);
FileIO::WriteMatDouble(cluster_label, cluster_label.rows, cluster_label.cols, output_dir);

sprintf(output_dir, "D:/Document/HKUST/Year 5/Research/Solutions/unified_framework/output/para_%d/weight_mat.bin", 61);
FileIO::WriteMatDouble(weight_mat, weight_mat.rows, weight_mat.cols, output_dir);
*/