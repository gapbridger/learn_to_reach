#ifndef _EXPLORER_H
#define _EXPLORER_H

#define PI 3.14159265

#include <iostream>
#include <random>
#include <time.h>
#include <queue>
#include <deque>
#include <array>
#include <vector>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/types_c.h"
#include "opencv2/nonfree/features2d.hpp"

#include "../inc/fio.h"
#include "../inc/loader.h"
#include "../inc/general_transform.h"
#include "../inc/hmm.h"

#include <pcl/features/normal_3d.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/visualization/pcl_visualizer.h>

#include "gtest/gtest.h"
#include "rigid_transform_3d.h"


// typedef std::vector<double> fL;

class Explorer{
private:
       
	int id_; 	
	int dim_feature_;
	int num_joints_;
	int num_segments_;	
	int num_trend_;
	int train_data_size_;
	int test_data_size_;
	long train_iteration_;	
	long expand_iteration_;
	int path_count_;	
	int dim_transform_;
	int num_weights_;
	int icm_iteration_;
	int max_num_neighbors_;
	char data_set_[100];
	
    double max_exploration_range_;
    double ini_exploration_range_;	
	double avg_cost_;	
	double normal_learning_rate_;
	double neighborhood_range_;
	double transform_alpha_;
	double icm_sigma_;
	
	std::random_device rd_;

	std::vector<double> targets_;
	std::vector<double> prev_targets_;
	std::vector<std::vector<double>> path_;
	std::vector<double> kernel_list_;
	cv::Mat weight_label_probabilities_;
	// std::vector<cv::Mat> home_cloud_link_label_probabilities_;

	cv::Mat joint_idx_;
	cv::Mat joint_range_limit_;
	cv::Mat action_;
    cv::Mat train_prop_;
	cv::Mat train_target_idx_;    
	cv::Mat test_prop_;
	cv::Mat test_target_idx_;   
    cv::Mat home_prop_;
	cv::Mat curr_prop_;
	// cv::Mat curr_prop_matrix_;
	cv::Mat prop_diff_;
	cv::Mat prop_dist_;
	cv::Mat aim_idx_matrix_;
	std::vector<cv::Mat> feature_;
	cv::Mat feature_home_;

	cv::Mat explore_path_target_;
	cv::Mat prev_explore_path_target_;
	cv::Mat explore_path_kdtree_indices_;
	cv::Mat explore_path_kdtree_dists_;

	cv::Mat cloud_;
	cv::Mat prev_cloud_;
	cv::Mat home_cloud_;
	cv::Mat home_cloud_weight_label_; // number of clouds are treated as number of columns
	cv::Mat home_cloud_indices_;
	cv::Mat home_cloud_min_dists_;
	std::vector<cv::Mat> predicted_cloud_;
	cv::Mat tmp_cloud_;

	// GeneralTransform transform_;
	RigidTransform3D transform_;


	std::vector<cv::Mat> original_error_mean_old_;
	std::vector<cv::Mat> original_error_mean_new_;
	std::vector<cv::Mat> original_error_cov_old_;
	std::vector<cv::Mat> original_error_cov_new_;

	std::vector<cv::Mat> predicted_error_mean_old_;
	std::vector<cv::Mat> predicted_error_mean_new_;
	std::vector<cv::Mat> predicted_error_cov_old_;
	std::vector<cv::Mat> predicted_error_cov_new_;

	cv::Mat original_error_explained_variance_count_;
	cv::Mat predicted_error_explained_variance_count_;

	int dim_state_;
	int batch_size_;

	// int cloud_scale_;
    
public:
    // initialization
    Explorer(int dir_id, char* data_set, int train_iteration, int expand_iteration, int dim_transform, int num_joints, int num_segments, int batch_size, double normal_learning_rate, double ini_exploration_range, 
		int train_data_size, int test_data_size, const cv::Mat& joint_idx, const cv::Mat& joint_range_limit, double neighborhood_range, int icm_iteration, double icm_beta, double icm_sigma, int max_num_neighbors);
    ~Explorer();	
	void RecordData(Loader& loader, std::vector<std::vector<double>>& trend_array, cv::Mat& home_cloud_label, cv::Mat& explained_variance, cv::Mat& correlation, int aim_idx, int iteration_count, int record_trend_interval, int record_diagnosis_interval);
	// void Segment(std::vector<cv::Mat>& matched_target_cloud, const cv::Mat& home_cloud_label, const cv::Mat& target_cloud, const cv::Mat& index, int num_joints);
	void Segment(std::vector<cv::Mat>& segmented_target_cloud, std::vector<cv::Mat>& segmented_home_cloud, std::vector<cv::Mat>& segmented_prediction_cloud, const cv::Mat& home_cloud_label, const cv::Mat& target_cloud, const cv::Mat& home_cloud, 
					   const std::vector<cv::Mat>& prediction_cloud, const std::vector<cv::Mat>& indices, int num_joints, int iteration_count);
	void ShowLearningProgress(int iteration_count);
	void GenerateLinePath(std::vector<std::vector<double>>& path, std::vector<double>& targets, std::vector<double>& prev_targets);
	int GenerateAimIndex(std::mt19937& engine, cv::flann::Index& kd_trees, std::vector<int>& path, int iteration_count, const cv::Mat& scale);
	void LoadHomeCloud(Loader& loader);
	int GenerateAimIndexLinePath(std::mt19937& engine, int current_iteration);
	void Train();
	// void Test(int single_frame_flag, int display_flag, int test_idx);
	void Test(bool single_frame, bool display, int test_idx, int data_set_id);
	void ShowTransformationGrid(int num_grid, int weight_idx);
	void Explorer::LearningFromPointCloudTest();
	void DownSamplingPointCloud(double voxel_size, pcl::VoxelGrid<pcl::PointXYZ>& voxel_grid, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr down_sampled_cloud);
	void DepthFiltering(float depth, pcl::PassThrough<pcl::PointXYZ>& pass, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud);
	void ShowCloudSequence();
	static void SetFeature(cv::Mat& feature, cv::Mat& feature_home, int num_joints, const cv::Mat& curr_prop);
	void PreprocessingAndSavePointCloud();
	void PCD2Mat(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, cv::Mat& cloud_mat);
	void Mat2PCD(cv::Mat& cloud_mat, pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud);
	void Mat2PCDWithMask(cv::Mat& cloud_mat, pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, cv::Mat& mask);
	void ReOrder(cv::Mat& input, cv::Mat& output, cv::Mat& input_indices);
	void RecordingTrend(GeneralTransform& transform, Loader& loader, std::vector<std::vector<double>>& trend_array, int iter, int write_trend_interval, int aim_idx);
	static void BuildModelGraph(const cv::Mat& home_cloud, int num_joints, cv::Mat& home_cloud_indices, cv::Mat& home_cloud_min_dists, double neighborhood_range, int max_num_neighbors);
	static void BuildMultiScaleNeighborhood(const cv::Mat& home_cloud, std::vector<std::vector<int>>& home_cloud_neighbor_indices, std::vector<std::vector<float>>& home_cloud_neighbor_dists, cv::Mat& scale_indices, cv::Mat& neighborhood_scales);
	static void InitializeModelLabel(const std::vector<cv::Mat>& min_dists, int num_joints, cv::Mat& home_cloud_label);
	static void LabelByPointToPointMatchingDistsBatch(const std::vector<std::vector<cv::Mat>>& min_dists, int num_joints,int max_num_neighbors, double beta, double sigma, cv::Mat& home_cloud_label, cv::Mat& averaged_dists, cv::Mat& home_cloud_neighbor_indices);
	static void LabelByPointToPlaneMatchingDistsBatch(const std::vector<std::vector<cv::Mat>>& indices, const std::vector<std::vector<cv::Mat>>& predicted_cloud, const std::vector<cv::Mat>& data_cloud, 
		const std::vector<cv::Mat>& target_cloud_normal, int num_joints, int max_num_neighbors, double beta, double sigma, cv::Mat& home_cloud_label, cv::Mat& averaged_projected_dists, cv::Mat& home_cloud_weight_label);
	static void IteratedConditionalModes(const cv::Mat& home_cloud_neighbor_indices, const std::vector<cv::Mat>& min_dists, cv::Mat& home_cloud_label, cv::Mat& potential, int num_joints, int icm_iterations, int max_num_neighbors, double beta, double sigma);
	static void IteratedConditionalModesWithoutContinuityConstraint(const cv::Mat& home_cloud_neighbor_indices, const std::vector<cv::Mat>& min_dists, cv::Mat& home_cloud_label, cv::Mat& potential, int num_joints, int icm_iterations, int max_num_neighbors, double beta, double sigma);
	static void ICMWithoutContinuityBatch(const cv::Mat& home_cloud_neighbor_indices, const std::vector<cv::Mat>& min_dists, cv::Mat& home_cloud_label, cv::Mat& potential, int num_joints, int icm_iterations, int max_num_neighbors, double beta, double sigma);
	void UpdateTransform();
	void CalculateOriginalPredictedError(const cv::Mat& home_cloud, const cv::Mat& target_cloud, const std::vector<cv::Mat>& predicted_cloud,
		const std::vector<cv::Mat>& indices, int num_joints, cv::Mat& original_error, cv::Mat& predicted_error);
	void SegmentWithHMM(std::vector<cv::Mat>& hmm_observation_sequence, std::vector<cv::Mat>& segmented_target_cloud, std::vector<cv::Mat>& segmented_home_cloud, std::vector<cv::Mat>& segmented_predicted_cloud, 
					   const cv::Mat& target_cloud, const cv::Mat& home_cloud, const cv::Mat& home_cloud_label, const cv::Mat& predicted_error, const cv::Mat& original_error,
					   const std::vector<cv::Mat>& predicted_cloud, const std::vector<cv::Mat>& indices, int num_joints, int hmm_dim_state, int hmm_time_idx);
	void AccumulateRunningVariance(std::vector<cv::Mat>& old_mean, std::vector<cv::Mat>& new_mean, std::vector<cv::Mat>& old_variance, std::vector<cv::Mat>& new_variance, 
				std::vector<cv::Mat>& data_list, int joint_idx, int state_idx, cv::Mat& current_count);
	void SegmentByLabel(std::vector<cv::Mat>& segmented_target_cloud, std::vector<cv::Mat>& segmented_home_cloud, std::vector<cv::Mat>& segmented_prediction_cloud, std::vector<cv::Mat>& segmented_dists, const cv::Mat& home_cloud_label_probabilities, const std::vector<cv::Mat>& prediction_cloud, const std::vector<cv::Mat>& min_dists, int num_joints);
	void SegmentByLabelBatch(std::vector<std::vector<cv::Mat>>& segmented_target_cloud, std::vector<std::vector<cv::Mat>>& segmented_home_cloud, std::vector<std::vector<cv::Mat>>& segmented_predicted_cloud, const cv::Mat& home_cloud_label, 
								   const cv::Mat& home_cloud, const std::vector<cv::Mat>& target_cloud, const std::vector<std::vector<cv::Mat>>& predicted_cloud, const std::vector<std::vector<cv::Mat>>& indices, int num_joints, int cloud_dim);
	void SegmentByLabelWithNormalBatch(std::vector<std::vector<cv::Mat>>& segmented_target_cloud, std::vector<std::vector<cv::Mat>>& segmented_target_cloud_normal, std::vector<std::vector<cv::Mat>>& segmented_home_cloud, std::vector<std::vector<cv::Mat>>& segmented_predicted_cloud, 
		const std::vector<cv::Mat>& target_cloud, const std::vector<cv::Mat>& target_cloud_normal, const cv::Mat& home_cloud, const std::vector<std::vector<cv::Mat>>& predicted_cloud, const std::vector<std::vector<cv::Mat>>& indices, 
		const cv::Mat& home_cloud_label, int num_joints, int cloud_dim);
	void TestBatchLearning();
	void PrepareBatchData();
	void BatchLearningFromRealData();
	void EstimateNormal();
	void ConvertMatchingDistsToPointWeights(const cv::Mat& matching_dists, const cv::Mat& original_dists, cv::Mat& point_weight);
	void Mat2PCDWithLabel(cv::Mat& cloud_mat, pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const cv::Mat& cloud_label, int target_label_idx);
	void Diagnosis(int num_diagnosis, int diagnosis_interval, bool single_frame, bool display, int test_idx, int data_set_id);
	void TestForWeight(Transform& transform, Loader& loader, cv::Mat& error, bool single_frame, bool display, bool diagnosis_flag, int test_idx, int data_set_id, int diagnosis_idx);
	void SaveTransform();
	void ReachTarget(int test_idx, int data_set_id, int weight_idx);
	void DetectTargetByTemplate(cv::Mat& target_mat, cv::Mat& ball);
	void ReadCloudFromPCD(char input_dir[], pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud);
	void EuclideanDistanceClustering(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, std::vector<boost::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>>& cloud_cluster);
	void ReadTargetTemplate(pcl::PointCloud<pcl::PointXYZ>::Ptr& target_template_ptr);
};

struct DistCompare{
    // the fifth component is the distance...
    inline bool operator()(const cv::Mat& a, const cv::Mat& b){
        return a.at<double>(0, 0) < b.at<double>(0, 0);
    }
};


#endif
