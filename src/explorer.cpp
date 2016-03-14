// with google c++ coding style here
#include "../inc/explorer.h"
#include "../inc/critic.h"
#include "../inc/actor.h"
#include "../inc/point_to_point_segmentation.h"
#include "../inc/point_to_point_segmentation_with_weights.h"
#include <pcl/features/boundary.h>
using namespace cv;
using namespace std;

// constructor
Explorer::Explorer(int id, 
				   char* data_set,
				   int train_iteration, 
				   int expand_iteration, 
				   int dim_transform, 
				   int num_joints, 
				   int num_segments,
				   int batch_size,
				   double normal_learning_rate,
				   double ini_exploration_range,
				   int train_data_size,
				   int test_data_size,
				   const Mat& joint_idx,
				   const Mat& joint_range_limit,
				   double neighborhood_range, 
				   int icm_iteration, 
				   double transform_alpha, 
				   double icm_sigma,
				   int max_num_neighbors) 
	: transform_(
	num_segments, 
	pow(3.0, num_joints) - 1, 
	batch_size,
	normal_learning_rate,
	transform_alpha) 
{    	
	sprintf(data_set_, data_set);		
	id_ = id; 	
	batch_size_ = batch_size;
	train_iteration_ = train_iteration;
	expand_iteration_ = expand_iteration;
	num_joints_ = num_joints;
	num_segments_ = num_segments;
	dim_transform_ = dim_transform;
	num_weights_ = 6; // dim_transform_ * (dim_transform_ - 1);
	train_data_size_ = train_data_size;
	test_data_size_ = test_data_size;
	normal_learning_rate_ = normal_learning_rate;
	// 3 is the dim of sinusoidal component, 1 sin cos
	dim_feature_ = pow(3.0, num_joints_) - 1;
	// this should also be changable and don't want to put it in config...
	num_trend_ = (num_weights_ + 1) * num_segments_ + 1 + num_segments_ * (dim_transform_ - 1);
	path_count_ = 0;		
    max_exploration_range_ = 1;
    ini_exploration_range_ = ini_exploration_range;	
	avg_cost_ = 0;	
	
	targets_ = vector<double>(num_joints_);
	prev_targets_ = vector<double>(num_joints_);
	explore_path_target_ = Mat::zeros(1, num_joints_, CV_64F);
	prev_explore_path_target_ = Mat::zeros(1, num_joints_, CV_64F);
	train_prop_ = Mat::zeros(train_data_size_, num_joints_, CV_64F);    
	test_prop_ = Mat::zeros(test_data_size_, num_joints_, CV_64F);    
    home_prop_ = Mat::zeros(1, num_joints_, CV_64F); // previous column, now row
	curr_prop_ = Mat::zeros(1, num_joints_, CV_64F); // previous column, now row
	prop_diff_ = Mat::zeros(train_data_size_, num_joints_, CV_64F);
	prop_dist_ = Mat::zeros(train_data_size_, 1, CV_64F);
	train_target_idx_ = Mat::zeros(train_data_size_, 1, CV_64F);    
	test_target_idx_ = Mat::zeros(test_data_size_, 1, CV_64F);    
	aim_idx_matrix_ = Mat::zeros(train_data_size_, 1, CV_64F);
	feature_ = vector<Mat>(batch_size_);
	for(int i = 0; i < batch_size_; i++)
	{
		feature_[i] = Mat::zeros(dim_feature_, 1, CV_64F);	
	}
	feature_home_ = Mat::zeros(dim_feature_, 1, CV_64F);	
	// home_cloud_ = std::vector<cv::Mat>(num_joints_);
	predicted_cloud_ = vector<Mat>(num_segments_);
	joint_idx.copyTo(joint_idx_);
	explore_path_kdtree_indices_ = Mat::zeros(train_data_size_, 1, CV_32S);
	explore_path_kdtree_dists_ = Mat::zeros(train_data_size_, 1, CV_32F);

	joint_idx_ = Mat::zeros(num_joints_, 1, CV_64F);
	joint_range_limit_ = Mat::zeros(num_joints_, 2, CV_64F);
	joint_idx.copyTo(joint_idx_);
	joint_range_limit.copyTo(joint_range_limit_);
	
	neighborhood_range_ = neighborhood_range;
	transform_alpha_ = transform_alpha;
	icm_sigma_ = icm_sigma;
	icm_iteration_ = icm_iteration;
	max_num_neighbors_ = max_num_neighbors;

	dim_state_ = num_segments_;
	original_error_mean_old_ = vector<Mat>(dim_state_);
	original_error_mean_new_ = vector<Mat>(dim_state_);
	original_error_cov_old_ = vector<Mat>(dim_state_);
	original_error_cov_new_ = vector<Mat>(dim_state_);

	predicted_error_mean_old_ = vector<Mat>(dim_state_);
	predicted_error_mean_new_ = vector<Mat>(dim_state_);
	predicted_error_cov_old_ = vector<Mat>(dim_state_);
	predicted_error_cov_new_ = vector<Mat>(dim_state_);

	for(int i = 0; i < dim_state_; i++)
	{
		original_error_mean_old_[i] = Mat::zeros(1, num_joints, CV_64F);
		original_error_mean_new_[i] = Mat::zeros(1, num_joints, CV_64F);
		original_error_cov_old_[i] = Mat::zeros(1, num_joints, CV_64F);
		original_error_cov_new_[i] = Mat::zeros(1, num_joints, CV_64F);

		predicted_error_mean_old_[i] = Mat::zeros(1, num_joints, CV_64F);
		predicted_error_mean_new_[i] = Mat::zeros(1, num_joints, CV_64F);
		predicted_error_cov_old_[i] = Mat::zeros(1, num_joints, CV_64F);
		predicted_error_cov_new_[i] = Mat::zeros(1, num_joints, CV_64F);
	}
	original_error_explained_variance_count_ = Mat::zeros(dim_state_, num_joints, CV_64F);
	predicted_error_explained_variance_count_ = Mat::zeros(dim_state_, num_joints, CV_64F);
}

Explorer::~Explorer()
{
}

void Explorer::Train()
{
	char input_dir[400];
	char output_dir[400];	
	int aim_idx = 0; // sorted idx 
	int aim_frame_idx = 0; // actual frame idx pointed to
	int record_trend_interval = 1000;
	int record_diagnosis_interval = 20;
	int query_cloud_size = 0;
	Mat predicted_cloud_f; // matched_target_cloud, transformed_query_cloud, indices, min_dists;
	Mat train_prop_f;
	Mat segmented_home_cloud_size = Mat::zeros(1, 1, CV_64F);
	vector<vector<Mat>> point_weight(batch_size_);
	for(int i = 0; i < batch_size_; i++)
	{
		point_weight[i] = vector<Mat>(num_segments_);
	}
	mt19937 engine(rd_());		
	vector<vector<double>> trend_array(num_trend_, vector<double>(0));
	vector<vector<Mat>> indices(batch_size_);
	vector<vector<Mat>> min_dists(batch_size_);
	vector<Mat> data_cloud(batch_size_);
	vector<Mat> data_cloud_f(batch_size_);
	vector<vector<Mat>> segmented_target_cloud(batch_size_);
	vector<vector<Mat>> segmented_home_cloud(batch_size_);
	vector<vector<Mat>> segmented_prediction_cloud(batch_size_);
	vector<vector<Mat>> predicted_cloud(batch_size_);
	
	vector<int> path(0);
	Loader loader(num_weights_, num_segments_, dim_feature_, num_trend_, id_, data_set_);
	loader.FormatWeightsForTestDirectory();
	loader.FormatTrendDirectory();
	loader.LoadProprioception(train_data_size_, test_data_size_, train_prop_, test_prop_, home_prop_, train_target_idx_, test_target_idx_, joint_idx_);	
	/***************** read in hand segmented home cloud ************/
	/*sprintf(input_dir, "D:/Document/HKUST/Year 5/Research/Solutions/unified_framework/output/para_%d/home_cloud_segmented_size.bin", id_);
	FileIO::ReadMatDouble(segmented_home_cloud_size, 1, 1, input_dir);
	int home_cloud_size = segmented_home_cloud_size.at<double>(0, 0);
	home_cloud_ = Mat::ones(home_cloud_size, 4, CV_64F);
	sprintf(input_dir, "D:/Document/HKUST/Year 5/Research/Solutions/unified_framework/output/para_%d/home_cloud_segmented.bin", id_);
	FileIO::ReadMatDouble(home_cloud_.colRange(0, 3), home_cloud_size, 3, input_dir);*/
	/***************** read in hand segmented home cloud ************/
	LoadHomeCloud(loader);
	// initialize probabilities to be equal...
	weight_label_probabilities_ = Mat::zeros(home_cloud_.rows, num_segments_, CV_64F) + 1 / num_segments_;
	Mat potential = Mat::zeros(home_cloud_.rows, num_segments_, CV_64F);
	Mat original_error = Mat::zeros(home_cloud_.rows, num_segments_, CV_64F);
	Mat predicted_error = Mat::zeros(home_cloud_.rows, num_segments_, CV_64F);
	Mat averaged_dists = Mat::zeros(home_cloud_.rows, num_segments_, CV_64F);
	Mat explained_variance = Mat::zeros(num_segments_, 1, CV_64F);
	Mat correlation = Mat::zeros(num_segments_ - 1, home_cloud_.cols, CV_64F);
	Mat original_averaged_dists = Mat::zeros(home_cloud_.rows, num_segments_, CV_64F);
	Mat home_cloud_label;
	vector<int> aim_idx_list(batch_size_);
	vector<int> aim_frame_idx_list(batch_size_);
	vector<Mat> curr_prop_batch(batch_size_);
	// some parameters need to be externalized
	Mat home_cloud_neighbor_indices, home_cloud_neighbor_dists;
	BuildModelGraph(home_cloud_, num_segments_, home_cloud_neighbor_indices, home_cloud_neighbor_dists, neighborhood_range_, max_num_neighbors_);

	/********** multi scale neighborhood **********/
	/*int num_scales = 3;
	Mat neighborhood_scales = Mat::zeros(num_scales, 1, CV_64F);
	neighborhood_scales.at<double>(0, 0) = 0.06; neighborhood_scales.at<double>(1, 0) = 0.12; neighborhood_scales.at<double>(2, 0) = 0.20; 
	vector<vector<int>> multi_scale_neighbor_indices;
	vector<vector<float>> multi_scale_neighbor_dists;
	Mat scale_indices = Mat::zeros(home_cloud_.rows, num_scales, CV_32S);
	BuildMultiScaleNeighborhood(home_cloud_, multi_scale_neighbor_indices, multi_scale_neighbor_dists, scale_indices, neighborhood_scales);*/
	/********** multi scale neighborhood **********/

	train_prop_.convertTo(train_prop_f, CV_32F);
	cv::flann::Index kd_trees(train_prop_f, cv::flann::KDTreeIndexParams(4), cvflann::FLANN_DIST_EUCLIDEAN); // build kd tree
	
	Mat feature_zero = Mat::zeros(dim_feature_, 1, CV_64F);	
	SetFeature(feature_home_, feature_zero, num_joints_, home_prop_);
	home_prop_.copyTo(prev_explore_path_target_);
	// need to put this in configuration
	Mat scale = Mat::zeros(num_joints_, 2, CV_64F);
	for(int i = 0; i < num_joints_; i++)
	{
		scale.at<double>(i, 0) = joint_range_limit_.at<double>(i, 0) - home_prop_.at<double>(0, i);
		scale.at<double>(i, 1) = joint_range_limit_.at<double>(i, 1) - home_prop_.at<double>(0, i);
	}
	// main loop
	// always start from home pose

	/********************* just for display ************************/
	char viewer_name[20];
	sprintf(viewer_name, "cloud_viewer_%d", id_);
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
	viewer = boost::shared_ptr<pcl::visualization::PCLVisualizer>(new pcl::visualization::PCLVisualizer(viewer_name));
	viewer->setBackgroundColor(0, 0, 0);	
	viewer->initCameraParameters();	
	/********************* just for display ************************/

	PointToPointSegmentation segmentation(home_cloud_.rows, home_cloud_.cols, num_segments_, batch_size_, max_num_neighbors_);
	// PointToPointSegmentationWithWeights segmentation(home_cloud_.rows, home_cloud_.cols, num_segments_, batch_size_, max_num_neighbors_, transform_alpha_, icm_sigma_);
	segmentation.set_home_cloud(home_cloud_);
	segmentation.set_home_cloud_neighbor_indices(home_cloud_neighbor_indices);
	vector<cv::flann::Index*> kd_trees_ptr(batch_size_);
	Mat home_cloud_f = Mat::zeros(home_cloud_.rows, home_cloud_.cols, CV_32F);
	home_cloud_.convertTo(home_cloud_f, CV_32F);
	for(int batch_idx = 0; batch_idx < batch_size_; batch_idx++)
	{
		curr_prop_batch[batch_idx] = Mat::zeros(1, home_prop_.cols, CV_64F);
		predicted_cloud[batch_idx] = vector<Mat>(num_segments_);
		indices[batch_idx] = vector<Mat>(num_segments_);
		min_dists[batch_idx] = vector<Mat>(num_segments_);    
		segmented_target_cloud[batch_idx] = vector<Mat>(num_segments_);
		segmented_home_cloud[batch_idx] = vector<Mat>(num_segments_);
		segmented_prediction_cloud[batch_idx] = vector<Mat>(num_segments_);
		kd_trees_ptr[batch_idx] = new cv::flann::Index(home_cloud_f, cv::flann::KDTreeIndexParams(4), cvflann::FLANN_DIST_EUCLIDEAN); // build kd tree 
	}

	for(unsigned long iteration_count = 0; iteration_count < train_iteration_; iteration_count++)
	{		
		// feature
		for(int batch_idx = 0; batch_idx < batch_size_; batch_idx++)
		{
			aim_idx = iteration_count == 0 ? 0 : GenerateAimIndex(engine, kd_trees, path, iteration_count, scale);
			aim_frame_idx = train_target_idx_.at<double>(aim_idx, 0);
			// frames below 30 is not good...
			if(iteration_count > 0)
			{
				// while(aim_frame_idx < 30 || aim_frame_idx == 0)
				// {
				while(aim_idx == 0)
				{
					aim_idx = GenerateAimIndex(engine, kd_trees, path, iteration_count, scale);
					aim_frame_idx = train_target_idx_.at<double>(aim_idx, 0);
				}
			}
			aim_idx_list[batch_idx] = aim_idx;
			aim_frame_idx_list[batch_idx] = aim_frame_idx;
			curr_prop_ = train_prop_.rowRange(aim_idx, aim_idx + 1);
			curr_prop_.copyTo(curr_prop_batch[batch_idx]);
			SetFeature(feature_[batch_idx], feature_home_, num_joints_, curr_prop_);
			// load cloud
			loader.LoadBinaryPointCloud(data_cloud[batch_idx], aim_frame_idx);
			data_cloud_f[batch_idx] = Mat::zeros(data_cloud[batch_idx].rows, data_cloud[batch_idx].cols, CV_32F);
			data_cloud[batch_idx].convertTo(data_cloud_f[batch_idx], CV_32F);
			delete kd_trees_ptr[batch_idx];
			kd_trees_ptr[batch_idx] = new cv::flann::Index(data_cloud_f[batch_idx], cv::flann::KDTreeIndexParams(4), cvflann::FLANN_DIST_EUCLIDEAN); // build kd tree 
		}

		if(iteration_count != 0)
		{
			// calc transformation and transform cloud
			transform_.CalcTransformation(feature_);
			transform_.TransformCloud(home_cloud_, predicted_cloud); // need to investigate home cloud issue
			for(int batch_idx = 0; batch_idx < batch_size_; batch_idx++)
			{
				for(int joint_idx = 0; joint_idx < num_segments_; joint_idx++)
				{       
					predicted_cloud[batch_idx][joint_idx].convertTo(predicted_cloud_f, CV_32F); 		       
					kd_trees_ptr[batch_idx]->knnSearch(predicted_cloud_f, indices[batch_idx][joint_idx], min_dists[batch_idx][joint_idx], 1, cv::flann::SearchParams(64)); // kd tree search, index indices the matches in query cloud
				}
			}
			segmentation.set_kd_tree_indices(indices);
			segmentation.set_kd_tree_min_dists(min_dists);
			segmentation.set_target_cloud(data_cloud);
			segmentation.set_predicted_cloud(predicted_cloud);
			// need to set following variables for neighborhood based segmentation
			// multi_scale_neighbor_indices, multi_scale_neighbor_dists, scale_indices, neighborhood_scales
			// segmentation.set_multi_scale_neighbor_indices(multi_scale_neighbor_indices);
			// segmentation.set_scale_indices(scale_indices);
			segmentation.Match(iteration_count, icm_iteration_);
			segmentation.Segment();
			// segmentation.UpdateKinematicStructure(curr_prop_batch);
			segmented_target_cloud = segmentation.segmented_target_cloud();
			segmented_prediction_cloud = segmentation.segmented_predicted_cloud();
			segmented_home_cloud = segmentation.segmented_home_cloud();
			home_cloud_label = segmentation.home_cloud_label();
			point_weight = segmentation.segmented_point_weight();

			// original_averaged_dists = segmentation.original_averaged_dists();
			// averaged_dists = segmentation.averaged_dists();
			// explained_variance = segmentation.explained_variance();
			// correlation = segmentation.correlation();
			// ConvertMatchingDistsToPointWeights(averaged_dists, original_averaged_dists, point_weight);
			
			transform_.CalculateGradient(segmented_target_cloud, segmented_prediction_cloud, segmented_home_cloud, point_weight, feature_); // target, prediction, query, without label...
			transform_.Update();
			// transform_.UpdateKinematicStructure();
			
			// record data
			RecordData(loader, trend_array, home_cloud_label, explained_variance, correlation, aim_idx, iteration_count, record_trend_interval, record_diagnosis_interval);
			
			/******************** just for display ******************/
			if(iteration_count % 50 == 1)
			{
				/*Mat point_weight_mat = Mat::zeros(point_weight[0][0].rows, 2, CV_64F);
				point_weight[0][0].copyTo(point_weight_mat.colRange(0, 1));
				point_weight[0][1].copyTo(point_weight_mat.colRange(1, 2));
				sprintf(output_dir, "D:/Document/HKUST/Year 5/Research/Solutions/unified_framework/output/para_%d/matching_error/point_weight_%d.bin", id_, iteration_count);
				FileIO::WriteMatDouble(point_weight_mat, point_weight_mat.rows, point_weight_mat.cols, output_dir);*/

				sprintf(output_dir, "D:/Document/HKUST/Year 5/Research/Solutions/unified_framework/output/para_%d/home_cloud_label.bin", id_);
				FileIO::WriteMatDouble(home_cloud_label, home_cloud_label.rows, home_cloud_label.cols, output_dir);
				for(int i = 0; i < num_segments_; i++)
				{
					COLOUR c = GetColour(i, 0, num_segments_ - 1);
					pcl::PointCloud<pcl::PointXYZ>::Ptr segmented_home_cloud_pcd(new pcl::PointCloud<pcl::PointXYZ>); 
					Mat2PCD(segmented_home_cloud[0][i], segmented_home_cloud_pcd);
					// Mat2PCDWithLabel(segmented_home_cloud[0][i], segmented_home_cloud_pcd, home_cloud_label, i);
					pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> segmented_home_cloud_color(segmented_home_cloud_pcd, (int)(c.r * 255.0), (int)(c.g * 255.0), (int)(c.b * 255.0));
					char cloud_name[20];
					sprintf(cloud_name, "cloud_segment_%d", i);
					if(iteration_count == 1)
					{			
						viewer->addPointCloud<pcl::PointXYZ>(segmented_home_cloud_pcd, segmented_home_cloud_color, cloud_name);	
						viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, cloud_name);
					}
					else
					{
						viewer->updatePointCloud<pcl::PointXYZ>(segmented_home_cloud_pcd, segmented_home_cloud_color, cloud_name);
						viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, cloud_name);
					}
				}
				viewer->spinOnce(10.0); // ms as unit
			}
			/******************** just for display ******************/
		}
		ShowLearningProgress(iteration_count);
	}	
}

void Explorer::TestForWeight(Transform& transform, Loader& loader, Mat& error, bool single_frame, bool display, bool diagnosis_flag, int test_idx, int data_set_id, int diagnosis_idx)
{
	char input_dir[400];
	char output_dir[400];
	int aim_idx = 0;
	int aim_frame_idx;
	int start_idx = 0;
	int end_idx;
	int test_idx_1 = test_idx;
	if(data_set_id == 1)
	{
		end_idx = train_data_size_;
	}
	else
	{
		end_idx = test_data_size_;
	}
	if(single_frame)
	{
		start_idx = test_idx_1;
		end_idx = test_idx_1 + 1;
	}
	Mat home_cloud_label = Mat::zeros(home_cloud_.rows, num_segments_, CV_64F);
	Mat cloud_f, predicted_cloud_f, home_cloud_f;
	
	Mat matching_averaged_dists = Mat::zeros(num_segments_, end_idx - start_idx, CV_64F);
	vector<vector<Mat>> predicted_cloud(1);
	vector<vector<Mat>> indices(1);
	vector<vector<Mat>> min_dists(1);
	predicted_cloud[0] = vector<Mat>(num_segments_);
	indices[0] = vector<Mat>(num_segments_);
	min_dists[0] = vector<Mat>(num_segments_);

	vector<vector<Mat>> home_indices(1);
	vector<vector<Mat>> home_min_dists(1);
	home_indices[0] = vector<Mat>(num_segments_);
	home_min_dists[0] = vector<Mat>(num_segments_);
	// load home cloud label
	sprintf(input_dir, "D:/Document/HKUST/Year 5/Research/Solutions/unified_framework/output/para_%d/home_cloud_label.bin", id_);
	// sprintf(input_dir, "D:/Document/HKUST/Year 5/Research/Solutions/unified_framework/output/diagnosis_%d/home_cloud_label_%d.bin", id_, test_idx);
	FileIO::ReadMatDouble(home_cloud_label, home_cloud_.rows, num_segments_, input_dir);

	Mat feature_zero = Mat::zeros(dim_feature_, 1, CV_64F);	
	SetFeature(feature_home_, feature_zero, num_joints_, home_prop_);
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
	if(display)
	{		
		viewer = boost::shared_ptr<pcl::visualization::PCLVisualizer>(new pcl::visualization::PCLVisualizer("cloud Viewer"));
		// viewer->setBackgroundColor(0, 0, 0);	
		viewer->setBackgroundColor(255, 255, 255);
		viewer->initCameraParameters();	
		// viewer->setCameraPosition(0, 0, 0, 1, 1, 0, 0, 0, 0);
		
	}
	vector<Mat> segmented_target_cloud(num_segments_);
	vector<Mat> segmented_home_cloud(num_segments_);
	vector<Mat> segmented_prediction_cloud(num_segments_);
	vector<Mat> segmented_dists(num_segments_);
	vector<int> path(0);
	
	float depth = 1.2;
	pcl::PassThrough<pcl::PointXYZ> pass;

	for(aim_idx = start_idx; aim_idx < end_idx; aim_idx++)
	{
		if(data_set_id == 1)
		{
			aim_frame_idx = train_target_idx_.at<double>(aim_idx, 0);	
			curr_prop_ = train_prop_.rowRange(aim_idx, aim_idx + 1);
		}
		else
		{
			aim_frame_idx = test_target_idx_.at<double>(aim_idx, 0);
			curr_prop_ = test_prop_.rowRange(aim_idx, aim_idx + 1);
		}
		loader.LoadBinaryPointCloud(cloud_, aim_frame_idx);
		cloud_.convertTo(cloud_f, CV_32F);
		home_cloud_.convertTo(home_cloud_f, CV_32F);
		cv::flann::Index target_cloud_kd_trees(cloud_f, cv::flann::KDTreeIndexParams(4), cvflann::FLANN_DIST_EUCLIDEAN); // build kd tree
		cv::flann::Index home_cloud_kd_trees(home_cloud_f, cv::flann::KDTreeIndexParams(4), cvflann::FLANN_DIST_EUCLIDEAN); // build kd tree
		SetFeature(feature_[0], feature_home_, num_joints_, curr_prop_);
		transform.CalcTransformation(feature_);
		transform.TransformCloud(home_cloud_, predicted_cloud); // need to investigate home cloud issue
		if(diagnosis_flag)
		{
			
			// calculate distances
			// load indices
			for(int joint_idx = 0; joint_idx < num_segments_; joint_idx++)
			{   
				// read in the calculated indices...
				sprintf(input_dir, "D:/Document/HKUST/Year 5/Research/Solutions/unified_framework/output/para_%d/home_cloud_label/indices_%d_%d.bin", id_, aim_idx, joint_idx);
				indices[0][joint_idx] = Mat::zeros(home_cloud_.rows, 1, CV_32S);
				FileIO::ReadMatInt(indices[0][joint_idx], indices[0][joint_idx].rows, indices[0][joint_idx].cols, input_dir);
				min_dists[0][joint_idx] = Mat::zeros(home_cloud_.rows, 1, CV_32F);
				predicted_cloud[0][joint_idx].convertTo(predicted_cloud_f, CV_32F); 
				for(int point_idx = 0; point_idx < home_cloud_.rows; point_idx++)
				{
					int curr_idx = indices[0][joint_idx].at<int>(point_idx, 0);
					// min_dists[0][joint_idx].at<float>(point_idx, 0) = pow(norm(predicted_cloud[0][joint_idx].rowRange(point_idx, point_idx + 1) - cloud_.rowRange(curr_idx, curr_idx + 1), NORM_L2), 2);
					min_dists[0][joint_idx].at<float>(point_idx, 0) = norm(predicted_cloud[0][joint_idx].rowRange(point_idx, point_idx + 1) - cloud_.rowRange(curr_idx, curr_idx + 1), NORM_L2);
				}
			}
			// we have a sqrt inside the segment by label function... delete that...
			SegmentByLabel(segmented_target_cloud, segmented_home_cloud, segmented_prediction_cloud, segmented_dists, home_cloud_label, predicted_cloud[0], min_dists[0], num_segments_);
			for(int segment_idx = 0; segment_idx < num_segments_; segment_idx++)
			{       
				sprintf(output_dir, "D:/Document/HKUST/Year 5/Research/Solutions/unified_framework/output/para_%d/matching_error/error_%d_%d_%d.bin", id_, diagnosis_idx, aim_idx, segment_idx);
				FileIO::WriteMatDouble(segmented_dists[segment_idx], segmented_dists[segment_idx].rows, segmented_dists[segment_idx].cols, output_dir);
			}
		}
		else
		{

			for(int segment_idx = 0; segment_idx < num_segments_; segment_idx++)
			{       
				predicted_cloud[0][segment_idx].convertTo(predicted_cloud_f, CV_32F); 		
				target_cloud_kd_trees.knnSearch(predicted_cloud_f, indices[0][segment_idx], min_dists[0][segment_idx], 1, cv::flann::SearchParams(64)); // kd tree search, index indices the matches in query cloud
				sprintf(output_dir, "D:/Document/HKUST/Year 5/Research/Solutions/unified_framework/output/para_%d/home_cloud_label/indices_%d_%d.bin", id_, aim_idx, segment_idx);
				FileIO::WriteMatInt(indices[0][segment_idx], indices[0][segment_idx].rows, indices[0][segment_idx].cols, output_dir);
			}

			vector<Mat> tmp = vector<Mat>(num_segments_);
			for(int si = 0; si < num_segments_; si++)
			{
				tmp[si] = Mat::zeros(min_dists[0][si].rows, min_dists[0][si].cols, CV_32F);	
				sqrt(min_dists[0][si], tmp[si]);
			}
			SegmentByLabel(segmented_target_cloud, segmented_home_cloud, segmented_prediction_cloud, segmented_dists, home_cloud_label, predicted_cloud[0], tmp, num_segments_);
		}
	
		/*home_cloud_kd_trees.knnSearch(cloud_f, home_indices[0][0], home_min_dists[0][0], 1, cv::flann::SearchParams(64)); // kd tree search, index indices the matches in query cloud
		Mat mask = Mat::zeros(cloud_f.rows, 1, CV_64F);
		for(int point_idx = 0; point_idx < cloud_f.rows; point_idx++)
		{
			int curr_idx = home_indices[0][0].at<int>(point_idx, 0);
			if(home_cloud_label.at<double>(curr_idx, 0) == 1) // assume background label is 0
			{
				mask.at<double>(point_idx, 0) = 1;
			}
		}*/
		
		Mat curr_averaged_dist = Mat::zeros(num_segments_, 1, CV_64F);
		for(int segment_idx = 0; segment_idx < num_segments_; segment_idx++)
		{  
			Mat tmp_avg_dist = Mat::zeros(1, 1, CV_64F);
			reduce(segmented_dists[segment_idx], tmp_avg_dist, 0, CV_REDUCE_AVG);
			curr_averaged_dist.at<double>(segment_idx, 0) = tmp_avg_dist.at<double>(0, 0);
		}
		curr_averaged_dist.copyTo(matching_averaged_dists.colRange(aim_idx - start_idx, aim_idx - start_idx + 1));
		// matching_averaged_dists.at<double>(aim_idx - start_idx, 0) = curr_averaged_dist.at<double>(0, 0);
		
		if(display)
		{
			char** segment_names = new char*[num_segments_];
			for(int joint_idx = 0; joint_idx < num_segments_; joint_idx++)
			{
				segment_names[joint_idx] = new char[50];
			}
			for(int segment_idx = 0; segment_idx < num_segments_; segment_idx++)
			{
				/*if(segment_idx == 1 || segment_idx == 3)
				{*/
					sprintf(segment_names[segment_idx], "transformed_cloud_segments_%d", segment_idx);
					//pcl::PointCloud<pcl::PointXYZ>::Ptr predicted_cloud_pcd_tmp(new pcl::PointCloud<pcl::PointXYZ>);
					pcl::PointCloud<pcl::PointXYZ>::Ptr predicted_cloud_pcd(new pcl::PointCloud<pcl::PointXYZ>);
					Mat2PCD(segmented_prediction_cloud[segment_idx], predicted_cloud_pcd);
					char tmp_dir[400];
					sprintf(tmp_dir, "D:/Document/HKUST/Year 5/Research/Solutions/unified_framework/output/para_79/%d.bin", segment_idx);
					FileIO::WriteMatDouble(segmented_prediction_cloud[segment_idx], segmented_prediction_cloud[segment_idx].rows, segmented_prediction_cloud[segment_idx].cols, tmp_dir);
					//DepthFiltering(depth, pass, predicted_cloud_pcd_tmp, predicted_cloud_pcd);
					COLOUR c = GetColour(segment_idx, 0, num_segments_ - 1);
					if(segment_idx == 0)
					{
						c.r = 0; c.g = 0; c.b = 1;
					}
					if(segment_idx == 1)
					{
						c.r = 0; c.g = 1; c.b = 0;
					}
					if(segment_idx == 2)
					{
						c.r = 1; c.g = 0; c.b = 0;
					}
					if(segment_idx == 3)
					{
						c.r = 1; c.g = 1; c.b = 0;
					}
					pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> predicted_cloud_color(predicted_cloud_pcd, (int)(c.r * 255.0), (int)(c.g * 255.0), (int)(c.b * 255.0));			
					if(aim_idx - start_idx == 0)
					{			
						viewer->addPointCloud<pcl::PointXYZ>(predicted_cloud_pcd, predicted_cloud_color, segment_names[segment_idx]);		
						viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, segment_names[segment_idx]);
					}
					else
					{
						viewer->updatePointCloud<pcl::PointXYZ>(predicted_cloud_pcd, predicted_cloud_color, segment_names[segment_idx]);
						viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, segment_names[segment_idx]);
					}
				// }
			}

			Mat end_effector_point = predicted_cloud[0][3].rowRange(985, 985 + 1);
			cout << home_cloud_.at<double>(839, 0) << " " << home_cloud_.at<double>(839, 1) << " " << home_cloud_.at<double>(839, 2) << endl;
			cout << predicted_cloud[0][3].at<double>(839, 0) << " " << predicted_cloud[0][3].at<double>(839, 1) << " " << predicted_cloud[0][3].at<double>(839, 2) << endl;
			pcl::PointCloud<pcl::PointXYZ>::Ptr end_effector_pcd(new pcl::PointCloud<pcl::PointXYZ>);
			Mat2PCD(end_effector_point, end_effector_pcd);
			pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> end_effector_color(end_effector_pcd, 112, 48, 160); // 108,151,246
			viewer->addPointCloud<pcl::PointXYZ>(end_effector_pcd, end_effector_color, "end_effector");		
			viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 8, "end_effector");
			//pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud_pcd_tmp(new pcl::PointCloud<pcl::PointXYZ>);
			/*pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud_pcd(new pcl::PointCloud<pcl::PointXYZ>);
			Mat2PCD(cloud_, target_cloud_pcd);
			//DepthFiltering(depth, pass, target_cloud_pcd_tmp, target_cloud_pcd);
			pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> target_cloud_color(target_cloud_pcd, 112, 48, 160); // 108,151,246
			if(aim_idx - start_idx == 0)
			{
				viewer->addPointCloud<pcl::PointXYZ>(target_cloud_pcd, target_cloud_color, "target_cloud");		
				viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, "target_cloud");
			}
			else
			{
				viewer->updatePointCloud<pcl::PointXYZ>(target_cloud_pcd, target_cloud_color, "target_cloud");
				viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, "target_cloud");
			}*/
			// overlay the home cloud
			/*pcl::PointCloud<pcl::PointXYZ>::Ptr home_cloud_pcd(new pcl::PointCloud<pcl::PointXYZ>);
			Mat2PCD(home_cloud_, home_cloud_pcd);
			pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> home_cloud_color(home_cloud_pcd, 96,26,114);			
			if(aim_idx - start_idx == 0)
			{
				viewer->addPointCloud<pcl::PointXYZ>(home_cloud_pcd, home_cloud_color, "home_cloud");		
				viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "home_cloud");
			}
			else
			{
				viewer->updatePointCloud<pcl::PointXYZ>(home_cloud_pcd, home_cloud_color, "home_cloud");
				viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "home_cloud");
			}*/

			// viewer->resetCameraViewpoint("home_cloud");




			if(end_idx - start_idx == 1)
				viewer->spin();
			else
				viewer->spinOnce(50);

			for(int joint_idx = 0; joint_idx < num_segments_; joint_idx++)
			{
				delete segment_names[joint_idx];
			}
			delete [] segment_names;
		}

		// cout << "iteration: " << aim_idx << endl;
	}
	
	matching_averaged_dists.copyTo(error);
	Mat total_averaged_dist = Mat::zeros(num_segments_, 1, CV_64F);
	reduce(matching_averaged_dists, total_averaged_dist, 1, CV_REDUCE_AVG);
	for(int segment_idx = 0; segment_idx < num_segments_; segment_idx++)
	{
		cout << "segment: " << segment_idx << " averaged error: " << total_averaged_dist.at<double>(segment_idx, 0) << " ";
	}
	cout << endl;
	/*sprintf(output_dir, "D:/Document/HKUST/Year 5/Research/Solutions/unified_framework/output/para_%d/total_averaged_dist.bin", id_);
	FileIO::ReadMatDouble(total_averaged_dist, total_averaged_dist.rows, total_averaged_dist.cols, output_dir);*/
}

void Explorer::ReachTarget(int test_idx, int data_set_id, int weight_idx)
{
	int point_idx = 839; // 985; // 839;
	int home_frame_idx;
	double lambda = 1e-6;
	double beta = 0.2;
	Mat matching_error = Mat::zeros(1, 1, CV_64F);
	RigidTransform3D transform(num_segments_, pow(3.0, num_joints_) - 1, 1, 1.0, transform_alpha_);
	Loader loader(num_weights_, num_segments_, dim_feature_, num_trend_, id_, data_set_);
	loader.FormatWeightsForTestDirectory();
	loader.LoadProprioception(train_data_size_, test_data_size_, train_prop_, test_prop_, home_prop_, train_target_idx_, test_target_idx_, joint_idx_);	
	loader.LoadWeightsForTest(transform);
	home_frame_idx = train_target_idx_.at<double>(0, 0);
	loader.LoadBinaryPointCloud(home_cloud_, home_frame_idx);
	// setup target point position
	Mat original_position = home_cloud_.rowRange(point_idx, point_idx + 1);
	Mat curr_position = Mat::zeros(original_position.rows, original_position.cols, CV_64F);
	Mat target_position = Mat::zeros(original_position.rows, original_position.cols, CV_64F);
	Mat feature_zero = Mat::zeros(dim_feature_, 1, CV_64F);	
	Mat curr_feature = Mat::zeros(dim_feature_, 1, CV_64F);	
	Mat target_feature = Mat::zeros(dim_feature_, 1, CV_64F);	
	SetFeature(feature_home_, feature_zero, num_joints_, home_prop_);

	// plot the target plane
	/*Mat targets = Mat::zeros(train_data_size_, original_position.cols, CV_64F);
	for(int p_idx = 0; p_idx < train_data_size_; p_idx++)
	{
		Mat target_theta = train_prop_.rowRange(p_idx, p_idx + 1);
		SetFeature(target_feature, feature_home_, num_joints_, target_theta);
		transform.CalcSingleTransform(target_feature, weight_idx, 0);
		transform.TransformPoint(original_position, target_position, weight_idx, 0);
		target_position.copyTo(targets.rowRange(p_idx, p_idx + 1));
	}
	stringstream ss;
	ss << "target_positions.bin";
	FileIO::WriteMatDouble(targets, targets.rows, targets.cols, ss.str());*/
	// int ball_size = 227;
	 //  = Mat::zeros(ball_size, 4, CV_64F);
	/*FileIO::ReadMatDouble(ball, ball.rows, ball.cols, "ball.bin");
	Mat dist = Mat::zeros(ball_size, 1, CV_64F);*/

	Mat ball;
	Mat full_ball;
	DetectTargetByTemplate(ball, full_ball);

	Mat dist = Mat::ones(ball.rows, 1, CV_64F) * -1.0;
	Mat mean_position = Mat::zeros(1, ball.cols, CV_64F);
	reduce(ball, mean_position, 0, CV_REDUCE_AVG);
	double mean_distance = norm(home_cloud_.rowRange(point_idx, point_idx + 1) - mean_position, NORM_L2);

	double min_dist = 100.0;
	Mat min_dist_theta = Mat::zeros(1, 3, CV_64F);
	Mat min_dist_position = Mat::zeros(1, 4, CV_64F);
	Mat min_dist_target_position = Mat::zeros(1, 4, CV_64F);
	int min_dist_idx = 0;
	for(int idx = 0; idx < ball.rows; idx++)
	{
		target_position = ball.rowRange(idx, idx + 1);
		double current_distance = norm(home_cloud_.rowRange(point_idx, point_idx + 1) - target_position, NORM_L2);
		if(target_position.at<double>(0, 0) > mean_position.at<double>(0, 0)) // mean_distance)
		{
			// pseudo-inverse reaching
			Mat original_theta = train_prop_.rowRange(0, 1);
			Mat curr_theta = Mat::zeros(1, dim_transform_ - 1, CV_64F);
			original_theta.copyTo(curr_theta);
			Mat jacobian = Mat::eye(dim_transform_ - 1, dim_transform_ - 1, CV_64F);
			int num_steps = 100;
			for(int step_idx = 1; step_idx <= num_steps; step_idx++)
			{
				SetFeature(curr_feature, feature_home_, num_joints_, curr_theta);
				transform.CalcSingleTransform(curr_feature, weight_idx, 0);
				transform.TransformPoint(original_position, curr_position, weight_idx, 0);
				transform.CalculateJacobian(original_position, curr_theta, curr_feature, jacobian, weight_idx);
				Mat tmp_invert = Mat::zeros(dim_transform_ - 1, dim_transform_ - 1, CV_64F);
				invert(jacobian * jacobian.t() + Mat::eye(dim_transform_ - 1, dim_transform_ - 1, CV_64F) * lambda, tmp_invert);
				Mat inv_jacobian = jacobian.t() * tmp_invert;
				Mat delta_pos = beta * (target_position - curr_position);
				Mat delta_theta = inv_jacobian * delta_pos.colRange(0, dim_transform_ - 1).t();
				curr_theta = curr_theta + delta_theta.t();
			}
			dist.at<double>(idx, 0) = norm(target_position - curr_position);
			if(dist.at<double>(idx, 0) < min_dist)
			{
				curr_position.copyTo(min_dist_position);	
				curr_theta.copyTo(min_dist_theta);
				target_position.copyTo(min_dist_target_position);
				min_dist_idx = idx;
				min_dist = dist.at<double>(idx, 0);
			}
		}
	}

	char input_dir[400];
	Mat home_cloud_label = Mat::zeros(home_cloud_.rows, num_segments_, CV_64F);
	Mat cloud_f, predicted_cloud_f, home_cloud_f;
	vector<vector<Mat>> predicted_cloud(1);
	vector<vector<Mat>> indices(1);
	vector<vector<Mat>> min_dists(1);
	predicted_cloud[0] = vector<Mat>(num_segments_);
	indices[0] = vector<Mat>(num_segments_);
	min_dists[0] = vector<Mat>(num_segments_);
	vector<vector<Mat>> home_indices(1);
	vector<vector<Mat>> home_min_dists(1);
	home_indices[0] = vector<Mat>(num_segments_);
	home_min_dists[0] = vector<Mat>(num_segments_);
	// load home cloud label
	sprintf(input_dir, "D:/Document/HKUST/Year 5/Research/Solutions/unified_framework/output/para_%d/home_cloud_label.bin", id_);
	// sprintf(input_dir, "D:/Document/HKUST/Year 5/Research/Solutions/unified_framework/output/diagnosis_%d/home_cloud_label_%d.bin", id_, test_idx);
	FileIO::ReadMatDouble(home_cloud_label, home_cloud_.rows, num_segments_, input_dir);
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
	viewer = boost::shared_ptr<pcl::visualization::PCLVisualizer>(new pcl::visualization::PCLVisualizer("cloud Viewer"));
	viewer->setBackgroundColor(255, 255, 255);
	viewer->initCameraParameters();	
		
	vector<Mat> segmented_target_cloud(num_segments_);
	vector<Mat> segmented_home_cloud(num_segments_);
	vector<Mat> segmented_prediction_cloud(num_segments_);
	vector<Mat> segmented_dists(num_segments_);
	vector<int> path(0);
	home_cloud_.convertTo(home_cloud_f, CV_32F);

	SetFeature(feature_[0], feature_home_, num_joints_, min_dist_theta);
	// SetFeature(feature_[0], feature_home_, num_joints_, home_prop_);
	transform.CalcTransformation(feature_);
	transform.TransformCloud(home_cloud_, predicted_cloud); // need to investigate home cloud issue
	for(int joint_idx = 0; joint_idx < num_segments_; joint_idx++)
	{   
		min_dists[0][joint_idx] = Mat::zeros(home_cloud_.rows, 1, CV_32F);
	}
	SegmentByLabel(segmented_target_cloud, segmented_home_cloud, segmented_prediction_cloud, segmented_dists, home_cloud_label, predicted_cloud[0], min_dists[0], num_segments_);

	char** segment_names = new char*[num_segments_];
	
	for(int segment_idx = 0; segment_idx < num_segments_; segment_idx++)
	{
		segment_names[segment_idx] = new char[50];
		sprintf(segment_names[segment_idx], "transformed_cloud_segments_%d", segment_idx);
		//pcl::PointCloud<pcl::PointXYZ>::Ptr predicted_cloud_pcd_tmp(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointCloud<pcl::PointXYZ>::Ptr predicted_cloud_pcd(new pcl::PointCloud<pcl::PointXYZ>);
		Mat2PCD(segmented_prediction_cloud[segment_idx], predicted_cloud_pcd);
		COLOUR c = GetColour(segment_idx, 0, num_segments_ - 1);
		if(segment_idx == 0)
		{
			c.r = 0; c.g = 0; c.b = 1;
		}
		if(segment_idx == 1)
		{
			c.r = 0; c.g = 1; c.b = 0;
		}
		if(segment_idx == 2)
		{
			c.r = 1; c.g = 0; c.b = 0;
		}
		if(segment_idx == 3)
		{
			c.r = 1; c.g = 1; c.b = 0;
		}
		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> predicted_cloud_color(predicted_cloud_pcd, (int)(c.r * 255.0), (int)(c.g * 255.0), (int)(c.b * 255.0));			
		viewer->addPointCloud<pcl::PointXYZ>(predicted_cloud_pcd, predicted_cloud_color, segment_names[segment_idx]);		
		viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 8, segment_names[segment_idx]);
	}

	Mat end_effector_point = predicted_cloud[0][3].rowRange(point_idx, point_idx + 1);
	pcl::PointCloud<pcl::PointXYZ>::Ptr end_effector_pcd(new pcl::PointCloud<pcl::PointXYZ>);
	Mat2PCD(end_effector_point, end_effector_pcd);
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> end_effector_color(end_effector_pcd, 112, 48, 160); // 108,151,246
	viewer->addPointCloud<pcl::PointXYZ>(end_effector_pcd, end_effector_color, "end_effector");		
	viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 8, "end_effector");

	pcl::PointCloud<pcl::PointXYZ>::Ptr full_ball_pcd(new pcl::PointCloud<pcl::PointXYZ>);
	Mat2PCD(full_ball, full_ball_pcd);
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> full_ball_color(full_ball_pcd, 0, 255, 0); 
	viewer->addPointCloud<pcl::PointXYZ>(full_ball_pcd, full_ball_color, "full_target");
	viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 8, "full_target");

	pcl::PointCloud<pcl::PointXYZ>::Ptr ball_pcd(new pcl::PointCloud<pcl::PointXYZ>);
	Mat2PCD(ball, ball_pcd);
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> ball_color(ball_pcd, 255, 0, 0); 
	viewer->addPointCloud<pcl::PointXYZ>(ball_pcd, ball_color, "target");		
	viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 8, "target");


	cout << min_dist_theta.at<double>(0, 0) << " " << min_dist_theta.at<double>(0, 1) << " " << min_dist_theta.at<double>(0, 2) << endl;
	cout << min_dist_target_position.at<double>(0, 0) << " " << min_dist_target_position.at<double>(0, 1) << " " << min_dist_target_position.at<double>(0, 2) << endl;
	cout << min_dist_position.at<double>(0, 0) << " " << min_dist_position.at<double>(0, 1) << " " << min_dist_position.at<double>(0, 2) << endl;
	cout << mean_position.at<double>(0, 0) << " " << mean_position.at<double>(0, 1) << " " << mean_position.at<double>(0, 2) << endl;

	cout << min_dist_idx << endl;
	cout << min_dist << endl;

	viewer->spin();
	/*stringstream ss;
	ss << "ball_dists.bin";
	FileIO::WriteMatDouble(dist, dist.rows, dist.cols, ss.str());*/
}



void Explorer::SaveTransform()
{

	RigidTransform3D test_transform(num_segments_, pow(3.0, num_joints_) - 1, 1, 1.0, transform_alpha_);
	Loader loader(num_weights_, num_segments_, dim_feature_, num_trend_, id_, data_set_);
	loader.FormatWeightsForTestDirectory();
	loader.LoadProprioception(train_data_size_, test_data_size_, train_prop_, test_prop_, home_prop_, train_target_idx_, test_target_idx_, joint_idx_);	
	loader.LoadWeightsForTest(test_transform);


	Mat feature_zero = Mat::zeros(dim_feature_, 1, CV_64F);	
	vector<Mat> curr_feature = vector<Mat>(1);
	vector<vector<Mat>> calculated_transform = vector<vector<Mat>>(1);
	calculated_transform[0] = vector<Mat>(num_segments_);
	for(int i = 0; i < num_segments_; i++)
	{
		calculated_transform[0][i] = Mat::eye(4, 4, CV_64F);
	}
	curr_feature[0] = Mat::zeros(dim_feature_, 1, CV_64F);
	SetFeature(feature_home_, feature_zero, num_joints_, home_prop_);
	
	for(int idx = 0; idx < train_data_size_; idx++)
	{
		curr_prop_ = train_prop_.rowRange(idx, idx + 1);
		SetFeature(curr_feature[0], feature_home_, num_joints_, curr_prop_);
		test_transform.CalcTransformation(curr_feature);
		calculated_transform = test_transform.transform();
		loader.SaveTransform(calculated_transform[0], idx, true);
	}
	for(int idx = 0; idx < test_data_size_; idx++)
	{
		curr_prop_ = test_prop_.rowRange(idx, idx + 1);
		SetFeature(curr_feature[0], feature_home_, num_joints_, curr_prop_);
		test_transform.CalcTransformation(curr_feature);
		calculated_transform = test_transform.transform();
		loader.SaveTransform(calculated_transform[0], idx, false);
	}
}

void Explorer::Test(bool single_frame, bool display, int test_idx, int data_set_id)
{
	int home_frame_idx;
	char input_dir[400];
	Mat segmented_home_cloud_size = Mat::zeros(1, 1, CV_64F);
	Mat matching_error;
	if(single_frame)
	{
		matching_error = Mat::zeros(1, 1, CV_64F);
	}
	else
	{
		matching_error = Mat::zeros(1, test_data_size_, CV_64F);
	}
	RigidTransform3D test_transform(num_segments_, pow(3.0, num_joints_) - 1, 1, 1.0, transform_alpha_);
	Loader loader(num_weights_, num_segments_, dim_feature_, num_trend_, id_, data_set_);
	loader.FormatWeightsForTestDirectory();
	loader.LoadProprioception(train_data_size_, test_data_size_, train_prop_, test_prop_, home_prop_, train_target_idx_, test_target_idx_, joint_idx_);	
	loader.LoadWeightsForTest(test_transform);
	home_frame_idx = train_target_idx_.at<double>(0, 0);
	/************** only for hand segmented home cloud *******************/
	/*sprintf(input_dir, "D:/Document/HKUST/Year 5/Research/Solutions/unified_framework/output/para_%d/home_cloud_segmented_size.bin", id_);
	FileIO::ReadMatDouble(segmented_home_cloud_size, 1, 1, input_dir);
	int home_cloud_size = segmented_home_cloud_size.at<double>(0, 0);
	home_cloud_ = Mat::ones(home_cloud_size, 4, CV_64F);
	sprintf(input_dir, "D:/Document/HKUST/Year 5/Research/Solutions/unified_framework/output/para_%d/home_cloud_segmented.bin", id_);
	FileIO::ReadMatDouble(home_cloud_.colRange(0, 3), home_cloud_size, 3, input_dir);*/
	loader.LoadBinaryPointCloud(home_cloud_, home_frame_idx);
	TestForWeight(test_transform, loader, matching_error, single_frame, display, false, test_idx, data_set_id, 0);
}

void Explorer::Diagnosis(int num_diagnosis, int diagnosis_interval, bool single_frame, bool display, int test_idx, int data_set_id)
{

	int home_frame_idx;
	char output_dir[400];
	char input_dir[400];
	Mat segmented_home_cloud_size = Mat::zeros(1, 1, CV_64F);
	vector<Mat> matching_error;
	Mat curr_error;
	if(single_frame)
	{
		matching_error = vector<Mat>(num_diagnosis);
		for(int i = 0; i < num_diagnosis; i++)
		{
			matching_error[i] = Mat::zeros(num_segments_, 1, CV_64F);
		}
		curr_error = Mat::zeros(num_segments_, 1, CV_64F);
	}
	else
	{
		matching_error = vector<Mat>(num_diagnosis);
		for(int i = 0; i < num_diagnosis; i++)
		{
			matching_error[i] = Mat::zeros(num_segments_, test_data_size_, CV_64F);
		}
		curr_error = Mat::zeros(num_segments_, test_data_size_, CV_64F);
	}
	RigidTransform3D test_transform(num_segments_, pow(3.0, num_joints_) - 1, 1, 1.0, transform_alpha_);
	Loader loader(num_weights_, num_segments_, dim_feature_, num_trend_, id_, data_set_);
	loader.FormatWeightsForTestDirectory();
	loader.FormatWeightsForDiagnosisDirectory();
	loader.LoadProprioception(train_data_size_, test_data_size_, train_prop_, test_prop_, home_prop_, train_target_idx_, test_target_idx_, joint_idx_);	
	home_frame_idx = train_target_idx_.at<double>(0, 0);
	/************** only for hand segmented home cloud *******************/
	/*sprintf(input_dir, "D:/Document/HKUST/Year 5/Research/Solutions/unified_framework/output/para_%d/home_cloud_segmented_size.bin", id_);
	FileIO::ReadMatDouble(segmented_home_cloud_size, 1, 1, input_dir);
	int home_cloud_size = segmented_home_cloud_size.at<double>(0, 0);
	home_cloud_ = Mat::ones(home_cloud_size, 4, CV_64F);
	sprintf(input_dir, "D:/Document/HKUST/Year 5/Research/Solutions/unified_framework/output/para_%d/home_cloud_segmented.bin", id_);
	FileIO::ReadMatDouble(home_cloud_.colRange(0, 3), home_cloud_size, 3, input_dir);*/

	loader.LoadBinaryPointCloud(home_cloud_, home_frame_idx);

	int diagnosis_iterations[] =  {13500}; // {0, 40, 160, 320, 640, 1280, 2560, 5000, 13500}; // {13500};
	int count = 0;
	// for(int diagnosis_idx = 0; diagnosis_idx < 0 + num_diagnosis * diagnosis_interval; diagnosis_idx += diagnosis_interval)
	for(int idx = 0; idx < 1; idx ++)
	{
		int diagnosis_idx = diagnosis_iterations[idx];
		loader.LoadWeightsForDiagnosis(test_transform, diagnosis_idx);
		TestForWeight(test_transform, loader, curr_error, single_frame, display, true, test_idx, data_set_id, diagnosis_idx);
		curr_error.copyTo(matching_error[count]);
		// sprintf(output_dir, "D:/Document/HKUST/Year 5/Research/Solutions/unified_framework/output/para_%d/matching_error/diagnosis_errors_%d.bin", id_, diagnosis_idx);
		// FileIO::WriteMatDouble(matching_error[count], matching_error[count].rows, matching_error[count].cols, output_dir);
		cout << "diagnosis idx: " << diagnosis_idx << endl;
		count++;
	}

	
}

void Explorer::SegmentByLabel(vector<Mat>& segmented_target_cloud, 
					   vector<Mat>& segmented_home_cloud, 
					   vector<Mat>& segmented_prediction_cloud, 
					   std::vector<cv::Mat>& segmented_dists, 
					   const Mat& home_cloud_label, 
					   const vector<Mat>& prediction_cloud, 
					   const vector<Mat>& min_dists, 
					   int num_segments)
{
	// shuffle the cloud to make it match with the template
	double min_value = 0;
	double max_value = 0;
	Point min_location, max_location;
	Mat count = Mat::zeros(num_segments, 1, CV_32S);
	int num_points = home_cloud_label.rows;
	int cloud_dim = prediction_cloud[0].cols;
	// segmented_dists = Mat::zeros(num_points, 1, CV_64F);
	for(int i = 0; i < num_segments; i++)
	{
		segmented_target_cloud[i] = Mat::zeros(num_points, cloud_dim, CV_64F);
		segmented_home_cloud[i] = Mat::zeros(num_points, cloud_dim, CV_64F);
		segmented_prediction_cloud[i] = Mat::zeros(num_points, cloud_dim, CV_64F);
		segmented_dists[i] = Mat::zeros(num_points, 1, CV_64F);
	}
	for(int p = 0; p < num_points; p++)
	{
		
		/*for(int label_idx = 0; label_idx < num_segments; label_idx++)
		{
			int curr_count = count.at<int>(label_idx, 0);
			prediction_cloud[label_idx].rowRange(p, p + 1).copyTo(segmented_prediction_cloud[label_idx].rowRange(curr_count, curr_count + 1));
			double curr_dist = min_dists[label_idx].at<float>(p, 0);
			segmented_dists[label_idx].at<double>(curr_count, 0) = sqrt(curr_dist);
			count.at<int>(label_idx, 0) = count.at<int>(label_idx, 0) + 1;
		}	*/

		minMaxLoc(home_cloud_label.rowRange(p, p + 1), &min_value, &max_value, &min_location, &max_location);
		if(min_value != 0 || max_value != 0)
		{
			int label_idx = max_location.x;
			int curr_count = count.at<int>(label_idx, 0);
			prediction_cloud[label_idx].rowRange(p, p + 1).copyTo(segmented_prediction_cloud[label_idx].rowRange(curr_count, curr_count + 1));
			double curr_dist = min_dists[label_idx].at<float>(p, 0);
			segmented_dists[label_idx].at<double>(curr_count, 0) = curr_dist;
			count.at<int>(label_idx, 0) = count.at<int>(label_idx, 0) + 1;
		}
	}
	for(int i = 0; i < num_segments; i++)
	{
		segmented_target_cloud[i] = segmented_target_cloud[i].rowRange(0, count.at<int>(i, 0));
		segmented_home_cloud[i] = segmented_home_cloud[i].rowRange(0, count.at<int>(i, 0));
		segmented_prediction_cloud[i] = segmented_prediction_cloud[i].rowRange(0, count.at<int>(i, 0));
		segmented_dists[i] = segmented_dists[i].rowRange(0, count.at<int>(i, 0));
	}
}

// build the neighbor graph of the home point cloud by radius search...
void Explorer::BuildModelGraph(const Mat& home_cloud, int num_joints, Mat& home_cloud_neighbor_indices, Mat& home_cloud_neighbor_dists, double neighborhood_range, int max_num_neighbor)
{
	Mat home_cloud_f; 
	home_cloud.convertTo(home_cloud_f, CV_32F);
	// initialize home cloud related matrices
	home_cloud_neighbor_indices = Mat::zeros(home_cloud.rows, max_num_neighbor, CV_32S) - 1; // all initialized to -1, which is the marker of non-used cells...
	home_cloud_neighbor_dists = Mat::zeros(home_cloud.rows, max_num_neighbor, CV_32F) - 1; // all initialized to -1, which is the marker
	cv::flann::Index home_cloud_kd_trees(home_cloud_f, cv::flann::KDTreeIndexParams(4), cvflann::FLANN_DIST_EUCLIDEAN); // build kd tree 
	for(int i = 0; i < home_cloud.rows; i++)
	{
		// the radius search thing only works for one row at a time...
		Mat query_point = home_cloud_f.rowRange(i, i + 1);
		Mat curr_neighbor_indices = Mat::zeros(1, max_num_neighbor + 1, CV_32S) - 1;
		Mat curr_neighbor_dists = Mat::zeros(1, max_num_neighbor + 1, CV_32F) - 1;
		home_cloud_kd_trees.radiusSearch(query_point, curr_neighbor_indices, curr_neighbor_dists, neighborhood_range * neighborhood_range, max_num_neighbor + 1, cv::flann::SearchParams(64)); // kd tree search, index indices the matches in query cloud, need to exclude self in the end
		curr_neighbor_indices.colRange(1, max_num_neighbor + 1).copyTo(home_cloud_neighbor_indices.rowRange(i, i + 1));
		curr_neighbor_dists.colRange(1, max_num_neighbor + 1).copyTo(home_cloud_neighbor_dists.rowRange(i, i + 1));
	}
	/*char output_dir[200];
	sprintf(output_dir, "D:/Document/HKUST/Year 5/Research/Solutions/unified_framework/output/home_cloud_indices.bin");
	FileIO::WriteMatInt(home_cloud_neighbor_indices, home_cloud.rows, max_num_neighbor, output_dir);
	sprintf(output_dir, "D:/Document/HKUST/Year 5/Research/Solutions/unified_framework/output/home_cloud_min_dists.bin");
	FileIO::WriteMatFloat(home_cloud_neighbor_dists, home_cloud.rows, max_num_neighbor, output_dir);*/
}

void Explorer::LoadHomeCloud(Loader& loader)
{
	int home_frame_idx = train_target_idx_.at<double>(0, 0);
	loader.LoadBinaryPointCloud(home_cloud_, home_frame_idx);
}

void Explorer::ShowLearningProgress(int iteration_count)
{
	if(iteration_count % 5 == 1)			
	{
		cout << "iteration: " << iteration_count << endl;			
	}
}

void Explorer::RecordData(Loader& loader, vector<vector<double>>& trend_array, Mat& home_cloud_label, Mat& explained_variance, Mat& correlation, int aim_idx, int iteration_count, int record_trend_interval, int record_diagnosis_interval)
{
	int trend_count;
	vector<Mat> w_r_x = transform_.w_r_x();
	vector<Mat> w_r_y = transform_.w_r_y();
	vector<Mat> w_r_z = transform_.w_r_z();
	vector<Mat> w_t_x = transform_.w_t_x();
	vector<Mat> w_t_y = transform_.w_t_y();
	vector<Mat> w_t_z = transform_.w_t_z();
	for(int i = 0; i < num_segments_; i++)
	{
		trend_count = i * 6 + 0;
		trend_array[trend_count].push_back(cv::norm(w_r_x[i], NORM_L2));
		trend_count = i * 6 + 1;
		trend_array[trend_count].push_back(cv::norm(w_r_y[i], NORM_L2));
		trend_count = i * 6 + 2;
		trend_array[trend_count].push_back(cv::norm(w_r_z[i], NORM_L2));
		trend_count = i * 6 + 3;
		trend_array[trend_count].push_back(cv::norm(w_t_x[i], NORM_L2));
		trend_count = i * 6 + 4;
		trend_array[trend_count].push_back(cv::norm(w_t_y[i], NORM_L2));
		trend_count = i * 6 + 5;
		trend_array[trend_count].push_back(cv::norm(w_t_z[i], NORM_L2));
		/*for(int j = 0; j < num_weights_; j++)
		{
			trend_count = i * num_weights_ + j;
			vector<Mat> w_t_x = transform_.w_t_x();
			trend_array[trend_count].push_back(cv::norm(w_t_x[0], NORM_L2));
		}*/
	}
	trend_array[num_segments_ * num_weights_].push_back(aim_idx);
	trend_count = num_segments_ * num_weights_ + 1;
	for(int curr_count = trend_count; curr_count < trend_count + num_segments_; curr_count++)
	{
		trend_array[curr_count].push_back(explained_variance.at<double>(curr_count - trend_count, 0));
	}
	trend_count = trend_count + num_segments_;
	int num_corr = (num_segments_) * (dim_transform_ - 1);
	for(int curr_count = trend_count; curr_count < trend_count + num_corr; curr_count++)
	{
		int row_idx = (curr_count - trend_count) / (dim_transform_ - 1);
		int col_idx = (curr_count - trend_count) % (dim_transform_ - 1);
		trend_array[curr_count].push_back(correlation.at<double>(row_idx, col_idx));
	}

	// record trend
	if(iteration_count % record_trend_interval == 1)
	{			
		int append_flag = iteration_count == 1 ? 0 : 1;			
		loader.SaveTrend(trend_array, num_trend_, append_flag);								
		for(int i = 0; i < num_trend_; i++)
			trend_array[i].clear();
		// record testing weight
		loader.SaveWeightsForTest(transform_);
	}
	// record diagnosis
	if(iteration_count % record_diagnosis_interval == 1)
	{
		loader.SaveWeightsForDiagnosis(transform_, iteration_count / record_diagnosis_interval);	
		loader.SaveHomeCloudLabelDiagnosis(home_cloud_label, iteration_count / record_diagnosis_interval);
	}
}

void Explorer::SetFeature(Mat& feature, Mat& feature_home, int num_joints, const Mat& curr_prop)
{
	int sinusoidal_dim = 3;
	int feature_dim = feature.rows;
	Mat count = Mat::zeros(num_joints, 1, CV_64F);
	Mat curr_prop_sinusoidal = Mat::zeros(num_joints, sinusoidal_dim, CV_64F);
	// set the sinusoidal value
	for(int i = 0; i < num_joints; i++)
	{
		curr_prop_sinusoidal.at<double>(i, 0) = 1;
		curr_prop_sinusoidal.at<double>(i, 1) = sin(curr_prop.at<double>(0, i) / 180.0 * PI);
		curr_prop_sinusoidal.at<double>(i, 2) = cos(curr_prop.at<double>(0, i) / 180.0 * PI);
	}
	
	for(int idx = 0; idx <= feature_dim; idx++)
	{
		if(idx != 0)
		{
			feature.at<double>(idx - 1, 0) = 1;
			int factor = sinusoidal_dim;
			for(int joint_idx = num_joints - 1; joint_idx >= 0; joint_idx--)
			{
				if(joint_idx == num_joints - 1)
				{
					count.at<double>(joint_idx, 0) = idx % factor;	
				}
				else
				{
					count.at<double>(joint_idx, 0) = idx / factor % sinusoidal_dim;	
					factor *= sinusoidal_dim;
				}
				feature.at<double>(idx - 1, 0) *= curr_prop_sinusoidal.at<double>(joint_idx, count.at<double>(joint_idx, 0));	
				// cout << joint_idx << " " << count.at<double>(joint_idx, 0) << " ";
			}
			// cout << endl;
		}
	}
	feature = feature - feature_home;
}

int Explorer::GenerateAimIndex(mt19937& engine, cv::flann::Index& kd_trees, vector<int>& path, int iteration_count, const Mat& scale)
{
	int aim_idx = 0;
	double current_range = 0;
	double max_speed = 10000.0; // no exploring path requested here... // 0.6 * scale.at<double>(0, 0); // 0.4 * scale
	double path_length = 0;
	int num_frame_path = 0;
	if(path.size() == 0)
	{		
        // planar exploration range, starting from the center, range is 0 to 1
		current_range = ini_exploration_range_ + (max_exploration_range_ - ini_exploration_range_) * iteration_count / expand_iteration_;	
		current_range = current_range > max_exploration_range_ ? max_exploration_range_ : current_range;
		for(int i = 0; i < num_joints_; i++)
		{
			uniform_real_distribution<double> uniform(scale.at<double>(i, 0) * current_range + home_prop_.at<double>(0, i), scale.at<double>(i, 1) * current_range + home_prop_.at<double>(0, i));
			explore_path_target_.at<double>(0, i) = uniform(engine); // row vector
		}
		path_length = cv::norm(explore_path_target_ - prev_explore_path_target_, NORM_L2);
		num_frame_path = (int)(path_length / max_speed) + 1;
		path.clear();
		for(int i = 1; i <= num_frame_path; i++)
		{
			Mat tmp_target = Mat::zeros(1, num_joints_, CV_64F);
			tmp_target = prev_explore_path_target_ + (explore_path_target_ - prev_explore_path_target_) * i / num_frame_path;
			tmp_target.convertTo(tmp_target, CV_32F);
			kd_trees.knnSearch(tmp_target, explore_path_kdtree_indices_, explore_path_kdtree_dists_, 1, cv::flann::SearchParams(64));
			path.push_back(explore_path_kdtree_indices_.at<int>(0, 0));			
		}	
		explore_path_target_.copyTo(prev_explore_path_target_);
	}
	aim_idx = path[0];
	path.erase(path.begin());		
	return aim_idx;
}

void Explorer::PCD2Mat(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, Mat& cloud_mat)
{
	int size = cloud->points.size();
	int dim = 4;
	cloud_mat = Mat::zeros(size, dim, CV_64F);
	for(int i = 0; i < size; i++)
	{
		cloud_mat.at<double>(i, 0) = cloud->points[i].x;
		cloud_mat.at<double>(i, 1) = cloud->points[i].y;
		cloud_mat.at<double>(i, 2) = cloud->points[i].z;
		cloud_mat.at<double>(i, 3) = 1.0;
	}
}
//
void Explorer::Mat2PCD(Mat& cloud_mat, pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
{
	int size = cloud_mat.rows;
	vector<pcl::PointXYZ> points_vec(size);
	cloud.reset(new pcl::PointCloud<pcl::PointXYZ>());
	for(int i = 0; i < size; i++)
	{
		pcl::PointXYZ point;
		point.x = cloud_mat.at<double>(i, 0);
		point.y = cloud_mat.at<double>(i, 1);
		point.z = cloud_mat.at<double>(i, 2);
		cloud->push_back(point);
	}	
}

void Explorer::Mat2PCDWithMask(Mat& cloud_mat, pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, Mat& mask)
{
	int size = cloud_mat.rows;
	vector<pcl::PointXYZ> points_vec(size);
	cloud.reset(new pcl::PointCloud<pcl::PointXYZ>());
	for(int i = 0; i < size; i++)
	{
		if(mask.at<double>(i, 0) == 0)
		{
			pcl::PointXYZ point;
			point.x = cloud_mat.at<double>(i, 0);
			point.y = cloud_mat.at<double>(i, 1);
			point.z = cloud_mat.at<double>(i, 2);
			cloud->push_back(point);
		}
	}	
}

void Explorer::Mat2PCDWithLabel(Mat& cloud_mat, pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const Mat& cloud_label, int target_label_idx)
{
	int size = cloud_mat.rows;
	vector<pcl::PointXYZ> points_vec(size);
	cloud.reset(new pcl::PointCloud<pcl::PointXYZ>());
	for(int i = 0; i < size; i++)
	{
		if(cloud_label.at<double>(i, target_label_idx) == 1)
		{
			pcl::PointXYZ point;
			point.x = cloud_mat.at<double>(i, 0);
			point.y = cloud_mat.at<double>(i, 1);
			point.z = cloud_mat.at<double>(i, 2);
			cloud->push_back(point);
		}
	}	
}

void Explorer::EstimateNormal()
{
	char input_dir[400];
	char output_dir[400];	
	int aim_idx = 0;	
	int aim_frame_idx = 0;
	int query_cloud_size = 0;
	unsigned long iteration_count = 0;
	vector<Mat> indices(num_segments_);
    vector<Mat> min_dists(num_segments_);    
	mt19937 engine(rd_());		
	int start_idx = 0;
	int end_idx = train_data_size_;
	Loader loader(num_weights_, num_segments_, dim_feature_, num_trend_, id_, data_set_);
	loader.FormatWeightsForTestDirectory();
	loader.LoadProprioception(train_data_size_, test_data_size_, train_prop_, test_prop_, home_prop_, train_target_idx_, test_target_idx_, joint_idx_);	
	// loader.LoadWeightsForTest(transform_);
	int home_frame_idx = train_target_idx_.at<double>(0, 0);
	loader.LoadBinaryPointCloud(home_cloud_, home_frame_idx);
	Mat home_cloud_label = Mat::zeros(home_cloud_.rows, num_segments_, CV_64F);
	Mat feature_zero = Mat::zeros(dim_feature_, 1, CV_64F);	
	SetFeature(feature_home_, feature_zero, num_joints_, home_prop_);
	Mat home_cloud_neighbor_indices, home_cloud_neighbor_dists;
	// BuildModelGraph(home_cloud_, num_joints_, home_cloud_neighbor_indices, home_cloud_neighbor_dists, neighborhood_range_, max_num_neighbors_);
	Mat cloud_f;
	vector<int> path(0);

	/*boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("cloud viewer"));
	viewer->setBackgroundColor(0, 0, 0);
	viewer->initCameraParameters();*/
	for(int cloud_idx = start_idx; cloud_idx < end_idx; cloud_idx++)
	{
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_pcd (new pcl::PointCloud<pcl::PointXYZ>);
		loader.LoadBinaryPointCloud(cloud_, cloud_idx);
		Mat2PCD(cloud_, cloud_pcd);
		pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimator;
		normal_estimator.setInputCloud(cloud_pcd);
		pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ> ());
		normal_estimator.setSearchMethod(tree);
		pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);
		normal_estimator.setRadiusSearch (neighborhood_range_);
		normal_estimator.compute (*cloud_normals);
		COLOUR c = GetColour(0, 0, num_segments_);
		// std::cout << (*cloud_normals.get()).points[0].normal_x << std::endl;
		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cloud_color(cloud_pcd, (int)(c.r * 255.0), (int)(c.g * 255.0), (int)(c.b * 255.0));			

		Mat normal_mat = Mat::zeros(cloud_.rows, 3, CV_64F);
		for(int point_idx = 0; point_idx < cloud_.rows; point_idx++)
		{
			normal_mat.at<double>(point_idx, 0) = (*cloud_normals.get()).points[point_idx].normal_x;
			normal_mat.at<double>(point_idx, 1) = (*cloud_normals.get()).points[point_idx].normal_y;
			normal_mat.at<double>(point_idx, 2) = (*cloud_normals.get()).points[point_idx].normal_z;
		}
		if(cloud_idx % 100 == 1)
		{
			cout << cloud_idx << endl;
		}
		sprintf(output_dir, "D:/Document/HKUST/Year 5/Research/Data/PointClouds/december_13_2013/binary_normal_dense/normal_%d.bin", cloud_idx);
		FileIO::WriteMatDouble(normal_mat, normal_mat.rows, 3, output_dir);

		//viewer->addPointCloud<pcl::PointXYZ>(cloud_pcd, cloud_color, "cloud");									
		//viewer->addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(cloud_pcd, cloud_normals, 5, 0.01, "normals");
		//viewer->spin(); // ms as unit
		// pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> transformed_cloud_color(transformed_cloud_pcd, 0, 255, 0);
	}
}

void Explorer::PreprocessingAndSavePointCloud()
{
	unsigned long iteration_count = 0;
	double depth_threshold = 1.25; // 0.8 for arm data
	double voxel_grid_size = 0.015; // 0.010; // 0.005 is dense and 0.01 is normal...

	Loader loader(num_weights_, num_segments_, dim_feature_, num_trend_, id_, data_set_);
	loader.FormatWeightsForTestDirectory();
	loader.FormatTrendDirectory();
	loader.LoadProprioception(train_data_size_, test_data_size_, train_prop_, test_prop_, home_prop_, train_target_idx_, test_target_idx_, joint_idx_);	
	pcl::PassThrough<pcl::PointXYZ> pass;
	pcl::PCDReader reader;
	pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;	
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr tmp_cloud(new pcl::PointCloud<pcl::PointXYZ>);	
	// point clouds		
	for(iteration_count = 0; iteration_count < 15625; iteration_count++)
	{					
		loader.LoadPointCloud(cloud, reader, iteration_count); // load point cloud		
		DepthFiltering(depth_threshold, pass, cloud, tmp_cloud);
		DownSamplingPointCloud(voxel_grid_size, voxel_grid, tmp_cloud, cloud);
		loader.SavePointCloudAsBinaryMat(cloud, iteration_count);
		if(iteration_count % 100 == 1)
			cout << "iteration: " << iteration_count << endl;
	}
}

void Explorer::DepthFiltering(float depth, pcl::PassThrough<pcl::PointXYZ>& pass, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud)
{
	pass.setInputCloud(cloud);
	pass.setFilterFieldName ("z");
	pass.setFilterLimits(0.0, depth);
	pass.filter(*filtered_cloud);
}

void Explorer::DownSamplingPointCloud(double voxel_size, pcl::VoxelGrid<pcl::PointXYZ>& voxel_grid, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr down_sampled_cloud)
{
	voxel_grid.setInputCloud(cloud);
	voxel_grid.setLeafSize(voxel_size, voxel_size, voxel_size);
	voxel_grid.filter(*down_sampled_cloud);
}

void Explorer::ReadCloudFromPCD(char input_dir[], pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
{
	double depth_threshold = 1.25; // 0.8 for arm data
	double voxel_grid_size = 0.005; // 0.010; // 0.005 is dense and 0.01 is normal...
	pcl::PCDReader reader;
	pcl::PassThrough<pcl::PointXYZ> pass;
	pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;	
	pcl::PointCloud<pcl::PointXYZ>::Ptr tmp_cloud(new pcl::PointCloud<pcl::PointXYZ>);	
	reader.read(input_dir, *cloud);	
	DepthFiltering(depth_threshold, pass, cloud, tmp_cloud);
	DownSamplingPointCloud(voxel_grid_size, voxel_grid, tmp_cloud, cloud);
}

void Explorer::EuclideanDistanceClustering(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& cloud_cluster)
{
	int original_cloud_size = (int) cloud->points.size ();
	pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
	pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane (new pcl::PointCloud<pcl::PointXYZ> ());
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_f(new pcl::PointCloud<pcl::PointXYZ> ());
	// pcl::PCDWriter writer;
	pcl::SACSegmentation<pcl::PointXYZ> seg;
	seg.setOptimizeCoefficients (true);
	seg.setModelType (pcl::SACMODEL_PLANE);
	seg.setMethodType (pcl::SAC_RANSAC);
	seg.setMaxIterations (100);
	seg.setDistanceThreshold (0.02);

	while (cloud->points.size () > 0.2 * original_cloud_size)
	{
		// Segment the largest planar component from the remaining cloud
		seg.setInputCloud (cloud);
		seg.segment (*inliers, *coefficients);
		if (inliers->indices.size () == 0)
		{
			cout << "Could not estimate a planar model for the given dataset." << endl;
			break;
		}
		// Extract the planar inliers from the input cloud
		pcl::ExtractIndices<pcl::PointXYZ> extract;
		extract.setInputCloud (cloud);
		extract.setIndices (inliers);
		extract.setNegative (false);
		// Get the points associated with the planar surface
		extract.filter (*cloud_plane);
		cout << "PointCloud representing the planar component: " << cloud_plane->points.size () << " data points." << endl;
		// Remove the planar inliers, extract the rest
		extract.setNegative (true);
		extract.filter (*cloud_f);
		*cloud = *cloud_f;
	}
	// Creating the KdTree object for the search method of the extraction
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
	tree->setInputCloud (cloud);
	vector<pcl::PointIndices> cluster_indices;
	pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
	ec.setClusterTolerance(0.02); // 2cm

	ec.setMinClusterSize(100);
	ec.setMaxClusterSize(25000);
	ec.setSearchMethod(tree);
	ec.setInputCloud(cloud);
	ec.extract(cluster_indices);

	int num_cluster = cluster_indices.size();
	cloud_cluster = vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>(num_cluster);
	int cluster_idx = 0;
	for (vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
	{
		cloud_cluster[cluster_idx] = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
		for (vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit)
			cloud_cluster[cluster_idx]->points.push_back (cloud->points[*pit]); 
		cloud_cluster[cluster_idx]->width = cloud_cluster[cluster_idx]->points.size ();
		cloud_cluster[cluster_idx]->height = 1;
		cloud_cluster[cluster_idx]->is_dense = true;
		cluster_idx++;
	}
}

void Explorer::ReadTargetTemplate(pcl::PointCloud<pcl::PointXYZ>::Ptr& target_template_ptr)
{
	int target_template_size = 227;
	int target_template_dim = 4;
	Mat target_template = Mat::zeros(target_template_size, target_template_dim, CV_64F);
	FileIO::ReadMatDouble(target_template, target_template_size, target_template_dim, "D:/Document/HKUST/Year 5/Research/Solutions/unified_framework/ball.bin");
	Mat2PCD(target_template, target_template_ptr);
}

void Explorer::DetectTargetByTemplate(Mat& target_mat, Mat& ball)
{
	char input_dir[400];
	sprintf(input_dir, "D:/Document/HKUST/Year 5/Research/Data/PointClouds/target_templates/mar_12_2016/pos_4/3.pcd");
	pcl::PointCloud<pcl::PointXYZ>::Ptr target_template_ptr(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> cloud_cluster;
	ReadTargetTemplate(target_template_ptr);
	ReadCloudFromPCD(input_dir, cloud);
	EuclideanDistanceClustering(cloud, cloud_cluster);
	int num_cluster = cloud_cluster.size();

	int matched_template_index = 0;
	double min_fitness_score = 10000.0;
	pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
	icp.setInputTarget(target_template_ptr);
	for(int cluster_idx = 0; cluster_idx < num_cluster; cluster_idx++)
	{
		if(cloud_cluster[cluster_idx]->points.size() < 5000)
		{
			pcl::PointCloud<pcl::PointXYZ> aligned;
			icp.setInputCloud(cloud_cluster[cluster_idx]);
			icp.align(aligned);
			double score = icp.getFitnessScore();
			cout << "has converged:" << icp.hasConverged() << " score: " << score << std::endl;
			if(score < min_fitness_score)
			{
				matched_template_index = cluster_idx;
				min_fitness_score = score;
			}
		}
	}

	pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
	ne.setInputCloud (cloud_cluster[matched_template_index]);
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ> ());
	ne.setSearchMethod (tree);
	pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);
	// Use all neighbors in a sphere of radius 3cm
	ne.setRadiusSearch (0.01);
	// Compute the features
	ne.compute (*normals);

	pcl::PointCloud<pcl::Boundary> boundaries;
	pcl::BoundaryEstimation<pcl::PointXYZ, pcl::Normal, pcl::Boundary> est;
	est.setInputCloud(cloud_cluster[matched_template_index]);
	est.setInputNormals(normals);
	est.setRadiusSearch (0.01);   // 2cm radius
	est.setSearchMethod(pcl::search::KdTree<pcl::PointXYZ>::Ptr (new pcl::search::KdTree<pcl::PointXYZ>));
	est.compute(boundaries);

	int size = cloud_cluster[matched_template_index]->points.size();
	int dim = 4;
	int boundary_points_count = 0;
	target_mat = Mat::zeros(size, dim, CV_64F);
	for(int i = 0; i < size; i++)
	{
		
		if(boundaries.points[i].boundary_point != 0)
		{
			target_mat.at<double>(boundary_points_count, 0) = cloud_cluster[matched_template_index]->points[i].x;
			target_mat.at<double>(boundary_points_count, 1) = cloud_cluster[matched_template_index]->points[i].y;
			target_mat.at<double>(boundary_points_count, 2) = cloud_cluster[matched_template_index]->points[i].z;
			target_mat.at<double>(boundary_points_count, 3) = 1.0;
			boundary_points_count++;
		}
	}
	target_mat = target_mat.rowRange(0, boundary_points_count);
	// cout << boundary_points_count << endl;
	PCD2Mat(cloud_cluster[matched_template_index], ball);
}

/*stringstream ss;
ss << "clusters/cloud_cluster_" << j << ".bin";
Mat cloud_mat;
PCD2Mat(cloud_cluster[j], cloud_mat);
FileIO::WriteMatDouble(cloud_mat, cloud_mat.rows, cloud_mat.cols, ss.str());*/
// writer.write<pcl::PointXYZ> (ss.str (), *cloud_cluster[j], false);