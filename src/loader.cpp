#include "../inc/loader.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/types_c.h"


Loader::Loader(int num_weights, int num_segments, int feature_dim, int trend_number, int trial_id, char* dataset)
{
	// initialize weights
	num_weights_ = num_weights;
	feature_dim_ = feature_dim;
	// diagnosis_number_ = weight_number;
	trend_number_ = trend_number;  
	trial_id_ = trial_id;
	num_segments_ = num_segments;
	int len = 400;
	test_weights_dir_ = new char*[num_segments];  
	for(int i = 0; i < num_segments; i++)
		test_weights_dir_[i] = new char[len]; 
	diagnosis_weights_dir_ = new char[len];  
	test_probabilities_dir_ = new char[len];  
	diagnosis_probabilities_dir_ = new char[len];  
	trend_dir_ = new char*[trend_number_]; 
	for(int i = 0; i < trend_number_; i++)
		trend_dir_[i] = new char[len]; 	

	sprintf(common_output_prefix_, "D:/Document/HKUST/Year 5/Research/Solutions/unified_framework/output/para_%d/", trial_id_);
	sprintf(common_diagnosis_prefix_, "D:/Document/HKUST/Year 5/Research/Solutions/unified_framework/output/diagnosis_%d/", trial_id_);
	sprintf(common_data_prefix_, "D:/Document/HKUST/Year 5/Research/Data/PointClouds/");
	strcat(common_data_prefix_, dataset);
	strcat(common_data_prefix_, "/"); // march 10 2014/"); // feb 23

}

void Loader::LoadWeightsForTest(Transform& transform)
{
	cv::Mat current_weight = cv::Mat::zeros(num_weights_, feature_dim_, CV_64F);	
	for(int i = 0; i < num_segments_; i++)
	{
		FileIO::ReadMatDouble(current_weight, num_weights_, feature_dim_, test_weights_dir_[i]); 
		transform.set_w(current_weight, i);
	}
}

void Loader::SaveWeightsForTest(Transform& transform)
{
	cv::Mat current_weight = cv::Mat::zeros(num_weights_, feature_dim_, CV_64F);
	for(int i = 0; i < num_segments_; i++)
	{
		current_weight = transform.w(i);
		FileIO::WriteMatDouble(current_weight, num_weights_, feature_dim_, test_weights_dir_[i]);	
	}
}

void Loader::SaveWeightsForDiagnosis(Transform& transform, int diagnosis_idx)
{
	char tmp_dir[20];	
	for(int i = 0; i < num_segments_; i++)
	{
		FormatWeightsForDiagnosisDirectory();
		sprintf(tmp_dir, "_%d_%d.bin", diagnosis_idx, i);	
		strcat(diagnosis_weights_dir_, tmp_dir);	
		cv::Mat current_weight = cv::Mat::zeros(num_weights_, feature_dim_, CV_64F);
		current_weight = transform.w(i);
		FileIO::WriteMatDouble(current_weight, num_weights_, feature_dim_, diagnosis_weights_dir_);
	}
}

void Loader::SaveHomeCloudLabelDiagnosis(cv::Mat& label, int diagnosis_idx)
{
	char tmp_dir[20];	
	memset(diagnosis_weights_dir_, 0, sizeof(diagnosis_weights_dir_));
	strcpy(diagnosis_weights_dir_, common_diagnosis_prefix_);		
	strcat(diagnosis_weights_dir_, "home_cloud_label");	

	sprintf(tmp_dir, "_%d.bin", diagnosis_idx);	
	strcat(diagnosis_weights_dir_, tmp_dir);	
	FileIO::WriteMatDouble(label, label.rows, label.cols, diagnosis_weights_dir_);
}

void Loader::LoadWeightsForDiagnosis(Transform& transform, int diagnosis_idx)
{
	char tmp_dir[20];	
	
	cv::Mat current_weight = cv::Mat::zeros(num_weights_, feature_dim_, CV_64F);
	for(int i = 0; i < num_segments_; i++)
	{
		FormatWeightsForDiagnosisDirectory();
		sprintf(tmp_dir, "_%d_%d.bin", diagnosis_idx, i);	
		strcat(diagnosis_weights_dir_, tmp_dir);	
		FileIO::ReadMatDouble(current_weight, num_weights_, feature_dim_, diagnosis_weights_dir_); 
		transform.set_w(current_weight, i);	
		// current_weight = transform.w(i);
		// FileIO::WriteMatDouble(current_weight, num_weights_, feature_dim_, test_weights_dir_[i]);	
	}
	
}

// save value trend: either output average value or weight norm...
void Loader::SaveTrend(std::vector<std::vector<double>>& trend_array, int trend_number, int append_flag)
{
  // FormatValueDirectory(dir_idx, true); // should be called in constructor
	int data_len = 0;
	cv::Mat data;
	for(int i = 0; i < trend_number; i++){
		data_len = trend_array[i].size();
		data = cv::Mat::zeros(data_len, 1, CV_64F);
		for(int j = 0; j < data_len; j++)
			data.at<double>(j, 0) = trend_array[i][j];
		FileIO::RecordMatDouble(data, data_len, 1, trend_dir_[i], append_flag); 
	}
}
// format test weight directory (final weight loaded for test)
void Loader::FormatWeightsForTestDirectory()
{
	for(int i = 0; i < num_segments_; i++)
	{
		memset(test_weights_dir_[i], 0, sizeof(test_weights_dir_));
		strcpy(test_weights_dir_[i], common_output_prefix_);	
		char tmp_dir[10];
		sprintf(tmp_dir, "w_%d.bin", i);
		strcat(test_weights_dir_[i], tmp_dir);
	}
}

void Loader::FormatWeightsForDiagnosisDirectory()
{
	memset(diagnosis_weights_dir_, 0, sizeof(diagnosis_weights_dir_));
	strcpy(diagnosis_weights_dir_, common_diagnosis_prefix_);		
	strcat(diagnosis_weights_dir_, "w");		
}

// format trend directory
void Loader::FormatTrendDirectory()
{
	// char dir_idx_str[5];
	// sprintf(dir_idx_str, "%d/", dir_id_);
	for(int i = 0; i < trend_number_; i++)
	{
		memset(&trend_dir_[i][0], 0, sizeof(trend_dir_[i]));
		strcpy(trend_dir_[i], common_output_prefix_); // "D:/Document/HKUST/Year 5/Research/Solutions/expansion/output/para_"
		// strcat(trend_dir_[i], dir_idx_str);
	}
	AppendTrendName(trend_dir_);
	for(int i = 0; i < trend_number_; i++)	
		strcat(trend_dir_[i], "_trend.bin");
		
}

void Loader::AppendTrendName(char** trend_dir_str_array)
{
	int dim_transform = 3;
	if(trend_dir_str_array != NULL)
	{
		for(int i = 0; i < num_segments_; i++)
		{
			for(int j = 0; j < num_weights_; j++)
			{
				char curr_w_dir[10];
				sprintf(curr_w_dir, "w_%d_%d", i, j);
				strcat(trend_dir_str_array[i * num_weights_ + j], curr_w_dir);
			}
		}
		strcat(trend_dir_str_array[num_segments_ * num_weights_], "idx");
		int trend_count = num_segments_ * num_weights_ + 1;
		for(int segment_idx = 0; segment_idx < num_segments_; segment_idx++)
		{
			char curr_dir[10];
			sprintf(curr_dir, "evar_%d", segment_idx);
			strcat(trend_dir_str_array[trend_count + segment_idx], curr_dir);
		}
		trend_count = trend_count + num_segments_;
		
		int num_corr = (num_segments_) * dim_transform;
		for(int curr_count = trend_count; curr_count < trend_count + num_corr; curr_count++)
		{
			int row_idx = (curr_count - trend_count) / dim_transform;
			int col_idx = (curr_count - trend_count) % dim_transform;
			char curr_dir[10];
			sprintf(curr_dir, "corr_%d_%d", row_idx, col_idx);
			strcat(trend_dir_str_array[curr_count], curr_dir);
		}
	}
}

void Loader::LoadLearningRates(GeneralTransform& transform) // empty parameter, need to be specialized
{
	char input_dir[400];	
	int n_w = 1;
	cv::Mat rates = cv::Mat::zeros(2, 1, CV_64F);
	sprintf(input_dir, "D:/Document/HKUST/Year 5/Research/Solutions/unified_framework/input/rate_%d.bin", trial_id_);
	FileIO::ReadMatDouble(rates, 2, 1, input_dir);
	// need to write set learning rates routine here
	transform.set_w_rate(rates.at<double>(0, 0));
	transform.set_w_natural_rate(rates.at<double>(1, 0));
	std::cout << "learning rates: ";
	for(int i = 0; i < 2; i++)
		std::cout << rates.at<double>(i, 0) << " ";
	std::cout << std::endl;	
}

void Loader::LoadProprioception(int train_data_size, int test_data_size, cv::Mat& train_prop, cv::Mat& test_prop, cv::Mat& home_prop, cv::Mat& train_target_idx, cv::Mat& test_target_idx, const cv::Mat& joint_idx)
{
	char input_dir[400];
	char prop_dir[40];
	int num_joints = train_prop.cols;
	
	for(int i = 0; i < num_joints; i++)
	{
		if(train_data_size != 0)
		{
			cv::Mat p_tmp_train = cv::Mat::zeros(train_data_size, 1, CV_64F);
			strcpy(input_dir, common_data_prefix_);
			sprintf(prop_dir, "train_p%d.bin", (int)joint_idx.at<double>(i, 0));
			strcat(input_dir, prop_dir);	
			FileIO::ReadFloatMatToDouble(p_tmp_train, train_data_size, 1, input_dir);
			p_tmp_train.copyTo(train_prop.colRange(i, i + 1));
		}

		if(test_data_size != 0)
		{
			cv::Mat p_tmp_test = cv::Mat::zeros(test_data_size, 1, CV_64F);
			strcpy(input_dir, common_data_prefix_);
			sprintf(prop_dir, "test_p%d.bin", (int)joint_idx.at<double>(i, 0));
			strcat(input_dir, prop_dir);	
			FileIO::ReadFloatMatToDouble(p_tmp_test, test_data_size, 1, input_dir);
			p_tmp_test.copyTo(test_prop.colRange(i, i + 1));
		}
	}
	
	strcpy(input_dir, common_data_prefix_);
	strcat(input_dir, "prop_home.bin");	
	FileIO::ReadFloatMatToDouble(home_prop, 1, num_joints, input_dir);
	
	// train frame index
	if(train_data_size != 0)
	{
		strcpy(input_dir, common_data_prefix_);
		strcat(input_dir, "train_prop_idx.bin");	
		FileIO::ReadFloatMatToDouble(train_target_idx, train_data_size, 1, input_dir);
	}
	// test frame index
	if(test_data_size != 0)
	{
		strcpy(input_dir, common_data_prefix_);
		strcat(input_dir, "test_prop_idx.bin");	
		FileIO::ReadFloatMatToDouble(test_target_idx, test_data_size, 1, input_dir);
	}
}

void Loader::LoadPointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PCDReader& reader, int idx)
{	
	char tmp_dir[40];
	char input_dir[400];
	sprintf(tmp_dir, "pcd/%d.pcd", idx);
	strcpy(input_dir, common_data_prefix_);
	strcat(input_dir, tmp_dir);
	reader.read(input_dir, *cloud);	
}

void Loader::LoadBinaryPointCloud(cv::Mat& cloud, int idx)
{	
	char tmp_dir[40];
	char input_dir[400];
	sprintf(tmp_dir, "binary_background/size_%d.bin", idx);
	strcpy(input_dir, common_data_prefix_);
	strcat(input_dir, tmp_dir);
	cv::Mat size_mat = cv::Mat::zeros(1, 1, CV_64F);
	FileIO::ReadMatDouble(size_mat, 1, 1, input_dir);
	int cloud_size = size_mat.at<double>(0, 0);	
	int dim = 4;
	cloud = cv::Mat::ones(cloud_size, dim, CV_64F);	
	sprintf(tmp_dir, "binary_background/%d.bin", idx);
	strcpy(input_dir, common_data_prefix_);
	strcat(input_dir, tmp_dir);
	FileIO::ReadMatDouble(cloud.colRange(0, dim - 1), cloud_size, dim - 1, input_dir);		
}

void Loader::FormatProbabilitiesForTestDirectory()
{
	memset(test_probabilities_dir_, 0, sizeof(test_probabilities_dir_));
	strcpy(test_probabilities_dir_, common_output_prefix_);		
	strcat(test_probabilities_dir_, "probabilities.bin");		
}

void Loader::FormatProbabilitiesForDiagnosisDirectory()
{
	memset(diagnosis_probabilities_dir_, 0, sizeof(diagnosis_probabilities_dir_));
	strcpy(diagnosis_probabilities_dir_, common_diagnosis_prefix_);		
	strcat(diagnosis_probabilities_dir_, "probabilities");		
}

void Loader::LoadCloudProbabilities(cv::Mat& probabilities, bool diagnosis_flag, int diagnosis_idx)
{
	if(diagnosis_flag == true)
	{
		char tmp_dir[20];	
		FormatProbabilitiesForDiagnosisDirectory();
		sprintf(tmp_dir, "_%d.bin", diagnosis_idx);	
		strcat(diagnosis_probabilities_dir_, tmp_dir);	
		FileIO::ReadMatDouble(probabilities, probabilities.rows, probabilities.cols, diagnosis_probabilities_dir_); 
	}
	else
	{
		FormatProbabilitiesForTestDirectory();
		FileIO::ReadMatDouble(probabilities, probabilities.rows, probabilities.cols, test_probabilities_dir_);
	}
}

void Loader::SavePointCloudAsBinaryMat(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, int idx)
{	
	char tmp_dir[40];
	char output_dir[400];
	int dim = 3;	
	int cloud_size = cloud->points.size();
	cv::Mat cloud_mat = cv::Mat::zeros(cloud_size, dim, CV_64F);
	for(int i = 0; i < cloud_size; i++)
	{
		cloud_mat.at<double>(i, 0) = cloud->points[i].x;
		cloud_mat.at<double>(i, 1) = cloud->points[i].y;
		cloud_mat.at<double>(i, 2) = cloud->points[i].z;
	}
	sprintf(tmp_dir, "binary_background/%d.bin", idx);
	strcpy(output_dir, common_data_prefix_);
	strcat(output_dir, tmp_dir);
	FileIO::WriteMatDouble(cloud_mat, cloud_size, dim, output_dir);
	// save size...
	sprintf(tmp_dir, "binary_background/size_%d.bin", idx);
	strcpy(output_dir, common_data_prefix_);
	strcat(output_dir, tmp_dir);
	cv::Mat size_mat = cv::Mat::zeros(1, 1, CV_64F);
	size_mat.at<double>(0, 0) = cloud_size;
	FileIO::WriteMatDouble(size_mat, 1, 1, output_dir);
}

void Loader::LoadCloudNormal(cv::Mat& normal, int idx)
{	
	char tmp_dir[40];
	char input_dir[400];
	sprintf(tmp_dir, "binary/size_%d.bin", idx);
	strcpy(input_dir, common_data_prefix_);
	strcat(input_dir, tmp_dir);
	cv::Mat size_mat = cv::Mat::zeros(1, 1, CV_64F);
	FileIO::ReadMatDouble(size_mat, 1, 1, input_dir);
	int cloud_size = size_mat.at<double>(0, 0);	
	int dim = 3;
	normal = cv::Mat::ones(cloud_size, dim, CV_64F);	
	sprintf(tmp_dir, "binary_normal/normal_%d.bin", idx);
	strcpy(input_dir, common_data_prefix_);
	strcat(input_dir, tmp_dir);
	FileIO::ReadMatDouble(normal, cloud_size, dim, input_dir);		
}

void Loader::SaveFeature(cv::Mat& feature, int idx, bool train_flag)
{
	char output_dir[400];
	char curr_dir[50];

	if(train_flag)
	{
		sprintf(curr_dir, "feature/train_prop_feature_%d.bin", idx);
	}
	else
	{
		sprintf(curr_dir, "feature/test_prop_feature_%d.bin", idx);
	}
	strcpy(output_dir, common_data_prefix_);
	strcat(output_dir, curr_dir);
	FileIO::WriteMatDouble(feature, feature.rows, feature.cols, output_dir);
}

void Loader::SaveTransform(std::vector<cv::Mat>& transform, int idx, bool train_flag)
{
	char output_dir[400];
	char curr_dir[50];

	int num_segments = transform.size();
	for(int i = 0; i < num_segments; i++)
	{
		if(train_flag)
		{
			sprintf(curr_dir, "transform/train_transform_%d_%d.bin", idx, i);
		}
		else
		{
			sprintf(curr_dir, "transform/test_transform_%d_%d.bin", idx, i);
		}
		strcpy(output_dir, common_output_prefix_);
		strcat(output_dir, curr_dir);
		FileIO::WriteMatDouble(transform[i], transform[i].rows, transform[i].cols, output_dir);	
	}
}