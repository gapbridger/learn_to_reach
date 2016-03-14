#ifndef LOADER_H
#define LOADER_H 

#include "../inc/fio.h"
#include "../inc/general_transform.h"

#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/calib3d/calib3d_c.h"
#include "opencv2/highgui/highgui_c.h"

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

typedef std::vector<double> fL;

class Loader
{

private:
	int num_weights_; // output dimension always 1...
	int diagnosis_number_;
	int trend_number_;
	int trial_id_;
	int feature_dim_;
	int num_segments_;
	// trained weights for testing
	char** test_weights_dir_;
	char* test_probabilities_dir_; 
	char* diagnosis_probabilities_dir_;
	char* diagnosis_weights_dir_;
	char** trend_dir_;
	char common_data_prefix_[400];
	char common_output_prefix_[400];
	char common_diagnosis_prefix_[400];

	double width_shift_;
	double height_shift_;
	double width_scale_;
	double height_scale_;

public:
	// Initialization
	Loader(int weight_number, int num_joints, int input_dim, int trend_number, int dir_id, char* dir);
	void LoadProprioception(int num_train_data, int num_test_data, cv::Mat& train_prop, cv::Mat& test_prop, cv::Mat& home_prop, cv::Mat& train_target_idx, cv::Mat& test_target_idx, const cv::Mat& joint_idx);
	void RecordSiftKeyPoints();
	void LoadLearningRates(GeneralTransform& transform);

	void LoadPointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PCDReader& reader, int idx);
	void LoadBinaryPointCloud(cv::Mat& cloud, int idx);
	void SavePointCloudAsBinaryMat(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, int idx);
	void LoadWeightsForTest(Transform& transform);
	void SaveWeightsForTest(Transform& transform);
	void SaveWeightsForDiagnosis(Transform& transform, int diagnosis_idx);
	void LoadWeightsForDiagnosis(Transform& transform, int diagnosis_idx);
	void SaveTrend(std::vector<std::vector<double>>& trend_array, int trend_number, int append_flag);
	void FormatWeightsForTestDirectory();
	void FormatWeightsForDiagnosisDirectory();
	void FormatTrendDirectory();
	void AppendTrendName(char** trend_dir_str_array);
	void LoadCloudProbabilities(cv::Mat& probabilities, bool diagnosis_flag, int diagnosis_idx = 0);
	void FormatProbabilitiesForTestDirectory();
	void FormatProbabilitiesForDiagnosisDirectory();
	void LoadCloudNormal(cv::Mat& normal, int idx);
	void SaveFeature(cv::Mat& feature, int idx, bool train_flag);
	void SaveTransform(std::vector<cv::Mat>& transform, int idx, bool train_flag);
	void SaveHomeCloudLabelDiagnosis(cv::Mat& label, int diagnosis_idx);
};

#endif


/*void FormatWeightsForTestDirectory();
void FormatWeightsForDiagnosisDirectory();
void FormatTrendDirectory();
void LoadWeightsForTest(GeneralTransform& transform, int output_dim, int input_dim);
void SaveWeightsForTest(GeneralTransform& transform, int output_dim, int input_dim);
void SaveWeightsForDiagnosis(GeneralTransform& transform, ellipse::Ellipse& ellipse, int output_dim, int input_dim, int diagnosis_idx);
void LoadWeightsForDiagnosis(GeneralTransform& transform, ellipse::Ellipse& ellipse, int output_dim, int input_dim, int diagnosis_idx);
void SaveTrend(fL* trend_array, int trend_number, int append_flag);
void SaveEllipse(ellipse::Ellipse& ellipse);
void LoadEllipse(ellipse::Ellipse& ellipse);
void ReadImage(int frame_idx, cv::Mat& disp_img);
void LoadSiftKeyPoint(cv::Mat& descriptors, cv::Mat& key_points, int frame_idx, cv::Mat& ini_transformation);
void LoadAllSiftKeyPoint(MatL& descriptors, MatL& key_points, int start_idx, int end_idx, cv::Mat& ini_transformation);
	
void AppendTestWeightName(char** dir_str_array);	
void AppendDiagnosisWeightName(char** dir_str_array);
void AppendTrendName(char** dir_str_array);	*/