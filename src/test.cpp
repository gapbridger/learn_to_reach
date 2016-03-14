#include <iostream>
#include "../inc/test.h"
#include "../inc/point_to_point_segmentation.h"
#include "../inc/rigid_transform_3d.h"
#include "../inc/point_to_point_segmentation_with_weights.h"

using namespace cv;
using namespace std;

void UnifiedLearningTest::SetUp()
{
	test_data_dir_prefix_ = new char[200];
	sprintf(test_data_dir_prefix_, "D:/Document/HKUST/Year 5/Research/Solutions/unified_framework/test/");
	double_epsilon_ = 1e-5;
}

void UnifiedLearningTest::TearDown()
{
	delete test_data_dir_prefix_;
}

typedef vector<Mat> (RigidTransform3D::*Getter)() const;
typedef void (RigidTransform3D::*Setter)(const vector<Mat>&);

double CalcNumericalGradient(Getter getter, Setter setter, RigidTransform3D& transform, vector<Mat>& feature, const vector<vector<Mat>>& query_cloud, vector<vector<Mat>>& predicted_cloud, vector<vector<Mat>>& target_cloud, const Mat& point_weight, int num_transform, int dim_feature, int feature_dim_idx, double disturb_value);

/*TEST_F(UnifiedLearningTest, TestCalcFeature2D)
{
	int dim_sinusoidal = 3;
	int num_joints_2d = 2;
	int dim_feature_2d = pow((double)dim_sinusoidal, num_joints_2d) - 1;

	char test_prop_2d_dir[200];
	char test_prop_feature_2d_dir[200];
	strcpy(test_prop_2d_dir, test_data_dir_prefix_);
	strcat(test_prop_2d_dir, "test_prop_2d.bin");
	strcpy(test_prop_feature_2d_dir, test_data_dir_prefix_);
	strcat(test_prop_feature_2d_dir, "test_prop_feature_2d.bin");

	// column vectors
	cv::Mat test_prop_2d = cv::Mat::zeros(1, num_joints_2d, CV_64F);
	cv::Mat test_prop_feature_2d = cv::Mat::zeros(dim_feature_2d, 1, CV_64F);
	cv::Mat expected_test_prop_feature_2d = cv::Mat::zeros(dim_feature_2d, 1, CV_64F);
	cv::Mat feature_zero = cv::Mat::zeros(dim_feature_2d, 1, CV_64F);

	FileIO::ReadMatDouble(test_prop_2d, 1, num_joints_2d, test_prop_2d_dir);
	FileIO::ReadMatDouble(expected_test_prop_feature_2d, dim_feature_2d, 1, test_prop_feature_2d_dir);

	Explorer::SetFeature(test_prop_feature_2d, feature_zero, num_joints_2d, test_prop_2d);
	for(int i = 0; i < dim_feature_2d; i++)
	{
		EXPECT_NEAR(expected_test_prop_feature_2d.at<double>(i, 0), test_prop_feature_2d.at<double>(i, 0), double_epsilon_);
	}
}*/

TEST_F(UnifiedLearningTest, TestCalcFeature3D)
{
	int dim_sinusoidal = 3;
	int num_joints_3d = 3;
	int dim_feature_3d = pow((double)dim_sinusoidal, num_joints_3d) - 1;

	char test_prop_3d_dir[200];
	char test_prop_feature_3d_dir[200];
	strcpy(test_prop_3d_dir, test_data_dir_prefix_);
	strcat(test_prop_3d_dir, "test_prop_3d.bin");
	strcpy(test_prop_feature_3d_dir, test_data_dir_prefix_);
	strcat(test_prop_feature_3d_dir, "test_prop_feature_3d.bin");

	// column vectors
	Mat test_prop_3d = Mat::zeros(1, num_joints_3d, CV_64F);
	Mat test_prop_feature_3d = Mat::zeros(dim_feature_3d, 1, CV_64F);
	Mat expected_test_prop_feature_3d = Mat::zeros(dim_feature_3d, 1, CV_64F);
	Mat feature_zero = Mat::zeros(dim_feature_3d, 1, CV_64F);

	FileIO::ReadMatDouble(test_prop_3d, 1, num_joints_3d, test_prop_3d_dir);
	FileIO::ReadMatDouble(expected_test_prop_feature_3d, dim_feature_3d, 1, test_prop_feature_3d_dir);

	Explorer::SetFeature(test_prop_feature_3d, feature_zero, num_joints_3d, test_prop_3d);
	for(int i = 0; i < dim_feature_3d; i++)
	{
		EXPECT_NEAR(expected_test_prop_feature_3d.at<double>(i, 0), test_prop_feature_3d.at<double>(i, 0), double_epsilon_);
	}
}
/*
TEST_F(UnifiedLearningTest, TestCalculateGradient)
{
	char test_query_cloud_dir[200];
	char test_target_cloud_dir[200];
	int num_joint = 1; // currently only have data for 1 joint
	int num_cloud_points = 1000;
	int dim_transform = 4;
	int dim_feature = pow(3.0, num_joint) - 1;
	int num_transform_elements = dim_transform * (dim_transform - 1);
	double disturb_value = 0.0001;
	double numerical_gradient = 0;
	double analytical_gradient = 0;

	std::vector<cv::Mat> predicted_cloud(num_joint);   
    std::vector<cv::Mat> target_cloud(num_joint); 
    std::vector<cv::Mat> query_cloud(num_joint);   
	std::vector<cv::Mat> w_grad(num_joint);   

	for(int i = 0; i < num_joint; i++)
	{
		predicted_cloud[i] = cv::Mat::zeros(num_cloud_points, dim_transform, CV_64F);
		target_cloud[i] = cv::Mat::zeros(num_cloud_points, dim_transform, CV_64F);
		query_cloud[i] = cv::Mat::zeros(num_cloud_points, dim_transform, CV_64F);
		w_grad[i] = cv::Mat::zeros(num_transform_elements, dim_feature, CV_64F);
	}

	strcpy(test_query_cloud_dir, test_data_dir_prefix_);
	strcat(test_query_cloud_dir, "test_query_cloud.bin");
	strcpy(test_target_cloud_dir, test_data_dir_prefix_);
	strcat(test_target_cloud_dir, "test_target_cloud.bin");

	FileIO::ReadMatDouble(target_cloud[0], num_cloud_points, dim_transform, test_target_cloud_dir);
	FileIO::ReadMatDouble(query_cloud[0], num_cloud_points, dim_transform, test_query_cloud_dir);

	cv::Mat feature = cv::Mat::zeros(dim_feature, 1, CV_64F);
	cv::randu(feature, cv::Scalar::all(0), cv::Scalar::all(1));
	GeneralTransform transform(dim_transform, num_joint, 0.01);
	transform.CalcTransformation(feature);
	transform.TransformCloud(query_cloud[0], transform.get_transform(), predicted_cloud);
	transform.CalculateGradient(target_cloud, predicted_cloud, query_cloud, feature);
	w_grad = transform.w_grad();

	cv::Mat diff; // , filtered_diff, filtered_query_cloud;
	cv::Mat disturb, dist, new_w;
	double e_1, e_2;
	for(int idx = 0; idx < num_joint; idx++)
	{
		for(int i = 0; i < num_transform_elements; i++)
		{
			for(int j = 0; j < dim_feature; j++)
			{
				disturb = cv::Mat::zeros(num_transform_elements, dim_feature, CV_64F);
				disturb.at<double>(i, j) = disturb_value;
				// e_1
				new_w = transform.get_w(idx) + disturb;
				transform.set_w(new_w, idx);
				transform.CalcTransformation(feature);
				transform.TransformCloud(query_cloud[0], transform.get_transform(), predicted_cloud);
				diff = predicted_cloud[idx] - target_cloud[idx];
				cv::reduce(diff.mul(diff) / 2, diff, 1, CV_REDUCE_SUM);
				cv::reduce(diff, dist, 0, CV_REDUCE_AVG);
				e_1 = dist.at<double>(0, 0);

				// e_2
				new_w = transform.get_w(idx) - 2 * disturb;
				transform.set_w(new_w, idx);
				transform.CalcTransformation(feature);
				transform.TransformCloud(query_cloud[0], transform.get_transform(), predicted_cloud);
				diff = predicted_cloud[idx] - target_cloud[idx];
				cv::reduce(diff.mul(diff) / 2, diff, 1, CV_REDUCE_SUM);
				cv::reduce(diff, dist, 0, CV_REDUCE_AVG);
				e_2 = dist.at<double>(0, 0);

				new_w = transform.get_w(idx) + disturb;
				transform.set_w(new_w, idx);

				numerical_gradient = (e_1 - e_2) / (2 * disturb_value);
				analytical_gradient = w_grad[idx].at<double>(i, j);

				EXPECT_NEAR(numerical_gradient, analytical_gradient, double_epsilon_);
			}
		}
	}
}

TEST_F(UnifiedLearningTest, TestBuildModelGraph)
{
	// load testing home cloud
	char test_input_dir[400];
	int cloud_size;
	int dim = 4;
	double neighborhood_range = 1e-2;
	double max_num_neighbor = 20;
	cv::Mat home_cloud_indices, home_cloud_min_dists;
	strcpy(test_input_dir, test_data_dir_prefix_);
	strcat(test_input_dir, "test_home_cloud_size.bin");
	cv::Mat size_mat = cv::Mat::zeros(1, 1, CV_64F);
	FileIO::ReadMatDouble(size_mat, 1, 1, test_input_dir);
	cloud_size = size_mat.at<double>(0, 0);	
	
	cv::Mat test_home_cloud = cv::Mat::ones(cloud_size, dim, CV_64F);	
	strcpy(test_input_dir, test_data_dir_prefix_);
	strcat(test_input_dir, "test_home_cloud.bin");
	FileIO::ReadMatDouble(test_home_cloud.colRange(0, dim - 1), cloud_size, dim - 1, test_input_dir);		

	Explorer::BuildModelGraph(test_home_cloud, 1, home_cloud_indices, home_cloud_min_dists, neighborhood_range, max_num_neighbor);

	// randomly assert several values...
	EXPECT_EQ(test_home_cloud.rows, home_cloud_indices.rows);
	EXPECT_EQ(max_num_neighbor, home_cloud_indices.cols);
	EXPECT_EQ(test_home_cloud.rows, home_cloud_min_dists.rows);
	EXPECT_EQ(max_num_neighbor, home_cloud_min_dists.cols);

	strcpy(test_input_dir, test_data_dir_prefix_);
	strcat(test_input_dir, "test_home_cloud_neighbor_dists.bin");
	cv::Mat expected_test_home_cloud_neighbor_dists = cv::Mat::ones(cloud_size, max_num_neighbor, CV_64F);	
	FileIO::ReadMatDouble(expected_test_home_cloud_neighbor_dists, cloud_size, max_num_neighbor, test_input_dir);		

	for(int i = 0; i < cloud_size; i++)
	{
		for(int j = 0; j < max_num_neighbor; j++)
		{
			EXPECT_NEAR((float)expected_test_home_cloud_neighbor_dists.at<double>(i, j), home_cloud_min_dists.at<float>(i, j), double_epsilon_);
		}
	}
}*/


TEST_F(UnifiedLearningTest, TestFeatureGradient)
{
	int num_joints = 3; // currently only have data for 1 joint
	int dim_feature = pow(3.0, num_joints) - 1;
	Mat joint_angle = Mat::zeros(1, num_joints, CV_64F);
	joint_angle.at<double>(0, 0) = -40;
	joint_angle.at<double>(0, 1) = 40;
	joint_angle.at<double>(0, 2) = 40;
	Mat home_joint_angle = Mat::zeros(1, num_joints, CV_64F);
	home_joint_angle.at<double>(0, 0) = -35;
	home_joint_angle.at<double>(0, 1) = 35;
	home_joint_angle.at<double>(0, 2) = 35;
	Mat feature_zero = Mat::zeros(dim_feature, 1, CV_64F);
	Mat home_feature = Mat::zeros(dim_feature, 1, CV_64F);
	Mat feature = Mat::zeros(dim_feature, 1, CV_64F);
	Mat feature1 = Mat::zeros(dim_feature, 1, CV_64F);
	Mat feature2 = Mat::zeros(dim_feature, 1, CV_64F);
	Mat feature_gradient_analytical = Mat::zeros(dim_feature, num_joints, CV_64F);
	Mat feature_gradient_numerical = Mat::zeros(dim_feature, num_joints, CV_64F);


	Explorer::SetFeature(home_feature, feature_zero, num_joints, home_joint_angle);
	Explorer::SetFeature(feature, home_feature, num_joints, joint_angle);
	
	RigidTransform3D::CalculateFeatureGradient(joint_angle, feature_gradient_analytical, dim_feature);
	double disturb_value = 1e-5;
	
	for (int joint_idx = 0; joint_idx < num_joints; joint_idx++)
	{
		joint_angle.at<double>(0, joint_idx) += disturb_value;
		Explorer::SetFeature(feature1, home_feature, num_joints, joint_angle);
		joint_angle.at<double>(0, joint_idx) -= 2 * disturb_value;
		Explorer::SetFeature(feature2, home_feature, num_joints, joint_angle);
		Mat grad = (feature1 - feature2) / (2 * disturb_value);
		joint_angle.at<double>(0, joint_idx) += disturb_value;
		grad.copyTo(feature_gradient_numerical.colRange(joint_idx, joint_idx + 1));
	}
	for(int joint_idx = 0; joint_idx < num_joints; joint_idx++)
	{
		for(int dim_idx = 0; dim_idx < dim_feature; dim_idx++)
		{
			EXPECT_NEAR(feature_gradient_analytical.at<double>(dim_idx, joint_idx), feature_gradient_numerical.at<double>(dim_idx, joint_idx), 1e-6);
		}
	}
}

TEST_F(UnifiedLearningTest, TestCalculateJacobian)
{
	int dim_transform = 4;
	int num_joints = 3;
	int dim_feature = pow(3.0, num_joints) - 1;
	int batch_size =  1;
	double disturb_value = 1e-5;
	int weight_idx = 0;
	int batch_idx = 0;

	Mat curr_pos = Mat::ones(1, dim_transform, CV_64F);
	Mat transformed_pos = Mat::ones(1, dim_transform, CV_64F);
	Mat transformed_pos_1 = Mat::ones(1, dim_transform, CV_64F);
	Mat transformed_pos_2 = Mat::ones(1, dim_transform, CV_64F);
	
	RigidTransform3D transform(num_joints, dim_feature, 1, 0.01, 4);
	
	Mat joint_angle = Mat::zeros(1, num_joints, CV_64F);
	joint_angle.at<double>(0, 0) = -40.0;
	joint_angle.at<double>(0, 1) = 40.0;
	joint_angle.at<double>(0, 2) = 40.0;
	Mat home_joint_angle = Mat::zeros(1, num_joints, CV_64F);
	home_joint_angle.at<double>(0, 0) = -35;
	home_joint_angle.at<double>(0, 1) = 35;
	home_joint_angle.at<double>(0, 2) = 35;
	
	Mat feature_zero = Mat::zeros(dim_feature, 1, CV_64F);
	Mat home_feature = Mat::zeros(dim_feature, 1, CV_64F);
	Mat feature = Mat::zeros(dim_feature, 1, CV_64F);
	Mat feature_1 = Mat::zeros(dim_feature, 1, CV_64F);
	Mat feature_2 = Mat::zeros(dim_feature, 1, CV_64F);
	Explorer::SetFeature(home_feature, feature_zero, num_joints, home_joint_angle);

	Mat jacobian_numerical = Mat::zeros(dim_transform - 1, num_joints, CV_64F);
	Mat jacobian_analytical = Mat::zeros(dim_transform - 1, num_joints, CV_64F);

	Explorer::SetFeature(feature, home_feature, num_joints, joint_angle);
	transform.CalcSingleTransform(feature, weight_idx, batch_idx);
	transform.TransformPoint(curr_pos, transformed_pos, weight_idx, batch_idx);
	transform.CalculateJacobian(transformed_pos, joint_angle, feature, jacobian_analytical, weight_idx);
	
	for(int joint_idx = 0; joint_idx < num_joints; joint_idx++)
	{
		joint_angle.at<double>(0, joint_idx) += disturb_value;
		Explorer::SetFeature(feature_1, home_feature, num_joints, joint_angle);
		transform.CalcSingleTransform(feature_1, weight_idx, batch_idx);
		transform.TransformPoint(curr_pos, transformed_pos_1, weight_idx, batch_idx);
		joint_angle.at<double>(0, joint_idx) -= 2 * disturb_value;
		Explorer::SetFeature(feature_2, home_feature, num_joints, joint_angle);
		transform.CalcSingleTransform(feature_2, weight_idx, batch_idx);
		transform.TransformPoint(curr_pos, transformed_pos_2, weight_idx, batch_idx);
		Mat grad = (transformed_pos_1 - transformed_pos_2) / (2 * disturb_value);
		grad = grad.colRange(0, dim_transform - 1).t();
		joint_angle.at<double>(0, joint_idx) += disturb_value;
		grad.copyTo(jacobian_numerical.colRange(joint_idx, joint_idx + 1));
	}
	for(int joint_idx = 0; joint_idx < num_joints; joint_idx++)
	{
		for(int dim_idx = 0; dim_idx < dim_transform - 1; dim_idx++)
		{
			EXPECT_NEAR(jacobian_numerical.at<double>(dim_idx, joint_idx), jacobian_analytical.at<double>(dim_idx, joint_idx), 1e-7);
		}
	}
}

TEST_F(UnifiedLearningTest, TestRigidTransform3D)
{
	char test_query_cloud_dir[200];
	char test_target_cloud_dir[200];
	int num_transform = 1; // currently only have data for 1 joint
	int num_cloud_points = 1000;
	int dim_transform = 4;
	int dim_feature = pow(3.0, num_transform) - 1;
	int batch_size =  1;
	double disturb_value = 1e-5;
	double alpha = 4;
	vector<vector<Mat>> predicted_cloud(batch_size);
    vector<vector<Mat>> target_cloud(batch_size);
    vector<vector<Mat>> query_cloud(batch_size);

	vector<Mat> w_r_x_grad;
	vector<Mat> w_r_y_grad;
	vector<Mat> w_r_z_grad;
	vector<Mat> w_t_x_grad;
	vector<Mat> w_t_y_grad;
	vector<Mat> w_t_z_grad;

	vector<vector<Mat>> point_weight(batch_size);
	point_weight[0] = vector<Mat>(num_transform);
	point_weight[0][0] = Mat::ones(num_cloud_points, num_transform, CV_64F);
	// RNG rng(getTickCount());
	// rng.fill(point_weight[0][0], RNG::UNIFORM, 0, 1);


	predicted_cloud[0] = vector<Mat>(num_transform);
    target_cloud[0] = vector<Mat>(num_transform);
	query_cloud[0] = vector<Mat>(num_transform);
	for(int i = 0; i < num_transform; i++)
	{
		predicted_cloud[0][i] = Mat::zeros(num_cloud_points, dim_transform, CV_64F);
		target_cloud[0][i] = Mat::zeros(num_cloud_points, dim_transform, CV_64F);
		query_cloud[0][i] = Mat::zeros(num_cloud_points, dim_transform, CV_64F);
	}

	RigidTransform3D transform(num_transform, dim_feature, 1, 0.01, alpha);
	strcpy(test_query_cloud_dir, test_data_dir_prefix_);
	strcat(test_query_cloud_dir, "test_query_cloud.bin");
	strcpy(test_target_cloud_dir, test_data_dir_prefix_);
	strcat(test_target_cloud_dir, "test_target_cloud.bin");
	FileIO::ReadMatDouble(target_cloud[0][0], num_cloud_points, dim_transform, test_target_cloud_dir);
	FileIO::ReadMatDouble(query_cloud[0][0], num_cloud_points, dim_transform, test_query_cloud_dir);

	vector<Mat> feature(batch_size);
    feature[0] = Mat::zeros(dim_feature, 1, CV_64F);
	randu(feature[0], Scalar::all(0), Scalar::all(0.1));

	transform.CalcTransformation(feature);
	transform.TransformCloud(query_cloud[0][0], predicted_cloud);
	transform.CalculateGradient(target_cloud, predicted_cloud, query_cloud, point_weight, feature);

	w_r_x_grad = transform.w_r_x_grad();
	w_r_y_grad = transform.w_r_y_grad();
	w_r_z_grad = transform.w_r_z_grad();
	w_t_x_grad = transform.w_t_x_grad();
	w_t_y_grad = transform.w_t_y_grad();
	w_t_z_grad = transform.w_t_z_grad();

	Mat diff; // , filtered_diff, filtered_query_cloud;
	Mat disturb, dist;
	vector<Mat> w_r_x(num_transform);
	double e_1, e_2;
	for(int feature_dim_idx = 0; feature_dim_idx < dim_feature; feature_dim_idx++)
	{
		double w_r_x_g_n = CalcNumericalGradient(&RigidTransform3D::w_r_x, &RigidTransform3D::set_w_r_x, transform, feature, query_cloud, predicted_cloud, target_cloud, point_weight[0][0], num_transform, dim_feature, feature_dim_idx, disturb_value);
		double w_r_y_g_n = CalcNumericalGradient(&RigidTransform3D::w_r_y, &RigidTransform3D::set_w_r_y, transform, feature, query_cloud, predicted_cloud, target_cloud, point_weight[0][0], num_transform, dim_feature, feature_dim_idx, disturb_value);
		double w_r_z_g_n = CalcNumericalGradient(&RigidTransform3D::w_r_z, &RigidTransform3D::set_w_r_z, transform, feature, query_cloud, predicted_cloud, target_cloud, point_weight[0][0], num_transform, dim_feature, feature_dim_idx, disturb_value);
		double w_t_x_g_n = CalcNumericalGradient(&RigidTransform3D::w_t_x, &RigidTransform3D::set_w_t_x, transform, feature, query_cloud, predicted_cloud, target_cloud, point_weight[0][0], num_transform, dim_feature, feature_dim_idx, disturb_value);
		double w_t_y_g_n = CalcNumericalGradient(&RigidTransform3D::w_t_y, &RigidTransform3D::set_w_t_y, transform, feature, query_cloud, predicted_cloud, target_cloud, point_weight[0][0], num_transform, dim_feature, feature_dim_idx, disturb_value);
		double w_t_z_g_n = CalcNumericalGradient(&RigidTransform3D::w_t_z, &RigidTransform3D::set_w_t_z, transform, feature, query_cloud, predicted_cloud, target_cloud, point_weight[0][0], num_transform, dim_feature, feature_dim_idx, disturb_value);

		double w_r_x_g_a = w_r_x_grad[0].at<double>(0, feature_dim_idx);
		double w_r_y_g_a = w_r_y_grad[0].at<double>(0, feature_dim_idx);
		double w_r_z_g_a = w_r_z_grad[0].at<double>(0, feature_dim_idx);
		double w_t_x_g_a = w_t_x_grad[0].at<double>(0, feature_dim_idx);
		double w_t_y_g_a = w_t_y_grad[0].at<double>(0, feature_dim_idx);
		double w_t_z_g_a = w_t_z_grad[0].at<double>(0, feature_dim_idx);

		cout << w_r_x_g_n << " " << w_r_x_g_a << " " << endl;
		cout << w_r_y_g_n << " " << w_r_y_g_a << " " << endl;
		cout << w_r_z_g_n << " " << w_r_z_g_a << " " << endl;
		cout << w_t_x_g_n << " " << w_t_x_g_a << " " << endl;
		cout << w_t_y_g_n << " " << w_t_y_g_a << " " << endl;
		cout << w_t_z_g_n << " " << w_t_z_g_a << " " << endl;

		EXPECT_NEAR(w_r_x_g_n, w_r_x_g_a, 1e-6);
		EXPECT_NEAR(w_r_y_g_n, w_r_y_g_a, 1e-6);
		EXPECT_NEAR(w_r_z_g_n, w_r_z_g_a, 1e-6);
		EXPECT_NEAR(w_t_x_g_n, w_t_x_g_a, 1e-6);
		EXPECT_NEAR(w_t_y_g_n, w_t_y_g_a, 1e-6);
		EXPECT_NEAR(w_t_z_g_n, w_t_z_g_a, 1e-6);
	}
}

double CalcNumericalGradient(
	Getter getter, 
	Setter setter, 
	RigidTransform3D& transform, 
	vector<Mat>& feature, 
	const vector<vector<Mat>>& query_cloud, 
	vector<vector<Mat>>& predicted_cloud, 
	vector<vector<Mat>>& target_cloud, 
	const Mat& point_weight, 
	int num_transform, 
	int dim_feature, 
	int feature_dim_idx, 
	double disturb_value)
{
	Mat diff, exp_diff; // , filtered_diff, filtered_query_cloud;
	Mat disturb, dist;
	vector<Mat> w(num_transform);
	double e_1, e_2, numerical_gradient;
	double alpha = transform.alpha();
	int dim_transform = 4;
	
	disturb = Mat::zeros(1, dim_feature, CV_64F);
	disturb.at<double>(0, feature_dim_idx) = disturb_value;
	// e_1
	w = (transform.*getter)();
	w[0] = w[0] + disturb;
	(transform.*setter)(w);
	transform.CalcTransformation(feature);
	transform.TransformCloud(query_cloud[0][0], predicted_cloud);
	diff = predicted_cloud[0][0] - target_cloud[0][0];
	diff = diff.colRange(0, dim_transform - 1);
	reduce(diff.mul(diff), diff, 1, CV_REDUCE_SUM);
	/*diff = -0.5 * alpha * diff;
	exp(diff, exp_diff);
	exp_diff = 1 - exp_diff;
	reduce(exp_diff, dist, 0, CV_REDUCE_AVG);*/
	reduce(diff, dist, 0, CV_REDUCE_AVG);
	e_1 = 0.5 * dist.at<double>(0, 0);
	// e_1 = dist.at<double>(0, 0); // 1 - exp(-0.5 * alpha * dist.at<double>(0, 0));

	// e_2
	w = (transform.*getter)();
	w[0] = w[0] - 2 * disturb;
	(transform.*setter)(w);
	transform.CalcTransformation(feature);
	transform.TransformCloud(query_cloud[0][0], predicted_cloud);

	diff = predicted_cloud[0][0] - target_cloud[0][0];
	diff = diff.colRange(0, dim_transform - 1);
	reduce(diff.mul(diff), diff, 1, CV_REDUCE_SUM);
	/*diff = -0.5 * alpha * diff;
	exp(diff, exp_diff);
	exp_diff = 1 - exp_diff;
	reduce(exp_diff, dist, 0, CV_REDUCE_AVG);*/
	reduce(diff, dist, 0, CV_REDUCE_AVG);
	e_2 = 0.5 * dist.at<double>(0, 0);
	// e_2 = dist.at<double>(0, 0); // 1 - exp(-0.5 * alpha * dist.at<double>(0, 0));

	/*diff = predicted_cloud[0][0] - target_cloud[0][0];
	reduce(diff.mul(diff), diff, 1, CV_REDUCE_SUM);
	reduce(diff.mul(point_weight.colRange(0, 1)), dist, 0, CV_REDUCE_AVG);
	// e_2 = dist.at<double>(0, 0);
	e_2 = 1 - exp(-0.5 * alpha * dist.at<double>(0, 0));*/

	w = (transform.*getter)();
	w[0] = w[0] + disturb;
	(transform.*setter)(w);

	numerical_gradient = (e_1 - e_2) / (2 * disturb_value);
	return numerical_gradient;
}


TEST_F(UnifiedLearningTest, TestGetPointWeight)
{
	// load testing home cloud
	char test_input_dir[400];
	int num_joints = 2;
	int cloud_size = 1306;
	int batch_size = 1;
	Mat averaged_dists = Mat::zeros(cloud_size, num_joints, CV_64F);
	Mat original_averaged_dists = Mat::zeros(cloud_size, num_joints, CV_64F);
	Mat em_means;
	vector<Mat> em_covs;
	Mat em_weights;
	vector<vector<Mat>> segmented_point_weight(batch_size);
	segmented_point_weight[0] = vector<Mat>(num_joints);
	// read in data
	strcpy(test_input_dir, test_data_dir_prefix_);
	strcat(test_input_dir, "test_predicted_dists.bin");
	FileIO::ReadMatDouble(averaged_dists, cloud_size, num_joints, test_input_dir);
	strcpy(test_input_dir, test_data_dir_prefix_);
	strcat(test_input_dir, "test_original_dists.bin");
	FileIO::ReadMatDouble(original_averaged_dists, cloud_size, num_joints, test_input_dir);


	PointToPointSegmentationWithWeights::SetPointWeightWithGaussianMixture(original_averaged_dists, averaged_dists, segmented_point_weight, num_joints, batch_size);
	
}

TEST_F(UnifiedLearningTest, TestInitializeModelLabel)
{
	// load testing home cloud
	char test_input_dir[400];
	char min_dist_dir[40];
	int cloud_size;
	int num_joints = 3;
	cv::Mat home_cloud_label, expected_home_cloud_label;
	std::vector<cv::Mat> min_dists(num_joints);
	// specify cloud size
	strcpy(test_input_dir, test_data_dir_prefix_);
	strcat(test_input_dir, "test_home_cloud_size.bin");
	cv::Mat size_mat = cv::Mat::zeros(1, 1, CV_64F);
	FileIO::ReadMatDouble(size_mat, 1, 1, test_input_dir);
	cloud_size = size_mat.at<double>(0, 0);	
	// initialize cloud label
	home_cloud_label = cv::Mat::zeros(cloud_size, num_joints, CV_64F);
	expected_home_cloud_label = cv::Mat::zeros(cloud_size, num_joints, CV_64F);
	strcpy(test_input_dir, test_data_dir_prefix_);
	strcat(test_input_dir, "test_home_cloud_label.bin");
	FileIO::ReadMatDouble(expected_home_cloud_label, cloud_size, num_joints, test_input_dir);
	for(int i = 0; i < num_joints; i++)
	{
		min_dists[i] = cv::Mat::zeros(cloud_size, 1, CV_32F);
		strcpy(test_input_dir, test_data_dir_prefix_);
		sprintf(min_dist_dir, "min_dist_%d.bin", i);
		strcat(test_input_dir, min_dist_dir);
		FileIO::ReadMatFloat(min_dists[i], cloud_size, 1, test_input_dir);
	}
	PointToPointSegmentation segmentation(cloud_size, 4, num_joints, 1, 40);
	// segmentation.set_home_cloud(home_cloud_);
	// segmentation.set_home_cloud_neighbor_indices(home_cloud_neighbor_indices);
	vector<vector<Mat>> min_dists_wrapper(1);
	min_dists_wrapper[0] = min_dists;
	// segmentation.set_kd_tree_indices(indices);
	segmentation.set_kd_tree_min_dists(min_dists_wrapper);
	segmentation.set_home_cloud_label(home_cloud_label);
	// segmentation.set_target_cloud(data_cloud_wrapper);
	// segmentation.set_predicted_cloud(predicted_cloud_wrapper);
	segmentation.Match(1, 1);
	home_cloud_label = segmentation.home_cloud_label();
	// Explorer::InitializeModelLabel(min_dists, num_joints, home_cloud_label);
	for(int i = 0; i < cloud_size; i++)
	{
		for(int j = 0; j < num_joints; j++)
		{
			EXPECT_EQ(expected_home_cloud_label.at<double>(i, j), home_cloud_label.at<double>(i, j));
		}
	}
}

TEST_F(UnifiedLearningTest, TestIteratedConditionalMode)
{
	// load testing home cloud
	char test_input_dir[400];
	char min_dist_dir[40];
	int cloud_size;
	int num_joints = 3;
	int max_num_neighbors = 20;
	std::vector<cv::Mat> min_dists(num_joints);
	cv::Mat home_cloud_label, home_cloud_neighbor_indices, potential, expected_potential, expected_home_cloud_label;
	strcpy(test_input_dir, test_data_dir_prefix_);
	strcat(test_input_dir, "test_home_cloud_size.bin");
	cv::Mat size_mat = cv::Mat::zeros(1, 1, CV_64F);
	FileIO::ReadMatDouble(size_mat, 1, 1, test_input_dir);
	cloud_size = size_mat.at<double>(0, 0);	
	// read in cloud label
	home_cloud_label = cv::Mat::zeros(cloud_size, num_joints, CV_64F);
	potential = cv::Mat::zeros(cloud_size, num_joints, CV_64F);
	strcpy(test_input_dir, test_data_dir_prefix_);
	strcat(test_input_dir, "test_home_cloud_label.bin");
	FileIO::ReadMatDouble(home_cloud_label, cloud_size, num_joints, test_input_dir);
	// read in neighbor indices
	home_cloud_neighbor_indices = cv::Mat::zeros(cloud_size, max_num_neighbors, CV_64F);
	strcpy(test_input_dir, test_data_dir_prefix_);
	strcat(test_input_dir, "test_home_cloud_neighbor_indices.bin");
	FileIO::ReadMatDouble(home_cloud_neighbor_indices, cloud_size, max_num_neighbors, test_input_dir);
	home_cloud_neighbor_indices.convertTo(home_cloud_neighbor_indices, CV_32S);
	// read in expected potential
	expected_potential = cv::Mat::zeros(cloud_size, num_joints, CV_64F);
	strcpy(test_input_dir, test_data_dir_prefix_);
	strcat(test_input_dir, "test_home_cloud_potential.bin");
	FileIO::ReadMatDouble(expected_potential, cloud_size, num_joints, test_input_dir);
	// read in expected label after update
	expected_home_cloud_label = cv::Mat::zeros(cloud_size, num_joints, CV_64F);
	strcpy(test_input_dir, test_data_dir_prefix_);
	strcat(test_input_dir, "test_home_cloud_label_after_icm.bin");
	FileIO::ReadMatDouble(expected_home_cloud_label, cloud_size, num_joints, test_input_dir);
	// read in min dists
	for(int i = 0; i < num_joints; i++)
	{
		min_dists[i] = cv::Mat::zeros(cloud_size, 1, CV_32F);
		strcpy(test_input_dir, test_data_dir_prefix_);
		sprintf(min_dist_dir, "min_dist_%d.bin", i);
		strcat(test_input_dir, min_dist_dir);
		FileIO::ReadMatFloat(min_dists[i], cloud_size, 1, test_input_dir);
	}
	// execute the ICM algorithm for one iteration with beta equal to 1
	// Explorer::IteratedConditionalModes(home_cloud_neighbor_indices, home_cloud_label, potential, num_joints, 1, max_num_neighbors, 1.0);
	PointToPointSegmentation segmentation(cloud_size, 4, num_joints, 1, max_num_neighbors);
	// segmentation.set_home_cloud(home_cloud_);
	segmentation.set_home_cloud_neighbor_indices(home_cloud_neighbor_indices);
	vector<vector<Mat>> min_dists_wrapper(1);
	min_dists_wrapper[0] = min_dists;
	// segmentation.set_kd_tree_indices(indices);
	segmentation.set_kd_tree_min_dists(min_dists_wrapper);
	segmentation.set_home_cloud_label(home_cloud_label);
	// segmentation.set_target_cloud(data_cloud_wrapper);
	// segmentation.set_predicted_cloud(predicted_cloud_wrapper);
	segmentation.Match(2, 1);
	home_cloud_label = segmentation.home_cloud_label();
	// Explorer::IteratedConditionalModes(home_cloud_neighbor_indices, min_dists, home_cloud_label, potential, num_joints, 1, max_num_neighbors, 1.0, 1.0);
	for(int i = 0; i < cloud_size; i++)
	{
		for(int j = 0; j < num_joints; j++)
		{
			EXPECT_EQ(expected_home_cloud_label.at<double>(i, j), home_cloud_label.at<double>(i, j));
			// EXPECT_NEAR(expected_potential.at<double>(i, j), potential.at<double>(i, j), 1e-4);
		}
	}
}



/*TEST_F(UnifiedLearningTest, TestGetLabelFromProbability)
{
	// load testing home cloud
	char test_input_dir[400];
	char min_dist_dir[40];
	int cloud_size = 0;
	int num_joints = 3;
	int averaging_period = 100;
	std::random_device rd;
	cv::Mat test_probabilities = cv::Mat::zeros(1, num_joints, CV_64F);
	test_probabilities.at<double>(0, 0) = 0.2; test_probabilities.at<double>(0, 1) = 0.3; test_probabilities.at<double>(0, 2) = 0.5;
	EXPECT_EQ(2, Explorer::GetLabelFromProbability(test_probabilities, 0.1, num_joints));
	EXPECT_EQ(2, Explorer::GetLabelFromProbability(test_probabilities, 0.4, num_joints));
	EXPECT_EQ(2, Explorer::GetLabelFromProbability(test_probabilities, 0.7, num_joints));
}*/

/*TEST_F(UnifiedLearningTest, TestCalculateGradientWithNormal)
{
	char test_query_cloud_dir[200];
	char test_target_cloud_dir[200];
	int num_joint = 1; // currently only have data for 1 joint
	int dim_transform = 4;
	int dim_feature = pow(3.0, num_joint) - 1;
	int num_transform_elements = dim_transform * (dim_transform - 1);
	double disturb_value = 0.0001;
	double numerical_gradient = 0;
	double analytical_gradient = 0;
	std::vector<cv::Mat> predicted_cloud(num_joint);   
	std::vector<cv::Mat> query_cloud(num_joint);   
	std::vector<cv::Mat> query_cloud_normal(num_joint);   
	std::vector<cv::Mat> target_cloud(num_joint);   
	std::vector<cv::Mat> target_cloud_normal(num_joint);   
	std::vector<cv::Mat> w_grad(num_joint);   

	// home cloud 12443
	strcpy(test_query_cloud_dir, test_data_dir_prefix_);
	strcat(test_query_cloud_dir, "size_12243.bin");
	cv::Mat query_cloud_size_mat = cv::Mat::zeros(1, 1, CV_64F);
	FileIO::ReadMatDouble(query_cloud_size_mat, 1, 1, test_query_cloud_dir);
	int query_cloud_size = query_cloud_size_mat.at<double>(0, 0);
	strcpy(test_query_cloud_dir, test_data_dir_prefix_);
	strcat(test_query_cloud_dir, "12243.bin");
	query_cloud[0] = cv::Mat::ones(query_cloud_size, dim_transform, CV_64F);
	FileIO::ReadMatDouble(query_cloud[0].colRange(0, dim_transform -1), query_cloud_size, dim_transform - 1, test_query_cloud_dir);
	strcpy(test_query_cloud_dir, test_data_dir_prefix_);
	strcat(test_query_cloud_dir, "normal_12243.bin");
	query_cloud_normal[0] = cv::Mat::ones(query_cloud_size, dim_transform - 1, CV_64F);
	FileIO::ReadMatDouble(query_cloud_normal[0], query_cloud_size, dim_transform - 1, test_query_cloud_dir);

	// target cloud 10460
	strcpy(test_target_cloud_dir, test_data_dir_prefix_);
	strcat(test_target_cloud_dir, "size_10460.bin");
	cv::Mat target_cloud_size_mat = cv::Mat::zeros(1, 1, CV_64F);
	FileIO::ReadMatDouble(target_cloud_size_mat, 1, 1, test_target_cloud_dir);
	int target_cloud_size = target_cloud_size_mat.at<double>(0, 0);
	strcpy(test_target_cloud_dir, test_data_dir_prefix_);
	strcat(test_target_cloud_dir, "10460.bin");
	target_cloud[0] = cv::Mat::ones(target_cloud_size, dim_transform, CV_64F);
	FileIO::ReadMatDouble(target_cloud[0].colRange(0, dim_transform -1), target_cloud_size, dim_transform - 1, test_target_cloud_dir);
	strcpy(test_target_cloud_dir, test_data_dir_prefix_);
	strcat(test_target_cloud_dir, "normal_10460.bin");
	target_cloud_normal[0] = cv::Mat::ones(target_cloud_size, dim_transform - 1, CV_64F);
	FileIO::ReadMatDouble(target_cloud_normal[0], target_cloud_size, dim_transform - 1, test_query_cloud_dir);
	// trick to just shrink target cloud to the same size...
	query_cloud[0] = query_cloud[0].rowRange(0, target_cloud_size);
	query_cloud_normal[0] = query_cloud_normal[0].rowRange(0, target_cloud_size);

	for(int i = 0; i < num_joint; i++)
	{
		predicted_cloud[i] = cv::Mat::zeros(query_cloud_size, dim_transform, CV_64F);
		w_grad[i] = cv::Mat::zeros(num_transform_elements, dim_feature, CV_64F);
	}

	cv::Mat feature = cv::Mat::zeros(dim_feature, 1, CV_64F);
	cv::randu(feature, cv::Scalar::all(0), cv::Scalar::all(1));
	GeneralTransform transform(dim_transform, num_joint, 0.01);
	transform.CalcTransformation(feature);
	transform.TransformCloud(query_cloud[0], transform.get_transform(), predicted_cloud);
	// transform.CalculateGradient(target_cloud, predicted_cloud, query_cloud, feature);
	// here clouds are not matched in a closest distance sense.. but for testing gradient, should be sufficient
	transform.CalculateGradient(target_cloud, target_cloud_normal, predicted_cloud, query_cloud, feature);
	w_grad = transform.w_grad();

	cv::Mat diff; // , filtered_diff, filtered_query_cloud;
	cv::Mat disturb, dist, new_w;
	double e_1, e_2;
	for(int idx = 0; idx < num_joint; idx++)
	{
		for(int i = 0; i < num_transform_elements; i++)
		{
			for(int j = 0; j < dim_feature; j++)
			{
				disturb = cv::Mat::zeros(num_transform_elements, dim_feature, CV_64F);
				disturb.at<double>(i, j) = disturb_value;
				// e_1
				new_w = transform.get_w(idx) + disturb;
				transform.set_w(new_w, idx);
				transform.CalcTransformation(feature);
				transform.TransformCloud(query_cloud[0], transform.get_transform(), predicted_cloud);
				diff = predicted_cloud[idx] - target_cloud[idx];
				cv::Mat tmp_error = diff.colRange(0, dim_transform - 1) * target_cloud_normal[0].t();
				cv::Mat error = tmp_error.diag();
				cv::Mat squared_error = error.mul(error) / 2;
				cv::reduce(squared_error, squared_error, 0, CV_REDUCE_AVG);
				e_1 = squared_error.at<double>(0, 0);

				// e_2
				new_w = transform.get_w(idx) - 2 * disturb;
				transform.set_w(new_w, idx);
				transform.CalcTransformation(feature);
				transform.TransformCloud(query_cloud[0], transform.get_transform(), predicted_cloud);
				diff = predicted_cloud[idx] - target_cloud[idx];
				tmp_error = diff.colRange(0, dim_transform - 1) * target_cloud_normal[0].t();
				error = tmp_error.diag();
				squared_error = error.mul(error) / 2;
				cv::reduce(squared_error, squared_error, 0, CV_REDUCE_AVG);
				e_2 = squared_error.at<double>(0, 0);

				new_w = transform.get_w(idx) + disturb;
				transform.set_w(new_w, idx);

				numerical_gradient = (e_1 - e_2) / (2 * disturb_value);
				analytical_gradient = w_grad[idx].at<double>(i, j);

				EXPECT_NEAR(numerical_gradient, analytical_gradient, double_epsilon_);
			}
		}
	}
}

TEST_F(UnifiedLearningTest, TestCalculateObservationLikelihood)
{
	char input_dir[200];
	char curr_observ_dir[50];
	int dim_state = 2;
	int dim_observation = 2;
	int sequence_length = 10;
	int num_observation_sequence = 2;
	std::vector<cv::Mat> mu = std::vector<cv::Mat>(dim_state); 
	std::vector<cv::Mat> sigma = std::vector<cv::Mat>(dim_state); 
	std::vector<cv::Mat> likelihood = std::vector<cv::Mat>(num_observation_sequence);
	std::vector<cv::Mat> data = std::vector<cv::Mat>(num_observation_sequence); // cv::Mat::zeros(sequence_length, dim_observation, CV_64F);
	std::vector<cv::Mat> expected_likelihood = std::vector<cv::Mat>(num_observation_sequence);
	// fixed parameters for mu and sigma
	for(int i = 0; i < dim_state; i++)
	{
		mu[i] = cv::Mat::zeros(dim_observation, 1, CV_64F);
		sigma[i] = cv::Mat::zeros(dim_observation, dim_observation, CV_64F);
	}
	mu[0].at<double>(0, 0) = 1; mu[0].at<double>(1, 0) = 3;
	mu[1].at<double>(0, 0) = 2; mu[1].at<double>(1, 0) = 4;
	sigma[0] = cv::Mat::eye(2, 2, CV_64F); sigma[0] = sigma[0] * 0.25;
	sigma[1] = cv::Mat::eye(2, 2, CV_64F); sigma[1] = sigma[1] * 0.25;

	for(int i = 0; i < num_observation_sequence; i++)
	{
		data[i] = cv::Mat::zeros(sequence_length, dim_observation, CV_64F);
		expected_likelihood[i] = cv::Mat::zeros(sequence_length, dim_state, CV_64F);
		likelihood[i] = cv::Mat::zeros(sequence_length, dim_state, CV_64F);
	}

	// expected likelihood
	for(int i = 0; i < num_observation_sequence; i++)
	{
		strcpy(input_dir, test_data_dir_prefix_);
		sprintf(curr_observ_dir, "hmm_expected_likelihood_%d.bin", i);
		strcat(input_dir, curr_observ_dir);
		FileIO::ReadMatDouble(expected_likelihood[i], sequence_length, dim_state, input_dir);

		// hmm data
		strcpy(input_dir, test_data_dir_prefix_);
		sprintf(curr_observ_dir, "hmm_data_%d.bin", i);
		strcat(input_dir, curr_observ_dir);
		FileIO::ReadMatDouble(data[i], sequence_length, dim_observation, input_dir);
	}

	// hmm model and calculate likelihood
	HMM hmm(dim_state, dim_observation, sequence_length, num_observation_sequence);
	hmm.set_emission_mu(mu);
	hmm.set_emission_sigma(sigma);
	hmm.CalculateInvSigma();
	hmm.CalculateObservationLikelihood(likelihood, data);

	for(int i = 0; i < num_observation_sequence; i++)
	{
		for(int j = 0; j < sequence_length; j++)
		{
			for(int k = 0; k < dim_state; k++)
			{
				EXPECT_NEAR(expected_likelihood[i].at<double>(j, k), likelihood[i].at<double>(j, k), double_epsilon_);
			}
		}
	}
}

TEST_F(UnifiedLearningTest, TestForwardBackwardProcedure)
{
	char input_dir[200];
	char curr_observ_dir[50];
	int dim_state = 2;
	int dim_observation = 2;
	int sequence_length = 10;
	int num_observation_sequence = 2;
	// fixed parameters for mu and sigma
	cv::Mat prior = cv::Mat::zeros(dim_state, 1, CV_64F); 
	cv::Mat trans = cv::Mat::zeros(dim_state, dim_state, CV_64F);
	std::vector<cv::Mat> likelihood = std::vector<cv::Mat>(num_observation_sequence);
	std::vector<cv::Mat> expected_alpha = std::vector<cv::Mat>(num_observation_sequence);
	std::vector<cv::Mat> expected_beta = std::vector<cv::Mat>(num_observation_sequence);
	std::vector<cv::Mat> expected_gamma = std::vector<cv::Mat>(num_observation_sequence);
	std::vector<cv::Mat> expected_kesi_summed = std::vector<cv::Mat>(num_observation_sequence);
	
	// set prior & transition probability values
	prior.at<double>(0, 0) = 0.5; prior.at<double>(1, 0) = 0.5; 
	trans.at<double>(0, 0) = 0.8; trans.at<double>(0, 1) = 0.2; trans.at<double>(1, 0) = 0.2; trans.at<double>(1, 1) = 0.8; 

	// expected data
	for(int i = 0; i < num_observation_sequence; i++)
	{
		likelihood[i] = cv::Mat::zeros(sequence_length, dim_state, CV_64F);
		expected_alpha[i] = cv::Mat::zeros(sequence_length, dim_state, CV_64F);
		expected_beta[i] = cv::Mat::zeros(sequence_length, dim_state, CV_64F);
		expected_gamma[i] = cv::Mat::zeros(sequence_length, dim_state, CV_64F);
		expected_kesi_summed[i] = cv::Mat::zeros(dim_state, dim_state, CV_64F);
	}
	// expected likelihood
	for(int i = 0; i < num_observation_sequence; i++)
	{
		// hmm gamma
		strcpy(input_dir, test_data_dir_prefix_);
		sprintf(curr_observ_dir, "hmm_alpha_%d.bin", i);
		strcat(input_dir, curr_observ_dir);
		FileIO::ReadMatDouble(expected_alpha[i], sequence_length, dim_state, input_dir);

		// hmm beta
		strcpy(input_dir, test_data_dir_prefix_);
		sprintf(curr_observ_dir, "hmm_beta_%d.bin", i);
		strcat(input_dir, curr_observ_dir);
		FileIO::ReadMatDouble(expected_beta[i], sequence_length, dim_state, input_dir);

		// hmm gamma
		strcpy(input_dir, test_data_dir_prefix_);
		sprintf(curr_observ_dir, "hmm_gamma_%d.bin", i);
		strcat(input_dir, curr_observ_dir);
		FileIO::ReadMatDouble(expected_gamma[i], sequence_length, dim_state, input_dir);

		// hmm summed
		strcpy(input_dir, test_data_dir_prefix_);
		sprintf(curr_observ_dir, "hmm_kesi_summed_%d.bin", i);
		strcat(input_dir, curr_observ_dir);
		FileIO::ReadMatDouble(expected_kesi_summed[i], dim_state, dim_state, input_dir);

		// hmm likelihood
		strcpy(input_dir, test_data_dir_prefix_);
		sprintf(curr_observ_dir, "hmm_expected_likelihood_%d.bin", i);
		strcat(input_dir, curr_observ_dir);
		FileIO::ReadMatDouble(likelihood[i], sequence_length, dim_state, input_dir);
	}

	// hmm model and calculate likelihood
	HMM hmm(dim_state, dim_observation, sequence_length, num_observation_sequence);
	hmm.set_prior(prior);
	hmm.set_trans(trans);
	hmm.ForwardBackwardProcedure(likelihood);

	for(int i = 0; i < num_observation_sequence; i++)
	{
		for(int j = 0; j < sequence_length; j++)
		{
			for(int k = 0; k < dim_state; k++)
			{
				EXPECT_NEAR(expected_alpha[i].at<double>(j, k), hmm.get_alpha()[i].at<double>(j, k), double_epsilon_);
				EXPECT_NEAR(expected_beta[i].at<double>(j, k), hmm.get_beta()[i].at<double>(j, k), double_epsilon_);
				EXPECT_NEAR(expected_gamma[i].at<double>(j, k), hmm.get_gamma()[i].at<double>(j, k), double_epsilon_);
				
			}
		}
		for(int p = 0; p < dim_state; p++)
		{
			for(int q = 0; q < dim_state; q++)
			{
				EXPECT_NEAR(expected_kesi_summed[i].at<double>(p, q), hmm.get_kesi_summed()[i].at<double>(p, q), double_epsilon_);		
			}
		}
	}
}

TEST_F(UnifiedLearningTest, Test_HMM_EM)
{
	char input_dir[200];
	char curr_observ_dir[50];
	int dim_state = 2;
	int dim_observation = 2;
	int sequence_length = 10;
	int num_observation_sequence = 2;
	// fixed parameters for mu and sigma
	std::vector<cv::Mat> hmm_data = std::vector<cv::Mat>(num_observation_sequence); 
	std::vector<cv::Mat> likelihood = std::vector<cv::Mat>(num_observation_sequence);
	cv::Mat prior_0 = cv::Mat::zeros(dim_state, 1, CV_64F); 
	cv::Mat trans_0 = cv::Mat::zeros(dim_state, dim_state, CV_64F);
	cv::Mat prior_1 = cv::Mat::zeros(dim_state, 1, CV_64F); 
	cv::Mat trans_1 = cv::Mat::zeros(dim_state, dim_state, CV_64F);
	cv::Mat expected_prior = cv::Mat::zeros(dim_state, 1, CV_64F); 
	cv::Mat expected_trans = cv::Mat::zeros(dim_state, dim_state, CV_64F);
	std::vector<cv::Mat> mu_0 = std::vector<cv::Mat>(dim_state); 
	std::vector<cv::Mat> sigma_0 = std::vector<cv::Mat>(dim_state); 
	std::vector<cv::Mat> mu_1 = std::vector<cv::Mat>(dim_state); 
	std::vector<cv::Mat> sigma_1 = std::vector<cv::Mat>(dim_state); 
	std::vector<cv::Mat> expected_mu = std::vector<cv::Mat>(dim_state); 
	std::vector<cv::Mat> expected_sigma = std::vector<cv::Mat>(dim_state); 
	for(int i = 0; i < dim_state; i++)
	{
		mu_0[i] = cv::Mat::zeros(dim_observation, 1, CV_64F);
		sigma_0[i] = cv::Mat::zeros(dim_observation, dim_observation, CV_64F);
		mu_1[i] = cv::Mat::zeros(dim_observation, 1, CV_64F);
		sigma_1[i] = cv::Mat::zeros(dim_observation, dim_observation, CV_64F);
		expected_mu[i] = cv::Mat::zeros(dim_observation, 1, CV_64F);
		expected_sigma[i] = cv::Mat::zeros(dim_observation, dim_observation, CV_64F);
	}
	mu_0[0].at<double>(0, 0) = 1.5; mu_0[0].at<double>(1, 0) = 2.5;
	mu_0[1].at<double>(0, 0) = 2.5; mu_0[1].at<double>(1, 0) = 3.5;
	sigma_0[0] = cv::Mat::eye(2, 2, CV_64F); sigma_0[0] = sigma_0[0] * 0.4;
	sigma_0[1] = cv::Mat::eye(2, 2, CV_64F); sigma_0[1] = sigma_0[1] * 0.4;

	prior_0.at<double>(0, 0) = 0.4; prior_0.at<double>(1, 0) = 0.6; 
	trans_0.at<double>(0, 0) = 0.7; trans_0.at<double>(0, 1) = 0.3; trans_0.at<double>(1, 0) = 0.3; trans_0.at<double>(1, 1) = 0.7; 

	// expected likelihood
	for(int i = 0; i < num_observation_sequence; i++)
	{
		// hmm gamma
		// multi observation...
		hmm_data[i] = cv::Mat::zeros(sequence_length, dim_observation, CV_64F);
		strcpy(input_dir, test_data_dir_prefix_);
		sprintf(curr_observ_dir, "hmm_data_0.bin");
		strcat(input_dir, curr_observ_dir);
		FileIO::ReadMatDouble(hmm_data[i], sequence_length, dim_state, input_dir);

		likelihood[i] = cv::Mat::zeros(sequence_length, dim_state, CV_64F);
	}
	// hmm parameters
	// prior
	strcpy(input_dir, test_data_dir_prefix_);
	sprintf(curr_observ_dir, "hmm_expected_prior.bin");
	strcat(input_dir, curr_observ_dir);
	FileIO::ReadMatDouble(expected_prior, dim_state, 1, input_dir);

	// transition
	strcpy(input_dir, test_data_dir_prefix_);
	sprintf(curr_observ_dir, "hmm_expected_transition.bin");
	strcat(input_dir, curr_observ_dir);
	FileIO::ReadMatDouble(expected_trans, dim_state, dim_state, input_dir);

	for(int s = 0; s < dim_state; s++)
	{
		// hmm summed
		strcpy(input_dir, test_data_dir_prefix_);
		sprintf(curr_observ_dir, "hmm_expected_mu_state_%d.bin", s);
		strcat(input_dir, curr_observ_dir);
		FileIO::ReadMatDouble(expected_mu[s], dim_observation, 1, input_dir);

		// hmm likelihood
		strcpy(input_dir, test_data_dir_prefix_);
		sprintf(curr_observ_dir, "hmm_expected_sigma_state_%d.bin", s);
		strcat(input_dir, curr_observ_dir);
		FileIO::ReadMatDouble(expected_sigma[s], dim_observation, dim_observation, input_dir);
	}

	// hmm model and calculate likelihood
	HMM hmm(dim_state, dim_observation, sequence_length, num_observation_sequence);
	hmm.set_prior(prior_0);
	hmm.set_trans(trans_0);
	hmm.set_emission_mu(mu_0);
	hmm.set_emission_sigma(sigma_0);

	hmm.ExpectationStep(likelihood, hmm_data);
	hmm.MaximizationStep(hmm_data);

	mu_1 = hmm.get_emission_mu();
	sigma_1 = hmm.get_emission_sigma();
	prior_1 = hmm.get_prior();
	trans_1 = hmm.get_trans();
	for(int p = 0; p < dim_state; p++)
	{
		for(int q = 0; q < dim_observation; q++)
		{
			EXPECT_NEAR(expected_mu[p].at<double>(q, 0), mu_1[p].at<double>(q, 0), double_epsilon_);		
			for(int m = 0; m < dim_observation; m++)
			{
				if(q == m)
				{
					EXPECT_NEAR(expected_sigma[p].at<double>(q, m), sigma_1[p].at<double>(q, m), double_epsilon_);		
				}
			}
		}
		
		EXPECT_NEAR(expected_prior.at<double>(p, 0), prior_1.at<double>(p, 0), double_epsilon_);		
		for(int m = 0; m < dim_state; m++)
		{
			EXPECT_NEAR(expected_trans.at<double>(p, m), trans_1.at<double>(p, m), double_epsilon_);		
		}
	}
}

// externally dependent test... temporary...
TEST_F(UnifiedLearningTest, TestHmmOnArmData)
{
	char data_dir[200];
	char explained_ratio_dir[200];
	char input_dir[200];
	char cloud_size_dir[200];
	char tmp_dir[200];
	int dim_state = 3;
	int dim_observation = 2;
	int sequence_length = 200;
	int para_idx = 51;
	int prop_train_size = 17005;
	double seed = time(NULL);
	cv::RNG rng(seed);
	cv::Mat upper_bound = cv::Mat::zeros(1, 1, CV_64F) + 1e-3;
	cv::Mat lower_bound = cv::Mat::zeros(1, 1, CV_64F) + 1e-4;
	sprintf(data_dir, "D:/Document/HKUST/Year 5/Research/Data/PointClouds/december_13_2013/");
	sprintf(explained_ratio_dir, "D:/Document/HKUST/Year 5/Research/Solutions/unified_framework/output/");
	strcpy(input_dir, data_dir);
	strcat(input_dir, "train_prop_idx.bin");
	cv::Mat train_idx = cv::Mat::zeros(prop_train_size, 1, CV_64F);
	FileIO::ReadFloatMatToDouble(train_idx, prop_train_size, 1, input_dir);
	int home_cloud_idx = train_idx.at<double>(0, 0);
	cv::Mat cloud_size_mat = cv::Mat::zeros(1, 1, CV_64F);
	sprintf(cloud_size_dir, "binary_dense/size_%d.bin", home_cloud_idx);
	strcpy(input_dir, data_dir);
	strcat(input_dir, cloud_size_dir);
	FileIO::ReadMatDouble(cloud_size_mat, 1, 1, input_dir);
	
	int num_observation_sequence = cloud_size_mat.at<double>(0, 0);
	std::vector<cv::Mat> matching_error_sequences = std::vector<cv::Mat>(num_observation_sequence);
	for(int i = 0; i < num_observation_sequence; i++)
	{
		matching_error_sequences[i] = cv::Mat::zeros(sequence_length, dim_observation, CV_64F);
	}

	// load explained ratios...
	int max_sequence_length = 10000;
	int start_sequence_idx = 9001;
	int end_sequence_idx = start_sequence_idx + sequence_length - 1;
	if(end_sequence_idx > max_sequence_length)
	{
		std::cout << "sequence length too large..." << std::endl;
		exit(0);
	}
	// load explained ratio with noise...
	for(int sequence_idx = start_sequence_idx; sequence_idx <= end_sequence_idx; sequence_idx++)
	{
		sprintf(tmp_dir, "para_%d/matching_error/matching_error_%d.bin", para_idx, sequence_idx);
		strcpy(input_dir, explained_ratio_dir);
		strcat(input_dir, tmp_dir);
		cv::Mat curr_matching_error = cv::Mat::zeros(num_observation_sequence, dim_observation, CV_64F);
		FileIO::ReadMatDouble(curr_matching_error, num_observation_sequence, dim_observation, input_dir);
		for(int observation_idx = 0; observation_idx < num_observation_sequence; observation_idx++)
		{
			curr_matching_error.rowRange(observation_idx, observation_idx + 1).copyTo(matching_error_sequences[observation_idx].rowRange(sequence_idx - start_sequence_idx, sequence_idx - start_sequence_idx + 1));
		}

	}

	// fixed parameters for mu and sigma
	std::vector<cv::Mat> likelihood = std::vector<cv::Mat>(num_observation_sequence);
	cv::Mat prior_0 = cv::Mat::zeros(dim_state, 1, CV_64F); 
	cv::Mat prior_1 = cv::Mat::zeros(dim_state, 1, CV_64F); 
	cv::Mat trans_0 = cv::Mat::zeros(dim_state, dim_state, CV_64F);
	cv::Mat trans_1 = cv::Mat::zeros(dim_state, dim_state, CV_64F);
	std::vector<cv::Mat> mu_0 = std::vector<cv::Mat>(dim_state); 
	std::vector<cv::Mat> mu_1 = std::vector<cv::Mat>(dim_state); 
	std::vector<cv::Mat> sigma_0 = std::vector<cv::Mat>(dim_state); 
	std::vector<cv::Mat> sigma_1 = std::vector<cv::Mat>(dim_state); 
	for(int i = 0; i < dim_state; i++)
	{
		mu_0[i] = cv::Mat::zeros(dim_observation, 1, CV_64F);
		mu_1[i] = cv::Mat::zeros(dim_observation, 1, CV_64F);
		sigma_0[i] = cv::Mat::zeros(dim_observation, dim_observation, CV_64F);
		sigma_1[i] = cv::Mat::zeros(dim_observation, dim_observation, CV_64F);
	}
	// initialization value
	mu_0[0].at<double>(0, 0) = 0.02; mu_0[0].at<double>(1, 0) = 0.001; 
	mu_0[1].at<double>(0, 0) = 0.001; mu_0[1].at<double>(1, 0) = 0.02; 
	mu_0[2].at<double>(0, 0) = 0.001; mu_0[2].at<double>(1, 0) = 0.001; 
	sigma_0[0] = cv::Mat::eye(2, 2, CV_64F); sigma_0[0] = sigma_0[0] * 0.01; 
	sigma_0[1] = cv::Mat::eye(2, 2, CV_64F); sigma_0[1] = sigma_0[1] * 0.01; 
	sigma_0[2] = cv::Mat::eye(2, 2, CV_64F); sigma_0[2] = sigma_0[1] * 0.01; 

	prior_0.at<double>(0, 0) = 0.3334; prior_0.at<double>(1, 0) = 0.3334; prior_0.at<double>(2, 0) = 0.3332; 
	trans_0.at<double>(0, 0) = 0.8; trans_0.at<double>(0, 1) = 0.1; trans_0.at<double>(0, 2) = 0.1; 
	trans_0.at<double>(1, 0) = 0.1; trans_0.at<double>(1, 1) = 0.8; trans_0.at<double>(1, 2) = 0.1; 
	trans_0.at<double>(2, 0) = 0.1; trans_0.at<double>(2, 1) = 0.1; trans_0.at<double>(2, 2) = 0.8; 
	// expected likelihood
	for(int i = 0; i < num_observation_sequence; i++)
	{
		likelihood[i] = cv::Mat::zeros(sequence_length, dim_state, CV_64F);
	}

	// hmm model and calculate likelihood
	HMM hmm(dim_state, dim_observation, sequence_length, num_observation_sequence);
	hmm.set_prior(prior_0);
	hmm.set_trans(trans_0);
	hmm.set_emission_mu(mu_0);
	hmm.set_emission_sigma(sigma_0);

	for(int iter = 0; iter < 20; iter++)
	{
		hmm.ExpectationStep(likelihood, matching_error_sequences);
		hmm.MaximizationStep(matching_error_sequences);
		mu_1 = hmm.get_emission_mu();
		sigma_1 = hmm.get_emission_sigma();
		prior_1 = hmm.get_prior();
		trans_1 = hmm.get_trans();
		std::cout << "iteration: " << iter << std::endl;
		std::cout << "mu 0: " << mu_1[0].at<double>(0, 0) << " " << mu_1[0].at<double>(1, 0) << std::endl; 
		std::cout << "mu 1: " << mu_1[1].at<double>(0, 0) << " " << mu_1[1].at<double>(1, 0) << std::endl; 
		std::cout << "mu 2: " << mu_1[2].at<double>(0, 0) << " " << mu_1[2].at<double>(1, 0) << std::endl; 
		std::cout << "sigma 0: " << sigma_1[0].at<double>(0, 0) << " " << sigma_1[0].at<double>(1, 1) << std::endl;
		std::cout << "sigma 1: " << sigma_1[1].at<double>(0, 0) << " " << sigma_1[1].at<double>(1, 1) << std::endl;
		std::cout << "sigma 2: " << sigma_1[2].at<double>(0, 0) << " " << sigma_1[2].at<double>(1, 1) << std::endl;
		std::cout << "prior: " << prior_1.at<double>(0, 0) << " " << prior_1.at<double>(1, 0) << " " << prior_1.at<double>(2, 0) << std::endl;
		std::cout << "trans: " << trans_1.at<double>(0, 0) << " " << trans_1.at<double>(1, 1) << " " << trans_1.at<double>(2, 2) << std::endl;
	}

	std::vector<cv::Mat> learned_gamma = hmm.get_gamma();
	cv::Mat label = cv::Mat::zeros(num_observation_sequence, 1, CV_64F);
	cv::Point min_location, max_location;
	min_location.x = 0; min_location.y = 0; max_location.x = 0; max_location.y = 0;
	
	for(int j = 0; j < sequence_length; j++)
	{
		for(int i = 0; i < num_observation_sequence; i++)
		{
			double min_value = 0;
			double max_value = 0;
			cv::minMaxLoc(learned_gamma[i].rowRange(j, j + 1), &min_value, &max_value, &min_location, &max_location);
			label.at<double>(i, 0) = max_location.x;
		}
		strcpy(input_dir, test_data_dir_prefix_);
		sprintf(explained_ratio_dir, "learned_label/learned_label_%d.bin", j);
		strcat(input_dir, explained_ratio_dir);
		FileIO::WriteMatDouble(label, num_observation_sequence, 1, input_dir);
	}
}*/