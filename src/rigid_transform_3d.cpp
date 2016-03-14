#include "../inc/rigid_transform_3d.h"
#include "opencv2/highgui/highgui_c.h"

using namespace cv;
using namespace std;

RigidTransform3D::RigidTransform3D(int num_transform, int feature_dim, int batch_size, double learning_rate, double alpha)
{
	double weight_range = 1e-4;
	transform_dim_ = 4;
	RNG rng(getTickCount());
	num_transform_ = num_transform;	
	feature_dim_ = feature_dim;
	batch_size_ = batch_size;
	learning_rate_ = learning_rate;
	alpha_ = alpha;
	// initialize transform
	transform_ = vector<vector<Mat>>(batch_size);
	tmp_rotation_mat_x_ = vector<vector<Mat>>(batch_size);
	tmp_rotation_mat_y_ = vector<vector<Mat>>(batch_size);
	tmp_rotation_mat_z_ = vector<vector<Mat>>(batch_size);
	tmp_translation_mat_ = vector<vector<Mat>>(batch_size);
	for(int batch_idx = 0; batch_idx < batch_size; batch_idx++)
	{
		transform_[batch_idx] = vector<Mat>(num_transform);	
		tmp_rotation_mat_x_[batch_idx] = vector<Mat>(num_transform);	
		tmp_rotation_mat_y_[batch_idx] = vector<Mat>(num_transform);	
		tmp_rotation_mat_z_[batch_idx] = vector<Mat>(num_transform);	
		tmp_translation_mat_[batch_idx] = vector<Mat>(num_transform);	
		for(int transform_idx = 0; transform_idx < num_transform; transform_idx++)
		{
			transform_[batch_idx][transform_idx] = Mat::eye(transform_dim_, transform_dim_, CV_64F);
			tmp_rotation_mat_x_[batch_idx][transform_idx] = Mat::eye(transform_dim_, transform_dim_, CV_64F);
			tmp_rotation_mat_y_[batch_idx][transform_idx] = Mat::eye(transform_dim_, transform_dim_, CV_64F);
			tmp_rotation_mat_z_[batch_idx][transform_idx] = Mat::eye(transform_dim_, transform_dim_, CV_64F);
			tmp_translation_mat_[batch_idx][transform_idx] = Mat::eye(transform_dim_, transform_dim_, CV_64F);
		}
	}
	// initialize weights
	w_r_x_ = vector<Mat>(num_transform);
	w_r_y_ = vector<Mat>(num_transform);
	w_r_z_ = vector<Mat>(num_transform);
	w_t_x_ = vector<Mat>(num_transform);
	w_t_y_ = vector<Mat>(num_transform);
	w_t_z_ = vector<Mat>(num_transform);

	w_r_x_grad_avg_ = vector<Mat>(num_transform);
	w_r_y_grad_avg_ = vector<Mat>(num_transform);
	w_r_z_grad_avg_ = vector<Mat>(num_transform);
	w_t_x_grad_avg_ = vector<Mat>(num_transform);
	w_t_y_grad_avg_ = vector<Mat>(num_transform);
	w_t_z_grad_avg_ = vector<Mat>(num_transform);

	w_r_x_grad_ = vector<Mat>(num_transform);
	w_r_y_grad_ = vector<Mat>(num_transform);
	w_r_z_grad_ = vector<Mat>(num_transform);
	w_t_x_grad_ = vector<Mat>(num_transform);
	w_t_y_grad_ = vector<Mat>(num_transform);
	w_t_z_grad_ = vector<Mat>(num_transform);

	for(int i = 0; i < num_transform; i++)
	{
		/*w_r_x_[i] = Mat::ones(1, feature_dim, CV_64F) * weight_range;
		w_r_y_[i] = Mat::ones(1, feature_dim, CV_64F) * weight_range;
		w_r_z_[i] = Mat::ones(1, feature_dim, CV_64F) * weight_range;
		w_t_x_[i] = Mat::ones(1, feature_dim, CV_64F) * weight_range;
		w_t_y_[i] = Mat::ones(1, feature_dim, CV_64F) * weight_range;
		w_t_z_[i] = Mat::ones(1, feature_dim, CV_64F) * weight_range;*/

		w_r_x_[i] = Mat::zeros(1, feature_dim, CV_64F);
		w_r_y_[i] = Mat::zeros(1, feature_dim, CV_64F);
		w_r_z_[i] = Mat::zeros(1, feature_dim, CV_64F);
		w_t_x_[i] = Mat::zeros(1, feature_dim, CV_64F);
		w_t_y_[i] = Mat::zeros(1, feature_dim, CV_64F);
		w_t_z_[i] = Mat::zeros(1, feature_dim, CV_64F);

		rng.fill(w_r_x_[i], RNG::UNIFORM, -weight_range, weight_range);
		rng.fill(w_r_y_[i], RNG::UNIFORM, -weight_range, weight_range);
		rng.fill(w_r_z_[i], RNG::UNIFORM, -weight_range, weight_range);
		rng.fill(w_t_x_[i], RNG::UNIFORM, -weight_range, weight_range);
		rng.fill(w_t_y_[i], RNG::UNIFORM, -weight_range, weight_range);
		rng.fill(w_t_z_[i], RNG::UNIFORM, -weight_range, weight_range);
		

		w_r_x_grad_avg_[i] = Mat::zeros(1, feature_dim, CV_64F);
		w_r_y_grad_avg_[i] = Mat::zeros(1, feature_dim, CV_64F);
		w_r_z_grad_avg_[i] = Mat::zeros(1, feature_dim, CV_64F);
		w_t_x_grad_avg_[i] = Mat::zeros(1, feature_dim, CV_64F);
		w_t_y_grad_avg_[i] = Mat::zeros(1, feature_dim, CV_64F);
		w_t_z_grad_avg_[i] = Mat::zeros(1, feature_dim, CV_64F);

		w_r_x_grad_[i] = Mat::zeros(1, feature_dim, CV_64F);
		w_r_y_grad_[i] = Mat::zeros(1, feature_dim, CV_64F);
		w_r_z_grad_[i] = Mat::zeros(1, feature_dim, CV_64F);
		w_t_x_grad_[i] = Mat::zeros(1, feature_dim, CV_64F);
		w_t_y_grad_[i] = Mat::zeros(1, feature_dim, CV_64F);
		w_t_z_grad_[i] = Mat::zeros(1, feature_dim, CV_64F);
	}
}

void RigidTransform3D::CalcSingleTransform(const Mat& feature, int transform_idx, int batch_idx)
{
	Mat rotation_x_angle = w_r_x_[transform_idx] * feature;
	Mat rotation_y_angle = w_r_y_[transform_idx] * feature;
	Mat rotation_z_angle = w_r_z_[transform_idx] * feature;
	double cosine_x_angle =  cos(rotation_x_angle.at<double>(0, 0)); double sine_x_angle =  sin(rotation_x_angle.at<double>(0, 0));
	double cosine_y_angle =  cos(rotation_y_angle.at<double>(0, 0)); double sine_y_angle =  sin(rotation_y_angle.at<double>(0, 0));
	double cosine_z_angle =  cos(rotation_z_angle.at<double>(0, 0)); double sine_z_angle =  sin(rotation_z_angle.at<double>(0, 0));
	// x rotation

	tmp_rotation_mat_x_[batch_idx][transform_idx].at<double>(1, 1) = cosine_x_angle; tmp_rotation_mat_x_[batch_idx][transform_idx].at<double>(1, 2) = -sine_x_angle; 
	tmp_rotation_mat_x_[batch_idx][transform_idx].at<double>(2, 1) = sine_x_angle; tmp_rotation_mat_x_[batch_idx][transform_idx].at<double>(2, 2) = cosine_x_angle; 
	// y rotation

	tmp_rotation_mat_y_[batch_idx][transform_idx].at<double>(0, 0) = cosine_y_angle; tmp_rotation_mat_y_[batch_idx][transform_idx].at<double>(0, 2) = sine_y_angle; 
	tmp_rotation_mat_y_[batch_idx][transform_idx].at<double>(2, 0) = -sine_y_angle; tmp_rotation_mat_y_[batch_idx][transform_idx].at<double>(2, 2) = cosine_y_angle; 
	// z rotation

	tmp_rotation_mat_z_[batch_idx][transform_idx].at<double>(0, 0) = cosine_z_angle; tmp_rotation_mat_z_[batch_idx][transform_idx].at<double>(0, 1) = -sine_z_angle; 
	tmp_rotation_mat_z_[batch_idx][transform_idx].at<double>(1, 0) = sine_z_angle; tmp_rotation_mat_z_[batch_idx][transform_idx].at<double>(1, 1) = cosine_z_angle; 
	// translation

	Mat translation_x = w_t_x_[transform_idx] * feature;
	Mat translation_y = w_t_y_[transform_idx] * feature;
	Mat translation_z = w_t_z_[transform_idx] * feature;
	tmp_translation_mat_[batch_idx][transform_idx].at<double>(0, 3) = translation_x.at<double>(0, 0);
	tmp_translation_mat_[batch_idx][transform_idx].at<double>(1, 3) = translation_y.at<double>(0, 0);
	tmp_translation_mat_[batch_idx][transform_idx].at<double>(2, 3) = translation_z.at<double>(0, 0);
	// transformation

	transform_[batch_idx][transform_idx] = tmp_translation_mat_[batch_idx][transform_idx] * tmp_rotation_mat_x_[batch_idx][transform_idx] * tmp_rotation_mat_y_[batch_idx][transform_idx] 
		* tmp_rotation_mat_z_[batch_idx][transform_idx];
}

void RigidTransform3D::CalcTransformation(vector<Mat>& feature)
{
	for(int batch_idx = 0; batch_idx < batch_size_; batch_idx++)
	{
		// rotation on x
		for(int transform_idx = 0; transform_idx < num_transform_; transform_idx++)
		{
			CalcSingleTransform(feature[batch_idx], transform_idx, batch_idx);
		}
	}
}

void RigidTransform3D::TransformCloud(const Mat& input_cloud, vector<vector<Mat>>& output_cloud)
{
	for(int batch_idx = 0; batch_idx < batch_size_; batch_idx++)
	{
		for(int transform_idx = 0; transform_idx < num_transform_; transform_idx++)
		{
			output_cloud[batch_idx][transform_idx] = input_cloud * transform_[batch_idx][transform_idx].t();          
		}
	}
}

void RigidTransform3D::TransformPoint(Mat& input_point, Mat& output_point, int index, int batch_idx)
{
	output_point = input_point * transform_[batch_idx][index].t();          
}

// each weight gradient is different, six different weights...
void RigidTransform3D::CalculateGradient(const vector<vector<Mat>>& matched_target_cloud, const vector<vector<Mat>>& matched_predicted_cloud, const vector<vector<Mat>>& matched_query_cloud, const vector<vector<Mat>>& point_weight, const std::vector<cv::Mat>& feature)
{
	// the cloud should be n by 4...        
	Mat weight_sum, tmp_diff, exp_tmp_diff, repeated_exp_tmp_diff;

	for(int transform_idx = 0; transform_idx < num_transform_; transform_idx++)
	{
		w_r_x_grad_avg_[transform_idx] = Mat::zeros(1, feature_dim_, CV_64F);
		w_r_y_grad_avg_[transform_idx] = Mat::zeros(1, feature_dim_, CV_64F);
		w_r_z_grad_avg_[transform_idx] = Mat::zeros(1, feature_dim_, CV_64F);
		w_t_x_grad_avg_[transform_idx] = Mat::zeros(1, feature_dim_, CV_64F);
		w_t_y_grad_avg_[transform_idx] = Mat::zeros(1, feature_dim_, CV_64F);
		w_t_z_grad_avg_[transform_idx] = Mat::zeros(1, feature_dim_, CV_64F);
	}
	// std::cout << "calculating batch gradient" << std::endl;
	for(int batch_idx = 0; batch_idx < batch_size_; batch_idx++)
	{
		for(int transform_idx = 0; transform_idx < num_transform_; transform_idx++)
		{
			// calculate gradient...
			if(matched_predicted_cloud[batch_idx][transform_idx].rows != 0)
			{
				int num_points = matched_predicted_cloud[batch_idx][transform_idx].rows;
				Mat tmp = Mat::zeros(1, transform_dim_ - 1, CV_64F);
				Mat tmp_row = Mat::zeros(1, transform_dim_ - 1, CV_64F);
				Mat tmp_col = Mat::zeros(num_points, 1, CV_64F);
				Mat tmp_replicated = Mat::zeros(num_points, transform_dim_ - 1, CV_64F);
				Mat tmp_reduced = Mat::zeros(num_points, transform_dim_ - 1, CV_64F);
				Mat tmp_single = Mat::zeros(1, 1, CV_64F);
				Mat curr_transform_point_weight = point_weight[batch_idx][transform_idx].t();
				// double average_size_inv = 1.0 / curr_transform_point_weight.cols;
				reduce(curr_transform_point_weight, weight_sum, 1, CV_REDUCE_SUM);
				double average_size_inv = 1.0 / weight_sum.at<double>(0, 0);
				// get the sine, cosine and translate
				double sx = tmp_rotation_mat_x_[batch_idx][transform_idx].at<double>(2, 1); double cx = tmp_rotation_mat_x_[batch_idx][transform_idx].at<double>(1, 1);
				double sy = tmp_rotation_mat_y_[batch_idx][transform_idx].at<double>(0, 2); double cy = tmp_rotation_mat_y_[batch_idx][transform_idx].at<double>(0, 0);
				double sz = tmp_rotation_mat_z_[batch_idx][transform_idx].at<double>(1, 0); double cz = tmp_rotation_mat_z_[batch_idx][transform_idx].at<double>(0, 0);
				// double tx = tmp_translation_mat_[batch_idx][transform_idx].at<double>(0, 3); double ty = tmp_translation_mat_[batch_idx][transform_idx].at<double>(1, 3); double tz = tmp_translation_mat_[batch_idx][transform_idx].at<double>(2, 3);
				Mat diff = matched_predicted_cloud[batch_idx][transform_idx] - matched_target_cloud[batch_idx][transform_idx];
				// saturation...
				/*tmp_diff = diff.colRange(0, transform_dim_ - 1).mul(diff.colRange(0, transform_dim_ - 1));
				reduce(tmp_diff, tmp_diff, 1, CV_REDUCE_SUM);
				tmp_diff = -0.5 * alpha_ * tmp_diff;
				exp(tmp_diff, exp_tmp_diff);
				exp_tmp_diff = exp_tmp_diff * alpha_;
				repeat(exp_tmp_diff, 1, transform_dim_ - 1, repeated_exp_tmp_diff);*/
				// r_x
				CalcTmpDerivative(matched_query_cloud[batch_idx][transform_idx], diff.colRange(0, transform_dim_ - 1), tmp, tmp_replicated, tmp_reduced,
					0, 0, 0, 0, num_points);
				CalcTmpDerivative(matched_query_cloud[batch_idx][transform_idx], diff.colRange(0, transform_dim_ - 1), tmp, tmp_replicated, tmp_reduced,
					cx * sy * cz - sx * sz, -(cx * sy * sz + sx * cz), -cx * cy, 1, num_points);
				CalcTmpDerivative(matched_query_cloud[batch_idx][transform_idx], diff.colRange(0, transform_dim_ - 1), tmp, tmp_replicated, tmp_reduced, 
					sx * sy * cz + cx * sz, (-sx * sy * sz + cx * cz), -sx * cy, 2, num_points);
				// tmp_row = curr_transform_point_weight * repeated_exp_tmp_diff.mul(tmp_reduced.mul(diff.colRange(0, transform_dim_ - 1))) * average_size_inv;
				tmp_row = curr_transform_point_weight * tmp_reduced.mul(diff.colRange(0, transform_dim_ - 1)) * average_size_inv;
				reduce(tmp_row, tmp_single, 1, CV_REDUCE_SUM);
				w_r_x_grad_avg_[transform_idx] = w_r_x_grad_avg_[transform_idx] + tmp_single.at<double>(0, 0) * feature[batch_idx].t();
				// r_y
				CalcTmpDerivative(matched_query_cloud[batch_idx][transform_idx], diff.colRange(0, transform_dim_ - 1), tmp, tmp_replicated, tmp_reduced,
					-sy * cz, sy * sz, cy, 0, num_points);
				CalcTmpDerivative(matched_query_cloud[batch_idx][transform_idx], diff.colRange(0, transform_dim_ - 1), tmp, tmp_replicated, tmp_reduced,
					sx * cy * cz, -sx * cy * sz, sx * sy, 1, num_points);
				CalcTmpDerivative(matched_query_cloud[batch_idx][transform_idx], diff.colRange(0, transform_dim_ - 1), tmp, tmp_replicated, tmp_reduced, 
					-cx * cy * cz, cx * cy * sz, -cx * sy, 2, num_points); // -cx * cy * cz, cx * cy * sz, -cx * sz, 2, num_points);
				// tmp_row = curr_transform_point_weight * repeated_exp_tmp_diff.mul(tmp_reduced.mul(diff.colRange(0, transform_dim_ - 1))) * average_size_inv;
				tmp_row = curr_transform_point_weight * tmp_reduced.mul(diff.colRange(0, transform_dim_ - 1)) * average_size_inv;
				reduce(tmp_row, tmp_single, 1, CV_REDUCE_SUM);
				w_r_y_grad_avg_[transform_idx] = w_r_y_grad_avg_[transform_idx] + tmp_single.at<double>(0, 0) * feature[batch_idx].t();
				// r_z 
				CalcTmpDerivative(matched_query_cloud[batch_idx][transform_idx], diff.colRange(0, transform_dim_ - 1), tmp, tmp_replicated, tmp_reduced,
					-cy * sz, -cy * cz, 0, 0, num_points);
				CalcTmpDerivative(matched_query_cloud[batch_idx][transform_idx], diff.colRange(0, transform_dim_ - 1), tmp, tmp_replicated, tmp_reduced,
					-sx * sy * sz + cx * cz, -sx * sy * cz - cx * sz, 0, 1, num_points);
				CalcTmpDerivative(matched_query_cloud[batch_idx][transform_idx], diff.colRange(0, transform_dim_ - 1), tmp, tmp_replicated, tmp_reduced, 
					cx * sy * sz + sx * cz, cx * sy * cz - sx * sz, 0, 2, num_points);
				/*tmp_row = curr_transform_point_weight * repeated_exp_tmp_diff.mul(tmp_reduced.mul(diff.colRange(0, transform_dim_ - 1))) * average_size_inv;*/
				tmp_row = curr_transform_point_weight * tmp_reduced.mul(diff.colRange(0, transform_dim_ - 1)) * average_size_inv;
				reduce(tmp_row, tmp_single, 1, CV_REDUCE_SUM);
				w_r_z_grad_avg_[transform_idx] = w_r_z_grad_avg_[transform_idx] + tmp_single.at<double>(0, 0) * feature[batch_idx].t();
				// gradient of w_t_x
				/*tmp_single = curr_transform_point_weight * exp_tmp_diff.mul(diff.colRange(0, 1)) * average_size_inv;*/
				tmp_single = curr_transform_point_weight * diff.colRange(0, 1) * average_size_inv;
				w_t_x_grad_avg_[transform_idx] = w_t_x_grad_avg_[transform_idx] + tmp_single.at<double>(0, 0) * feature[batch_idx].t();
				// gradient of w_t_y
				// tmp_single = curr_transform_point_weight * exp_tmp_diff.mul(diff.colRange(1, 2)) * average_size_inv;
				tmp_single = curr_transform_point_weight * diff.colRange(1, 2) * average_size_inv;
				w_t_y_grad_avg_[transform_idx] = w_t_y_grad_avg_[transform_idx] + tmp_single.at<double>(0, 0) * feature[batch_idx].t();
				// gradient of w_t_z
				// tmp_single = curr_transform_point_weight * exp_tmp_diff.mul(diff.colRange(2, 3)) * average_size_inv;
				tmp_single = curr_transform_point_weight * diff.colRange(2, 3) * average_size_inv;
				w_t_z_grad_avg_[transform_idx] = w_t_z_grad_avg_[transform_idx] + tmp_single.at<double>(0, 0) * feature[batch_idx].t();
			}
		}
	}
	// std::cout << "calculate mean gradient" << std::endl;
	for(int i = 0; i < num_transform_; i++)
	{
		w_r_x_grad_[i] = w_r_x_grad_avg_[i] / batch_size_;
		w_r_y_grad_[i] = w_r_y_grad_avg_[i] / batch_size_;
		w_r_z_grad_[i] = w_r_z_grad_avg_[i] / batch_size_;
		w_t_x_grad_[i] = w_t_x_grad_avg_[i] / batch_size_;
		w_t_y_grad_[i] = w_t_y_grad_avg_[i] / batch_size_;
		w_t_z_grad_[i] = w_t_z_grad_avg_[i] / batch_size_;
	}
}

void RigidTransform3D::CalculateFeatureGradient(const Mat& joint_angle, Mat& feature_gradient, int feature_dim)
{
	double pi = 3.14159265;
	int sinusoidal_dim = 3;
	int num_joints = joint_angle.cols; // joint angle row vector
	feature_gradient = Mat::ones(feature_dim, num_joints, CV_64F);
	Mat count = Mat::zeros(num_joints, 1, CV_64F);
	Mat sinusoidal_gradient = Mat::zeros(num_joints, sinusoidal_dim, CV_64F);
	double scale = 1.0 / 180.0 * pi;
	// set the sinusoidal gradient value
	for(int i = 0; i < num_joints; i++)
	{
		sinusoidal_gradient.at<double>(i, 0) = 0;
		sinusoidal_gradient.at<double>(i, 1) = cos(joint_angle.at<double>(0, i) / 180.0 * pi) * scale;
		sinusoidal_gradient.at<double>(i, 2) = -sin(joint_angle.at<double>(0, i) / 180.0 * pi) * scale;
	}
	Mat sinusoidal_value = Mat::zeros(num_joints, sinusoidal_dim, CV_64F);
	// set the sinusoidal value
	for(int i = 0; i < num_joints; i++)
	{
		sinusoidal_value.at<double>(i, 0) = 1;
		sinusoidal_value.at<double>(i, 1) = sin(joint_angle.at<double>(0, i) / 180.0 * pi);
		sinusoidal_value.at<double>(i, 2) = cos(joint_angle.at<double>(0, i) / 180.0 * pi);
	}

	for(int gradient_joint_idx = 0; gradient_joint_idx < num_joints; gradient_joint_idx++)
	{
		for(int dim_idx = 0; dim_idx <= feature_dim; dim_idx++)
		{
			if(dim_idx != 0)
			{
				feature_gradient.at<double>(dim_idx - 1, gradient_joint_idx) = 1;
				int factor = sinusoidal_dim;
				for(int joint_idx = num_joints - 1; joint_idx >= 0; joint_idx--)
				{
					if(joint_idx == num_joints - 1)
					{
						count.at<double>(joint_idx, 0) = dim_idx % factor;	
					}
					else
					{
						count.at<double>(joint_idx, 0) = dim_idx / factor % sinusoidal_dim;	
						factor *= sinusoidal_dim;
					}
					if(joint_idx == gradient_joint_idx)
					{
						feature_gradient.at<double>(dim_idx - 1, gradient_joint_idx) *= sinusoidal_gradient.at<double>(joint_idx, count.at<double>(joint_idx, 0));
					}
					else
					{
						feature_gradient.at<double>(dim_idx - 1, gradient_joint_idx) *= sinusoidal_value.at<double>(joint_idx, count.at<double>(joint_idx, 0));	
					}
				}
			}
		}
	}
	
}

// each weight gradient is different, six different weights...
void RigidTransform3D::CalculateJacobian(const Mat& original_position, const Mat& joint_angle, const Mat& feature, Mat& jacobian, int transform_idx)
{
	int num_joints = joint_angle.cols;
	int dim_points = original_position.cols - 1; // row vector
	Mat rotation_x_angle = w_r_x_[transform_idx] * feature;
	Mat rotation_y_angle = w_r_y_[transform_idx] * feature;
	Mat rotation_z_angle = w_r_z_[transform_idx] * feature;
	Mat translation_x = w_t_x_[transform_idx] * feature;
	Mat translation_y = w_t_y_[transform_idx] * feature;
	Mat translation_z = w_t_z_[transform_idx] * feature;
	double cx =  cos(rotation_x_angle.at<double>(0, 0)); double sx =  sin(rotation_x_angle.at<double>(0, 0));
	double cy =  cos(rotation_y_angle.at<double>(0, 0)); double sy =  sin(rotation_y_angle.at<double>(0, 0));
	double cz =  cos(rotation_z_angle.at<double>(0, 0)); double sz =  sin(rotation_z_angle.at<double>(0, 0));
	Mat w_rx = w_r_x_[transform_idx];
	Mat w_ry = w_r_y_[transform_idx];
	Mat w_rz = w_r_z_[transform_idx];
	Mat w_tx = w_t_x_[transform_idx];
	Mat w_ty = w_t_y_[transform_idx];
	Mat w_tz = w_t_z_[transform_idx];

	double x0 = original_position.at<double>(0, 0);
	double y0 = original_position.at<double>(0, 1);
	double z0 = original_position.at<double>(0, 2);
	Mat feature_gradient;
	CalculateFeatureGradient(joint_angle, feature_gradient, feature.rows);
	Mat grad_x0_phi = (-sy * cz * x0 + sy * sz * y0 + cy * z0) * w_ry + (-cy * sz * x0 - cy * cz * y0) * w_rz + w_tx;
	Mat grad_y0_phi = (cx * sy * cz * x0 - cx * sy * sz * y0 - sx * cz * y0 - cx * cy * z0 - sx * sz * x0) * w_rx
		+ (sx * cy * cz * x0 - sx * cy * cz * y0 + sx * sy * z0) * w_ry 
		+ (-sx * sy * sz * x0 - sx * sy * cz * y0 - cx * sz * y0 + cx * cz * x0) * w_rz
		+ w_ty;
	Mat grad_z0_phi = (sx * sy * cz * x0 + cx * sz * x0 - sx * sy * sz * y0 + cx * cz * y0 - sx * cy * z0) * w_rx  
		+ (-cx * cy * cz * x0 + cx * cy * sz * y0 - cx * sy * z0) * w_ry 
		+ (cx * sy * sz * x0 + sx * cz * x0 + cx * sy * cz * y0 - sx * sz * y0) * w_rz 
		+ w_tz;
	jacobian = Mat::eye(dim_points, num_joints, CV_64F);
	Mat x0_jacobian = grad_x0_phi * feature_gradient;
	Mat y0_jacobian = grad_y0_phi * feature_gradient;
	Mat z0_jacobian = grad_z0_phi * feature_gradient;

	x0_jacobian.copyTo(jacobian.rowRange(0, 1));
	y0_jacobian.copyTo(jacobian.rowRange(1, 2));
	z0_jacobian.copyTo(jacobian.rowRange(2, 3));
}

void RigidTransform3D::CalcTmpDerivative(const Mat& query_points, const Mat& diff, Mat& tmp, Mat& tmp_replicated, Mat& tmp_reduced, double first, double second, double third, int col_idx, int num_points)
{
	tmp.at<double>(0, 0) = first; tmp.at<double>(0, 1) = second; tmp.at<double>(0, 2) = third;
	repeat(tmp, num_points, 1, tmp_replicated);
	tmp_replicated = tmp_replicated.mul(query_points.colRange(0, transform_dim_ - 1));
	reduce(tmp_replicated, tmp_reduced.colRange(col_idx, col_idx + 1), 1, CV_REDUCE_SUM);
}

void RigidTransform3D::Update()
{
	for(int i = 0; i < num_transform_; i++)
	{
		w_r_x_[i] = w_r_x_[i] - learning_rate_ * w_r_x_grad_[i];
		w_r_y_[i] = w_r_y_[i] - learning_rate_ * w_r_y_grad_[i];
		w_r_z_[i] = w_r_z_[i] - learning_rate_ * w_r_z_grad_[i];
		w_t_x_[i] = w_t_x_[i] - learning_rate_ * w_t_x_grad_[i];
		w_t_y_[i] = w_t_y_[i] - learning_rate_ * w_t_y_grad_[i];
		w_t_z_[i] = w_t_z_[i] - learning_rate_ * w_t_z_grad_[i];
	}
	
}

Mat RigidTransform3D::w(int idx)
{
	int num_weights = 6;

	Mat weight = Mat::zeros(num_weights, feature_dim_, CV_64F);
	w_r_x_[idx].copyTo(weight.rowRange(0, 1));
	w_r_y_[idx].copyTo(weight.rowRange(1, 2));
	w_r_z_[idx].copyTo(weight.rowRange(2, 3));
	w_t_x_[idx].copyTo(weight.rowRange(3, 4));
	w_t_y_[idx].copyTo(weight.rowRange(4, 5));
	w_t_z_[idx].copyTo(weight.rowRange(5, 6));

	return weight;
}

void RigidTransform3D::set_w(const Mat& w, int idx)
{
	w.rowRange(0, 1).copyTo(w_r_x_[idx]);
	w.rowRange(1, 2).copyTo(w_r_y_[idx]);
	w.rowRange(2, 3).copyTo(w_r_z_[idx]);
	w.rowRange(3, 4).copyTo(w_t_x_[idx]);
	w.rowRange(4, 5).copyTo(w_t_y_[idx]);
	w.rowRange(5, 6).copyTo(w_t_z_[idx]);
}

// divide(average_size, tmp_single, tmp_single);
// tmp_col = diff.colRange(2, 3);
// reduce(tmp_col.mul(point_weight[batch_idx][transform_idx]), tmp_single, 0, CV_REDUCE_AVG);

// divide(average_size, tmp_single, tmp_single);
// tmp_col = diff.colRange(1, 2);
// reduce(tmp_col.mul(point_weight[batch_idx][transform_idx]), tmp_single, 0, CV_REDUCE_AVG);

// divide(average_size, tmp_single, tmp_single);
// reduce(tmp_col.mul(point_weight[batch_idx][transform_idx]), tmp_single, 0, CV_REDUCE_AVG);

// tmp_row = tmp_row * average_size_inv;
// divide(average_size, tmp_row, tmp_row);
// reduce(tmp_reduced.mul(diff.colRange(0, transform_dim_ - 1)).mul(curr_transform_point_weight), tmp, 0, CV_REDUCE_AVG);

// tmp_row = tmp_row * average_size_inv;
// divide(average_size, tmp_row, tmp_row);
// reduce(tmp_reduced.mul(diff.colRange(0, transform_dim_ - 1)).mul(curr_transform_point_weight), tmp, 0, CV_REDUCE_AVG);

// tmp_row = tmp_row * average_size_inv;
// divide(average_size, tmp_row, tmp_row);
// reduce(tmp_reduced.mul(diff.colRange(0, transform_dim_ - 1)).mul(curr_transform_point_weight), tmp_row, 0, CV_REDUCE_AVG);