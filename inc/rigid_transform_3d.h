#ifndef _RIGID_TRANSFORM_3D_H
#define _RIGID_TRANSFORM_3D_H
#include "../inc/transform.h"

class RigidTransform3D: public Transform
{
private: 
	std::vector<std::vector<cv::Mat>> transform_;
	// three rotation weights
	std::vector<cv::Mat> w_r_x_;
	std::vector<cv::Mat> w_r_y_;
	std::vector<cv::Mat> w_r_z_;
	// three translation weights
	std::vector<cv::Mat> w_t_x_;
	std::vector<cv::Mat> w_t_y_;
	std::vector<cv::Mat> w_t_z_;

	// gradients
	std::vector<cv::Mat> w_r_x_grad_avg_;
	std::vector<cv::Mat> w_r_y_grad_avg_;
	std::vector<cv::Mat> w_r_z_grad_avg_;
	std::vector<cv::Mat> w_t_x_grad_avg_;
	std::vector<cv::Mat> w_t_y_grad_avg_;
	std::vector<cv::Mat> w_t_z_grad_avg_;

	std::vector<cv::Mat> w_r_x_grad_;
	std::vector<cv::Mat> w_r_y_grad_;
	std::vector<cv::Mat> w_r_z_grad_;
	std::vector<cv::Mat> w_t_x_grad_;
	std::vector<cv::Mat> w_t_y_grad_;
	std::vector<cv::Mat> w_t_z_grad_;

	int feature_dim_;
	int num_transform_;
	int batch_size_;
	int transform_dim_;
	double learning_rate_;
	double alpha_;
	std::vector<std::vector<cv::Mat>> tmp_rotation_mat_x_;
	std::vector<std::vector<cv::Mat>> tmp_rotation_mat_y_;
	std::vector<std::vector<cv::Mat>> tmp_rotation_mat_z_;
	std::vector<std::vector<cv::Mat>> tmp_translation_mat_;

public:
	
	RigidTransform3D(int num_transform, int feature_dim, int batch_size, double learning_rate, double alpha);
	void CalcSingleTransform(const cv::Mat& feature, int transform_idx, int batch_idx);
	void RigidTransform3D::TransformPoint(cv::Mat& input_point, cv::Mat& output_point, int index, int batch_idx);
	void CalcTransformation(std::vector<cv::Mat>& feature) override;
	void TransformCloud(const cv::Mat& input_cloud, std::vector<std::vector<cv::Mat>>& output_cloud) override;
	void CalcTmpDerivative(const cv::Mat& query_points, const cv::Mat& diff, cv::Mat& tmp, cv::Mat& tmp_replicated, cv::Mat& tmp_reduced, double first, double second, double third, int row_idx, int num_points);
	void CalculateGradient(const std::vector<std::vector<cv::Mat>>& matched_target_cloud, const std::vector<std::vector<cv::Mat>>& matched_predicted_cloud, const std::vector<std::vector<cv::Mat>>& matched_query_cloud, const std::vector<std::vector<cv::Mat>>& point_weight, const std::vector<cv::Mat>& feature) override;
	static void CalculateFeatureGradient(const cv::Mat& joint_angle, cv::Mat& feature_gradient, int feature_dim);
	void CalculateJacobian(const cv::Mat& current_position, const cv::Mat& joint_angle, const cv::Mat& feature, cv::Mat& jacobian, int transform_idx);
	void Update() override;
	cv::Mat w(int idx) override;
	//void UpdateKinematicStructure();
	void set_w(const cv::Mat& w, int idx) override;

	// inline functions
	std::vector<std::vector<cv::Mat>> transform() const
	{
		return transform_;
	}

	std::vector<cv::Mat> w_r_x() const
	{
		return w_r_x_;
	}

	std::vector<cv::Mat> w_r_y() const
	{
		return w_r_y_;
	}

	std::vector<cv::Mat> w_r_z() const
	{
		return w_r_z_;
	}

	std::vector<cv::Mat> w_t_x() const
	{
		return w_t_x_;
	}

	std::vector<cv::Mat> w_t_y() const
	{
		return w_t_y_;
	}

	std::vector<cv::Mat> w_t_z() const
	{
		return w_t_z_;
	}

	void set_w_r_x(const std::vector<cv::Mat>& mats)
	{
		w_r_x_ = mats;
	}

	void set_w_r_y(const std::vector<cv::Mat>& mats)
	{
		w_r_y_ = mats;
	}

	void set_w_r_z(const std::vector<cv::Mat>& mats)
	{
		w_r_z_ = mats;
	}

	void set_w_t_x(const std::vector<cv::Mat>& mats)
	{
		w_t_x_ = mats;
	}

	void set_w_t_y(const std::vector<cv::Mat>& mats)
	{
		w_t_y_ = mats;
	}

	void set_w_t_z(const std::vector<cv::Mat>& mats)
	{
		w_t_z_ = mats;
	}

	std::vector<cv::Mat> w_r_x_grad() const
	{
		return w_r_x_grad_;
	}

	std::vector<cv::Mat> w_r_y_grad() const
	{
		return w_r_y_grad_;
	}

	std::vector<cv::Mat> w_r_z_grad() const
	{
		return w_r_z_grad_;
	}

	std::vector<cv::Mat> w_t_x_grad() const
	{
		return w_t_x_grad_;
	}

	std::vector<cv::Mat> w_t_y_grad() const
	{
		return w_t_y_grad_;
	}

	std::vector<cv::Mat> w_t_z_grad() const
	{
		return w_t_z_grad_;
	}
	double alpha() const
	{
		return alpha_;
	}
	void set_alpha(double alpha)
	{
		alpha_ = alpha;
	}
};
#endif

/*cv::Mat w_r_x(int idx) const
{
	return w_r_x_[idx];
}*/

/*cv::Mat w_r_y(int idx) const
{
	return w_r_y_[idx];
}*/

/*cv::Mat w_r_z(int idx) const
{
	return w_r_z_[idx];
}*/

/*cv::Mat w_t_x(int idx) const
{
	return w_t_x_[idx];
}*/

/*cv::Mat w_t_y(int idx) const
{
	return w_t_y_[idx];
}*/

/*cv::Mat w_t_z(int idx) const
{
	return w_t_z_[idx];
}*/