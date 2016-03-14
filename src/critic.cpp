#include "../inc/critic.h"
#include <random>

using namespace std;

Critic::Critic(int input_dim, int output_dim, double alpha, double gamma)
{
	td_error_ = 0;
    prev_reward_ = 0;
	average_reward_ = 0;
    input_dim_ = input_dim; 
    output_dim_ = output_dim; // size
    alpha_ = alpha; 
    gamma_ = gamma; 
    new_value_ = cv::Mat::zeros(output_dim_, 1, CV_64F); // current output
    value_ = cv::Mat::zeros(output_dim_, 1, CV_64F); // previous output
	state_feature_ = cv::Mat::zeros(input_dim_, 1, CV_64F);
    v_ = cv::Mat::zeros(output_dim_, input_dim_, CV_64F);
	cv::RNG rng;
	rng.fill(v_, cv::RNG::UNIFORM, -0.1, 0.1);
}

Critic::~Critic()
{
}

double Critic::Update(double reward, const cv::Mat& new_state_feature, bool update)
{
	new_value_ = v_ * new_state_feature; // calculate value t + 1
	average_reward_ = (1 - gamma_) * average_reward_ + gamma_  * reward; // also for t + 1
	td_error_ = reward - average_reward_ + new_value_.at<double>(0, 0) - value_.at<double>(0, 0); // td error is only for t...
	if(update)
	{
		v_ = v_ + alpha_ * td_error_ * state_feature_.t();
		/*std::cout << "value weights: "; 
		for(int i = 0; i < input_dim_; i++)
		{
			std::cout << v_.at<double>(0, i) << " ";
		}
		std::cout << std::endl;*/
	}
	new_value_.copyTo(value_);
	new_state_feature.copyTo(state_feature_);
	return td_error_;
}

double Critic::td_error()
{
    return td_error_;
}

cv::Mat Critic::v()
{
    return v_;
}

double Critic::new_value()
{
  return new_value_.at<double>(0, 0);
}

double Critic::value()
{
  return value_.at<double>(0, 0);
}

void Critic::set_v(const cv::Mat& v)
{
    v.copyTo(v_);
}

void Critic::set_alpha(double alpha)
{
    alpha_ = alpha;
}

//void Critic::RecordNorm(Mat& data_mat, int count, double r){
//    if(data_mat.cols != 4){
//        cout << "data matrix size incorrect... " << endl;
//        exit(0);
//    }
//    // recording data...
//    data_mat.at<double>(count, 0) = norm(w_, NORM_L2); // recording data...
//    data_mat.at<double>(count, 1) = norm(delta_w_, NORM_L2);
//    data_mat.at<double>(count, 2) = td_error_;
//    data_mat.at<double>(count, 3) = r;
//}
