#include "../inc/actor.h"

using namespace std;

Actor::Actor(int input_dim, int output_dim, double beta, double eta, cv::Mat& actions)
{
	input_dim_ = input_dim;
    output_dim_ = output_dim; // size
    beta_ = beta; // policy learning rate
    eta_ = eta; // natural gradient learning rate
    pi_ = cv::Mat::zeros(output_dim_, 1, CV_64F); // policy probability
    theta_ = cv::Mat::zeros(output_dim_, input_dim_, CV_64F); // policy weights
    w_ = cv::Mat::zeros(output_dim_, input_dim_, CV_64F); // natural gradient 
    psi_ = cv::Mat::zeros(output_dim_, input_dim_, CV_64F); // compatible feature
	ones_ = cv::Mat::ones(output_dim_, 1, CV_64F);
	actions_ = cv::Mat::zeros(output_dim_, 1, CV_64F);
	actions.copyTo(actions_);
	cv::RNG rng;
	rng.fill(theta_, cv::RNG::UNIFORM, -0.1, 0.1);
	rng.fill(pi_, cv::RNG::UNIFORM, 0, 1);
	cv::Mat normalization = cv::Mat::zeros(1, 1, CV_64F);
	cv::reduce(pi_, normalization, 0, CV_REDUCE_SUM); // column-wise reduce
	cv::divide(pi_, cv::repeat(normalization, output_dim_, 1), pi_);
}

Actor::~Actor()
{
}

double Actor::GetAction(double random_num)
{
	if(random_num >= 0 && random_num <= 1)
	{
		double cummulative_probability = 0;
		for(int action_idx = 0; action_idx < output_dim_; action_idx++)
		{
			cummulative_probability = cummulative_probability + pi_.at<double>(action_idx, 0);	
			if(random_num < cummulative_probability)
			{
				return actions_.at<double>(action_idx, 0);	
			}
		}
	}
	else
	{
		double min_value = 0;
		double max_value = 0;
		cv::Point min_location, max_location;
		min_location.x = 0; min_location.y = 0; max_location.x = 0; max_location.y = 0;
		cv::minMaxLoc(pi_, &min_value, &max_value, &min_location, &max_location);
		return actions_.at<double>(max_location.y, 0);
	}
}

void Actor::Update(double td_error, const cv::Mat& state_feature, bool update)
{
	cv::Mat prop = theta_ * state_feature;
	cv::Mat normalization = cv::Mat::zeros(1, 1, CV_64F);
	cv::exp(prop, prop);
	cv::reduce(prop, normalization, 0, CV_REDUCE_SUM); // column-wise reduce
	cv::divide(prop, cv::repeat(normalization, output_dim_, 1), pi_);

	if(update)
	{
		psi_ = (ones_ - pi_) * state_feature.t();
		w_ = w_ - eta_ * psi_ * psi_.t() * w_ + eta_ * td_error * psi_;
		theta_ = theta_ + beta_ * w_;
	}
}

void Actor::set_w(const cv::Mat& w){
    w.copyTo(w_);
}

void Actor::set_theta(const cv::Mat& theta){
	theta.copyTo(theta_);
}

cv::Mat Actor::w(){
    return w_;
}

cv::Mat Actor::theta(){
    return theta_;
}

double Actor::beta(){
    return beta_;
}

void Actor::set_beta(double beta){
    beta_ = beta;
    // cout << "actual beta: " << beta_ << endl;
}

//void Actor::RecordNorm(Mat& data_mat, int count){
//    // check the size of matrix
//    if(data_mat.cols != 4){
//        // cout << "data matrix size incorrect... " << endl;
//        exit(0);
//    }
//    // recording data...
//    data_mat.at<double>(count, 0) = norm(w_, NORM_L2);
//    data_mat.at<double>(count, 1) = norm(norm_grad_w_, NORM_L2);
//    data_mat.at<double>(count, 2) = norm(nat_grad_w_, NORM_L2);
//    data_mat.at<double>(count, 3) = norm(delta_nat_grad_w_, NORM_L2);
//}
