#ifndef _ACTOR_H
#define _ACTOR_H

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/types_c.h"
#include "opencv2/nonfree/features2d.hpp"
#include <random>
#include <iostream>
#include <array>
class Actor{
private:
    int input_dim_, output_dim_; // size
    double beta_, eta_;
    cv::Mat pi_, theta_, w_, psi_, ones_, actions_;
	std::default_random_engine generator_;
public:
    Actor(int input_dim, int output_dim, double beta, double eta, cv::Mat& actions);
    ~Actor();
	double GetAction(double random_num);
	void Update(double td_error, const cv::Mat& state_feature, bool update);
    void UpdateNormalGradient(const cv::Mat& action, const cv::Mat& state);
    void UpdateNaturalGradient(double td_error);
    void UpdateActor();
    void RecordNorm(cv:: Mat& data_mat, int count);
    void set_cov(const cv::Mat& cov, int rate_flag);
    void set_w(const cv::Mat& w);
    void set_beta(double beta);
    double beta();
    cv::Mat output();
    cv::Mat cov();
    cv::Mat w();
	void set_theta(const cv::Mat& theta);
    cv::Mat theta();
    
};

#endif
