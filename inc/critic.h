#ifndef _CRITIC_H
#define _CRITIC_H

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/types_c.h"
#include "opencv2/nonfree/features2d.hpp"
#include <iostream>

class Critic
{
private:
    double td_error_, alpha_, gamma_, weight_range_;
    int input_dim_, output_dim_; // size
    double prev_reward_, average_reward_;
    cv::Mat new_value_, value_, state_feature_, v_;
public:
    Critic(int input_dim, int output_dim, double alpha, double gamma);
    ~Critic();
    void Init(int input_dim, int output_dim, double alpha, double gamma, cv::Mat weight);
    double Update(double reward, const cv::Mat& new_state_feature, bool update);
    double td_error(); // return td error
    cv::Mat v();
    double new_value();
    double value();
    double intrinsic_reward();
    void set_v(const cv::Mat& v);
    void set_alpha(double alpha);
};

#endif
