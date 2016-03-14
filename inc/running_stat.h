#ifndef _RUNNING_STAT_H
#define _RUNNING_STAT_H
#include <math.h>
#include "opencv2/core/core.hpp"

class RunningStat
{
public:
    RunningStat(int dim_y);
	RunningStat();

    void Clear()
    {
        x_n_ = 0;
    }

    void Push(double x, cv::Mat& y = cv::Mat());
    

    int NumDataValues() const;
    double Mean() const;
    double VarianceX() const;
	cv::Mat VarianceY() const;
	cv::Mat Covariance() const;
    double StandardDeviationX() const;
	cv::Mat StandardDeviationY() const;
	cv::Mat Correlation() const;

private:
    int x_n_, dim_y_;
	double alpha_;
    double x_old_mean_, x_new_mean_, x_old_var_, x_new_var_;
	cv::Mat y_old_mean_, y_new_mean_, y_old_var_, y_new_var_, cov_old_, cov_new_;
};

#endif