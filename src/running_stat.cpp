#include "../inc/running_stat.h"

using namespace cv;

RunningStat::RunningStat()
{
	RunningStat(0);
}

RunningStat::RunningStat(int dim_y) : x_n_(0), dim_y_(dim_y)
{
	alpha_ = 2.5e-4;
	if(dim_y != 0)
	{
		y_old_mean_ = Mat::zeros(1, dim_y, CV_64F);
		y_new_mean_ = Mat::zeros(1, dim_y, CV_64F);
		y_old_var_ = Mat::zeros(1, dim_y, CV_64F);
		y_new_var_ = Mat::zeros(1, dim_y, CV_64F);
		cov_old_ = Mat::zeros(1, dim_y, CV_64F);
		cov_new_ = Mat::zeros(1, dim_y, CV_64F);
	}
}

void RunningStat::Push(double x, Mat& y)
{
    x_n_++;

    // See Knuth TAOCP vol 2, 3rd edition, page 232
    if (x_n_ == 1)
    {
        x_new_mean_ = x;
        x_old_mean_ = x;
        x_new_var_ = 0.0;
        x_old_var_ = 0.0;
    }
    else
    {
        x_new_mean_ = alpha_ * x + (1 - alpha_) * x_old_mean_;
        x_new_var_ = alpha_ * (x - x_new_mean_) * (x - x_old_mean_) + (1 - alpha_) * x_old_var_;
    
        // set up for next iteration
        x_old_mean_ = x_new_mean_; 
        x_old_var_ = x_new_var_;
    }

	if(dim_y_ != 0 && !y.empty())
	{
		if (x_n_ == 1)
		{
			y.copyTo(y_old_mean_);
			y.copyTo(y_new_mean_);
		}
		else
		{
			y_new_mean_ = alpha_ * y  + (1 - alpha_) * y_old_mean_;
			Mat tmp = (y - y_new_mean_).t() * (y - y_old_mean_);
			Mat y_var_update = tmp.diag(0);
			y_new_var_ = alpha_ * y_var_update.t() + (1 - alpha_) * y_old_var_;
			cov_new_ = alpha_ * (y - y_new_mean_) * (x - x_new_mean_) + (1- alpha_) * cov_old_;;

			// set up for next iteration
			y_new_mean_.copyTo(y_old_mean_);
			y_new_var_.copyTo(y_old_var_);
			cov_new_.copyTo(cov_old_);
		}
	}
}

int RunningStat::NumDataValues() const
{
    return x_n_;
}

double RunningStat::Mean() const
{
    return x_n_ > 0 ? x_new_mean_ : 0.0;
}

double RunningStat::VarianceX() const
{
    // return (x_n_ > 1) ? x_new_var_ / (x_n_ - 1) : 0.0;
    return (x_n_ > 1) ? x_new_var_ : 0.0;
}

Mat RunningStat::Covariance() const
{
    // return (x_n_ > 1) ? cov_new_ / (x_n_ - 1) : Mat::zeros(1, dim_y_, CV_64F);
    return (x_n_ > 1) ? cov_new_ : Mat::zeros(1, dim_y_, CV_64F);
}

double RunningStat::StandardDeviationX() const
{
    return sqrt(VarianceX());
}

Mat RunningStat::VarianceY() const
{
    // return (dim_y_ != 0 && x_n_ > 1) ? y_new_var_ / (x_n_ - 1) : Mat::zeros(1, dim_y_, CV_64F);
    return (dim_y_ != 0 && x_n_ > 1) ? y_new_var_ : Mat::zeros(1, dim_y_, CV_64F);
}

Mat RunningStat::StandardDeviationY() const
{
	Mat y_std_dev = Mat::zeros(1, 1, CV_64F);
	if(dim_y_ != 0)
	{
		Mat y_var = VarianceY();
		sqrt(y_var, y_std_dev);
	}
	return y_std_dev;
}

Mat RunningStat::Correlation() const
{
	double x_std_dev = StandardDeviationX();
	Mat y_std_dev = StandardDeviationY();
	Mat std_dev = y_std_dev * x_std_dev;
	Mat covariance = Covariance();
	Mat corr = Mat::zeros(cov_new_.rows, cov_new_.cols, CV_64F);
	if(dim_y_ != 0)
	{
		divide(covariance, std_dev, corr);
	}
	return corr;
}


/*void RunningStat::Push(double x, Mat& y)
{
    x_n_++;

    // See Knuth TAOCP vol 2, 3rd edition, page 232
    if (x_n_ == 1)
    {
        x_old_mean_ = x_new_mean_ = x;
        x_old_var_ = 0.0;
    }
    else
    {
        x_new_mean_ = x_old_mean_ + (x - x_old_mean_)/x_n_;
        x_new_var_ = x_old_var_ + (x - x_old_mean_)*(x - x_new_mean_);
    
        // set up for next iteration
        x_old_mean_ = x_new_mean_; 
        x_old_var_ = x_new_var_;
    }

	if(dim_y_ != 0 && !y.empty())
	{
		if (x_n_ == 1)
		{
			y.copyTo(y_old_mean_);
			y.copyTo(y_new_mean_);
		}
		else
		{
			y_new_mean_ = y_old_mean_ + (y - y_old_mean_) / x_n_;
			Mat tmp = (y - y_old_mean_).t() * (y - y_new_mean_);
			Mat y_var_update = tmp.diag(0);
			y_new_var_ = y_old_var_ + y_var_update.t();
			cov_new_ = cov_old_ + (y - y_new_mean_) * (x - x_new_mean_);

			// set up for next iteration
			y_new_mean_.copyTo(y_old_mean_);
			y_new_var_.copyTo(y_old_var_);
			cov_new_.copyTo(cov_old_);
		}
	}
}*/