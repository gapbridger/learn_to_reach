#include "../inc/ellipse.h"
using namespace ellipse;
Ellipse::Ellipse(double initial_x, double initial_y, double initial_long_axis, double initial_short_axis, double initial_angle, double radius, cv::Mat ini_transformation) :
	home_mu_x_(initial_x),
	home_mu_y_(initial_y),
	home_long_axis_(initial_long_axis),
	home_short_axis_(initial_short_axis),
	home_angle_(initial_angle),
	mu_x_(initial_x),
	mu_y_(initial_y),
	long_axis_(initial_long_axis),
	short_axis_(initial_short_axis),
	angle_(initial_angle),
	radius_(radius),
	transform_(Transform()),
	home_transform_(Transform()),
	ini_transformation_(ini_transformation)
{
	mu_ = cv::Mat::zeros(2, 1, CV_64F);
	cov_ = cv::Mat::eye(2, 2, CV_64F);
	cov_inv_ = cv::Mat::eye(2, 2, CV_64F);
	prev_mu_ = cv::Mat::zeros(2, 1, CV_64F);
	prev_cov_ = cv::Mat::eye(2, 2, CV_64F);
	prev_cov_inv_ = cv::Mat::eye(2, 2, CV_64F);
	home_mu_ = cv::Mat::zeros(2, 1, CV_64F);
	home_cov_ = cv::Mat::eye(2, 2, CV_64F);
	home_cov_inv_ = cv::Mat::eye(2, 2, CV_64F);
	
	eigen_value_ = cv::Mat::zeros(2, 2, CV_64F);
	eigen_vector_ = cv::Mat::zeros(2, 2, CV_64F);
	home_eigen_value_ = cv::Mat::zeros(2, 2, CV_64F);
	home_eigen_vector_ = cv::Mat::zeros(2, 2, CV_64F);			
			
	a_home_ = cv::Mat::eye(3, 3, CV_64F); // home ellipse A
	inv_a_home_ = cv::Mat::eye(3, 3, CV_64F); // home ellipse inv A

	prev_a_ = cv::Mat::eye(3, 3, CV_64F); // ellipse A
	prev_inv_a_ = cv::Mat::eye(3, 3, CV_64F); // inverse of A

	a_ = cv::Mat::eye(3, 3, CV_64F); // ellipse A
	inv_a_ = cv::Mat::eye(3, 3, CV_64F); // inverse of A

	set_a_home(0);
	set_a();
	set_inv_a_home();
	set_inv_a();
	set_home_mu();
	set_home_cov();
	eta_ = 0;

	/*SetHomeDistMetric();	
	SetHomeElipsMuCov();*/
	

	/*SetMu(mu_x_, mu_y_, mu_);
	SetCovariance(long_axis_, short_axis_, angle_, cov_);
	CalcInvCovar(cov_, cov_inv_);*/
}

void Ellipse::TransformEllipse()
{		
	// this is where the ellipse is transformed or here we should update A like this...	
	inv_a_ = inv_a_home_ * transform_.transform_inv();
}
// calculate mahalanobis distance
double Ellipse::MahalanobisDistance(const cv::Mat& data, const cv::Mat& inv_a)
{
	// calculate mahalanobis distance 
	// size of data is 3 by 1, size of inv_a_ is 3 by 3
	cv::Mat dist_mat = data.t() * inv_a.t() * inv_a * data;	
	double dist = sqrt(dist_mat.at<double>(0, 0) - 1);
	return dist;
}
// judge whether inside an ellipse
bool Ellipse::CheckInsideEllipse(const cv::Mat& point, const cv::Mat& inv_a)
{
	// radius is in the sense of mahalanobis distance
	if(MahalanobisDistance(point, inv_a) <= radius_)
		return true;
	else
		return false;
}

void Ellipse::GetKeyPointInEllipse(const cv::Mat& descriptor, const cv::Mat& key_point, cv::Mat& elips_descriptor, cv::Mat& elips_key_point, cv::Mat& elips_distance, int curr_frame_flag)
{
	int num_key_point = key_point.rows;	
	int num_cols = descriptor.cols;
	int num_elips_key_point = 0;
	double distance = 0;
	std::vector<int> idx_list;
	std::vector<double> dist_list;	
	// homogenous representation
	cv::Mat pt = cv::Mat::zeros(3, 1, CV_64F);
	pt.at<double>(2, 0) = 1.0;
	
	// double distance = 0;
	for(int i = 0; i < num_key_point; i++)
	{		
		pt.at<double>(0, 0) = key_point.at<float>(i, 0); // x
		pt.at<double>(1, 0) = key_point.at<float>(i, 1); // y
		
		if(curr_frame_flag)
			distance = MahalanobisDistance(pt, inv_a_);
		else
			distance = MahalanobisDistance(pt, prev_inv_a_);

		if(distance <= radius_)
		{
			idx_list.push_back(i);
			dist_list.push_back(distance);
		}		
	}
	// allocate memory
	num_elips_key_point = idx_list.size();
	elips_descriptor.create(num_elips_key_point, num_cols, CV_32F);
	elips_key_point.create(num_elips_key_point, 2, CV_32F);
	elips_distance.create(num_elips_key_point, 1, CV_64F);
	// assign value
	for(int i = 0; i < num_elips_key_point; i++)
	{
		key_point.rowRange(idx_list[i], idx_list[i] + 1).copyTo(elips_key_point.rowRange(i, i + 1));
		descriptor.rowRange(idx_list[i], idx_list[i] + 1).copyTo(elips_descriptor.rowRange(i, i + 1));
		elips_distance.at<double>(i, 0) = dist_list[i];
	}
}

void Ellipse::ClassifyPointsHomeEllipse(MatL match_data, cv::Mat& classified_points, cv::Mat& motion_ratio, cv::Mat& maha_dist)
{
	int num_classified_data = 0;
	double classification_dist_threshold =  0.1;
	double improve_percent = 0;
	
	std::vector<int> idx_list;
	if(match_data[0].cols != 13)
	{
		std::cout << "match data size incorrect..." << std::endl;
		exit(0);
	}
	for(int i = 0; i < match_data.size(); i++)
	{		
		if(match_data[i].at<double>(0, 12) > 0.85)
			idx_list.push_back(i);
	}
	num_classified_data = idx_list.size();
	classified_points.create(num_classified_data, 2, CV_64F);
	motion_ratio.create(num_classified_data, 1, CV_64F);
	maha_dist.create(num_classified_data, 1, CV_64F);
	for(int i = 0; i < num_classified_data; i++)
	{
		classified_points.at<double>(i, 0) = match_data[idx_list[i]].at<double>(0, 13); // 9 current ref point x...
		classified_points.at<double>(i, 1) = match_data[idx_list[i]].at<double>(0, 14); // 10 current ref point y...
		motion_ratio.at<double>(i, 0) = match_data[idx_list[i]].at<double>(0, 12); // 5 12		
		maha_dist.at<double>(i, 0) = match_data[idx_list[i]].at<double>(0, 11); // 6 11
	}		
}


void Ellipse::UpdateHomeEllipse(cv::Mat& points, cv::Mat& motion_ratio, cv::Mat& maha_dist)
{
	char output[400];
	if(points.rows >= 5)
	{
		cv::Mat tmp_cov = cv::Mat::zeros(2, 2, CV_64F);				
		cv::Mat tmp_cov_inv = cv::Mat::zeros(2, 2, CV_64F);				
		cv::Mat tmp_mu = cv::Mat::zeros(2, 1, CV_64F);
		cv::Mat tmp_points = points.t(); // cv::Mat::zeros(points.rows, points.cols, CV_64F);
		
		MVEE(tmp_points, tmp_mu, tmp_cov_inv, 0.1);
		cv::invert(tmp_cov_inv, tmp_cov);
				
		home_mu_ = home_mu_ + eta_ * (tmp_mu - home_mu_);
		home_cov_ = home_cov_ + eta_ * (tmp_cov - home_cov_);

		set_a_home(1);
		set_inv_a_home();
		// UpdateHomeDistMetricByMuCov();			
	}	
}

void Ellipse::MVEE(cv::Mat& P, cv::Mat& c, cv::Mat& A, double tolerance)
{
	int d = P.rows; // P should be transposed...
	int N = P.cols;
	int count = 1;
	double err = 1;
	double maximum = 0;
	int j[2] = {0, 0};
	double step_size = 0;

	cv::Mat Q = cv::Mat::ones(d + 1, N, CV_64F);
	P.copyTo(Q.rowRange(0, d));	
	cv::Mat u = cv::Mat::ones(N, 1, CV_64F);	
	u = u * (1.0 / N);
	cv::Mat diag_u = cv::Mat::zeros(N, N, CV_64F);
	cv::Mat new_u = cv::Mat::zeros(N, 1, CV_64F);
	cv::Mat X = cv::Mat::zeros(N, N, CV_64F);
	cv::Mat inv_X = cv::Mat::zeros(N, N, CV_64F);
	cv::Mat M = cv::Mat::zeros(N, 1, CV_64F);	

		
	while(err > tolerance)
	{		
		// this->diag(u, diag_u);
		diag_u = cv::Mat::diag(u);
		X = Q * diag_u * Q.t(); //        % X = \sum_i ( u_i * q_i * q_i')  is a (d+1)x(d+1) matrix
		cv::invert(X, inv_X);
		M = (Q.t() * inv_X * Q).diag(0);
		cv::minMaxIdx(M, 0, &maximum, 0, j);		 
		// M = diag(Q.t() * inv_X * Q); // % M the diagonal vector of an NxN matrix
		// [maximum j] = max(M);
		step_size = (maximum - d -1) / ((d + 1) * (maximum - 1));
		new_u = (1 - step_size) * u;
		new_u.at<double>(j[0], 0) = new_u.at<double>(j[0], 0) + step_size;
		// new_u(j) = new_u(j) + step_size;
		count = count + 1;
		err = cv::norm(new_u - u, cv::NORM_L2);
		new_u.copyTo(u);
		// u = new_u;
	}
	diag_u = cv::Mat::diag(u);
	// this->diag(u, diag_u);
	c = P * u;
	cv::invert(P * diag_u * P.t() - c * c.t(), A);
	A = A * (1.0 / d);	
}

void Ellipse::CopyToPrev()
{
	prev_mu_x_ = mu_x_;
	prev_mu_y_ = mu_y_;
	prev_long_axis_ = long_axis_;
	prev_short_axis_ = short_axis_;
	prev_angle_ = angle_;

	mu_.copyTo(prev_mu_);
	cov_.copyTo(prev_cov_);

	transform_.CopyToPrev();
	home_transform_.CopyToPrev();

	a_.copyTo(prev_a_);
	inv_a_.copyTo(prev_inv_a_);
}

void Ellipse::UpdateEllipseVisualizationParameters(int initial_specification_flag)
{
	cv::Mat home_conic = cv::Mat::zeros(3, 3, CV_64F);
	cv::Mat conic = cv::Mat::zeros(3, 3, CV_64F);
	cv::Mat inv_transform = transform_.transform_inv();
	cv::Mat inv_ini_transformation = cv::Mat::eye(3, 3, CV_64F);
	// go to conic
	double a = 0, b= 0, c = 0, d = 0, e = 0, f = 0, g = 0;
	a = pow(cos(home_angle_), 2) / pow(home_long_axis_, 2) + pow(sin(home_angle_), 2) / pow(home_short_axis_, 2);	
	b = 2 * cos(home_angle_) * sin(home_angle_) * (1 / pow(home_long_axis_, 2) - 1 / pow(home_short_axis_, 2));
	c = pow(sin(home_angle_), 2) / pow(home_long_axis_, 2) + pow(cos(home_angle_), 2) / pow(home_short_axis_, 2);
	d = -(2 * a * home_mu_x_ + b * home_mu_y_);
	e = -(2 * c * home_mu_y_ + b * home_mu_x_);
	f = a * pow(home_mu_x_, 2) + b * home_mu_x_ * home_mu_y_ + c * pow(home_mu_y_, 2) - 1;
	
	home_conic.at<double>(0, 0) = a;         home_conic.at<double>(0, 1) = b / 2.0;
	home_conic.at<double>(0, 2) = d / 2.0;   home_conic.at<double>(1, 0) = b / 2.0;
	home_conic.at<double>(1, 1) = c;         home_conic.at<double>(1, 2) = e / 2.0;
	home_conic.at<double>(2, 0) = d / 2.0;   home_conic.at<double>(2, 1) = e / 2.0;
	home_conic.at<double>(2, 2) = f;

	// transform
	cv::invert(ini_transformation_, inv_ini_transformation);
	if(initial_specification_flag){		
		conic = inv_ini_transformation.t() * home_conic * inv_ini_transformation;
		// get back the visualization parameters
		a = conic.at<double>(0, 0); b = conic.at<double>(0, 1); c = conic.at<double>(1, 1); 
		d = conic.at<double>(0, 2); f = conic.at<double>(1, 2); g = conic.at<double>(2, 2);	
		// miu 
		home_mu_x_ = (c * d - b * f) / (b * b - a * c); // x
		home_mu_y_ = (a * f - b * d) / (b * b - a * c); // y
		// long and short axes
		home_long_axis_ = sqrt(2 * (a * f * f + c * d * d + g * b * b - 2 * b * d * f - a * c * g) / ((b * b - a * c) * (sqrt(pow((double)(a - c), (double)2.0) + 4 * b * b) - (a + c))));
		home_short_axis_ = sqrt(2 * (a * f * f + c * d * d + g * b * b - 2 * b * d * f - a * c * g) / ((b * b - a * c) * (-sqrt(pow((double)(a - c), (double)2.0) + 4 * b * b) - (a + c))));
		// angle
		if(b == 0 && a < c)
			home_angle_ = 0;
		else if(b == 0 && a >= c)
			home_angle_ = PI / 2;
		else if(a < c)
			home_angle_ = 1.0 / 2.0 * atan(2 * b / (a - c));
		else if(a >= c)
			home_angle_ = PI / 2 + 1.0 / 2.0 * atan(2 * b / (a - c));

		home_mu_.at<double>(0, 0) = home_mu_x_;
		home_mu_.at<double>(1, 0) = home_mu_y_;
		// set cov form of the home ellipse
		eigen_value_.at<double>(0, 0) = pow(long_axis_, 2); eigen_value_.at<double>(0, 1) = 0.0;
		eigen_value_.at<double>(1, 1) = pow(short_axis_, 2); eigen_value_.at<double>(1, 0) = 0.0;
		eigen_vector_.at<double>(0, 0) = cos(angle_); eigen_vector_.at<double>(0, 1) = -sin(angle_);
		eigen_vector_.at<double>(1, 0) = sin(angle_); eigen_vector_.at<double>(1, 1) = cos(angle_);
		// aim covar
		home_cov_ = eigen_vector_ * eigen_value_ * eigen_vector_.t();	

		std::cout << "home mu x: " << mu_x_ << " home mu y: " << mu_y_ << " home angle: " << angle_ << std::endl;
		std::cout << "home long aixs: " << long_axis_ << " home short axis: " << short_axis_ << std::endl;
	}
	else
	{
		conic = inv_transform.t() * home_conic * inv_transform;
		// reverse transform back to image coordinate... ini_transform is from image coordinate to a scaled coordinate...
		conic = ini_transformation_.t() * conic * ini_transformation_;
		// get back the visualization parameters
		a = conic.at<double>(0, 0); b = conic.at<double>(0, 1); c = conic.at<double>(1, 1); 
		d = conic.at<double>(0, 2); f = conic.at<double>(1, 2); g = conic.at<double>(2, 2);
	
		// miu 
		mu_x_ = (c * d - b * f) / (b * b - a * c); // x
		mu_y_ = (a * f - b * d) / (b * b - a * c); // y
		// long and short axes
		long_axis_ = sqrt(2 * (a * f * f + c * d * d + g * b * b - 2 * b * d * f - a * c * g) / ((b * b - a * c) * (sqrt(pow((double)(a - c), (double)2.0) + 4 * b * b) - (a + c))));
		short_axis_ = sqrt(2 * (a * f * f + c * d * d + g * b * b - 2 * b * d * f - a * c * g) / ((b * b - a * c) * (-sqrt(pow((double)(a - c), (double)2.0) + 4 * b * b) - (a + c))));
		// angle
		if(b == 0 && a < c)
			angle_ = 0;
		else if(b == 0 && a >= c)
			angle_ = PI / 2;
		else if(a < c)
			angle_ = 1.0 / 2.0 * atan(2 * b / (a - c));
		else if(a >= c)
			angle_ = PI / 2 + 1.0 / 2.0 * atan(2 * b / (a - c));

		mu_.at<double>(0, 0) = mu_x_;
		mu_.at<double>(1, 0) = mu_y_;
		// set cov form of the home ellipse
		eigen_value_.at<double>(0, 0) = pow(long_axis_, 2); eigen_value_.at<double>(0, 1) = 0.0;
		eigen_value_.at<double>(1, 1) = pow(short_axis_, 2); eigen_value_.at<double>(1, 0) = 0.0;
		eigen_vector_.at<double>(0, 0) = cos(angle_); eigen_vector_.at<double>(0, 1) = -sin(angle_);
		eigen_vector_.at<double>(1, 0) = sin(angle_); eigen_vector_.at<double>(1, 1) = cos(angle_);
		// aim covar
		cov_ = eigen_vector_ * eigen_value_ * eigen_vector_.t();	

		std::cout << "mu x: " << mu_x_ << " mu y: " << mu_y_ << " angle: " << angle_ << std::endl;
		std::cout << "long aixs: " << long_axis_ << " short axis: " << short_axis_ << std::endl;
	}
}

void Ellipse::DrawEllipse(cv::Mat& disp_img, double radius, COLOUR& color)
{
	cv::ellipse(disp_img, cv::Point2f(mu_x_, mu_y_), cv::Size(long_axis_ * radius, short_axis_ * radius), angle_ / PI * 180.0, 0.0, 360.0, cv::Scalar((int)(color.r * 255.0), (int)(color.g * 255.0), (int)(color.b * 255.0)), 2);
}

void Ellipse::DrawEllipse(cv::Mat& disp_img, double radius)
{
	cv::ellipse(disp_img, cv::Point2f(mu_x_, mu_y_), cv::Size(long_axis_ * radius, short_axis_ * radius), angle_ / PI * 180.0, 0.0, 360.0, cv::Scalar(255, 0, 0), 2);
}
  
void Ellipse::set_a_home(int home_mu_cov_flag)
{
	if(home_mu_cov_flag)
	{
		cv::eigen(home_cov_, home_eigen_value_, home_eigen_vector_);	
		home_mu_x_ = home_mu_.at<double>(0, 0);
		home_mu_y_ = home_mu_.at<double>(1, 0);
		home_long_axis_ = sqrt(home_eigen_value_.at<double>(0, 0));
		home_short_axis_ = sqrt(home_eigen_value_.at<double>(1, 0));
		home_angle_ = atan2(home_eigen_vector_.at<double>(0, 1), home_eigen_vector_.at<double>(0, 0)); // sin cos
	}
	// translate
	cv::Mat t = cv::Mat::eye(3, 3, CV_64F);	
	t.at<double>(0, 2) = home_mu_x_; t.at<double>(1, 2) = home_mu_y_;
	// rotation
	cv::Mat r = cv::Mat::eye(3, 3, CV_64F);
	r.at<double>(0, 0) = cos(home_angle_); r.at<double>(0, 1) = -sin(home_angle_);	
	r.at<double>(1, 0) = sin(home_angle_); r.at<double>(1, 1) = cos(home_angle_);	
	// scaling
	cv::Mat s = cv::Mat::eye(3, 3, CV_64F);
	s.at<double>(0, 0) = home_long_axis_; s.at<double>(1, 1) = home_short_axis_;

	a_home_ = t * r * s;
	if(!home_mu_cov_flag)
	{
		a_home_ = ini_transformation_ * a_home_;
		UpdateEllipseVisualizationParameters(1);
	}
}

void Ellipse::set_inv_a_home()
{
	cv::invert(a_home_, inv_a_home_);
}

void Ellipse::set_a()
{
	a_home_.copyTo(a_);
}

void Ellipse::set_inv_a()
{
	cv::invert(a_, inv_a_);
}

void Ellipse::set_home_mu()
{
	home_mu_.at<double>(0, 0) = home_mu_x_;
	home_mu_.at<double>(1, 0) = home_mu_y_;
}

void Ellipse::set_home_mu(const cv::Mat& mu)
{
	mu.copyTo(home_mu_);
}

void Ellipse::set_home_cov(const cv::Mat& cov)
{
	cov.copyTo(home_cov_);
}

void Ellipse::set_home_cov()
{
	eigen_value_.at<double>(0, 0) = pow(home_long_axis_, 2); eigen_value_.at<double>(0, 1) = 0.0;
	eigen_value_.at<double>(1, 1) = pow(home_short_axis_, 2); eigen_value_.at<double>(1, 0) = 0.0;
	eigen_vector_.at<double>(0, 0) = cos(home_angle_); eigen_vector_.at<double>(0, 1) = -sin(home_angle_);
	eigen_vector_.at<double>(1, 0) = sin(home_angle_); eigen_vector_.at<double>(1, 1) = cos(home_angle_);
	// aim covar
	home_cov_ = eigen_vector_ * eigen_value_ * eigen_vector_.t();	
}

void Ellipse::set_eta(double eta)
{
	eta_ = eta;
}

cv::Mat Ellipse::home_mu()
{
	return home_mu_;
}

cv::Mat Ellipse::home_cov()
{
	return home_cov_;
}

cv::Mat Ellipse::mu()
{
	return mu_;
}

cv::Mat Ellipse::cov()
{
	return cov_;
}

//void Ellipse::diag(cv::Mat& input, cv::Mat& output)
//{
//	int size = input.rows;
//	for(int i = 0; i < size; i++)
//	{
//		output.at<double>(i, i) = input.at<double>(i, 0);
//	}
//}






/*	
	mu_x_ = (ini_mu_x_ - ini_ref_mu_x_) + ref_mu_x_ + action.at<double>(0, 0);
	mu_y_ = (ini_mu_y_ - ini_ref_mu_y_) + ref_mu_y_ + action.at<double>(1, 0);
	angle_ = (ini_angle_ - ini_ref_angle_) + ref_angle_ + action.at<double>(2, 0);
	long_axis_ = (ini_long_axis_ - ini_ref_long_axis_) + ref_long_axis_ + action.at<double>(3, 0);
	short_axis_ = (ini_short_axis_ - ini_ref_short_axis_) + ref_short_axis_ + action.at<double>(4, 0);
*/

/*
	mu_x_ = ini_mu_x_ + action.at<double>(0, 0);
	mu_y_ = ini_mu_y_ + action.at<double>(1, 0);
	angle_ = ini_angle_ + action.at<double>(2, 0);
	long_axis_ = ini_long_axis_ + action.at<double>(3, 0);
	short_axis_ = ini_short_axis_ + action.at<double>(4, 0);
*/

/*	
	mu_x_ = ini_mu_x_ + action.at<double>(0, 0);
	mu_y_ = ini_mu_y_ + action.at<double>(1, 0);
	angle_ = ini_angle_ + action.at<double>(2, 0);
	long_axis_ = ini_long_axis_ + action.at<double>(3, 0);
	short_axis_ = ini_short_axis_ + action.at<double>(4, 0);
*/

// eigen values stored in descending order, eigen vectors stored in rows with the order corresponding to eigen values
/*	
	ref_cov_.at<double>(0, 0) = 7.000000000000002;
	ref_cov_.at<double>(0, 1) = 5.196152422706633;
	ref_cov_.at<double>(1, 0) = 5.196152422706633;
	ref_cov_.at<double>(1, 1) = 12.999999999999998;
*/

//ref_mu_.at<double>(0, 0) = 0;
//ref_mu_.at<double>(1, 0) = 0;
//ref_cov_.at<double>(0, 0) = 1.0;
//ref_cov_.at<double>(0, 1) = 0.0;
//ref_cov_.at<double>(1, 0) = 0.0;
//ref_cov_.at<double>(1, 1) = 1.0;
/*prev_classified_points.at<double>(i, 1) = match_data[idx_list[i]].at<double>(0, 13);*/

// prev_classified_points.at<double>(i, 0) = match_data[idx_list[i]].at<double>(0, 7); // current ref point x...
// prev_classified_points.at<double>(i, 1) = match_data[idx_list[i]].at<double>(0, 8); // current ref point y..
//if(match_data[idx_list[i]].at<double>(0, 13) >= match_data[idx_list[i]].at<double>(0, 12))
//	improve_percent = 0;
//else
//	improve_percent = (match_data[idx_list[i]].at<double>(0, 12) - match_data[idx_list[i]].at<double>(0, 13)) / match_data[idx_list[i]].at<double>(0, 12);
//// assign improve percentage...
//improvement.at<double>(i, 0) = improve_percent;

//if(improvement_sum.at<double>(0, 0) != 0)
		//{
		//	improvement = improvement / improvement_sum.at<double>(0, 0);			
		//	// tmp mu
		//	for(int i = 0; i < points.rows; i++)
		//		tmp_mu = tmp_mu + improvement.at<double>(i, 0) * points.rowRange(i, i + 1);									
		//	// update mu...
		//	ref_mu_ = ref_mu_ + eta_ * (tmp_mu.t() - ref_mu_);
		//	// tmp covariance
		//	for(int i = 0; i < points.rows; i++)		
		//		tmp_cov = tmp_cov + improvement.at<double>(i, 0) * (points.rowRange(i, i + 1).t() - ref_mu_) * (points.rowRange(i, i + 1) - ref_mu_.t());										
		//	// update ref cov
		//	ref_cov_ = ref_cov_ + eta_ * (tmp_cov - ref_cov_);
		//}		

// cv::Mat improvement_sum;
// cv::reduce(improvement, improvement_sum, 0, CV_REDUCE_SUM);	

/********* original update mu *********/
	// cv::reduce(points, tmp_mu, 0, CV_REDUCE_AVG);
	// ref_mu_ = ref_mu_ + eta_ * (tmp_mu.t() - ref_mu_);
		
	/********* original update cov *********/
	// update ref mu
	/*for(int i = 0; i < points.rows; i++)
	{
		tmp_cov = (points.rowRange(i, i + 1).t() - ref_mu_) * (points.rowRange(i, i + 1) - ref_mu_.t());
		ref_cov_ = ref_cov_ + eta_ * (tmp_cov - ref_cov_);				
	}*/

//void Ellipse::UpdateRefEllipse(cv::Mat& points, cv::Mat& prev_points, cv::Mat& maha_dist)
//{
//	if(points.rows != 0)
//	{
//		// previous update rule is wrong
//		// double eta = 1e-6;		
//		int farthest_point_idx = 0;
//		cv::Mat tmp_cov = cv::Mat::zeros(2, 2, CV_64F);				
//		cv::Mat tmp_mu = cv::Mat::zeros(1, 2, CV_64F);
//		cv::Mat tmp_idx;		
//		cv::sortIdx(maha_dist, tmp_idx, CV_SORT_EVERY_COLUMN + CV_SORT_DESCENDING);
//		farthest_point_idx = tmp_idx.at<int>(0, 0);
//
//		// mu
//		cv::reduce(points, tmp_mu, 0, CV_REDUCE_AVG);		
//		ref_mu_ = ref_mu_ + eta_ * (tmp_mu.t() - ref_mu_);
//		// ref_mu_ = ref_mu_ + eta_ * (points.rowRange(farthest_point_idx, farthest_point_idx + 1).t() - ref_mu_);
//
//		// cov				
//		// ref_mu_ = ref_mu_ + eta_ * (tmp_mu.t() - ref_mu_);
//		tmp_cov = (points.rowRange(farthest_point_idx, farthest_point_idx + 1).t() - ref_mu_) * (points.rowRange(farthest_point_idx, farthest_point_idx + 1) - ref_mu_.t());		
//		ref_cov_ = ref_cov_ + eta_ * (tmp_cov - ref_cov_);		
//
//		// original update rule backup
//		/*for(int i = 0; i < points.rows; i++)
//			tmp_cov = tmp_cov + 1.0 / points.rows * (points.rowRange(i, i + 1).t() - ref_mu_) * (points.rowRange(i, i + 1) - ref_mu_.t());
//		ref_cov_ = ref_cov_ + eta_ * (tmp_cov - ref_cov_);		*/
//		SetRefEllipseParameters();
//	}	
//}
