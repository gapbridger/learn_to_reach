#ifndef ELLIPSE_H
#define ELLIPSE_H
#define PI 3.14159265

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "../inc/transform.h"
typedef std::vector<cv::Mat> MatL;

typedef struct {
    double r,g,b;
} COLOUR;

namespace ellipse
{
class Ellipse
{
private:
	// initial parameters
	
	double home_mu_x_;
	double home_mu_y_;
	double home_long_axis_;
	double home_short_axis_;
	double home_angle_;

	// current parameters
	double mu_x_;
	double mu_y_;
	double long_axis_;
	double short_axis_;
	double angle_;
	// previous parameters
	double prev_mu_x_;
	double prev_mu_y_;
	double prev_long_axis_;
	double prev_short_axis_;
	double prev_angle_;
	
	double radius_;

	/*cv::Mat ref_transform_;
	cv::Mat aim_transform_;*/
	// cv::Mat inv_aim_transform_;
	// current matrix structures
	cv::Mat mu_;
	cv::Mat cov_;
	cv::Mat cov_inv_;
	// previous matrix structures
	cv::Mat prev_mu_;
	cv::Mat prev_cov_;
	cv::Mat prev_cov_inv_;
	// ref matrix
	cv::Mat home_mu_;
	cv::Mat home_cov_;
	cv::Mat home_cov_inv_;	

	cv::Mat eigen_value_;
	cv::Mat eigen_vector_;		
	cv::Mat home_eigen_value_;
	cv::Mat home_eigen_vector_;				

	Transform home_transform_;
	
	cv::Mat a_home_; // home ellipse A
	cv::Mat inv_a_home_; // home ellipse inv A

	cv::Mat prev_a_; // ellipse A
	cv::Mat prev_inv_a_; // inverse of A

	cv::Mat a_; // ellipse A
	cv::Mat inv_a_; // inverse of A

	double eta_;

	cv::Mat ini_transformation_;
		
public:
	Transform transform_;

	Ellipse(double initial_x, double initial_y, double initial_long_axis, double initial_short_axis, double initial_angle, double radius, cv::Mat ini_transformation);			
	void TransformEllipse();
	double MahalanobisDistance(const cv::Mat& data, const cv::Mat& inv_a);
	bool CheckInsideEllipse(const cv::Mat& point, const cv::Mat& inv_a);
	void GetKeyPointInEllipse(const cv::Mat& descriptor, const cv::Mat& key_point, cv::Mat& elips_descriptor, cv::Mat& elips_key_point, cv::Mat& elips_distance, int curr_frame_flag);
	void ClassifyPointsHomeEllipse(MatL match_data, cv::Mat& classified_points, cv::Mat& prev_classified_points, cv::Mat& maha_dist);
	void UpdateHomeEllipse(cv::Mat& points, cv::Mat& motion_ratio, cv::Mat& maha_dist);
	void UpdateEllipseVisualizationParameters(int initial_specification_flag);
	void MVEE(cv::Mat& P, cv::Mat& c, cv::Mat& A, double tolerance);
	void CopyToPrev();			
	void DrawEllipse(cv::Mat& disp_img, double radius, COLOUR& color);
	void DrawEllipse(cv::Mat& disp_img, double radius);
	// helper functions
	cv::Mat home_mu();
	cv::Mat home_cov();
	cv::Mat mu();
	cv::Mat cov();
	void set_a_home(int home_mu_cov_flag);
	void set_inv_a_home();
	void set_a();
	void set_inv_a();
	void set_home_mu();
	void set_home_cov();
	void set_home_mu(const cv::Mat& mu);
	void set_home_cov(const cv::Mat& cov);
	void set_eta(double eta);
	
	
};
}

#endif