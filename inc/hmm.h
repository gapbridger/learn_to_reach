#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/types_c.h"
#include "opencv2/nonfree/features2d.hpp"

#define PI 3.141592653589793

// single gaussian emmision probability with multiple observation sequence
class HMM
{
public:
	HMM(int dim_state, int dim_observation, int sequence_length, int num_observation_sequence);
	void ExpectationStep(std::vector<cv::Mat>& b, const std::vector<cv::Mat>& data, bool neighbor_flag);
	void MaximizationStep(const std::vector<cv::Mat>& data);
	void CalculateInvSigma();
	void CalculateObservationLikelihood(std::vector<cv::Mat>& b, const std::vector<cv::Mat>& data);
	void ForwardBackwardProcedure(std::vector<cv::Mat>& b, bool neighbor_flag);
	void MaximumLikelihoodEstimation();
	void NormalizationWithZeroCheck(cv::Mat& vector);
	void set_emission_mu(std::vector<cv::Mat>& emission_mu);
	void set_emission_sigma(std::vector<cv::Mat>& emission_sigma);
	void set_prior(cv::Mat& prior);
	void set_trans(cv::Mat& trans);
	void set_prior_based_on_neighborhood(cv::Mat& home_cloud_neighbor_idx, cv::Mat& home_cloud_neighbor_label);

	std::vector<cv::Mat> get_emission_mu();
	std::vector<cv::Mat> get_emission_sigma();
	std::vector<cv::Mat> get_alpha();
	std::vector<cv::Mat> get_beta();
	std::vector<cv::Mat> get_gamma();
	std::vector<cv::Mat> get_kesi_summed();
	cv::Mat get_prior();
	cv::Mat get_trans();


private:
	int dim_state_;
	int dim_observation_;
	int sequence_length_;
	int num_observation_sequence_;

	cv::Mat prior_;
	// prior based on neighborhood information
	cv::Mat neighborhood_prior_;
	cv::Mat trans_;
	std::vector<cv::Mat> emission_mu_;
	std::vector<cv::Mat> emission_sigma_;
	std::vector<cv::Mat> inv_emission_sigma_;

	std::vector<cv::Mat> alpha_;
	std::vector<cv::Mat> beta_;
	std::vector<cv::Mat> gamma_;
	std::vector<cv::Mat> kesi_summed_;
	std::vector<double> determinant_;
	cv::Mat scale_;
};