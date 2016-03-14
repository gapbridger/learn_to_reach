#include "../inc/hmm.h"
#include <iostream>

HMM::HMM(int dim_state, int dim_observation, int sequence_length, int num_observation_sequence)
{
	dim_state_ = dim_state;
	dim_observation_ = dim_observation;
	sequence_length_ = sequence_length;
	num_observation_sequence_ = num_observation_sequence;

	// shared hmm model parameters	
	prior_ = cv::Mat::zeros(dim_state_, 1, CV_64F); // std::vector<cv::Mat>(num_observation_sequence_);
	trans_ = cv::Mat::zeros(dim_state_, dim_state_, CV_64F); // std::vector<cv::Mat>(num_observation_sequence_);
	emission_mu_ = std::vector<cv::Mat>(dim_state_);
	emission_sigma_ = std::vector<cv::Mat>(dim_state_);
	inv_emission_sigma_ = std::vector<cv::Mat>(dim_state_);
	// neighborhood prior is with the same size of cloud, dimension is states
	neighborhood_prior_ = cv::Mat::zeros(num_observation_sequence_, dim_state_, CV_64F);
	for(int idx = 0; idx < dim_state_; idx++)
	{
		emission_mu_[idx] = cv::Mat::zeros(dim_observation_, 1, CV_64F);
		emission_sigma_[idx] = cv::Mat::zeros(dim_observation_, dim_observation_, CV_64F);
		inv_emission_sigma_[idx] = cv::Mat::zeros(dim_observation_, dim_observation_, CV_64F);
	}

	// observation specific parameters
	alpha_ = std::vector<cv::Mat>(num_observation_sequence_); // ::zeros(sequence_length, dim_state, CV_64F);
	beta_ = std::vector<cv::Mat>(num_observation_sequence_); // ::zeros(sequence_length, dim_state, CV_64F);
	gamma_ = std::vector<cv::Mat>(num_observation_sequence_);
	kesi_summed_ = std::vector<cv::Mat>(num_observation_sequence_);
	for(int idx = 0; idx < num_observation_sequence_; idx++)
	{
		alpha_[idx] = cv::Mat::zeros(sequence_length_, dim_state_, CV_64F);
		beta_[idx] = cv::Mat::zeros(sequence_length_, dim_state_, CV_64F);
		gamma_[idx] = cv::Mat::zeros(sequence_length_, dim_state_, CV_64F);
		kesi_summed_[idx] = cv::Mat::zeros(dim_state_, dim_state_, CV_64F);
	}
	determinant_ = std::vector<double>(dim_state_);
	scale_ = cv::Mat::zeros(dim_state_, 1, CV_64F);
	
	
}
void HMM::ExpectationStep(std::vector<cv::Mat>& b, const std::vector<cv::Mat>& data, bool neighbor_flag)
{
	CalculateInvSigma();
	CalculateObservationLikelihood(b, data);
	ForwardBackwardProcedure(b, neighbor_flag);
}

void HMM::MaximizationStep(const std::vector<cv::Mat>& data)
{
	// cv::Mat estimated_prior = cv::Mat::zeros(dim_state_, 1, CV_64F);
	// cv::Mat estimated_kesi = cv::Mat::zeros(dim_state_, dim_state_, CV_64F);
	// cv::Mat estimated_kesi_norm = cv::Mat::zeros(dim_state_, 1, CV_64F);
	std::vector<cv::Mat> gamma_summed = std::vector<cv::Mat>(num_observation_sequence_); // cv::Mat::zeros(1, dim_state_, CV_64F);
	cv::Mat gamma_summed_multi_observ = cv::Mat::zeros(1, dim_state_, CV_64F);
	std::vector<cv::Mat> data_weighted_sum = std::vector<cv::Mat>(dim_state_); // cv::Mat::zeros(1, dim_observation_, CV_64F);
	std::vector<cv::Mat> data_weighted_variance = std::vector<cv::Mat>(dim_state_); // cv::Mat::zeros(1, dim_observation_, CV_64F);
	for(int s = 0; s < dim_state_; s++)
	{
		data_weighted_sum[s] = cv::Mat::zeros(1, dim_observation_, CV_64F);	
		data_weighted_variance[s] = cv::Mat::zeros(dim_observation_, dim_observation_, CV_64F);	
	}
	for(int o = 0; o < num_observation_sequence_; o++)
	{
		// prior
		// estimated_prior = estimated_prior + gamma_[o].rowRange(0, 0 + 1).t();
		// trans
		gamma_summed[o] = cv::Mat::zeros(1, dim_state_, CV_64F);
		// estimated_kesi = estimated_kesi + kesi_summed_[o];
		cv::reduce(gamma_[o], gamma_summed[o], 0, CV_REDUCE_SUM);
		gamma_summed_multi_observ = gamma_summed_multi_observ + gamma_summed[o];
		// estimate mu
		cv::Mat tmp_mu = cv::Mat::zeros(1, dim_observation_, CV_64F);
		for(int s = 0; s < dim_state_; s++)
		{
			cv::Mat weighted_data = data[o].mul(cv::repeat(gamma_[o].colRange(s, s + 1), 1, dim_observation_));
			data_weighted_variance[s] = data_weighted_variance[s] + weighted_data.t() * data[o];
			cv::reduce(weighted_data, tmp_mu, 0, CV_REDUCE_SUM);
			data_weighted_sum[s] = data_weighted_sum[s] + tmp_mu;
		}
	}
	// prior
	// prior_ = estimated_prior / (double) num_observation_sequence_;
	// trans
	/*cv::reduce(estimated_kesi, estimated_kesi_norm, 1, CV_REDUCE_SUM);
	for(int i = 0; i < dim_state_; i++)
	{
		if(estimated_kesi_norm.at<double>(i, 0) == 0)
		{
			estimated_kesi_norm.at<double>(i, 0) = 1;
		}
	}*/
	// cv::divide(estimated_kesi, cv::repeat(estimated_kesi_norm, 1, dim_state_), trans_);
	// emission probability parameters: mu and sigma
	for(int s = 0; s < dim_state_; s++)
	{
		if(gamma_summed_multi_observ.at<double>(0, s) == 0)
		{
			gamma_summed_multi_observ.at<double>(0, s) = 1;
		}
		emission_mu_[s] = data_weighted_sum[s] / gamma_summed_multi_observ.at<double>(0, s);
		emission_mu_[s] = emission_mu_[s].t();
		emission_sigma_[s] = data_weighted_variance[s] / gamma_summed_multi_observ.at<double>(0, s) - emission_mu_[s] * emission_mu_[s].t();
		emission_sigma_[s] = cv::Mat::diag(emission_sigma_[s].diag());
		if(emission_mu_[s].at<double>(0, 0) != emission_mu_[s].at<double>(0, 0))
		{
			std::cout << "nan occured..." << std::endl;
			exit(0);
		}
	}
}

void HMM::CalculateInvSigma()
{
	double pi_scale = pow(2 * PI, dim_observation_);
	for(int state_idx = 0; state_idx < dim_state_; state_idx++)
	{
		determinant_[state_idx] = cv::determinant(emission_sigma_[state_idx]);
		inv_emission_sigma_[state_idx] = emission_sigma_[state_idx].inv();
		scale_.at<double>(state_idx, 0) = 1 / sqrt(pi_scale * determinant_[state_idx]);
	}
}

// b stands for the emmision likelihood
void HMM::CalculateObservationLikelihood(std::vector<cv::Mat>& b, const std::vector<cv::Mat>& data)
{
	// for each observation sequence
	for(int sequence_idx = 0; sequence_idx < data.size(); sequence_idx++)
	{
		// for each data
		for(int data_idx = 0; data_idx < data[sequence_idx].rows; data_idx++)
		{
			// for each state
			for(int state_idx = 0; state_idx < dim_state_; state_idx++)
			{
				cv::Mat diff = data[sequence_idx].rowRange(data_idx, data_idx + 1) - emission_mu_[state_idx].t();
				cv::Mat maha_dist = diff * inv_emission_sigma_[state_idx] * diff.t();
				b[sequence_idx].at<double>(data_idx, state_idx) = scale_.at<double>(state_idx) * exp(-0.5 * maha_dist.at<double>(0, 0)); 
			}
		}
	}
}

void HMM::ForwardBackwardProcedure(std::vector<cv::Mat>& b, bool neighbor_flag)
{
	// notation: b: observation likelihood, alpha: forward variable, beta: backward variable
	// forward pass
	for(int o = 0; o < num_observation_sequence_; o++)
	{
		for(int t = 0; t < sequence_length_; t++)
		{
			if(t == 0)
			{
				if(neighbor_flag)
				{
					alpha_[o].rowRange(t, t + 1) = b[o].rowRange(t, t + 1).mul(neighborhood_prior_.rowRange(o, o + 1));
				}
				else
				{
					alpha_[o].rowRange(t, t + 1) = b[o].rowRange(t, t + 1).mul(prior_.t());
				}
			}
			else
			{
				cv::Mat tmp = alpha_[o].rowRange(t - 1, t) * trans_;
				alpha_[o].rowRange(t, t + 1) = tmp.mul(b[o].rowRange(t, t + 1));
			}
			NormalizationWithZeroCheck(alpha_[o].rowRange(t, t + 1));
			// cv::normalize(alpha_[o].rowRange(t, t + 1), alpha_[o].rowRange(t, t + 1), 1, 0, cv::NORM_L1);
		}
	}
	// backward pass
	for(int o = 0; o < num_observation_sequence_; o++)
	{
		for(int t = sequence_length_ - 1; t >= 0; t--)
		{
			if(t == sequence_length_ - 1)
			{
				beta_[o].rowRange(t, t + 1) = cv::Mat::ones(1, dim_state_, CV_64F);
			}
			else
			{
				cv::Mat tmp_b = beta_[o].rowRange(t + 1, t + 2).mul(b[o].rowRange(t + 1, t + 2));
				beta_[o].rowRange(t, t + 1) = tmp_b * trans_.t(); 
				// cv::normalize(beta_[o].rowRange(t, t + 1), beta_[o].rowRange(t, t + 1), 1, 0, cv::NORM_L1);
				NormalizationWithZeroCheck(beta_[o].rowRange(t, t + 1));
				cv::Mat tmp_kesi = trans_.mul(alpha_[o].rowRange(t, t + 1).t() * tmp_b);
				NormalizationWithZeroCheck(tmp_kesi);
				kesi_summed_[o] = kesi_summed_[o] + tmp_kesi; 
				// kesi_summed_[o] = kesi_summed_[o] + tmp_kesi / cv::sum(tmp_kesi)[0]; 
			}
			gamma_[o].rowRange(t, t + 1) = alpha_[o].rowRange(t, t + 1).mul(beta_[o].rowRange(t, t + 1));
			NormalizationWithZeroCheck(gamma_[o].rowRange(t, t + 1));
			// cv::normalize(gamma_[o].rowRange(t, t + 1), gamma_[o].rowRange(t, t + 1), 1, 0, cv::NORM_L1);
		}
	}
}

void HMM::set_prior_based_on_neighborhood(cv::Mat& home_cloud_neighbor_idx, cv::Mat& home_cloud_neighbor_label)
{
	int cloud_size = num_observation_sequence_; 
	int max_num_neighbors = 80;
	double min_value = 0;
	double max_value = 0;
	double beta = 0.6;
	cv::Point min_location, max_location;
	cv::Mat tmp_mat_1, tmp_mat_2, tmp_zeros;
	min_location.x = 0; min_location.y = 0; max_location.x = 0; max_location.y = 0;
	cv::Mat new_home_cloud_label; 

	for(int point_idx = 0; point_idx < cloud_size; point_idx++)
	{
		cv::Mat label_count = cv::Mat::zeros(1, dim_state_, CV_64F);
		for(int neighbor_idx = 0; neighbor_idx < max_num_neighbors; neighbor_idx++)
		{
			int neighbor_point_idx = home_cloud_neighbor_idx.at<int>(point_idx, neighbor_idx);
			if(neighbor_point_idx != -1)
			{
				label_count = label_count + home_cloud_neighbor_label.rowRange(neighbor_idx, neighbor_idx + 1); 
			}
			else
			{
				break;
			}
		}
		label_count = label_count * beta;
		cv::exp(label_count, tmp_mat_1);
		cv::reduce(tmp_mat_1, tmp_mat_2, 1, CV_REDUCE_SUM); // row-wise reduce
		if(tmp_mat_2.at<double>(0, 0) == 0)
		{
			std::cout << "sum label error..." << std::endl;
			exit(0);
		}
		// neighborhood potential
		neighborhood_prior_.rowRange(point_idx, point_idx + 1) = tmp_mat_1 / cv::repeat(tmp_mat_2, label_count.rows, label_count.cols);
	}
}

void HMM::NormalizationWithZeroCheck(cv::Mat& vector)
{
	double normalization = cv::sum(vector)[0];
	if(normalization == 0)
	{
		normalization = 1;
	}
	vector = vector  / normalization; 
}

void HMM::set_emission_mu(std::vector<cv::Mat>& emission_mu)
{
	emission_mu_ = emission_mu;
}

void HMM::set_emission_sigma(std::vector<cv::Mat>& emission_sigma)
{
	emission_sigma_ = emission_sigma;
}

void HMM::set_prior(cv::Mat& prior)
{
	prior_ = prior;
}

void HMM::set_trans(cv::Mat& trans)
{
	trans_ = trans;
}

std::vector<cv::Mat> HMM::get_emission_mu()
{
	return emission_mu_;
}

std::vector<cv::Mat> HMM::get_emission_sigma()
{
	return emission_sigma_;
}

std::vector<cv::Mat> HMM::get_alpha()
{
	return alpha_;
}

std::vector<cv::Mat> HMM::get_beta()
{
	return beta_;
}

std::vector<cv::Mat> HMM::get_gamma()
{
	return gamma_;
}

std::vector<cv::Mat> HMM::get_kesi_summed()
{
	return kesi_summed_;
}

cv::Mat HMM::get_prior()
{
	return prior_;
}

cv::Mat HMM::get_trans()
{
	return trans_;
}
