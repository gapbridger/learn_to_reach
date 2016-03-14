#include "../inc/general_transform.h"
#include "../inc/loader.h"

COLOUR GetColour(double v, double vmin, double vmax)
{
    COLOUR c = {1.0,1.0,1.0}; // white
    double dv;

    if (v < vmin)
        v = vmin;
    if (v > vmax)
        v = vmax;

    dv = vmax - vmin;

    if (v < (vmin + 0.25 * dv)) 
    {
        c.r = 0;
        c.g = 4 * (v - vmin) / dv;
    }
    else if (v < (vmin + 0.5 * dv)) 
    {
        c.r = 0;
        c.b = 1 + 4 * (vmin + 0.25 * dv - v) / dv;
    }
    else if (v < (vmin + 0.75 * dv)) 
    {
        c.r = 4 * (v - vmin - 0.5 * dv) / dv;
        c.b = 0;
    }
    else
    {
        c.g = 1 + 4 * (vmin + 0.75 * dv - v) / dv;
        c.b = 0;
    }

    return c;
}

GeneralTransform::GeneralTransform(int transform_dim, int num_joints, double normal_learning_rate, int batch_size)
{
    // 
    feature_dim_ = pow(3.0, num_joints) - 1;
    transform_dim_ = transform_dim;
    num_weights_ = transform_dim_ * (transform_dim_ - 1);
    num_joints_ = num_joints;
    // learning rates
    w_rate_ = normal_learning_rate; 
    w_natural_rate_ = 2e-5;
    w_ = std::vector<cv::Mat>(num_joints_);
    w_grad_ = std::vector<cv::Mat>(num_joints_);    
    transform_inv_ = std::vector<cv::Mat>(num_joints_);
    transform_ = std::vector<cv::Mat>(num_joints_);
    transform_elements_ = std::vector<cv::Mat>(num_joints_);
    for(int i = 0; i < num_joints_; i++)
    {
        w_[i] = cv::Mat::zeros(num_weights_, feature_dim_, CV_64F);
        w_grad_[i] = cv::Mat::zeros(num_weights_, feature_dim_, CV_64F);        
        transform_inv_[i] = cv::Mat::eye(transform_dim_, transform_dim_, CV_64F);
        transform_[i] = cv::Mat::eye(transform_dim_, transform_dim_, CV_64F);
    }
    // initialize weights...
    double weight_range = 1e-5;
    // initialize all the weights
    for(int i = 0; i < num_joints_; i++)
		cv::randu(w_[i], cv::Scalar::all(-weight_range), cv::Scalar::all(weight_range));

    average_norm_ = std::vector<double>(num_joints_);
    ini_norm_ = std::vector<double>(num_joints_);
    lambda_ = 0.2;
    rejection_threshold_ = 0.1;

	batch_transform_inv_ = std::vector<std::vector<cv::Mat>>(batch_size);
	batch_transform_ = std::vector<std::vector<cv::Mat>>(batch_size);
	batch_transform_elements_ = std::vector<std::vector<cv::Mat>>(batch_size);
	for(int k = 0; k < batch_size; k++)
	{
		batch_transform_inv_[k] = std::vector<cv::Mat>(num_joints_);
		batch_transform_[k] = std::vector<cv::Mat>(num_joints_);
		batch_transform_elements_[k] = std::vector<cv::Mat>(num_joints_);
		for(int i = 0; i < num_joints_; i++)
		{
			batch_transform_inv_[k][i] = cv::Mat::eye(transform_dim_, transform_dim_, CV_64F);
			batch_transform_[k][i] = cv::Mat::eye(transform_dim_, transform_dim_, CV_64F);
		}
	}

    /*viewer_ = boost::shared_ptr<pcl::visualization::PCLVisualizer>(new pcl::visualization::PCLVisualizer("cloud Viewer"));
    viewer_->setBackgroundColor(0, 0, 0);   
    viewer_->initCameraParameters();*/
}

void GeneralTransform::InitBatchTransform(int batch_size)
{
	
}

std::vector<cv::Mat> GeneralTransform::get_transform()
{
	return transform_;
}

void GeneralTransform::CalcTransformationBatch(std::vector<cv::Mat>& batch_feature)
{
	int batch_size = batch_feature.size();
	for(int k = 0; k < batch_size; k++)
	{
		for(int i = 0; i < num_joints_; i++)
		{
			batch_transform_elements_[k][i] = w_[i] * batch_feature[k];       
			SetTransformationByElements(batch_transform_[k][i], batch_transform_elements_[k][i]);
			cv::invert(batch_transform_[k][i], batch_transform_inv_[k][i]);
		}
	}
}

void GeneralTransform::TransformCloudBatch(const cv::Mat& input_cloud, std::vector<std::vector<cv::Mat>>& output_cloud, int batch_size)
{
	for(int k = 0; k < batch_size; k++)
	{
		for(int i = 0; i < num_joints_; i++)
		{
			output_cloud[k][i] = input_cloud * batch_transform_[k][i].t();          
		}
	}
}

void GeneralTransform::CalcTransformation(std::vector<cv::Mat>& feature)
{
    for(int i = 0; i < num_joints_; i++)
    {
        transform_elements_[i] = w_[i] * feature[0];       
		SetTransformationByElements(transform_[i], transform_elements_[i]);
		cv::invert(transform_[i], transform_inv_[i]);
	}
}

void GeneralTransform::SetTransformationByElements(cv::Mat& transform, const cv::Mat& elements)
{
	int dim_transform = transform.rows;
	cv::Mat identity = cv::Mat::eye(dim_transform, dim_transform, CV_64F);
	transform.rowRange(0, dim_transform - 1) = identity.rowRange(0, dim_transform - 1) + elements.reshape(1, transform_dim_ - 1);
}

void GeneralTransform::TransformCloud(const cv::Mat& input_cloud, std::vector<std::vector<cv::Mat>>& output_cloud)
{
	for(int i = 0; i < num_joints_; i++)
		output_cloud[0][i] = input_cloud * transform_[i].t();          
}

void GeneralTransform::CalculateGradientBatch(const std::vector<std::vector<cv::Mat>>& matched_target_cloud, 
									   const std::vector<std::vector<cv::Mat>>& matched_predicted_cloud, 
									   const std::vector<std::vector<cv::Mat>>& matched_query_cloud, 
									   const std::vector<cv::Mat>& batch_feature)
{
	// the cloud should be n by 4...        
	int dim_transform = matched_query_cloud[0][0].cols;
	int batch_size = matched_query_cloud.size();
	std::vector<cv::Mat> aggregated_w_grad(num_joints_);
	for(int i = 0; i < num_joints_; i++)
	{
		aggregated_w_grad[i] = cv::Mat::zeros(num_weights_, feature_dim_, CV_64F);
	}
	// std::cout << "calculating batch gradient" << std::endl;
	for(int k = 0; k < batch_size; k++)
	{
		for(int i = 0; i < num_joints_; i++)
		{
			if(matched_predicted_cloud[k][i].rows != 0)
			{
				// std::cout << predicted_cloud[k][i].rows << " " << matched_target_cloud[k][i].rows << " " << query_cloud[k][i].rows << std::endl;;
				cv::Mat diff, transform_grad;      
				diff = matched_predicted_cloud[k][i] - matched_target_cloud[k][i];
				transform_grad = 1.0 / matched_query_cloud[k][i].rows * diff.colRange(0, dim_transform - 1).t() * matched_query_cloud[k][i];
				transform_grad = transform_grad.reshape(1, num_weights_);
				cv::Mat tmp_grad = transform_grad * batch_feature[k].t();
				aggregated_w_grad[i] = aggregated_w_grad[i] + tmp_grad;
			}
		}
	}
	// std::cout << "calculate mean gradient" << std::endl;
	for(int i = 0; i < num_joints_; i++)
	{
		w_grad_[i] = aggregated_w_grad[i] / batch_size;
	}
}


void GeneralTransform::CalculateGradientWithNormalBatch(
	const std::vector<std::vector<cv::Mat>>& matched_target_cloud, 
	const std::vector<std::vector<cv::Mat>>& matched_target_cloud_normal, 
	const std::vector<std::vector<cv::Mat>>& matched_predicted_cloud, 
	const std::vector<std::vector<cv::Mat>>& matched_query_cloud, 
	const std::vector<cv::Mat>& batch_feature)
{
    // the cloud should be n by 4...        
	int dim_transform = matched_query_cloud[0][0].cols;
	int batch_size = matched_query_cloud.size();
	std::vector<cv::Mat> aggregated_w_grad(num_joints_);
	for(int i = 0; i < num_joints_; i++)
	{
		aggregated_w_grad[i] = cv::Mat::zeros(num_weights_, feature_dim_, CV_64F);
	}
	for(int k = 0; k < batch_size; k++)
	{
		for(int i = 0; i < num_joints_; i++)
		{
			if(matched_predicted_cloud[k][i].rows != 0)
			{
				// std::cout << predicted_cloud[k][i].rows << " " << matched_target_cloud[k][i].rows << " " << query_cloud[k][i].rows << std::endl;;
				cv::Mat diff, transform_grad;      
				diff = matched_predicted_cloud[k][i] - matched_target_cloud[k][i]; // num points by dim_transform
				cv::Mat projection_mat = diff.colRange(0, dim_transform - 1) * matched_target_cloud_normal[k][i].t(); // num points by num points
				cv::Mat projection = cv::repeat(projection_mat.diag(), 1, matched_target_cloud_normal[k][i].cols);
				cv::Mat tmp_normal = matched_target_cloud_normal[k][i].mul(projection);
				transform_grad = 1.0 / matched_query_cloud[k][i].rows * tmp_normal.t() * matched_query_cloud[k][i];
				transform_grad = transform_grad.reshape(1, num_weights_);
				cv::Mat tmp_grad = transform_grad * batch_feature[k].t();
				aggregated_w_grad[i] = aggregated_w_grad[i] + tmp_grad;
			}
		}
	}
	// std::cout << "calculate mean gradient" << std::endl;
	for(int i = 0; i < num_joints_; i++)
	{
		w_grad_[i] = aggregated_w_grad[i] / batch_size;
	}
}

void GeneralTransform::CalculateGradient(const std::vector<std::vector<cv::Mat>>& matched_target_cloud, const std::vector<std::vector<cv::Mat>>& matched_predicted_cloud, const std::vector<std::vector<cv::Mat>>& matched_query_cloud, const std::vector<std::vector<cv::Mat>>& point_weight, const std::vector<cv::Mat>& feature)
//void GeneralTransform::CalculateGradient(const std::vector<cv::Mat>& matched_target_cloud, 
//								  const std::vector<cv::Mat>& prediction_cloud, 
//								  const std::vector<cv::Mat>& query_cloud, 
//								  const cv::Mat& feature,
//								  int id,
//								  int iteration_count)
{
    // the cloud should be n by 4...        
	int dim_transform = matched_query_cloud[0][0].cols;
	char output_dir[400];
    for(int i = 0; i < num_joints_; i++)
    {
		if(matched_predicted_cloud[0][i].rows != 0)
		{
			cv::Mat diff, transform_grad;      
			diff = matched_predicted_cloud[0][i] - matched_target_cloud[0][i];
			transform_grad = 1.0 / matched_query_cloud[0][i].rows * diff.colRange(0, dim_transform - 1).t() * matched_query_cloud[0][i];
			transform_grad = transform_grad.reshape(1, num_weights_);
			w_grad_[i] = transform_grad * feature[0].t();
			/********** record average gradient direction *********/
			// sprintf(output_dir, "D:/Document/HKUST/Year 5/Research/Solutions/unified_framework/output/para_%d/gradient_direction/avg_gradient_direction_%d_%d.bin", id, i, iteration_count);
			// FileIO::WriteMatDouble(transform_grad, transform_grad.rows, transform_grad.cols, output_dir);
			/********** record average gradient direction *********/
		}
		/*else
		{
			std::cout << "prediction cloud size 0..." << std::endl;
			exit(0);
		}*/
	}
}

void GeneralTransform::SegmentationAndUpdateFixedHomePos(std::vector<cv::Mat>& home_cloud, cv::Mat& query_cloud, cv::Mat& feature, int iteration_count)
{
    cv::Mat target_cloud, transformed_cloud;    
    int query_cloud_size = query_cloud.rows;
    int cloud_dim = home_cloud[0].cols;
    std::vector<cv::Mat> indices(num_joints_);
    std::vector<cv::Mat> min_dists(num_joints_);    
    int p_rates = 1e-1;

    // match different clouds, transformed by different weights, with the home cloud template...
	cv::flann::Index kd_trees(home_cloud_template_float_, cv::flann::KDTreeIndexParams(4), cvflann::FLANN_DIST_EUCLIDEAN); // build kd tree 
    for(int i = 0; i < num_joints_; i++)
    {       
        indices[i] = cv::Mat::zeros(query_cloud_size, 1, CV_32S);
        min_dists[i] = cv::Mat::zeros(query_cloud_size, 1, CV_32F);
        home_cloud[i].convertTo(transformed_cloud, CV_32F); 		        
        kd_trees.knnSearch(transformed_cloud, indices[i], min_dists[i], 1, cv::flann::SearchParams(64)); // kd tree search
    }

    /************* segmentation based on closest neighbor and update the probability according to distance *********************/

    // first accumulate the data...    
    for(int i = 0; i < query_cloud_size; i++)
    {
		int curr_idx_0 = indices[0].at<int>(i, 0);
		int curr_idx_1 = indices[1].at<int>(i, 0);
		// two joints here
		if(min_dists[0].at<float>(i, 0) < min_dists[1].at<float>(i, 0))
			vote_accumulation_.at<double>(curr_idx_0, 0) = vote_accumulation_.at<double>(curr_idx_0, 0) + 1;
		else
			vote_accumulation_.at<double>(curr_idx_0, 1) = vote_accumulation_.at<double>(curr_idx_0, 1) + 1;        
    }

    for(int i = 0; i < home_cloud_template_.rows; i++)
    {
		if(vote_accumulation_.at<double>(i, 0) == 0 && vote_accumulation_.at<double>(i, 1) == 0)
		{
			home_cloud_label_.at<double>(i, 0) = 0.5; home_cloud_label_.at<double>(i, 1) = 0.5;
		}
		else if(vote_accumulation_.at<double>(i, 0) == 0)
		{
			home_cloud_label_.at<double>(i, 0) = 0; home_cloud_label_.at<double>(i, 1) = 1;
		}
		else if(vote_accumulation_.at<double>(i, 1) == 0)
		{
			home_cloud_label_.at<double>(i, 0) = 1; home_cloud_label_.at<double>(i, 1) = 0;
		}
		else
		{
			double sum = vote_accumulation_.at<double>(i, 0) + vote_accumulation_.at<double>(i, 1);
			home_cloud_label_.at<double>(i, 0) = vote_accumulation_.at<double>(i, 0) / sum;
			home_cloud_label_.at<double>(i, 1) = vote_accumulation_.at<double>(i, 1) / sum;
		}       
    }

    // home_cloud_label_ = home_cloud_label_ + p_rates * (curr_probability - home_cloud_label_);

	if(iteration_count % 500 == 1)
    {       
        // just display the query cloud...		
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_template;        
		colored_template.reset(new pcl::PointCloud<pcl::PointXYZRGB>());
		// colored_template->resize(home_cloud_template_.rows);
		for(int i = 0; i < home_cloud_template_.rows; i++)
		{
			pcl::PointXYZRGB point;
			point.x = home_cloud_template_.at<double>(i, 0);
			point.y = home_cloud_template_.at<double>(i, 1);
			point.z = home_cloud_template_.at<double>(i, 2);
			COLOUR c = GetColour(home_cloud_label_.at<double>(i, 0), 0.0, 1.0);
			point.r = c.r * 255.0;
			point.g = c.g * 255.0;
			point.b = c.b * 255.0;

			colored_template->push_back(point);						
		}                
		pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(colored_template);
        if(iteration_count == 1)
			viewer_->addPointCloud<pcl::PointXYZRGB>(colored_template, rgb, "template");        
        else
            viewer_->updatePointCloud<pcl::PointXYZRGB>(colored_template, rgb, "template");
        
        viewer_->spinOnce(10);
    }

	std::vector<int> segmentation_count(num_joints_);
    std::vector<cv::Mat> segmented_target_cloud(num_joints_);
    std::vector<cv::Mat> segmented_transformed_cloud(num_joints_);
    std::vector<cv::Mat> segmented_query_cloud(num_joints_);
    std::vector<cv::Mat> segmented_idx(num_joints_);
    // pre allocate
	std::cout << "pre-allocate..." << std::endl;
    for(int i = 0; i < num_joints_; i++)
    {
        segmentation_count[i] = 0; // query_cloud.rows;     
        segmented_idx[i] = cv::Mat::zeros(query_cloud_size, 2, CV_64F); // first column original idx, second column matched idx
    }
    // get the data... only work for two joints here...
	std::cout << "segment..." << std::endl;
    for(int i = 0; i < query_cloud_size; i++)
    {
        int min_idx = 0;
		// this line has bug...
		if(home_cloud_label_.at<double>(i, 0) > home_cloud_label_.at<double>(i, 1))
			min_idx = 0;
		else
			min_idx = 1;
        int pos = segmentation_count[min_idx];
        segmented_idx[min_idx].at<double>(pos, 0) = i; 
		segmented_idx[min_idx].at<double>(pos, 1) = indices[min_idx].at<int>(i, 0);          
        segmentation_count[min_idx]++;
    }   
	// update...
	std::cout << "separate data..." << std::endl;
    for(int i = 0; i < num_joints_; i++)
    {
        segmented_target_cloud[i] = cv::Mat::zeros(segmentation_count[i], cloud_dim, CV_64F);
        segmented_transformed_cloud[i] = cv::Mat::zeros(segmentation_count[i], cloud_dim, CV_64F);
        segmented_query_cloud[i] = cv::Mat::zeros(segmentation_count[i], cloud_dim, CV_64F);
        for(int j = 0; j < segmentation_count[i]; j++)
        {
            int query_pos = segmented_idx[i].at<double>(j, 0);
            int matched_pos = segmented_idx[i].at<double>(j, 1);
			if(matched_pos >= home_cloud_template_.rows || j >= segmented_transformed_cloud[i].rows || query_pos >= home_cloud[i].rows)
				std::cout << "matched pos not correct..." << std::endl;
            home_cloud[i].rowRange(query_pos, query_pos + 1).copyTo(segmented_transformed_cloud[i].rowRange(j, j + 1));
            query_cloud.rowRange(query_pos, query_pos + 1).copyTo(segmented_query_cloud[i].rowRange(j, j + 1));
			home_cloud_template_.rowRange(matched_pos, matched_pos + 1).copyTo(segmented_target_cloud[i].rowRange(j, j + 1));
        }
    }


    // CalcGradient(matched_template, home_cloud, query_cloud_list, feature, matched_probabilities);
	std::cout << "calculating gradient..." << std::endl;
	// CalcGradient(segmented_target_cloud, segmented_transformed_cloud, segmented_query_cloud, feature, segmentation_count);
	std::cout << "updating..." << std::endl;
    Update(iteration_count);
}

//void GeneralTransform::ShowHomePoseLabel()
//{
//}

void GeneralTransform::SetHomeCloud(std::vector<cv::Mat>& home_cloud)
{
    int num_points = home_cloud[0].rows;
    double initial_probability = 1.0 / (double)num_joints_;	
    // set initial cloud	
	home_cloud_template_ = cv::Mat::zeros(num_points, home_cloud[0].cols, CV_64F);
    home_cloud[0].copyTo(home_cloud_template_);
    // establish the kd tree for nearest neighbor search
    home_cloud_template_.convertTo(home_cloud_template_float_, CV_32F);
    // kd_trees_ = cv::flann::Index(home_cloud_template_float_, cv::flann::KDTreeIndexParams(4), cvflann::FLANN_DIST_EUCLIDEAN); // build kd tree         
    // set initial probability
	home_cloud_label_ = cv::Mat::ones(num_points, num_joints_, CV_64F);
    home_cloud_label_ = home_cloud_label_.mul(initial_probability);

	// vote structures...
	vote_accumulation_ = cv::Mat::zeros(num_points, num_joints_, CV_64F);	
}

void GeneralTransform::SegmentationAndUpdate(std::vector<cv::Mat>& prev_home_cloud, std::vector<cv::Mat>& home_cloud, cv::Mat& query_cloud, cv::Mat& feature, int iteration_count)
{
    // all home cloud suppose to be the whole cloud thus same size...

    /************* nearest neighbor match part *********************/
    
    cv::Mat target_cloud, transformed_cloud;    
    int query_cloud_size = query_cloud.rows;
    int cloud_dim = home_cloud[0].cols;
    std::vector<cv::Mat> indices(num_joints_);
    std::vector<cv::Mat> min_dists(num_joints_);    

    for(int i = 0; i < num_joints_; i++)
    {       
        indices[i] = cv::Mat::zeros(query_cloud_size, 1, CV_32S);
        min_dists[i] = cv::Mat::zeros(query_cloud_size, 1, CV_32F);
    }
    // match different clouds, transformed by different weights...
    // for(int i = 0; i < num_joints_; i++)
    for(int i = 0; i < num_joints_; i++)
    {
        prev_home_cloud[i].convertTo(target_cloud, CV_32F); 
        home_cloud[i].convertTo(transformed_cloud, CV_32F); 
        cv::flann::Index kd_trees(target_cloud, cv::flann::KDTreeIndexParams(4), cvflann::FLANN_DIST_EUCLIDEAN); // build kd tree           
        kd_trees.knnSearch(transformed_cloud, indices[i], min_dists[i], 1, cv::flann::SearchParams(64)); // kd tree search
    }
    // segment the clouds by minimum distance...
    // the two segments are of the same length which is the length of the previous home cloud
    // maybe use vector first and do a whole conversion at the end... that should be good...    
    
    /************* segmentation based on closest neighbor part *********************/
        
    std::vector<int> segmentation_count(num_joints_);
    std::vector<cv::Mat> segmented_target_cloud(num_joints_);
    std::vector<cv::Mat> segmented_transformed_cloud(num_joints_);
    std::vector<cv::Mat> segmented_query_cloud(num_joints_);
    std::vector<cv::Mat> segmented_idx(num_joints_);
    // pre allocate
    for(int i = 0; i < num_joints_; i++)
    {
        segmentation_count[i] = 0; // query_cloud.rows;     
        segmented_idx[i] = cv::Mat::zeros(query_cloud_size, 2, CV_64F); // first column original idx, second column matched idx
    }
    // get the data...
    for(int i = 0; i < query_cloud_size; i++)
    {
        int min_idx = 0;
        double curr_min_dist = min_dists[0].at<float>(i, 0); 
        for(int j = 1; j < num_joints_; j++)
        {
            // find the minimum...
            if(min_dists[j].at<float>(i, 0) < curr_min_dist)
            {
                min_idx = j;
                curr_min_dist = min_dists[j].at<float>(i, 0);
            }
        }       
        int pos = segmentation_count[min_idx];
        segmented_idx[min_idx].at<double>(pos, 0) = i; segmented_idx[min_idx].at<double>(pos, 1) = indices[min_idx].at<int>(i, 0);          
        segmentation_count[min_idx]++;
    }   
    for(int i = 0; i < num_joints_; i++)
    {
        segmented_target_cloud[i] = cv::Mat::zeros(segmentation_count[i], cloud_dim, CV_64F);
        segmented_transformed_cloud[i] = cv::Mat::zeros(segmentation_count[i], cloud_dim, CV_64F);
        segmented_query_cloud[i] = cv::Mat::zeros(segmentation_count[i], cloud_dim, CV_64F);
        for(int j = 0; j < segmentation_count[i]; j++)
        {
            int query_pos = segmented_idx[i].at<double>(j, 0);
            int matched_pos = segmented_idx[i].at<double>(j, 1);
            home_cloud[i].rowRange(query_pos, query_pos + 1).copyTo(segmented_transformed_cloud[i].rowRange(j, j + 1));
            query_cloud.rowRange(query_pos, query_pos + 1).copyTo(segmented_query_cloud[i].rowRange(j, j + 1));
            prev_home_cloud[i].rowRange(matched_pos, matched_pos + 1).copyTo(segmented_target_cloud[i].rowRange(j, j + 1));
        }
    }
    
    /******************* display segmented data... *********************/

    if(iteration_count % 200 == 1)
    {       
        // just display the query cloud...
        std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> cloud_segments(num_joints_);   
        for(int i = 0; i < num_joints_; i++)
        {
            if(segmentation_count[i] != 0)
            {
                char cloud_name[10];
                sprintf(cloud_name, "%d", i);
                COLOUR c = GetColour(i * 1.0 / (num_joints_ - 1) * num_joints_, 0, num_joints_);
                cloud_segments[i] = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);                
                Mat2PCD_Trans(segmented_query_cloud[i], cloud_segments[i]);     
                pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cloud_color(cloud_segments[i], c.r * 255, c.g * 255, c.b * 255);
                if(iteration_count == 1)
                    viewer_->addPointCloud<pcl::PointXYZ>(cloud_segments[i], cloud_color, cloud_name);
                else
                    viewer_->updatePointCloud<pcl::PointXYZ>(cloud_segments[i], cloud_color, cloud_name);               
            }
        }
        viewer_->spinOnce(1);
    }
    
    /************* weights update part **************/
    // ReOrder_Trans(prev_home_cloud, segmented_target_cloud, indices);
    /*for(int i = 0; i < num_joints_; i++)
        query_cloud.copyTo(segmented_query_cloud[i]);*/
    // CalcGradient(segmented_target_cloud, segmented_transformed_cloud, segmented_query_cloud, feature, segmentation_count);
    // CalcGradient(segmented_target_cloud, segmented_transformed_cloud, segmented_query_cloud, feature, segmentation_count);
    Update(iteration_count);
    
}

void GeneralTransform::ReOrder_Trans(std::vector<cv::Mat>& input, std::vector<cv::Mat>& output, std::vector<cv::Mat>& input_indices)
{
    int size = output.size();
    for(int i = 0; i < size; i++)
    {
        output[i] = cv::Mat::zeros(input_indices[i].rows, input[i].cols, CV_64F);
        for(int p = 0; p < input_indices[i].rows; p++)
            for(int q = 0; q < input[i].cols; q++)
                output[i].at<double>(p, q) = input[i].at<double>(input_indices[i].at<int>(p, 0), q);        
    }
}

void GeneralTransform::Mat2PCD_Trans(cv::Mat& cloud_mat, pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
{
    int size = cloud_mat.rows;
    std::vector<pcl::PointXYZ> points_vec(size);
    cloud.reset(new pcl::PointCloud<pcl::PointXYZ>());
    for(int i = 0; i < size; i++)
    {
        pcl::PointXYZ point;
        point.x = cloud_mat.at<double>(i, 0);
        point.y = cloud_mat.at<double>(i, 1);
        point.z = cloud_mat.at<double>(i, 2);
        cloud->push_back(point);
    }   
}

void GeneralTransform::Rejection(cv::Mat& diff, cv::Mat& filtered_diff, cv::Mat& query_cloud, cv::Mat& filtered_query_cloud, double threshold)
{
    if(threshold != 0 && diff.rows > 10)
    {
        cv::Mat dist, idx, tmp_diff;
        tmp_diff = diff.mul(diff);
        cv::reduce(tmp_diff, dist, 1, CV_REDUCE_SUM);
        cv::sqrt(dist, dist);
        cv::sortIdx(dist, idx, CV_SORT_EVERY_COLUMN + CV_SORT_ASCENDING);
        int count = (int)(dist.rows * (1 - threshold));
        /*while(count < dist.rows && dist.at<double>(idx.at<int>(count, 0), 0) < threshold)
            count++;*/
        // std::cout << "original: " << diff.rows << " filtered: " << count << std::endl;
        filtered_diff = cv::Mat::zeros(count, diff.cols, CV_64F);
        filtered_query_cloud = cv::Mat::zeros(count, query_cloud.cols, CV_64F);
        
        for(int i = 0; i < count; i++)      
        {
            // diff
            for(int m = 0; m < diff.cols; m++)
                filtered_diff.at<double>(i, m) = diff.at<double>(idx.at<int>(i, 0), m);
            // query_cloud      
            for(int n = 0; n < query_cloud.cols; n++)
                filtered_query_cloud.at<double>(i, n) = query_cloud.at<double>(idx.at<int>(i, 0), n);       
        }
    }
    else
    {
        filtered_diff = cv::Mat::zeros(diff.rows, diff.cols, CV_64F);
        diff.copyTo(filtered_diff);
        filtered_query_cloud = cv::Mat::zeros(query_cloud.rows, query_cloud.cols, CV_64F);
        query_cloud.copyTo(filtered_query_cloud);
    }


}
void GeneralTransform::Update()
{   
	for(int i = 0; i < num_joints_; i++)
	{
		w_[i] = w_[i] - w_rate_ * w_grad_[i];
	}
}


void GeneralTransform::Update(int iter)
{   
    for(int i = 0; i < num_joints_; i++)
    {
        double curr_norm = cv::norm(natural_w_grad_[i], cv::NORM_L2);
        if(iter == 1)
        {
            ini_norm_[i] = curr_norm;
            average_norm_[i] = curr_norm;       
        }
        else    
            average_norm_[i] = (1 - lambda_) * average_norm_[i] + lambda_ * curr_norm;  
        w_[i] = w_[i] - (w_rate_ * ini_norm_[i] / average_norm_[i]) * natural_w_grad_[i];

        // w_[i] = w_[i] - w_rate_ * w_grad_[i];
    }
}

// copy to previous transformation
void GeneralTransform::CopyTransformToPrev()
{
    for(int i = 0; i < num_joints_; i++)
        transform_inv_[i].copyTo(prev_transform_inv_[i]);
}
cv::Mat GeneralTransform::fisher_inv(int idx)
{
    return fisher_inv_[idx];
}

cv::Mat GeneralTransform::natural_w_grad(int idx)
{
    return natural_w_grad_[idx];
}

cv::Mat GeneralTransform::w_grad(int idx)
{
    return w_grad_[idx];
}

std::vector<cv::Mat> GeneralTransform::w_grad()
{
    return w_grad_;
}

cv::Mat GeneralTransform::w(int idx)
{
    return w_[idx];
}

void GeneralTransform::set_w(const cv::Mat& w, int idx)
{
     w.copyTo(w_[idx]);
}

cv::Mat GeneralTransform::get_w(int idx)
{
     return w_[idx];
}

void GeneralTransform::set_w_rate(double w_rate)
{
    w_rate_ = w_rate;   
}

void GeneralTransform::set_w_natural_rate(double natural_rate)
{
    w_natural_rate_ = natural_rate;
}


void GeneralTransform::set_fisher_inv()
{
    for(int i = 0; i < num_joints_; i++)
        fisher_inv_[i] = cv::Mat::eye(feature_dim_ * num_weights_, feature_dim_ * num_weights_, CV_64F);
}

