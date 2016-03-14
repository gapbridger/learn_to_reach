#ifndef _SEGMENTATION_H
#define _SEGMENTATION_H

#include "opencv2/core/core.hpp"

class Segmentation
{
public:
	virtual void Match(int train_iteration, int icm_iterations) = 0;
	virtual void Segment() = 0;
	virtual void UpdateKinematicStructure(std::vector<cv::Mat>& curr_prop) = 0;
};

#endif