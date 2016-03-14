#ifndef FIO_H
#define FIO_H

#include <fstream>
#include "opencv2/core/core.hpp"

class FileIO{

public:
    static void ReadMatInt(cv::Mat& dst, int h, int w, std::string name);
    static void ReadMatFloat(cv::Mat& dst, int h, int w, std::string name);
	static void ReadFloatMatToDouble(cv::Mat& dst, int h, int w, std::string name);
    static void ReadMatDouble(cv::Mat& dst, int h, int w, std::string name);
    static void WriteMatInt(cv::Mat& src, int h, int w, std::string name);
    static void WriteMatFloat(cv::Mat& src, int h, int w, std::string name);
    static void AppendMatFloat(cv::Mat& src, int h, int w, std::string name);
    static void RecordMatDouble(cv::Mat& src, int h, int w, std::string name, int append_flag);
    static void WriteMatDouble(cv::Mat& src, int h, int w, std::string name);
	static std::string ReadFileString(std::string name);
};

#endif
