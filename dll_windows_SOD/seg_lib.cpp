#include "seg_lib.h"
#include <stdio.h>
#include <iostream>

#include <fstream>
#include <sstream>
#include <numeric>
#include <chrono>
#include <vector>

#include <opencv2/opencv.hpp>
#include <dirent.h>
#include <cstdio>
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"

#pragma comment(lib, "nvinfer.lib")
#pragma comment(lib, "nvinfer_plugin.lib")
#pragma comment(lib, "cudart.lib")
#pragma comment(lib, "opencv_world340.lib")


string Convert(float Num)

{
	ostringstream oss;
	oss << Num;
	string str(oss.str());
	return str;

}


#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

#define DEVICE 0  // GPU id
#define CONF_THRESH 0.65

using namespace nvinfer1;

// stuff we know about the network and the input/output blobs
static const int INPUT_W = 896;
static const int INPUT_H = 448;
static const int NUM_CLASSES = 1;
const char* INPUT_BLOB_NAME = "input_0";
const char* OUTPUT_BLOB_NAME = "output_0";
static Logger gLogger;

cv::Mat static_resize(cv::Mat& img) {
	float r = min(INPUT_W / (img.cols*1.0), INPUT_H / (img.rows*1.0));
	// r = std::min(r, 1.0f);
	int unpad_w = r * img.cols;
	int unpad_h = r * img.rows;
	cv::Mat re(unpad_h, unpad_w, CV_8UC3);
	cv::resize(img, re, re.size());
	cv::Mat out(INPUT_H, INPUT_W, CV_8UC3, cv::Scalar(114, 114, 114));
	re.copyTo(out(cv::Rect(0, 0, re.cols, re.rows)));
	return out;
}


float* blobFromImage(cv::Mat& img) {
	float* blob = new float[img.total() * 3];
	int channels = 3;
	int img_h = img.rows;
	int img_w = img.cols;
	 
	for (size_t c = 0; c < channels; c++)
	{
		for (size_t h = 0; h < img_h; h++)
		{
			for (size_t w = 0; w < img_w; w++)
			{
				blob[c * img_w * img_h + h * img_w + w] =
					(float)img.at<cv::Vec3b>(h, w)[c] / 255.0;
			}
		}
	}
 
 
	return blob;
}


int Clear_MicroConnected_Areas(cv::Mat src, cv::Mat &dst, double min_area)
{
	// 备份复制
	dst = src.clone();
	std::vector<std::vector<cv::Point> > contours;  // 创建轮廓容器
	std::vector<cv::Vec4i> 	hierarchy;

	// 寻找轮廓的函数
	// 第四个参数CV_RETR_EXTERNAL，表示寻找最外围轮廓
	// 第五个参数CV_CHAIN_APPROX_NONE，表示保存物体边界上所有连续的轮廓点到contours向量内
	cv::findContours(src, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE, cv::Point());

	int count_cell = contours.size();

	int count_small = 0;

	if (!contours.empty() && !hierarchy.empty())
	{
		std::vector<std::vector<cv::Point> >::const_iterator itc = contours.begin();
		// 遍历所有轮廓
		while (itc != contours.end())
		{
			// 定位当前轮廓所在位置
			cv::Rect rect = cv::boundingRect(cv::Mat(*itc));
			// contourArea函数计算连通区面积
			double area = contourArea(*itc);
			// 若面积小于设置的阈值
			if (area < min_area)
			{
				count_small++;
			}
			itc++;
		}
	}

	return count_cell - count_small;
}


float sigmoid(float x)
{
	return (1 / (1 + exp(-x)));
}


void doInference(IExecutionContext& context, float* input, float* output, const int output_size, cv::Size input_shape) {
	const ICudaEngine& engine = context.getEngine();

	// Pointers to input and output device buffers to pass to engine.
	// Engine requires exactly IEngine::getNbBindings() number of buffers.
	assert(engine.getNbBindings() == 2);
	void* buffers[2];

	// In order to bind the buffers, we need to know the names of the input and output tensors.
	// Note that indices are guaranteed to be less than IEngine::getNbBindings()
	const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);

	assert(engine.getBindingDataType(inputIndex) == nvinfer1::DataType::kFLOAT);
	const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);
	assert(engine.getBindingDataType(outputIndex) == nvinfer1::DataType::kFLOAT);
	int mBatchSize = engine.getMaxBatchSize();

	// Create GPU buffers on device
	CHECK(cudaMalloc(&buffers[inputIndex], 3 * input_shape.height * input_shape.width * sizeof(float)));
	CHECK(cudaMalloc(&buffers[outputIndex], output_size * sizeof(float)));

	// Create stream
	cudaStream_t stream;
	CHECK(cudaStreamCreate(&stream));

	// DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
	CHECK(cudaMemcpyAsync(buffers[inputIndex], input, 3 * input_shape.height * input_shape.width * sizeof(float), cudaMemcpyHostToDevice, stream));
	context.enqueue(1, buffers, stream, nullptr);
	CHECK(cudaMemcpyAsync(output, buffers[outputIndex], output_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
	cudaStreamSynchronize(stream);

	// Release stream and buffers
	cudaStreamDestroy(stream);
	CHECK(cudaFree(buffers[inputIndex]));
	CHECK(cudaFree(buffers[outputIndex]));
}

cv::Mat createLTU(int len)
{

	cv::Mat lookUpTable(1, 256, CV_8U);
	uchar *p = lookUpTable.data;
	for (int j = 0; j < 256; ++j)
	{
		p[j] = (j * (256 / len) > 255) ? uchar(255) : (uchar)(j * (256 / len));
	}
	return lookUpTable;
}


class MedicalSeg
{
public:
	int count;
	int percent;

	void init(char *engine_path) {
		std::string engine_file_path(engine_path);
		cudaSetDevice(DEVICE);
		// create a model using the API directly and serialize it to a stream
		char *trtModelStream{ nullptr };
		size_t size{ 0 };

		// load engine file
		std::ifstream file(engine_file_path, std::ios::binary);
		if (file.good()) {
			file.seekg(0, file.end);
			size = file.tellg();
			file.seekg(0, file.beg);
			trtModelStream = new char[size];
			assert(trtModelStream);
			file.read(trtModelStream, size);
			file.close();
		}

		//load engine
		std::cout << "load engine file" << std::endl;
		IRuntime* runtime = createInferRuntime(gLogger);
		assert(runtime != nullptr);
		std::cout << "xxxxxxxx 1" << std::endl;
		this->engine = runtime->deserializeCudaEngine(trtModelStream, size);
		assert(engine != nullptr);
		std::cout << "xxxxxxxx 2" << std::endl;
		this->context = engine->createExecutionContext();
		assert(context != nullptr);

		delete[] trtModelStream;
	}

	void prediction(int rows, int cols, unsigned __int32 *src_data) {

		auto out_dims = engine->getBindingDimensions(1);
		auto output_size = 1;
		for (int j = 0; j < out_dims.nbDims; j++) {
			//std::cout << "out_dims.d" << j << ": " << out_dims.d[j] << std::endl;
			output_size *= out_dims.d[j];
		}

		float* prob = new float[output_size];
		//std::cout << "The output size : " << output_size << std::endl;

		cv::Mat img0 = cv::Mat(rows, cols, CV_8UC3, src_data);
		cv::Mat img;
		cv::cvtColor(img0, img, cv::COLOR_BGR2RGB);
		int img_w = img.cols;
		int img_h = img.rows;

		cv::Mat preImg;
		cv::resize(img, preImg, cv::Size(INPUT_W, INPUT_H));

		float* blob;
		blob = blobFromImage(preImg);

		// run inference
		doInference(*context, blob, prob, output_size, preImg.size());
	
		// generate img
		//cout << "sigmoid" << endl;
		float * mask = new float[output_size];
		for (int i = 0; i < INPUT_W*INPUT_H; i++) {
			mask[i] = sigmoid(prob[i]);
		}

		int pixel_count = 0;
		//cout << "generate mask" << endl;
		cv::Mat mask_mat = cv::Mat(INPUT_H, INPUT_W, CV_8UC1);
		uchar *ptmp = NULL;
		for (int i = 0; i < INPUT_H; i++) {
			ptmp = mask_mat.ptr<uchar>(i);
			for (int j = 0; j < INPUT_W; j++) {
				float * pixcel = mask + i * INPUT_W + j;
				// std::cout << *pixcel << std::endl;
				if (*pixcel > CONF_THRESH) {
					ptmp[j] = 255;
					pixel_count++;
				}
				else {
					ptmp[j] = 0;
				}
			}
		}

		cv::Mat outImg;
		int count_cell = Clear_MicroConnected_Areas(mask_mat, outImg, 49);
		this->count = count_cell;
		this->percent = 100 * pixel_count / (INPUT_W * INPUT_H);

		cv::Mat im_color;

		//黑白变红色
		cv::cvtColor(outImg, im_color, cv::COLOR_GRAY2RGB);
		std::vector<cv::Mat> mv;
		split(im_color, mv);
		cv::Mat dst;
		mv[0] = 0;
		mv[1] = 0;
		merge(mv, dst);

		cv::Mat res;
		cv::add(preImg, dst, res);

		cv::Mat finalRes;
		cv::resize(res, finalRes, cv::Size(cols, rows));

		memcpy(src_data, finalRes.data, rows * cols * 3);

		delete blob;
		blob = NULL;
		delete prob;
		prob = NULL;
	}
private:
	ICudaEngine* engine;
	IExecutionContext* context;
};


extern "C" {
	MedicalSeg ms;
	extern "C" _declspec(dllexport) void init(char *engine_path) {
		ms.init(engine_path);
	}

	extern "C" _declspec(dllexport) void prediction(int rows, int cols, unsigned __int32 *src_data);
	
	void prediction(int rows, int cols, unsigned __int32 *src_data)
	{
		return ms.prediction(rows, cols, src_data);
	}

	extern "C" _declspec(dllexport) int getCount();

	int getCount()
	{
		return ms.count;
	}

	extern "C" _declspec(dllexport) int getPercent();

	int getPercent()
	{
		return ms.percent;
	}
}


