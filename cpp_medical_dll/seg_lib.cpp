#include "seg_lib.h"
#include <stdio.h>
#include <iostream>

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
#define NMS_THRESH 0.45
#define BBOX_CONF_THRESH 0.3

using namespace nvinfer1;

// stuff we know about the network and the input/output blobs
static const int INPUT_W = 1792;
static const int INPUT_H = 1024;
static const int NUM_CLASSES = 2;
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

void medicalSeg(std::string engine_file_path, std::string input_video_path) {

	cudaSetDevice(DEVICE);
	// create a model using the API directly and serialize it to a stream
	char *trtModelStream{ nullptr };
	size_t size{ 0 };

	std::cout << "starting resnet" << std::endl;

	std::cout << "engine_file_path: " << engine_file_path<<std::endl;
	std::cout << "input_video_path: " << input_video_path << std::endl;

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
	std::cout << "load engine file"<< std::endl;
	IRuntime* runtime = createInferRuntime(gLogger);
	assert(runtime != nullptr);
	std::cout << "xxxxxxxx 1" << std::endl;
	ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
	assert(engine != nullptr);
	std::cout << "xxxxxxxx 2" << std::endl;
	IExecutionContext* context = engine->createExecutionContext();
	assert(context != nullptr);

	delete[] trtModelStream;
	auto out_dims = engine->getBindingDimensions(1);
	auto output_size = 1;
	for (int j = 0; j < out_dims.nbDims; j++) {
		std::cout << "out_dims.d" << j << ": " << out_dims.d[j] << std::endl;
		output_size *= out_dims.d[j];
	}
	static float* prob = new float[output_size];
	std::cout << "The output size : " << output_size << std::endl;

	cv::VideoCapture capture(input_video_path);
	cv::Mat frame;
	while (true)
	{
		capture.read(frame);

		cv::Mat img = frame;
		int img_w = img.cols;
		int img_h = img.rows;

		cv::Mat pr_img;
		cv::resize(img, pr_img, cv::Size(INPUT_W, INPUT_H));
		// std::cout << "blob image" << std::endl;

		float* blob;
		blob = blobFromImage(pr_img);

		// run inference
		auto start = std::chrono::system_clock::now();
		doInference(*context, blob, prob, output_size, pr_img.size());
		auto end = std::chrono::system_clock::now();
		std::cout << "doInference done " << output_size << std::endl;
		float fps = 1000000 / std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
		// generate img
		
		cv::Mat outimg(INPUT_H, INPUT_W, CV_8UC1);
		int count_pixel = 0;
		for (int row = 0; row < INPUT_H; ++row)
		{
			uchar *uc_pixel = outimg.data + row * outimg.step;
			for (int col = 0; col < INPUT_W; ++col)
			{
				//uc_pixel[col] = (uchar)prob[row * INPUT_W + col];
				if (prob[row * INPUT_W + col] > prob[INPUT_H * INPUT_W + row * INPUT_W + col])
				{
					uc_pixel[col] = 0;
				}
				else
				{
					uc_pixel[col] = 1;
					count_pixel++;
				}
			}
		}

		std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

		// 计数
		cv::Mat Can_img;
		cv::Canny(outimg * 255, Can_img, 100, 250);
		vector<vector<cv::Point>> contours;
		vector<cv::Vec4i> hierarchy;
		cv::findContours(Can_img, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point());

		int count_cell = contours.size();
		int CELL_COUNT = count_cell;

		ostringstream buffer_fps;
		buffer_fps << fps;
		string text = "FPS: " + buffer_fps.str();
		std::cout << text << std::endl;

		int percent = (count_pixel * 100) / (INPUT_W * INPUT_H);
		int CELL_PERCENT = percent;
		ostringstream buffer_percent;
		buffer_percent << CELL_PERCENT;
		string text2 = "percent: " + buffer_percent.str() + "%";

		ostringstream buffer_cell;
		buffer_cell << CELL_COUNT;
		string text3 = "The number of cell: " + buffer_cell.str();
		std::cout << text3 << std::endl;

		int font_face = cv::FONT_HERSHEY_COMPLEX;
		double font_scale = 2;
		int thickness = 2;
		int baseline;
		cv::Size text_size = cv::getTextSize(text, font_face, font_scale, thickness, &baseline);
		cv::Size text_size2 = cv::getTextSize(text2, font_face, font_scale, thickness, &baseline);
		cv::Size text_size3 = cv::getTextSize(text3, font_face, font_scale, thickness, &baseline);

		cv::Mat im_color;
		cv::cvtColor(outimg, im_color, cv::COLOR_GRAY2RGB);
		cv::Mat lut = createLTU(NUM_CLASSES);
		cv::LUT(im_color, lut, im_color);
		cv::cvtColor(im_color, im_color, cv::COLOR_RGB2GRAY);
		cv::applyColorMap(im_color, im_color, cv::COLORMAP_HOT);

		//将文本框居中绘制
		cv::Point origin;
		origin.x = im_color.cols / 8 - text_size.width / 2;
		origin.y = im_color.rows / 8 + text_size.height / 2;
		cv::putText(im_color, text, origin, font_face, font_scale, cv::Scalar(0, 255, 255), thickness, 8, 0);

		cv::Point origin2;
		origin2.x = im_color.cols / 8 - text_size.width / 2;
		origin2.y = im_color.rows / 8 + text_size2.height / 2 + 100;
		cv::putText(im_color, text2, origin2, font_face, font_scale, cv::Scalar(0, 255, 255), thickness, 8, 0);

		cv::Point origin3;
		origin3.x = im_color.cols / 8 - text_size.width / 2;
		origin3.y = im_color.rows / 8 + text_size3.height / 2 + 200;
		cv::putText(im_color, text3, origin3, font_face, font_scale, cv::Scalar(0, 255, 255), thickness, 8, 0);

		cv::imshow("lane_vis", im_color);
		int c = cv::waitKey(10);
		if (c == 27) {
			break;
		}
	}
	// false color
}


