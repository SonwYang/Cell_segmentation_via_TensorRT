
#include<iostream>
#include <fstream>
#include <sstream>
#include <numeric>
#include <chrono>
#include <vector>
using namespace std;

#ifndef _SEG_LIB_H
#define _SEG_LIB_H
#ifdef SEG_API_EXPORTS
#define SEG_API _declspec(dllexport)
#else
#define SEG_API _declspec(dllimport)
#endif
extern "C" SEG_API void medicalSeg(std::string engine_file_path, std::string input_video_path);
extern "C" SEG_API int CELL_COUNT;
extern "C" SEG_API int CELL_PERCENT;
#endif
 