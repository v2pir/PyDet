#include <opencv2/opencv.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/core.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <pybind11/pybind11.h>
#include <math.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include "pybind11/numpy.h"
using namespace std;
using namespace cv;

namespace py = pybind11;

/*

g++ -std=c++11  $(pkg-config --cflags --libs opencv4) -undefined dynamic_lookup -I./pybind11/include/ `python3.10 -m pybind11 --includes` filters.cpp -o filters.so `python3.10-config --ldflags`

*/


py::array_t<uint8_t> saturate(py::array_t<uint8_t>& img, float saturation){

    unsigned int ro = img.shape(0);
    unsigned int co = img.shape(1);

    cv::Mat mat(img.shape(0), img.shape(1), CV_MAKETYPE(CV_8U, img.shape(2)),
                const_cast<uint8_t*>(img.data()), img.strides(0));

    cv::Mat image = mat.clone();
    //resize image

    cv::Mat g = cv::Mat::zeros(Size(image.cols, image.rows), CV_8UC1);

    //convert image to hsv
    cv::Mat hsv;
    cv::cvtColor(image,hsv, COLOR_BGR2HSV);

    cv::Mat sat;

    //split hsv image into channels to isolate saturation channel
    cv::Mat channels[3];
    cv::split(hsv, channels);

    //scale saturation channel and merge
    channels[1] *= saturation;
    cv::merge(channels, 3, sat);

    //convert saturated image back to BGR from HSV
    cv::Mat im;
    cv::cvtColor(sat, im, COLOR_HSV2BGR);

    py::array_t<uint8_t> imgout(
                                py::buffer_info(
                                    im.data,
                                    sizeof(uint8_t),
                                    py::format_descriptor<uint8_t>::format(),
                                    3,
                                    std::vector<size_t> {ro, co, 3},
                                    std::vector<size_t> {co * sizeof(uint8_t) * 3, sizeof(uint8_t) * 3, sizeof(uint8_t)}
                                )
    );

    return imgout;
}



PYBIND11_MODULE(filters, handle) {
    handle.doc() = "This is the module docs";
    handle.def("saturate", &saturate);
}