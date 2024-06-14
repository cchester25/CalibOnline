#include <opencv2/opencv.hpp>
#include <vector>
//*******************full kernels********************//
cv::Mat full_kernel3x3 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
cv::Mat full_kernel5x5 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
cv::Mat full_kernel7x7 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
cv::Mat full_kernel9x9 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9, 9));
cv::Mat full_kernel31x31 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(31, 31));
//*******************cross kernel********************//
cv::Mat cross_kernel3x3 = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));
cv::Mat cross_kernel5x5 = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(5, 5));
cv::Mat cross_kernel7x7 = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(7, 7));
//*******************diamond kernel********************//
cv::Mat one_kernel3x3 = (cv::Mat_<uint8_t>(3,3) << 0, 1, 0,
    0, 1, 0,
    0, 1, 0);
cv::Mat diamond_kernel5x5 = (cv::Mat_<uint8_t>(5,5) << 0, 0, 1, 0, 0,
0, 1, 1, 1, 0,
1, 1, 1, 1, 1,
0, 1, 1, 1, 0,
0, 0, 1, 0, 0);
cv::Mat diamond_kernel7x7 = (cv::Mat_<uint8_t>(7,7) << 0, 0, 0, 1, 0, 0, 0,
    0, 0, 1, 1, 1, 0, 0,
    0, 1, 1, 1, 1, 1, 0,
    1, 1, 1, 1, 1, 1, 1,
    0, 1, 1, 1, 1, 1, 0,
    0, 0, 1, 1, 1, 0, 0,
    0, 0, 0, 1, 0, 0, 0);
//*******************sharpen kernel********************//
cv::Mat sharpen_kernel3x3 = (cv::Mat_<char>(3,3) << 0, -1, 0,
    -1, 5, -1,
    0, -1, 0);
//*******************depth completion********************//

cv::Mat depth_completion(const cv::Mat& lidar_depth, int img_width, int img_height)
{
    // double time_start = clock();
    cv::Mat depth_ = lidar_depth.clone();

    cv::Mat mask;
    cv::Mat dilated_map;
    // //depth inversion
    mask = (depth_ > 0.1);
    cv::subtract(100.0, depth_, depth_, mask);
    // //custom kernel dilation
    cv::dilate(depth_, depth_, diamond_kernel5x5);

    // //small hole closure
    cv::morphologyEx(depth_, depth_, cv::MORPH_CLOSE, full_kernel5x5);
    //small hole fill
    mask = (depth_ < 0.1);
    cv::dilate(depth_, dilated_map, full_kernel7x7);
    dilated_map.copyTo(depth_, mask);
    //big hole fill 
    mask = (depth_ < 0.1);
    cv::dilate(depth_ ,dilated_map, full_kernel31x31);
    dilated_map.copyTo(depth_, mask);
    // //median and gaussion blur
    cv::medianBlur(depth_, depth_, 5);
    mask = (depth_ > 0.1);//valid_pixels
    cv::GaussianBlur(depth_, dilated_map, cv::Size(5,5), 0);
    dilated_map.copyTo(depth_, mask);
    // //depth inversion
    mask = (depth_ > 0.1);
    cv::subtract(100.0, depth_, depth_, mask);
    depth_ /= 80.0;
    depth_ *=255;
    cv::Mat color_depth;
    depth_.convertTo(color_depth, CV_8UC1);
    cv::applyColorMap(color_depth, color_depth, cv::COLORMAP_JET);

    return color_depth;

}