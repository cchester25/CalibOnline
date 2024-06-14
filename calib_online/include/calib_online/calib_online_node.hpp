#include <ros/ros.h>
#include <thread>
#include <boost/bind.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/time_synchronizer.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include <pcl/features/boundary.h>
#include <pcl/features/normal_3d.h>
#include <pcl/common/transforms.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/passthrough.h>
#include <cv_bridge/cv_bridge.h>
#include "depth_completion.hpp"
#include "miscalib_detction.hpp"
#include "measurement_noise.hpp"
#include <time.h>
#include <fstream>
#include <cstdlib> // Needed to use the exit function
#include <cmath>
#include <ceres/ceres.h>
std::string PointCloudTopic, ImageTopic, RawImageTopic, dataset;
std::vector<double> Intrinsic,Extrinsic,Dist_coeffs;
cv::Mat depth_edge,lidar_edge,mask_global,lidar_depth_cor,raw_image,accum_lidar;
bool lidar_ok = false,depth_ok = false, rawimage_ok = false, miscalib=true, use_deriction = true,
        high_noise=false, correct_ok = false, Distor = false;
int width=1226,height=370;
float fx,fy,cx,cy,pixel_inc=0,range_inc=0,degree_inc=0;
clock_t start,end;
double duration;
std::ofstream outfile;
Sophus::SE3d gt_pose_cor,gt_pose_;

// instrins matrix
Eigen::Matrix3d inner;
// Distortion coefficient
Eigen::Vector4d distor;

std::vector<PairData> Pair_list;
std::vector<std::vector<PairData>> Frame_list;
std::vector<cv::Mat> edge_list(2);
int accu_num = 0;
std::string outputFilename = "your path";


void get_random_RPYxyz(double& bR, double& bP, double& bY, double& bx, double& by, double& bz)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    double minAngle = -5.0;
    double maxAngle = 5.0;
    double minRadian = minAngle * M_PI /180.0;
    double maxRadian= maxAngle * M_PI /180.0;;
    double minTrans = -0.1;
    double maxTrans = 0.1;
    std::uniform_real_distribution<double> dis_RPY(minRadian,maxRadian);
    std::uniform_real_distribution<double> dis_XYZ(minTrans,maxTrans);
    // bR = dis_RPY(gen);
    // bP = dis_RPY(gen);
    // bY = dis_RPY(gen);
    // bx = dis_XYZ(gen);
    // by = dis_XYZ(gen);
    // bz = dis_XYZ(gen);
    bR = maxRadian;
    bP = maxRadian;
    bY = maxRadian;
    bx = maxTrans;
    by = maxTrans;
    bz = maxTrans;
    std::cout << " bR: " << bR << " bP: " << bP << " bY: " << bY << " bx: " << bx << " by: " << by << " bz: " << bz <<std::endl;
}


void get_projectimage(cv::Mat lidar, cv::Mat image)
{
    cv::Mat color_depth;
    lidar /= 10.0;
    lidar *=255;
    cv::dilate(lidar, lidar, full_kernel3x3);
    cv::Mat mask = (lidar > 0.1);

    lidar.convertTo(color_depth, CV_8UC1);
    cv::applyColorMap(color_depth, color_depth, cv::COLORMAP_JET);
    color_depth.copyTo(image,mask);
    std::string imagename = std::to_string(accu_num) + ".png";
    cv::imwrite(outputFilename + imagename , image);
}