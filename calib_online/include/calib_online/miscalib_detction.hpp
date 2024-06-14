#include <opencv2/opencv.hpp>
#include <numeric>
#include "sophus/se3.hpp"
bool use_mean = false;
struct PairData {
    double x, y, z, ul, vl, u, v;
    Eigen::Vector2d direction_cam;
    Eigen::Vector2d direction_lidar;
};
std::string matchname = "your path";
int matchnums = 0;
bool use_pair = true;
float miscalib_detection(cv::Mat lidar_edge,
                        cv::Mat lidar_points, 
                        cv::Mat depth_edge, 
                        std::vector<PairData>& Pair_list,
                        Sophus::SE3d gt_pose_,
                        float fx,
                        float fy,
                        float cx,
                        float cy)
{
    Pair_list.clear();

    std::vector<cv::Point2f> val_coordint,query_coordint;
    cv::findNonZero(depth_edge,val_coordint);
    cv::findNonZero(lidar_edge,query_coordint);

    cv::Mat val_features = cv::Mat(val_coordint).reshape(1);
    cv::Mat source;
    val_features.convertTo(source,CV_32F);
    // use kd-tree to search nearest piexls
    cv::flann::KDTreeIndexParams indexParams(1);
    cv::flann::Index kdtree(source,indexParams);
    int queryNum = 3; 
    std::vector<float> vecQuery(2);
    cv::flann::SearchParams params(32);

    std::vector<Eigen::Vector3d> effect_pts;
    std::vector<float> effect_mean(query_coordint.size(),0.0);
    std::vector<cv::Point2f> lidar_query;
    std::vector<cv::Point2f> mean_qi;
    std::vector<float> vec_mean;
    // calculate the miscalibration

    for(int idx = 0;idx < query_coordint.size();idx++)
    {
        auto query = query_coordint[idx];
        vecQuery[0] = query.x;
        vecQuery[1] = query.y;
        std::vector<int> vecInd(queryNum);
        std::vector<float> vecDist(queryNum);
        kdtree.knnSearch(vecQuery, vecInd, vecDist, queryNum, params);
        float mean = std::accumulate(vecDist.begin(),vecDist.end(),0.0)/vecDist.size();
        if(mean < 20.0)
        {
            vec_mean.push_back(mean);
        }
        if(mean < 2.0)
        {
            cv::Point2f qi;
            qi.x = val_coordint[vecInd[0]].x + val_coordint[vecInd[1]].x + val_coordint[vecInd[2]].x;
            qi.y = val_coordint[vecInd[0]].y + val_coordint[vecInd[1]].y + val_coordint[vecInd[2]].y;
            qi.x /= float(queryNum);
            qi.y /= float(queryNum);


            float depth_query = lidar_points.at<float>(vecQuery[1],vecQuery[0]);
            if(depth_query > 1.0)
            {
                Eigen::Vector3d lidar_edge_point;
                lidar_edge_point[2] = depth_query;
                lidar_edge_point[0] = (vecQuery[0]-cx)*lidar_edge_point[2]/fx;
                lidar_edge_point[1] = (vecQuery[1]-cy)*lidar_edge_point[2]/fy;
                Eigen::Vector3d q = gt_pose_.inverse() * lidar_edge_point;
                mean_qi.push_back(qi);
                lidar_query.push_back(query);
                effect_pts.push_back(q);
            }

        }
    }

    // get pair_points
    cv::Mat spare_edge(lidar_edge.rows, lidar_edge.cols , CV_8UC1, cv::Scalar(0));
    if(mean_qi.size() > 20 && use_pair)
    {
        // use_pair = false;
        for(int num = 0 ; num < mean_qi.size() ; num++)
        {
            PairData pair_data;
            pair_data.x = effect_pts[num][0];
            pair_data.y = effect_pts[num][1];
            pair_data.z = effect_pts[num][2];
            pair_data.ul = lidar_query[num].x;
            pair_data.vl = lidar_query[num].y;
            pair_data.u = mean_qi[num].x;
            pair_data.v = mean_qi[num].y;
            Pair_list.push_back(pair_data);
            spare_edge.at<u_char>(pair_data.v,pair_data.u)=255;
        }
        use_pair = false;
    }

    if(vec_mean.size() == 0)
    { 
        return 0.5;
    }

    float sum = std::accumulate(vec_mean.begin(),vec_mean.end(),0.0);
    float mean_ = sum/float(vec_mean.size());
    float mis_num = 0.0 , delta = 1.2;
    //******************draw the match image****************************
    /*cv::Mat match_image;
    cv::Mat lidar;
    lidar_edge.convertTo(lidar,CV_8UC1);
    cv::vconcat(lidar,depth_edge,match_image);
    cv::cvtColor(match_image,match_image,cv::COLOR_GRAY2BGR);

    for(int i=0;i<mean_qi.size();i++)
    {
        cv::Point2f q = mean_qi[i];
        q.y = q.y + 370;// change to the image height
        cv::Point2f p = lidar_query[i];
        cv::line(match_image,p,q,cv::Scalar(0,0,255),1);

    }
    cv::imshow("match image",match_image);
    cv::waitKey(1);*/
    //******************draw the match image****************************
    if(use_mean)
    {
        double u1 = 4.8,u2 =5.2,delta1 = 0.13,delta2 = 0.11;
        u1 = pow(mean_-u1,2);
        u2 = pow(mean_-u2,2);
        delta1 = 2*pow(delta1,2);
        delta2 = 2*pow(delta2,2);
        u1 = -u1 / delta1;
        u2 = -u2 / delta2;

        double pr_calib = exp(u2)/(exp(u1)+exp(u2));
        return pr_calib;
    }
    else
    {
        for(int i =0;i<vec_mean.size();i++)
        {
            if(vec_mean[i] > 4.8 * delta)
            {
                mis_num++;
            }
        }

        double pr = mis_num / float(vec_mean.size());

        double u1 = 0.09,u2 =0.515,delta1 = 0.13,delta2 = 0.11;
        u1 = pow(pr-u1,2);
        u2 = pow(pr-u2,2);
        delta1 = 2*pow(delta1,2);
        delta2 = 2*pow(delta2,2);
        u1 = -u1 / delta1;
        u2 = -u2 / delta2;

        double pr_calib = exp(u2)/(exp(u1)+exp(u2));
        return pr_calib;
    }
}

