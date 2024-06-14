#include "calib_online/calib_online_node.hpp"

class edge_calib {
public:
    edge_calib(PairData p) { pd = p; }
    template <typename T>
    bool operator()(const T *_q, const T *_t, T *residuals) const {
        bool is_distor = Distor;
        if(is_distor)
        {
            Eigen::Matrix<T, 3, 3> innerT = inner.cast<T>();
            Eigen::Matrix<T, 4, 1> distorT = distor.cast<T>();
            Eigen::Quaternion<T> q_incre{_q[3], _q[0], _q[1], _q[2]};
            Eigen::Matrix<T, 3, 1> t_incre{_t[0], _t[1], _t[2]};
            Eigen::Matrix<T, 3, 1> p_l(T(pd.x), T(pd.y), T(pd.z));
            Eigen::Matrix<T, 3, 1> p_c = q_incre.toRotationMatrix() * p_l + t_incre;
            Eigen::Matrix<T, 3, 1> p_2 = innerT * p_c;
            T uo = p_2[0] / p_2[2];
            T vo = p_2[1] / p_2[2];
            const T &fx_ = innerT.coeffRef(0, 0);
            const T &cx_ = innerT.coeffRef(0, 2);
            const T &fy_ = innerT.coeffRef(1, 1);
            const T &cy_ = innerT.coeffRef(1, 2);
            T xo = (uo - cx_) / fx_;
            T yo = (vo - cy_) / fy_;
            T r2 = xo * xo + yo * yo;
            T r4 = r2 * r2;
            T distortion = 1.0 + distorT[0] * r2 + distorT[1] * r4;
            T xd = xo * distortion + (distorT[2] * xo * yo + distorT[2] * xo * yo) +
                distorT[3] * (r2 + xo * xo + xo * xo);
            T yd = yo * distortion + distorT[3] * xo * yo + distorT[3] * xo * yo +
                distorT[2] * (r2 + yo * yo + yo * yo);
            T ud = fx_ * xd + cx_;
            T vd = fy_ * yd + cy_;
            if(use_deriction)
            {
                if (T(pd.direction_cam(0)) == T(0.0) && T(pd.direction_cam(1)) == T(0.0))
                {
                    residuals[0] = ud - T(pd.u);
                    residuals[1] = vd - T(pd.v);
                }
                else
                {
                    residuals[0] = ud - T(pd.u);
                    residuals[1] = vd - T(pd.v);
                    Eigen::Matrix<T, 2, 2> I = Eigen::Matrix<float, 2, 2>::Identity().cast<T>();
                    Eigen::Matrix<T, 2, 1> n = pd.direction_cam.cast<T>();
                    Eigen::Matrix<T, 1, 2> nt = pd.direction_cam.transpose().cast<T>();
                    Eigen::Matrix<T, 2, 2> V = n * nt;//get a direction projection mat
                    V = I - V;
                    Eigen::Matrix<T, 2, 1> R = Eigen::Matrix<float, 2, 1>::Zero().cast<T>();
                    R.coeffRef(0, 0) = residuals[0];
                    R.coeffRef(1, 0) = residuals[1];
                    R = V * R;//=R - VR : delta u delta v - their projection on direction_cam
                    residuals[0] = R.coeffRef(0, 0);
                    residuals[1] = R.coeffRef(1, 0);
                }
            }
            else
            {
                residuals[0] = uo - T(pd.u);
                residuals[1] = vo - T(pd.v);
            }
            return true;
        }
        else
        {
            Eigen::Matrix<T, 3, 3> innerT = inner.cast<T>();
            Eigen::Quaternion<T> q_incre{_q[3], _q[0], _q[1], _q[2]};
            Eigen::Matrix<T, 3, 1> t_incre{_t[0], _t[1], _t[2]};
            Eigen::Matrix<T, 3, 1> p_l(T(pd.x), T(pd.y), T(pd.z));
            Eigen::Matrix<T, 3, 1> p_c = q_incre.toRotationMatrix() * p_l + t_incre;
            Eigen::Matrix<T, 3, 1> p_2 = innerT * p_c;
            T uo = p_2[0] / p_2[2];
            T vo = p_2[1] / p_2[2];
            if(use_deriction)
            {
                if (T(pd.direction_cam(0)) == T(0.0) && T(pd.direction_cam(1)) == T(0.0))
                {
                    residuals[0] = uo - T(pd.u);
                    residuals[1] = vo - T(pd.v);
                }
                else
                {
                    residuals[0] = uo - T(pd.u);
                    residuals[1] = vo - T(pd.v);
                    Eigen::Matrix<T, 2, 2> I = Eigen::Matrix<float, 2, 2>::Identity().cast<T>();
                    Eigen::Matrix<T, 2, 1> n = pd.direction_cam.cast<T>();
                    Eigen::Matrix<T, 1, 2> nt = pd.direction_cam.transpose().cast<T>();
                    Eigen::Matrix<T, 2, 2> V = n * nt;//get a direction projection mat
                    V = I - V;
                    Eigen::Matrix<T, 2, 1> R = Eigen::Matrix<float, 2, 1>::Zero().cast<T>();
                    R.coeffRef(0, 0) = residuals[0];
                    R.coeffRef(1, 0) = residuals[1];
                    R = V * R;//=R - VR : delta u delta v - their projection on direction_cam
                    residuals[0] = R.coeffRef(0, 0);
                    residuals[1] = R.coeffRef(1, 0);
                }
            }
            else
            {
                residuals[0] = uo - T(pd.u);
                residuals[1] = vo - T(pd.v);
            }

            return true;
        }

    }
    static ceres::CostFunction *Create(PairData p) {
        return (
            new ceres::AutoDiffCostFunction<edge_calib, 2, 4, 3>(new edge_calib(p)));
    }

private:
    PairData pd;
};

void lidar_callback(const sensor_msgs::PointCloud2ConstPtr& pl_msg)
{
    // 处理点云数据
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*pl_msg, *cloud);
    if(high_noise)
    {
        for(int i=0;i<cloud->size();i++)
        {
            if(cloud->points[i].x > 0 && cloud->points[i].x < 80.0)
            {
                calcCovarance(cloud->points[i],range_inc,degree_inc);
            }
        }
    }
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_trans(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::transformPointCloud(*cloud, *cloud_trans, gt_pose_.matrix().cast<float>());
    cv::Mat lidar_depth(height, width , CV_32FC1, cv::Scalar(0));

    for(int i=0;i<cloud_trans->size();i++)
    {
        if(cloud_trans->points[i].z > 0 && cloud_trans->points[i].z < 80.0)
        {
            int u =  fx * cloud_trans->points[i].x / cloud_trans->points[i].z + cx;
            int v =  fy * cloud_trans->points[i].y / cloud_trans->points[i].z + cy;
            if(u >= 0 && u < width && v >=0 && v < height)
            {
                lidar_depth.at<float>(v,u) = cloud_trans->points[i].z;
            }
        }
    } 

    cv::Mat color_depth = depth_completion(lidar_depth,width,height);
    cv::GaussianBlur(color_depth,color_depth,cv::Size(5,5),0,0);
    cv::Canny(color_depth,lidar_edge,20,60);
    cv::Mat edge_depth;
    lidar_depth.copyTo(edge_depth,lidar_edge);
    lidar_depth_cor = edge_depth;
    lidar_ok = true;
}

void image_callback(const sensor_msgs::ImageConstPtr& depth_msg)
{
    // 处理图像数据
    cv_bridge::CvImageConstPtr cv_ptr;
    try
    {
        cv_ptr = cv_bridge::toCvCopy(depth_msg, sensor_msgs::image_encodings::MONO8);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }
    depth_edge = cv_ptr->image;
    if(high_noise)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dis(0, pixel_inc);
        Eigen::MatrixXf shift(2, 1);
        shift << dis(gen), dis(gen);
        std::vector<cv::Point> val_coordint;
        cv::findNonZero(depth_edge,val_coordint);
        cv::Mat edge_rect(height, width , CV_8UC1, cv::Scalar(0));
        for(int i=0;i<val_coordint.size();i++)
        {
            cv::Point shift_point = val_coordint[i];
            shift_point.x += shift(0,0);
            shift_point.y += shift(1,0);
            if(shift_point.x < width && shift_point.y < height
            && shift_point.x >= 0 && shift_point.y >= 0)
            {
                edge_rect.at<uchar>(shift_point.y,shift_point.x) = 255;
            }
        }
        depth_edge = edge_rect;
    }
    depth_ok=true;
}

void rawimage_callback(const sensor_msgs::ImageConstPtr& image_msg)
{
    // 处理图像数据
    cv_bridge::CvImageConstPtr cv_ptr;
    try
    {
        cv_ptr = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }
    raw_image = cv_ptr->image;
    rawimage_ok = true;

}


void MiscalibDet()
{
    int dec_num = 0;
    double probability_mean=0.0;
    ros::Rate loop_rate(10);
    while(ros::ok())
    {
        if(lidar_ok && depth_ok)
        {
            depth_ok = false;
            lidar_ok = false;            
            cv::Mat depth_edge_copy = depth_edge.clone();
            cv::Mat lidar_edge_copy= lidar_edge.clone();
            cv::Mat lidar_edge_ = lidar_depth_cor.clone();

            double probability = miscalib_detection(lidar_edge_copy,lidar_edge_,depth_edge_copy,Pair_list,gt_pose_,fx,fy,cx,cy);
            if(Pair_list.size() > 20)
            {
                cv::Mat lidar_line;
                lidar_edge_.convertTo(lidar_line, CV_8UC1);
                edge_list[0] = lidar_edge_;
                edge_list[1] = depth_edge_copy;
            }
            if(probability < 0.1)
                continue;
            probability_mean += probability;
            dec_num++; 
            probability_mean /= dec_num;
            if(dec_num == 50)
            {
                if(probability_mean > 0.4)
                {
                    std::cout << "\033[31;1mMiscalibration Detected!\033[0m" << std::endl;
                    miscalib = true;
                }
                else
                {
                    std::cout << "miscalib probability = " << probability_mean << "  " <<"\033[32;1mAllready calibrated!\033[0m" << std::endl;
                }

                dec_num = 0;
                probability_mean = 0.0;
            }
        }
        loop_rate.sleep();
    }
}

void get_direction(std::vector<PairData>& Pair_list)
{
    std::vector<cv::Point2f> val_coordint,query_coordint;
    cv::findNonZero(edge_list[0],query_coordint);
    cv::findNonZero(edge_list[1],val_coordint);
    cv::Mat val_features = cv::Mat(val_coordint).reshape(1);
    cv::Mat query_features = cv::Mat(query_coordint).reshape(1);
    cv::Mat source_c,source_l;
    val_features.convertTo(source_c,CV_32F);
    query_features.convertTo(source_l,CV_32F);

    cv::flann::KDTreeIndexParams indexParams(1);
    cv::flann::Index kdtree_c(source_c,indexParams);
    cv::flann::Index kdtree_l(source_l,indexParams);
    int queryNum = 3;
    std::vector<float> vecQuery(2);
    cv::flann::SearchParams params(32);

    for(int idl = 0; idl < Pair_list.size(); idl++)
    {
        PairData query = Pair_list[idl];
        vecQuery[0] = query.ul;
        vecQuery[1] = query.vl;
        std::vector<int> vecInd(queryNum);
        std::vector<float> vecDist(queryNum);
        kdtree_l.knnSearch(vecQuery, vecInd, vecDist, queryNum, params);
        Eigen::Vector2d mean_point(0, 0);
        for(int near_id = 0; near_id < queryNum; near_id++)
        {
            mean_point(0) += query_coordint[vecInd[near_id]].x;
            mean_point(1) += query_coordint[vecInd[near_id]].y;
        }
        mean_point(0) = mean_point(0) / queryNum;
        mean_point(1) = mean_point(1) / queryNum;

        Eigen::Matrix2d S;
        Eigen::Vector2d direction;
        S << 0, 0, 0, 0;
        for (size_t i = 0; i < queryNum; i++) {
            Eigen::Vector2d val_point(0, 0);
            val_point(0) = query_coordint[vecInd[i]].x;
            val_point(1) = query_coordint[vecInd[i]].y;
            Eigen::Matrix2d s = (val_point - mean_point) * (val_point - mean_point).transpose();
            S += s;
        }
        Eigen::EigenSolver<Eigen::Matrix<double, 2, 2>> es(S);
        Eigen::MatrixXcd evecs = es.eigenvectors();
        Eigen::MatrixXcd evals = es.eigenvalues();
        Eigen::MatrixXd evalsReal;
        evalsReal = evals.real();
        Eigen::MatrixXf::Index evalsMax;
        evalsReal.rowwise().sum().maxCoeff(&evalsMax);
        direction << evecs.real()(0, evalsMax), evecs.real()(1, evalsMax); 
        Pair_list[idl].direction_lidar = direction;
    }

    for(int idl = 0; idl < Pair_list.size(); idl++)
    {
        PairData query = Pair_list[idl];
        vecQuery[0] = query.u;
        vecQuery[1] = query.v;
        std::vector<int> vecInd(queryNum);
        std::vector<float> vecDist(queryNum);
        kdtree_c.knnSearch(vecQuery, vecInd, vecDist, queryNum, params);
        Eigen::Vector2d mean_point(0, 0);
        for(int near_id = 0; near_id < queryNum; near_id++)
        {
            mean_point(0) += val_coordint[vecInd[near_id]].x;
            mean_point(1) += val_coordint[vecInd[near_id]].y;
        }
        mean_point(0) = mean_point(0) / queryNum;
        mean_point(1) = mean_point(1) / queryNum;
        // move to corre
        Eigen::Matrix2d S;
        Eigen::Vector2d direction;
        S << 0, 0, 0, 0;
        for (size_t i = 0; i < queryNum; i++) {
            Eigen::Vector2d val_point(0, 0);
            val_point(0) = val_coordint[vecInd[i]].x;
            val_point(1) = val_coordint[vecInd[i]].y;
            Eigen::Matrix2d s = (val_point - mean_point) * (val_point - mean_point).transpose();
            S += s;
        }
        Eigen::EigenSolver<Eigen::Matrix<double, 2, 2>> es(S);
        Eigen::MatrixXcd evecs = es.eigenvectors();
        Eigen::MatrixXcd evals = es.eigenvalues();
        Eigen::MatrixXd evalsReal;
        evalsReal = evals.real();
        Eigen::MatrixXf::Index evalsMax;
        evalsReal.rowwise().sum().maxCoeff(&evalsMax);
        direction << evecs.real()(0, evalsMax), evecs.real()(1, evalsMax); 
        Pair_list[idl].direction_cam = direction;
        //move to corre
    }

}

void MiscalibCor()
{
    ros::Rate loop_rate(10);
    while(ros::ok())
    {
        if(Pair_list.size() > 10 && miscalib)
        {

            // pre_opti = false;
            double pose_error_init = 10;

            std::vector<PairData> Pair_list_cor;
            Pair_list_cor = Pair_list;

            
            // get direction
            if(use_deriction)
            {
                get_direction(Pair_list_cor);
            }

            Sophus::SE3d gt_pose_temp = gt_pose_cor;
            Eigen::Quaterniond q(gt_pose_temp.rotationMatrix());
            Eigen::Vector3d t = gt_pose_temp.translation();
            double ext[7];
            ext[0] = q.x();
            ext[1] = q.y();
            ext[2] = q.z();
            ext[3] = q.w();
            ext[4] = t[0];
            ext[5] = t[1];
            ext[6] = t[2];
            Eigen::Map<Eigen::Quaterniond> m_q = Eigen::Map<Eigen::Quaterniond>(ext);
            Eigen::Map<Eigen::Vector3d> m_t = Eigen::Map<Eigen::Vector3d>(ext + 4);
            ceres::LocalParameterization *q_parameterization = new ceres::EigenQuaternionParameterization();
            ceres::Problem problem;

            problem.AddParameterBlock(ext, 4, q_parameterization);
            problem.AddParameterBlock(ext + 4, 3);
            for (auto val : Pair_list_cor) 
            {
                ceres::CostFunction *cost_function;
                cost_function = edge_calib::Create(val);
                problem.AddResidualBlock(cost_function, NULL, ext, ext + 4);
            }
            ceres::Solver::Options options;
            options.preconditioner_type = ceres::JACOBI;
            options.linear_solver_type = ceres::SPARSE_SCHUR;
            options.minimizer_progress_to_stdout = true;
            options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
            ceres::Solver::Summary summary;
            ceres::Solve(options, &problem, &summary);
            std::cout << summary.BriefReport() << std::endl;
            gt_pose_temp.setRotationMatrix(m_q.toRotationMatrix());
            gt_pose_temp.translation() = m_t;
            
            pose_error_init = (gt_pose_.inverse() * gt_pose_temp).log().norm();
            std::cout << "pose error: "<< pose_error_init << std::endl;  
            if(pose_error_init < 0.1)
            {
                miscalib = false;
                Sophus::SE3d pose_error_final = gt_pose_.inverse() * gt_pose_temp;
                Eigen::Matrix4d T = pose_error_final.matrix();
                Eigen::Matrix3d R = T.block<3,3>(0,0);
                Eigen::Vector3d euler = R.eulerAngles(2, 1, 0);
                Eigen::Vector3d t = T.block<3,1>(0,3);
                std::cout << "eR: "<< euler[2] << "; eP: "<< euler[1] << "; eY: "<< euler[0] << "; ex: "<< t[0] << "; ey: "<< t[1] << "; ez: "<< t[2] << std::endl;
                std::cout << "gt_pose_cor : "<< gt_pose_temp.matrix().cast<float>() << std::endl;
                Pair_list.clear();
            }
               
        }
        loop_rate.sleep();
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "calib_online_node");
    ros::NodeHandle nh;
    nh.param<std::string>("common/PointCloudTopic",PointCloudTopic,"");
    nh.param<std::string>("common/ImageTopic",ImageTopic,"");
    nh.param<std::string>("common/RawImageTopic",RawImageTopic,"");
    nh.param<std::string>("dataset",dataset,"");
    nh.param<std::vector<double>>("Intrinsic",Intrinsic,{});
    nh.param<std::vector<double>>("Extrinsic",Extrinsic,{});
    nh.param<std::vector<double>>("DistCoeff",Dist_coeffs,{});
    nh.param<float>("inc/pixel_inc",pixel_inc,0);
    nh.param<float>("inc/range_inc",range_inc,0);
    nh.param<float>("inc/degree_inc",degree_inc,0);
    nh.param<bool>("high_noise",high_noise,false);
    nh.param<int>("width",width,0);
    nh.param<int>("height",height,0);
    outfile.open("/media/ubun/DATA/Projects/calibration/Code/CalibOnlineV3/src/calib_online/src/time.txt");
    fx = Intrinsic[0];
    fy = Intrinsic[4];
    cx = Intrinsic[2];
    cy = Intrinsic[5];
    inner << fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0;
    distor << Dist_coeffs[0], Dist_coeffs[1], Dist_coeffs[2], Dist_coeffs[3];
    Eigen::Matrix<double,3,4> T_velo_cam2;
    T_velo_cam2 << Extrinsic[0],Extrinsic[1],Extrinsic[2],Extrinsic[3],
                        Extrinsic[4],Extrinsic[5],Extrinsic[6],Extrinsic[7],
                        Extrinsic[8],Extrinsic[9],Extrinsic[10],Extrinsic[11];
    Eigen::Quaterniond q(T_velo_cam2.block<3,3>(0,0));
    Eigen::Vector3d t = T_velo_cam2.block<3,1>(0,3);
    Sophus::SE3d gt_pose(q, t);
    gt_pose_ = gt_pose;
    gt_pose_cor = gt_pose;

    //***************test******************//
    double bR, bP, bY, bx, by, bz;
    get_random_RPYxyz(bR, bP, bY, bx, by, bz);
    Eigen::Matrix3d rotation;
    rotation = Eigen::AngleAxisd(bY, Eigen::Vector3d::UnitZ()) *
                Eigen::AngleAxisd(bP, Eigen::Vector3d::UnitY()) *
                Eigen::AngleAxisd(bR, Eigen::Vector3d::UnitX());
    Eigen::Vector3d transation_bais(bx, by, bz);
    Sophus::SO3d so3_bais(rotation);
    gt_pose_cor.so3() = gt_pose.so3() * so3_bais;
    gt_pose_cor.translation() = gt_pose.translation() + transation_bais;
    std::cout << "gt_pose_cor_init" << gt_pose_cor.matrix().cast<float>() << std::endl; 
    double pose_error_init = (gt_pose_.inverse() * gt_pose_cor).log().norm();
    std::cout << "pose error: "<< pose_error_init << std::endl;
    //***************test******************//
    std::thread thread_det(MiscalibDet);
    if(thread_det.joinable())
    {
        thread_det.detach();
    }
    std::thread thread_cor(MiscalibCor);
    if(thread_cor.joinable())
    {
        thread_cor.detach();
    }

    //订阅点云和图像
    ros::Subscriber sub_pointcloud = nh.subscribe(PointCloudTopic, 10, lidar_callback);
    ros::Subscriber sub_image = nh.subscribe(ImageTopic, 10, image_callback);
    ros::Subscriber sub_rawimage = nh.subscribe(RawImageTopic, 10, rawimage_callback);
    ros::AsyncSpinner spinner(3);           //非阻塞式的spinner, 可以使用start和stop进行启停
    spinner.start();                        //启动线程
    ros::waitForShutdown();  
    return 0;
}