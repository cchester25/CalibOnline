#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <Eigen/Dense>
#include <random>
void calcCovarance( pcl::PointXYZ &lidar_position,
                    const float range_inc,
                    const float degree_inc)
{
    // the covarance of lidar position.
    //// get range 
    float range = sqrt(lidar_position.x * lidar_position.x + 
                        lidar_position.y * lidar_position.y +
                        lidar_position.z * lidar_position.z);
    //// get direction and the Antisymmetric matrix
    Eigen::Vector3f direction(lidar_position.x, lidar_position.y, lidar_position.z);
    direction.normalize();
    Eigen::Matrix3f direction_hat;
    direction_hat << 0, -direction(2), direction(1), direction(2), 0,
        -direction(0), -direction(1), direction(0), 0;
    float range_var = range_inc * range_inc;
    Eigen::Matrix2f direction_var;
    direction_var << pow(sin(DEG2RAD(degree_inc)), 2), 0, 0,
        pow(sin(DEG2RAD(degree_inc)), 2);
    Eigen::Vector3f base_vector1(1, 1,
                                -(direction(0) + direction(1)) / direction(2));
    base_vector1.normalize();
    Eigen::Vector3f base_vector2 = base_vector1.cross(direction);
    base_vector2.normalize();
    Eigen::Matrix<float, 3, 2> N;
    N << base_vector1(0), base_vector2(0), base_vector1(1), base_vector2(1),
        base_vector1(2), base_vector2(2);
    Eigen::Matrix<float, 3, 2> A = range * direction_hat * N;
    Eigen::Matrix3f lidar_position_var =
        direction * range_var * direction.transpose() +
        A * direction_var * A.transpose();
    //get the shift
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> normal(0, 1);
    Eigen::Matrix<float,3,1> random_normal(normal(gen), normal(gen), normal(gen));
    Eigen::Matrix3f cholesky_factor = lidar_position_var.llt().matrixL();
    Eigen::Matrix<float,3,1> shift = cholesky_factor * random_normal;
    //shift = random();
    lidar_position.x += shift(0,0);
    lidar_position.y += shift(1,0);
    lidar_position.z += shift(2,0);
}