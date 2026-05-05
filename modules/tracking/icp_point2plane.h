#pragma once
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include "sophus/se3.hpp"

struct Intrinsics
{
    float fx = 0, fy = 0, cx = 0, cy = 0;
};

struct IcpOptions
{
    int pyr_levels = 3;                   // 金字塔层数（0..L-1）
    int iters_per_level = 8;              // 每层迭代
    float depth_diff_thresh = 0.02f;      // m，相对你的“SLAM尺子”
    float normal_angle_thresh_deg = 60.f; // 法线夹角阈值
    bool use_huber = true;
    float huber_delta = 0.01f; // m
    int sample_stride = 1;     // 采样步长（>1更快）
};

struct IcpStats
{
    int num_corr = 0;
    float rmse = 0.f;
    int iters_done = 0;
};

struct IcpResult
{
    Sophus::SE3f T_ref_from_live; // 把 live 点变到 ref 相机坐标系
    IcpStats stats;
    bool ok = true;
    const char *fail_reason = "";
};

// 帧-帧点到平面 ICP：用 ref（参考/上一帧）做“模型”，live（当前帧）对齐到 ref。
class PointToPlaneICP
{
public:
    PointToPlaneICP(const Intrinsics &K, const IcpOptions &opt);

    // Z_ref/Z_live：正常深度（CV_32FC1），mask 可空（CV_8U）
    IcpResult Align(const cv::Mat &Z_ref, const cv::Mat &Z_live,
                    const cv::Mat &mask_ref, const cv::Mat &mask_live,
                    const Sophus::SE3f &T_init);

private:
    Intrinsics K_;
    IcpOptions opt_;

    struct Level
    {
        cv::Mat Z_ref, Z_live;
        cv::Mat V_ref, N_ref; // 顶点图/法线图（3通道32F，世界或相机坐标）
        cv::Mat V_live;       // 仅 live 顶点
        Intrinsics K;
    };
    std::vector<Level> lv_;

    static cv::Mat buildVertexMap(const cv::Mat &Z, const Intrinsics &K);
    static cv::Mat buildNormalMap(const cv::Mat &V); // PCA/差分法线（3通道32F, 单位向量或(0,0,0)无效）

    static inline Eigen::Vector3f se3Skew(const Eigen::Vector3f &v)
    {
        return Eigen::Vector3f(v.z(), v.x(), v.y()); // 占位，实际用下面的矩阵块
    }

    // 单层 GN 迭代一次：组装 H,b，返回 inlier 及 RMSE
    void gnIterationOne(const Level &L, const Sophus::SE3f &T_ref_from_live,
                        Eigen::Matrix<float, 6, 6> &H, Eigen::Matrix<float, 6, 1> &b,
                        int &inliers, double &sqerr_sum) const;

    // 下采样深度（最近邻/均值均可），同步缩放内参
    static void buildPyramid(const cv::Mat &Z_ref, const cv::Mat &Z_live,
                             const Intrinsics &K, int L,
                             std::vector<Level> &out, const cv::Mat &M_ref, const cv::Mat &M_live);
};
