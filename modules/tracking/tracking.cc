/*
 * This file is part of NR-SLAM
 *
 * Copyright (C) 2022-2023 Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
 *
 * NR-SLAM is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "tracking.h"
#include "icp_point2plane.h"

#include "features/shi_tomasi.h"
#include "optimization/g2o_optimization.h"
#include "utilities/dbscan.h"
#include "utilities/geometry_toolbox.h"
#include "utilities/statistics_toolbox.h"

#include "absl/log/log.h"
#include "absl/log/check.h"


using namespace std;

namespace {
    // 计算中位数
    static inline double median(std::vector<double> v) {
        if (v.empty()) return 0.0;
        size_t n = v.size()/2;
        std::nth_element(v.begin(), v.begin()+n, v.end());
        double m = v[n];
        if (v.size()%2==0) {
            std::nth_element(v.begin(), v.begin()+n-1, v.end());
            m = 0.5*(m + v[n-1]);
        }
        return m;
    }

    // Huber 权重（c≈1.345 对应高斯95%效率；可取 1.5~2.5）
    static inline double huberWeight(double r_over_sigma, double c=1.5) {
        double a = std::abs(r_over_sigma);
        if (a <= c) return 1.0;
        return c / a; // w = psi(r)/r
    }

    // 带权最小二乘的闭式解（用于 IRLS 内部，每次给定权重 w_i 重新估计 a,b）
    static inline bool weightedLS(const std::vector<float>& x, const std::vector<float>& y,
                                const std::vector<double>& w, float& a, float& b) {
        const int N = (int)x.size();
        if (N < 2) return false;
        double S=0,Sx=0,Sy=0,Sxx=0,Sxy=0;
        for (int i=0;i<N;++i) {
            double wi = w.empty()?1.0:w[i];
            double xi = x[i], yi = y[i];
            S   += wi;
            Sx  += wi*xi;
            Sy  += wi*yi;
            Sxx += wi*xi*xi;
            Sxy += wi*xi*yi;
        }
        double D = S*Sxx - Sx*Sx;
        if (std::abs(D) < 1e-12) return false;
        a = (float)((S*Sxy - Sx*Sy)/D);
        b = (float)((Sy - a*Sx)/S);
        return true;
    }

    // RANSAC + IRLS 拟合 a,b：y ≈ a*x + b
    struct ScaleShiftResult {
        float a=1.f, b=0.f;
        std::vector<int> inliers;
        bool ok=false;
    };

    static ScaleShiftResult robustFitScaleShiftRansacIrls(
            const std::vector<float>& x, const std::vector<float>& y,
            double p_success=0.99, int max_iters=1000, double k_inlier=2.5,
            int min_inliers_abs=20, double min_inlier_ratio=0.2, int irls_iters=5)
    {
        ScaleShiftResult res;
        const int N = (int)x.size();
        if (N < 2) return res;

        std::mt19937 rng(42);
        std::uniform_int_distribution<int> unif(0, N-1);

        int best_inliers = -1;
        float best_a=1.f, best_b=0.f;
        std::vector<int> best_set;

        // 初始迭代次数（可在循环内自适应更新）
        int iters = max_iters;
        double est_inlier_ratio = std::max(min_inlier_ratio, 0.2); // 初始估计
        int s = 2; // 每次采样 2 个点拟合线
        if (est_inlier_ratio > 0 && est_inlier_ratio < 1) {
            double logp = std::log(1.0 - p_success);
            double log1m = std::log(1.0 - std::pow(est_inlier_ratio, s));
            iters = (log1m<0) ? std::min(max_iters, (int)std::ceil(logp/log1m)) : max_iters;
        }

        std::vector<double> residuals(N);
        for (int it=0; it<iters; ++it) {
            // 随机采两个点（避免退化）
            int i=unif(rng), j=unif(rng);
            if (i==j) { --it; continue; }
            if (std::abs(x[i]-x[j]) < 1e-12) { --it; continue; }

            // 用两点解直线
            float a = (y[j] - y[i]) / (float)(x[j] - x[i]);
            float b = y[i] - a * x[i];

            // 计算全体残差
            for (int n=0;n<N;++n) residuals[n] = (double)a * x[n] + b - y[n];

            // 用 MAD 估计 sigma，并设阈值
            std::vector<double> absr(N);
            for (int n=0;n<N;++n) absr[n] = std::abs(residuals[n]);
            double mad = median(absr);                 // median(|r|)
            double sigma = 1.4826 * std::max(mad, 1e-9); // robust sigma
            double tau = k_inlier * sigma;

            // 选 inliers
            std::vector<int> inliers;
            inliers.reserve(N);
            for (int n=0;n<N;++n) if (std::abs(residuals[n]) <= tau) inliers.push_back(n);

            if ((int)inliers.size() > best_inliers) {
                // 在内点上最小二乘精化一次（无权）
                std::vector<float> xi, yi; xi.reserve(inliers.size()); yi.reserve(inliers.size());
                for (int id: inliers){ xi.push_back(x[id]); yi.push_back(y[id]); }
                float aa=a, bb=b;
                weightedLS(xi, yi, {}, aa, bb);

                best_inliers = (int)inliers.size();
                best_a = aa; best_b = bb;
                best_set = std::move(inliers);

                // 自适应更新 RANSAC 次数
                est_inlier_ratio = std::max(est_inlier_ratio, best_inliers/(double)N);
                if (est_inlier_ratio>0 && est_inlier_ratio<1) {
                    double logp = std::log(1.0 - p_success);
                    double log1m = std::log(1.0 - std::pow(est_inlier_ratio, s));
                    int new_iters = (log1m<0) ? (int)std::ceil(logp/log1m) : max_iters;
                    iters = std::min(iters, std::min(max_iters, new_iters));
                }
            }
        }

        if (best_inliers < std::max(min_inliers_abs, (int)(min_inlier_ratio*N))) {
            return res; // 失败
        }

        // —— IRLS(Huber) 在最佳内点集上做鲁棒精化 ——
        float a = best_a, b = best_b;
        std::vector<double> w(best_set.size(), 1.0);

        for (int it=0; it<irls_iters; ++it) {
            // 计算残差 & robust sigma
            std::vector<double> absr; absr.reserve(best_set.size());
            for (size_t k=0;k<best_set.size();++k) {
                int n = best_set[k];
                double r = (double)a * x[n] + b - y[n];
                absr.push_back(std::abs(r));
            }
            double sigma = 1.4826 * std::max(median(absr), 1e-9);

            // 更新 Huber 权
            for (size_t k=0;k<best_set.size();++k) {
                int n = best_set[k];
                double r = ((double)a * x[n] + b - y[n]) / sigma;
                w[k] = huberWeight(r, 1.5);
            }

            // 带权最小二乘更新 a,b
            std::vector<float> xi, yi; xi.reserve(best_set.size()); yi.reserve(best_set.size());
            for (int n: best_set){ xi.push_back(x[n]); yi.push_back(y[n]); }
            float new_a=a, new_b=b;
            if (!weightedLS(xi, yi, w, new_a, new_b)) break;

            // 收敛判断
            if (std::abs(new_a-a) + std::abs(new_b-b) < 1e-8) { a=new_a; b=new_b; break; }
            a=new_a; b=new_b;
        }

        res.a = a; res.b = b; res.inliers = std::move(best_set); res.ok = true;
        return res;
    }

} // 匿名命名空间

//! Edited 09.09
static inline bool sampleBilinear32F(const cv::Mat& img, float u, float v, float& out) {
    CV_Assert(img.type() == CV_32FC1);
    if (u < 0.f || v < 0.f || u > img.cols - 1.f || v > img.rows - 1.f) return false;
    const int x0 = (int)std::floor(u), y0 = (int)std::floor(v);
    const int x1 = std::min(x0 + 1, img.cols - 1);
    const int y1 = std::min(y0 + 1, img.rows - 1);
    const float ax = u - x0, ay = v - y0;

    const float I00 = img.at<float>(y0, x0);
    const float I10 = img.at<float>(y0, x1);
    const float I01 = img.at<float>(y1, x0);
    const float I11 = img.at<float>(y1, x1);
    if (!std::isfinite(I00) || !std::isfinite(I10) || !std::isfinite(I01) || !std::isfinite(I11)) return false;

    out = (1 - ax) * (1 - ay) * I00 + ax * (1 - ay) * I10 + (1 - ax) * ay * I01 + ax * ay * I11;
    return true;
}

Tracking::Tracking(const Tracking::Options options, std::shared_ptr<Map> map,
                   std::shared_ptr<CameraModel> calibration,
                   std::shared_ptr<StereoLucasKanade> stereo_matcher,
                   std::shared_ptr<ImageVisualizer> image_visualizer,
                   TimeProfiler* time_profiler) :
options_(options), map_(map), calibration_(calibration), stereo_matcher_(stereo_matcher),
image_visualizer_(image_visualizer), tracking_status_(NOT_INITIALIZED), time_profiler_(time_profiler)
{
    ShiTomasi::Options shi_tomasi_options;
    shi_tomasi_options.non_max_suprresion_window_size = 7;
    feature_extractor_ = make_shared<ShiTomasi>(shi_tomasi_options);

    klt_tracker_ = LucasKanadeTracker(cv::Size(options_.klt_window_size, options_.klt_window_size),
                                      options_.klt_max_level, options_.klt_max_iters,
                                      options_.klt_epsilon, options_.klt_min_eig_th);

    current_frame_ = make_shared<Frame>();
    current_frame_->SetCalibration(calibration);

    MonocularMapInitializer::Options monocular_map_initializer_options;
    monocular_map_initializer_options.klt_window_size = 21;
    monocular_map_initializer_options.klt_max_level = 4;
    monocular_map_initializer_options.klt_max_iters = 10;
    monocular_map_initializer_options.klt_epsilon = 0.0001;
    monocular_map_initializer_options.klt_min_eig_th = 0.0001;
    monocular_map_initializer_options.klt_min_SSIM = 0.5;

    monocular_map_initializer_options.rigid_initializer_max_features = 4000;
    monocular_map_initializer_options.rigid_initializer_min_sample_set_size = 8;
    monocular_map_initializer_options.rigid_initializer_min_parallax = 0.999;
    monocular_map_initializer_options.rigid_initializer_radians_per_pixel = options_.radians_per_pixel;
    monocular_map_initializer_options.rigid_initializer_epipolar_threshold = 0.005;

    monocular_map_initializer_ = make_unique<MonocularMapInitializer>(
            monocular_map_initializer_options, feature_extractor_, calibration_, image_visualizer_);

    SurfConfig cfg;
    cfg.window_size = options_.surfel_window_size;
    cfg.min_depth = options_.surfel_min_depth;
    cfg.max_depth = options_.surfel_max_depth;
    cfg.default_surfel_radius = options_.surfel_default_radius;
    cfg.use_adaptive_radius = options_.surfel_use_adaptive_radius;
    cfg.surfel_sample_step = options_.surfel_sample_step;
    cfg.max_new_surfels_per_frame = options_.surfel_max_new_per_frame;
    cfg.fusion_distance_threshold = options_.surfel_fusion_distance_threshold;
    cfg.fusion_normal_threshold = options_.surfel_fusion_normal_threshold;
    cfg.spatial_search_radius = options_.surfel_spatial_search_radius;
    cfg.active_min_depth = options_.surfel_active_min_depth;
    cfg.active_max_depth = options_.surfel_active_max_depth;
    cfg.confidence_decay_rate = options_.surfel_confidence_decay_rate;
    cfg.min_surfel_confidence = options_.surfel_min_confidence;
    cfg.warp_sigma = options_.surfel_warp_sigma;
    cfg.warp_max_disp = options_.surfel_warp_max_disp;
    cfg.quality_mode = options_.surfel_quality_mode;
    cfg.min_surfel_radius = options_.surfel_min_radius;
    cfg.max_surfel_radius = options_.surfel_max_radius;
    cfg.surfel_radius_scale = options_.surfel_radius_scale;
    cfg.max_depth_gradient = options_.surfel_max_depth_gradient;
    cfg.mask_border_pixels = options_.surfel_mask_border_pixels;
    cfg.max_fused_surfels_per_frame = options_.surfel_max_fused_per_frame;
    cfg.max_added_surfels_per_frame = options_.surfel_max_added_per_frame;
    cfg.color_update_max_residual = options_.surfel_color_update_max_residual;
    cfg.color_update_soft_residual = options_.surfel_color_update_soft_residual;
    cfg.sdf_constraint_enabled   = options_.surfel_sdf_constraint_enabled;
    cfg.sdf_phi_abs_threshold    = options_.surfel_sdf_phi_abs_threshold;
    cfg.sdf_normal_cos_threshold = options_.surfel_sdf_normal_cos_threshold;
    cfg.sdf_project_to_surface   = options_.surfel_sdf_project_to_surface;

    surf_map_ = std::make_unique<SurfMap>(cfg);

    SDFBuilder::Options sdf_options;
    sdf_options.voxel_size = options_.sdf_voxel_size;
    sdf_options.truncation_distance = options_.sdf_truncation_distance;
    sdf_options.dense_pixel_stride = options_.sdf_dense_pixel_stride;
    sdf_options.dense_weight = options_.sdf_dense_weight;
    sdf_options.sparse_weight = options_.sdf_sparse_weight;
    sdf_options.max_voxel_updates_per_frame = options_.sdf_max_voxel_updates_per_frame;
    sdf_options.min_depth = options_.sdf_min_depth;
    sdf_options.max_depth = options_.sdf_max_depth;
    sdf_options.sparse_min_points = options_.sdf_sparse_min_points;
    sdf_builder_ = std::make_unique<SDFBuilder>(sdf_options);
    surf_map_->SetSDFBuilder(sdf_builder_.get());

    // 内参矩阵延迟初始化
    K_initialized_ = false;
}

//! 第一版：放弃使用DDG，直接用全部观测到的特征点做EMDQ 
void Tracking::TrackImageNN(const cv::Mat &im, const cv::Mat& im_color, const absl::flat_hash_map<std::string, cv::Mat>& masks,
                            const cv::Mat &im_nn, const cv::Mat& im_clahe) {
    
    map_->SetAllMappointsToNonActive();

    if (map_->IsEmpty()) {
        MonocularMapInitializationNN(im, im_nn, masks.at("Global"), im_clahe);
    } else {

        UpdateTriangulatedPoints();

        absl::flat_hash_set<ID> lost_mappoint_ids = TrackCameraAndDeformationNN(im, im_nn, masks.at("Global"));

        // Point reuse.
        PointReuse(im, cv::Mat(), lost_mappoint_ids);

        if (current_frame_->GetKeypointsWithStatus({TRACKED_WITH_3D}).size() < 10) {
            LOG(INFO) << "Not enough points to track." << endl;
            exit(0);
        }

        // KeyFrame insertion using surf
        KeyFrameInsertion(im, im_color, masks);

        const uint64_t current_frame_id = static_cast<uint64_t>(current_frame_->GetId());
        if (surf_map_) {
            surf_map_->BeginFrame(current_frame_id);
        }

        // === P0: 将本帧估计的稀疏形变场传播到 SurfMap active surfel ===
        // 必须在 SDF/SurfMap 集成之前执行，使得新候选能与已形变的 surfel 正确融合。
        // 控制点：路标形变前的世界坐标；形变向量：g2o 输出的 d_i。
        if (surf_map_) {
            const auto& deformations  = current_frame_->Deformations();
            const auto  tracked_idxs  = current_frame_->GetIndexWithStatus({TRACKED_WITH_3D});
            const auto& idx_to_mp     = current_frame_->IndexToMapPointId();

            std::vector<Eigen::Vector3f> ctrl_pos, ctrl_deform;
            ctrl_pos.reserve(tracked_idxs.size());
            ctrl_deform.reserve(tracked_idxs.size());

            for (int idx : tracked_idxs) {
                if (idx >= static_cast<int>(deformations.size())) continue;
                const Eigen::Vector3f& d = deformations[idx];
                if (d.squaredNorm() < 1e-12f) continue;  // 零形变无需写入

                auto it = idx_to_mp.find(idx);
                if (it == idx_to_mp.end()) continue;
                auto mp = map_->GetMapPoint(it->second);
                if (!mp) continue;

                // GetLastWorldPosition() 是形变后坐标，减去 d 得到形变前控制点位置
                ctrl_pos.push_back(mp->GetLastWorldPosition() - d);
                ctrl_deform.push_back(d);
            }

            if (!ctrl_pos.empty()) {
                surf_map_->WarpSurfels(ctrl_pos, ctrl_deform);
            }
        }

        // === P3: NN 深度残差软校正（用对齐深度轻量修正路标 Z 深度）===
        // 策略：沿射线方向做 5% 的软拉伸，使路标深度向 NN 对齐深度靠拢。
        // 仅在深度差 < 1cm 时生效（避免遮挡/异常像素的错误拉伸）。
        // 调用时机：在 SDF/SurfMap 集成前，使 SDF 能写入更准确的稀疏点。
        if (!current_frame_->GetAlignedDepth().empty()) {
            const cv::Mat&      z_map       = current_frame_->GetAlignedDepth();
            const Sophus::SE3f  Tcw         = current_frame_->CameraTransformationWorld();
            const Sophus::SE3f  Twc         = Tcw.inverse();
            const auto          tracked_idx = current_frame_->GetIndexWithStatus({TRACKED_WITH_3D});
            const auto&         idx_to_mp   = current_frame_->IndexToMapPointId();

            constexpr float kAlpha     = 0.05f;   // 每帧拉伸比例（5%）
            constexpr float kMaxDeltaZ = 0.010f;  // 超过 1cm 差异则跳过

            int nudged = 0;
            for (int idx : tracked_idx) {
                auto it = idx_to_mp.find(idx);
                if (it == idx_to_mp.end()) continue;
                auto mp = map_->GetMapPoint(it->second);
                if (!mp) continue;

                const Eigen::Vector3f p_cam = Tcw * mp->GetLastWorldPosition();
                if (p_cam.z() <= 0.0f) continue;

                const cv::Point2f uv = calibration_->Project(p_cam);
                const int u = static_cast<int>(std::round(uv.x));
                const int v = static_cast<int>(std::round(uv.y));
                if (u < 0 || u >= z_map.cols || v < 0 || v >= z_map.rows) continue;

                const float z_nn = z_map.at<float>(v, u);
                if (!std::isfinite(z_nn) || z_nn <= 0.0f) continue;

                const float delta_z = z_nn - p_cam.z();
                if (std::abs(delta_z) > kMaxDeltaZ) continue;

                // 沿射线比例缩放：仅改变深度，投影方向不变
                const float new_z = p_cam.z() + kAlpha * delta_z;
                const float scale = new_z / p_cam.z();
                Eigen::Vector3f p_world_new = Twc * (p_cam * scale);

                mp->SetLastWorldPosition(p_world_new);
                current_frame_->LandmarkPositions()[idx] = p_world_new;
                ++nudged;
            }
            LOG(INFO) << "P3 depth nudge: " << nudged << "/" << tracked_idx.size()
                      << " landmarks nudged";
        }

        // === Dense surfel integration ===
        // SurfMap receives T_wc. SDF is optional and disabled by default for the
        // surfel-only baseline.
        if (!current_frame_->GetAlignedDepth().empty()) {
            const Sophus::SE3f    T_cw     = current_frame_->CameraTransformationWorld();
            const Eigen::Matrix4f T_wc     = T_cw.inverse().matrix();
            const Eigen::Matrix3f K        = calibration_->ToIntrinsicsMatrix();
            const uint64_t        frame_id = current_frame_id;
            const cv::Mat& aligned_depth   = current_frame_->GetAlignedDepth();
            const cv::Mat& global_mask     = masks.at("Global");

            // ── 1. 先建 SDF ──────────────────────────────────────────────────
            if (options_.surfel_sdf_constraint_enabled && sdf_builder_) {
                std::vector<Eigen::Vector3f> sparse_world_points;
                const auto tracked_idxs = current_frame_->GetIndexWithStatus({TRACKED_WITH_3D});
                const auto& idx_to_mp = current_frame_->IndexToMapPointId();
                const auto& landmark_positions = current_frame_->LandmarkPositions();
                sparse_world_points.reserve(tracked_idxs.size());

                for (int idx : tracked_idxs) {
                    if (idx < 0 || idx >= static_cast<int>(landmark_positions.size())) continue;
                    if (idx_to_mp.find(idx) == idx_to_mp.end()) continue;

                    const Eigen::Vector3f& p_world = landmark_positions[idx];
                    const Eigen::Vector3f p_cam = T_cw * p_world;
                    if (p_cam.z() < options_.sdf_min_depth ||
                        p_cam.z() > options_.sdf_max_depth) {
                        continue;
                    }

                    const cv::Point2f uv = calibration_->Project(p_cam);
                    const int u = static_cast<int>(std::round(uv.x));
                    const int v = static_cast<int>(std::round(uv.y));
                    if (u < 0 || u >= aligned_depth.cols ||
                        v < 0 || v >= aligned_depth.rows) {
                        continue;
                    }
                    if (!global_mask.empty() && global_mask.type() == CV_8UC1 &&
                        global_mask.at<uint8_t>(v, u) == 0) {
                        continue;
                    }

                    sparse_world_points.push_back(p_world);
                }

                sdf_builder_->IntegrateFrame(
                    aligned_depth, sparse_world_points, T_wc, K, global_mask, frame_id);

                const auto& sdf_stats = sdf_builder_->GetLastStats();
                LOG(INFO) << "SDF: voxels=" << sdf_stats.active_voxels
                          << ", dense_pts=" << sdf_stats.dense_points_used
                          << ", sparse_pts=" << sdf_stats.sparse_points_used
                          << ", updates=" << sdf_stats.voxel_updates
                          << ", dense_only_fallback=" << sdf_stats.dense_only_fallback;
            }

            // ── 2. 再建 SurfMap（ApplySDFConstraint 现在查到的是当帧 TSDF）──
            if (surf_map_) {
                surf_map_->IntegrateFrame(
                    im_color, aligned_depth, T_wc, K, frame_id, global_mask);
            }
        }

        LOG(INFO) << "SurfMap: active=" << surf_map_->NumActiveSurfels()
                  << ", finalized=" << surf_map_->NumFinalizedSurfels();


        // Insert frame to the temporal buffer.
        map_->SetLastFrame(current_frame_);

        // Draw current frame.
        image_visualizer_->DrawCurrentFrame(*current_frame_);
        image_visualizer_->DrawRegularizationGraph(*current_frame_, *(map_->GetRegularizationGraph()));
        image_visualizer_->DrawFeatures(current_frame_->Keypoints());
    }
}

void Tracking::TrackImage(const cv::Mat &im, const absl::flat_hash_map<std::string, cv::Mat>& masks,
                          const cv::Mat &additional_im, const cv::Mat& im_clahe) {
    map_->SetAllMappointsToNonActive();

    if (map_->IsEmpty()) {
        // If map is not initialized, perform map initialization.

        // For stereo experiment purposes.
        // StereoMapInitialization(im, additional_im, masks.at("Global"), im_clahe);

        // Depending on the type of sequence, the mask type used can be different.
        MonocularMapInitialization(im, masks.at("Global"), im_clahe);
        // MonocularMapInitialization(im, masks.at("PredefinedFilter"), im_clahe);
    } else {
        // Update points triangulated by the mapping in the last frame.
        UpdateTriangulatedPoints();

        LOG(INFO) << "Num of Keypoints 1: " << current_frame_->GetKeypointsWithStatus({TRACKED_WITH_3D}).size();
        // Otherwise perform normal tracking.
        // Depending on the type of sequence, the mask type used can be different.
        // absl::flat_hash_set<ID> lost_mappoint_ids = TrackCameraAndDeformation(im, masks.at("BorderFilter"));
        absl::flat_hash_set<ID> lost_mappoint_ids = TrackCameraAndDeformation(im, masks.at("Global"));

        LOG(INFO) << "Num of Keypoints 2: " << current_frame_->GetKeypointsWithStatus({TRACKED_WITH_3D}).size();

        // Point reuse.
        PointReuse(im, cv::Mat(), lost_mappoint_ids);

        LOG(INFO) << "Num of Keypoints 3: " << current_frame_->GetKeypointsWithStatus({TRACKED_WITH_3D}).size();

        if (current_frame_->GetKeypointsWithStatus({TRACKED_WITH_3D}).size() < 10) {
            LOG(INFO) << "Not enough points to track." << endl;
            exit(0);
        }

        // KeyFrame insertion.
        KeyFrameInsertion(im, im, masks);
        auto pose_se3 = current_frame_->CameraTransformationWorld();
        auto pose_t = pose_se3.translation();
        auto pose_q = pose_se3.so3().unit_quaternion();
        // 保存当前帧的位姿到 CSV 文件
        {
            // 定义 CSV 文件路径（可根据需要修改）
            std::string csv_filename = "camera_poses.csv";
            // 以追加模式打开文件
            std::ofstream ofs(csv_filename, std::ios::app);
            if (ofs.is_open()) {
                // 如果 current_frame_ 有时间戳或帧号，以下示例假设有 GetTimestamp() 方法
                // 若没有，可替换为其他表示帧号或固定数值
                ofs << pose_t.x() << "," << pose_t.y() << "," << pose_t.z() << ","
                    // 注意：这里假设表格格式为 (rX, rY, rZ, rW) ，而 Eigen::Quaternionf 的存储顺序为 (w, x, y, z)
                    << pose_q.x() << "," << pose_q.y() << "," << pose_q.z() << "," << pose_q.w() 
                    << "\n";
                ofs.close();
            } else {
                std::cerr << "Failed to open " << csv_filename << " for writing." << std::endl;
            }
        }

        poses.push_back(current_frame_->CameraTransformationWorld());
        LOG(INFO) << "Num Poses: " << poses.size() << endl;

        // Insert frame to the temporal buffer.
        map_->SetLastFrame(current_frame_);

        // Draw current frame.
        image_visualizer_->DrawCurrentFrame(*current_frame_);
        image_visualizer_->DrawRegularizationGraph(*current_frame_, *(map_->GetRegularizationGraph()));
        image_visualizer_->DrawFeatures(current_frame_->Keypoints());
    }
}

Tracking::TrackingStatus Tracking::GetTrackingStatus() const {
    return tracking_status_;
}

std::vector<Sophus::SE3f> Tracking::GetCameraPoses() {
    return poses;
}

bool Tracking::ExportSDF(const std::string& filename, float max_abs_tsdf) {
    if (!sdf_builder_) {
        LOG(WARNING) << "SDF builder is null, cannot export.";
        return false;
    }
    return sdf_builder_->ExportToPLY(filename, max_abs_tsdf);
}

Tracking::RuntimeMetrics Tracking::GetRuntimeMetrics() const {
    RuntimeMetrics metrics;
    metrics.frame_id = static_cast<uint64_t>(current_frame_ ? current_frame_->GetId() : 0);
    metrics.nn_align = last_nn_align_stats_;

    if (sdf_builder_) {
        metrics.sdf = sdf_builder_->GetLastStats();
    }
    if (surf_map_) {
        metrics.surfel_active = surf_map_->NumActiveSurfels();
        metrics.surfel_finalized = surf_map_->NumFinalizedSurfels();
        metrics.surfel_phi = surf_map_->GetPhiStats();
        metrics.surfel_fusion = surf_map_->GetLastFusionStats();
    }

    return metrics;
}

void Tracking::ApplyNnAlignResult(bool fit_ok, float a, float b,
                                   int num_inliers, const cv::Mat& nn,
                                   int correspondences_total) {
    float scale, shift;
    NnAlignMode mode;
    bool has_depth;
    int inliers;

    if (fit_ok) {
        last_nn_scale_ = a;
        last_nn_shift_ = b;
        has_last_nn_scale_shift_ = true;
        scale = a;
        shift = b;
        mode = NnAlignMode::SUCCESS;
        has_depth = true;
        inliers = num_inliers;
    } else if (has_last_nn_scale_shift_) {
        scale = last_nn_scale_;
        shift = last_nn_shift_;
        mode = NnAlignMode::REUSED_LAST;
        has_depth = true;
        inliers = num_inliers;
        LOG(WARNING) << "NN scale-shift fit failed; reuse previous a,b = ("
                     << last_nn_scale_ << ", " << last_nn_shift_ << ").";
    } else {
        scale = 1.0f;
        shift = 0.0f;
        mode = NnAlignMode::FAILED_EMPTY;
        has_depth = false;
        inliers = 0;
        LOG(WARNING) << "NN scale-shift fit failed and no previous a,b available.";
    }

    if (has_depth) {
        cv::Mat Z_aligned = scale * nn + shift;
        cv::Mat valid = (Z_aligned > 1e-9f) & (Z_aligned == Z_aligned);
        Z_aligned.setTo(0, ~valid);
        current_frame_->SetAlignedDepth(Z_aligned);
    } else {
        current_frame_->SetAlignedDepth(cv::Mat());
    }

    last_nn_align_stats_.frame_id = static_cast<uint64_t>(current_frame_->GetId());
    last_nn_align_stats_.mode = mode;
    last_nn_align_stats_.scale_a = scale;
    last_nn_align_stats_.shift_b = shift;
    last_nn_align_stats_.correspondences_total = correspondences_total;
    last_nn_align_stats_.inliers = inliers;
    last_nn_align_stats_.inlier_ratio = correspondences_total == 0 ? 0.0f :
            static_cast<float>(inliers) / static_cast<float>(correspondences_total);
    last_nn_align_stats_.has_aligned_depth = has_depth;

    LOG(INFO) << "NN align: frame=" << last_nn_align_stats_.frame_id
              << ", mode=" << static_cast<int>(last_nn_align_stats_.mode)
              << ", a=" << last_nn_align_stats_.scale_a
              << ", b=" << last_nn_align_stats_.shift_b
              << ", pairs=" << last_nn_align_stats_.correspondences_total
              << ", inliers=" << last_nn_align_stats_.inliers
              << ", inlier_ratio=" << last_nn_align_stats_.inlier_ratio;
}

void Tracking::ExtractFeatures(const cv::Mat& im, const cv::Mat& mask,
                               std::vector<cv::KeyPoint>& keypoints) {
    // Extract features.
    feature_extractor_->Extract(im, keypoints);

    // Mask out points.
    vector<cv::KeyPoint> masked_keypoints;
    for(size_t i = 0; i < keypoints.size(); i++){
        if(!mask.at<uchar>(keypoints[i].pt)){
            continue;
        } else{
            masked_keypoints.push_back(keypoints[i]);
        }
    }

    keypoints = masked_keypoints;
}

void Tracking::MonocularMapInitializationNN(const cv::Mat& im_left, const cv::Mat& im_nn,
                                       const cv::Mat& mask, const cv::Mat& im_clahe) {
    auto initialization_status = monocular_map_initializer_->ProcessNewImage(im_left, im_clahe, mask);

    if(!initialization_status.ok()) {
        LOG(INFO) << initialization_status.status().message();
        return;
    }

    poses.push_back(initialization_status->camera_transform_world);
    // LOG(INFO) << "Num Poses: " << poses.size() << endl;

    auto initialization_results = *initialization_status;

    // --- 1) 统计中位深度 → 工作尺度 ---
    vector<float> depths;
    for (int idx = 0; idx < initialization_results.current_keypoints.size(); idx++) {
        Eigen::Vector3f current_landmark_position = initialization_results.current_landmark_positions[idx];
        depths.push_back(current_landmark_position.z());
    }

    const int median_idx = depths.size() / 2;
    nth_element(depths.begin(), depths.begin() + median_idx, depths.end());
    const float median_depth = depths[median_idx];
    const float scale = 3.f / median_depth;
    map_->SetMapScale(scale);
    float sigma_scaled = Sigma(depths) * scale;

    Frame reference_frame;
    for (int idx = 0; idx < initialization_results.current_keypoints.size(); idx++) {
        cv::KeyPoint reference_keypoint = initialization_results.reference_keypoints[idx];
        cv::KeyPoint current_keypoint = initialization_results.current_keypoints[idx];

        Eigen::Vector3f reference_landmark_position = initialization_results.reference_landmark_positions[idx] * scale;
        Eigen::Vector3f current_landmark_position = initialization_results.current_landmark_positions[idx] * scale;

        ID mappoint_id = map_->CreateAndInsertMapPoint(reference_landmark_position,
                                                       reference_keypoint.class_id)->GetId();

        reference_frame.InsertObservation(reference_keypoint,
                                          reference_landmark_position,
                                          mappoint_id,
                                          TRACKED_WITH_3D);

        current_frame_->InsertObservation(current_keypoint,
                                          current_landmark_position,
                                          mappoint_id,
                                          TRACKED_WITH_3D);
    }
    reference_frame.SetCalibration(calibration_);
    reference_frame.MutableCameraTransformationWorld() = Sophus::SE3f();
    initialization_results.camera_transform_world.translation() = initialization_results.camera_transform_world.translation() * scale;
    current_frame_->MutableCameraTransformationWorld() = initialization_results.camera_transform_world;

    //! ===================== NEW: NN 对齐（拟合 a,b 并生成 Z） =====================
    cv::Mat nn = im_nn;
    if (nn.channels() == 3) { std::vector<cv::Mat> ch; cv::split(nn, ch); nn = ch[0]; }
    if (nn.type() != CV_32FC1) nn.convertTo(nn, CV_32FC1);

    const Sophus::SE3f Tcw_curr = current_frame_->CameraTransformationWorld();
    std::vector<int> tracked_indices = current_frame_->GetIndexWithStatus({TRACKED_WITH_3D});

    std::vector<float> x_midas; x_midas.reserve(tracked_indices.size());  // X：MiDaS 逆深度
    std::vector<float> y_depth; y_depth.reserve(tracked_indices.size()); // Y：SLAM 深度

    for (int idx : tracked_indices) {
        ID mappoint_id = current_frame_->IndexToMapPointId().at(idx);
        auto mp = map_->GetMapPoint(mappoint_id);
        if (!mp) continue;

        // 世界点 -> 当前相机坐标。CameraTransformationWorld() is T_cw.
        const Eigen::Vector3f Pw = mp->GetLastWorldPosition();
        const Eigen::Vector3f Pc_curr = Tcw_curr * Pw;
        if (Pc_curr.z() <= 0.f) continue; 

        // 投影到像素
        const cv::Point2f uv = calibration_->Project(Pc_curr);
        if (uv.x < 0 || uv.x > im_left.cols - 1 || uv.y < 0 || uv.y > im_left.rows - 1) continue;

        // 取 MiDaS 值（逆深度/视差）
        float d_nn = std::numeric_limits<float>::quiet_NaN();
        if (!sampleBilinear32F(nn, uv.x, uv.y, d_nn)) continue;
        if (!std::isfinite(d_nn) || d_nn <= 0.f) continue;

        x_midas.push_back(d_nn);
        y_depth.push_back(Pc_curr.z());     
    }

    auto r = robustFitScaleShiftRansacIrls(x_midas, y_depth,
                                       /*p_success=*/0.99,
                                       /*max_iters=*/1000,
                                       /*k_inlier=*/2.5,
                                       /*min_inliers_abs=*/20,
                                       /*min_inlier_ratio=*/0.2,
                                       /*irls_iters=*/5);

    ApplyNnAlignResult(r.ok, r.a, r.b,
                       static_cast<int>(r.inliers.size()), nn,
                       static_cast<int>(x_midas.size()));

    // Create Keyframes from the frames.
    auto first_keyframe = make_shared<KeyFrame>(reference_frame);
    auto current_keyframe = make_shared<KeyFrame>(*current_frame_);

    // Insert KeyFrame in the map.
    map_->InsertKeyFrame(first_keyframe);
    map_->InsertKeyFrame(current_keyframe);

    map_->SetLastFrame(current_frame_);

    // Initialize regularization graph.
    map_->InitializeRegularizationGraph(sigma_scaled * 3);

    // Set reference image to the KLT tracker.
    klt_tracker_.SetReferenceImage(im_left, current_frame_->Keypoints());

    // Save MapPoint photometric information
    for (const auto& [mappoint_id, idx] : current_frame_->MapPointIdToIndex()) {
        LucasKanadeTracker::PhotometricInformation photometric_information =
                klt_tracker_.GetPhotometricInformationOfPoint(idx);

        map_->GetMapPoint(mappoint_id)->SetPhotometricInformation(photometric_information);
    }

    tracking_status_ = TRACKING;
}

void Tracking::MonocularMapInitialization(const cv::Mat& im_left,
                                       const cv::Mat& mask, const cv::Mat& im_clahe) {
    auto initialization_status = monocular_map_initializer_->ProcessNewImage(im_left, im_clahe, mask);

    if(!initialization_status.ok()) {
        LOG(INFO) << initialization_status.status().message();
        return;
    }

    poses.push_back(initialization_status->camera_transform_world);
    // LOG(INFO) << "Num Poses: " << poses.size() << endl;

    auto initialization_results = *initialization_status;

    vector<float> depths;
    for (int idx = 0; idx < initialization_results.current_keypoints.size(); idx++) {
        Eigen::Vector3f current_landmark_position = initialization_results.current_landmark_positions[idx];
        depths.push_back(current_landmark_position.z());
    }

    const int median_idx = depths.size() / 2;
    nth_element(depths.begin(), depths.begin() + median_idx, depths.end());
    const float median_depth = depths[median_idx];
    const float scale = 3.f / median_depth;
    map_->SetMapScale(scale);
    float sigma_scaled = Sigma(depths) * scale;

    Frame reference_frame;
    for (int idx = 0; idx < initialization_results.current_keypoints.size(); idx++) {
        cv::KeyPoint reference_keypoint = initialization_results.reference_keypoints[idx];
        cv::KeyPoint current_keypoint = initialization_results.current_keypoints[idx];

        Eigen::Vector3f reference_landmark_position = initialization_results.reference_landmark_positions[idx] * scale;
        Eigen::Vector3f current_landmark_position = initialization_results.current_landmark_positions[idx] * scale;

        ID mappoint_id = map_->CreateAndInsertMapPoint(reference_landmark_position,
                                                       reference_keypoint.class_id)->GetId();

        reference_frame.InsertObservation(reference_keypoint,
                                          reference_landmark_position,
                                          mappoint_id,
                                          TRACKED_WITH_3D);

        current_frame_->InsertObservation(current_keypoint,
                                          current_landmark_position,
                                          mappoint_id,
                                          TRACKED_WITH_3D);
    }
    reference_frame.SetCalibration(calibration_);
    reference_frame.MutableCameraTransformationWorld() = Sophus::SE3f();
    initialization_results.camera_transform_world.translation() = initialization_results.camera_transform_world.translation() * scale;
    current_frame_->MutableCameraTransformationWorld() = initialization_results.camera_transform_world;


    // Create Keyframes from the frames.
    auto first_keyframe = make_shared<KeyFrame>(reference_frame);
    auto current_keyframe = make_shared<KeyFrame>(*current_frame_);

    // Insert KeyFrame in the map.
    map_->InsertKeyFrame(first_keyframe);
    map_->InsertKeyFrame(current_keyframe);

    map_->SetLastFrame(current_frame_);

    // Initialize regularization graph.
    map_->InitializeRegularizationGraph(sigma_scaled * 3);

    // Set reference image to the KLT tracker.
    klt_tracker_.SetReferenceImage(im_left, current_frame_->Keypoints());

    // Save MapPoint photometric information
    for (const auto& [mappoint_id, idx] : current_frame_->MapPointIdToIndex()) {
        LucasKanadeTracker::PhotometricInformation photometric_information =
                klt_tracker_.GetPhotometricInformationOfPoint(idx);

        map_->GetMapPoint(mappoint_id)->SetPhotometricInformation(photometric_information);
    }

    tracking_status_ = TRACKING;
}


void Tracking::StereoMapInitialization(const cv::Mat& im_left, const cv::Mat& im_right,
                                       const cv::Mat& mask, const cv::Mat& im_clahe) {
    current_frame_->Clear();

    vector<cv::KeyPoint> keypoints;
    ExtractFeatures(im_clahe, mask, keypoints);

    std::vector<Eigen::Vector3f> filtered_landmarks;
    std::vector<cv::KeyPoint> filtered_keypoints;

    auto stereo_matcher = StereoPatternMatching(calibration_, 3886.37);
    for (int idx = 0; idx < keypoints.size(); idx++) {
        auto landmark = stereo_matcher.computeStereo3D(keypoints[idx], im_left, im_right);
        if (landmark.ok() &&
            (*landmark).z() > 35.5f && (*landmark).z() < 70.5f) {
            filtered_landmarks.push_back(*landmark);
            filtered_keypoints.push_back(keypoints[idx]);
        }
    }

    // Apply dbscan to remove further outliers
    vector<int> labels = Dbscan3D(filtered_landmarks);

    std::vector<float> depths;

    for (int idx = 0; idx < labels.size(); idx++) {
        if (labels[idx] == 0) {
            depths.push_back(filtered_landmarks[idx].z());
        }
    }

    const int median_idx = depths.size() / 2;
    nth_element(depths.begin(), depths.begin() + median_idx, depths.end());
    float median_depth = depths[median_depth];
    const float scale = 1.f;

    current_frame_->MutableCameraTransformationWorld().translation() *= scale;

    for(int idx = 0; idx < labels.size(); idx++){
        if (labels[idx] == 0) {
            auto mappoint = map_->CreateAndInsertMapPoint(filtered_landmarks[idx] * scale,
                                                          filtered_keypoints[idx].class_id);
            current_frame_->InsertObservation(filtered_keypoints[idx],
                                             filtered_landmarks[idx],
                                             mappoint->GetId(),
                                             TRACKED_WITH_3D);
        }
    }

    map_->SetLastFrame(current_frame_);

    // Initialize regularization graph.
    map_->InitializeRegularizationGraph(10.5);
    map_->SetMapScale(1.f);

    // Set reference image to the KLT tracker.
    klt_tracker_.SetReferenceImage(im_left, current_frame_->Keypoints());

    // Save MapPoint photometric information
    for (const auto& [mappoint_id, idx] : current_frame_->MapPointIdToIndex()) {
        LucasKanadeTracker::PhotometricInformation photometric_information =
                klt_tracker_.GetPhotometricInformationOfPoint(idx);

        map_->GetMapPoint(mappoint_id)->SetPhotometricInformation(photometric_information);
    }

    // Create Keyframe from the current frame.
    auto keyframe = make_shared<KeyFrame>(*current_frame_);

    // Insert KeyFrame in the map.
    map_->InsertKeyFrame(keyframe);

    tracking_status_ = TRACKING;
}


//! Edited 09.09
absl::flat_hash_set<ID> Tracking::TrackCameraAndDeformationNN(const cv::Mat &im, const cv::Mat &im_nn, const cv::Mat& mask) {
    // Perform data association.
    DataAssociation(im, mask);

    // Coarse camera pose estimation.
    CameraPoseEstimation(); 

    //! NN 逆深度图 + NR-SLAM当前帧特征点深度估计 -> 求逆深度图的 scale + shift
    cv::Mat nn = im_nn;
    if (nn.channels() == 3) { std::vector<cv::Mat> ch; cv::split(nn, ch); nn = ch[0]; }
    if (nn.type() != CV_32FC1) nn.convertTo(nn, CV_32FC1);

    const Sophus::SE3f Tcw_curr = current_frame_->CameraTransformationWorld();
    std::vector<int> tracked_indices = current_frame_->GetIndexWithStatus({TRACKED_WITH_3D});

    std::vector<float> x_midas; x_midas.reserve(tracked_indices.size());  // X：MiDaS 逆深度
    std::vector<float> y_depth; y_depth.reserve(tracked_indices.size()); // Y：SLAM 深度

    for (int idx : tracked_indices) {
        ID mappoint_id = current_frame_->IndexToMapPointId().at(idx);
        auto mp = map_->GetMapPoint(mappoint_id);
        if (!mp) continue;

        // 世界点 -> 当前相机坐标。CameraTransformationWorld() is T_cw.
        const Eigen::Vector3f Pw = mp->GetLastWorldPosition();
        const Eigen::Vector3f Pc_curr = Tcw_curr * Pw;
        if (Pc_curr.z() <= 0.f) continue; 

        // 投影到像素
        const cv::Point2f uv = calibration_->Project(Pc_curr);
        if (uv.x < 0 || uv.x > im.cols - 1 || uv.y < 0 || uv.y > im.rows - 1) continue;

        // 取 MiDaS 值（逆深度/视差）
        float d_nn = std::numeric_limits<float>::quiet_NaN();
        if (!sampleBilinear32F(nn, uv.x, uv.y, d_nn)) continue;
        if (!std::isfinite(d_nn) || d_nn <= 0.f) continue;

        x_midas.push_back(d_nn);
        y_depth.push_back(Pc_curr.z());     
    }

    auto r = robustFitScaleShiftRansacIrls(x_midas, y_depth,
                                       /*p_success=*/0.99,
                                       /*max_iters=*/1000,
                                       /*k_inlier=*/2.5,
                                       /*min_inliers_abs=*/20,
                                       /*min_inlier_ratio=*/0.2,
                                       /*irls_iters=*/5);

    ApplyNnAlignResult(r.ok, r.a, r.b,
                       static_cast<int>(r.inliers.size()), nn,
                       static_cast<int>(x_midas.size()));

    CameraPoseEstimation(); 

    // 2. 点面ICP配准 + soft mask 权重同时估计相机位姿&环境形变
    

    // Deformation + camera pose estimation.
    return CameraPoseAndDeformationEstimationFinal(); 
}

absl::flat_hash_set<ID> Tracking::TrackCameraAndDeformation(const cv::Mat &im, const cv::Mat& mask) {
    // Perform data association.
    DataAssociation(im, mask);

    // Coarse camera pose estimation.
    CameraPoseEstimation(); 

    // bool use_deformation_field = false;
    bool use_deformation_field = true;

    //! Edited 06.03
    //todo 加入迭代控制：将相机PoseChange和DeformationFieldChange作为收敛条件
    if(use_deformation_field) {
        const int max_iters = 5;
        const float pose_epsilon = 1e-4;
        const float deformation_epsilon = 1e-3;
    
        Sophus::SE3f prev_pose = current_frame_->CameraTransformationWorld();
        std::vector<Eigen::Vector3f> prev_deformations;
    
        absl::flat_hash_set<ID> lost_ids;
        for (int iter = 0; iter < max_iters; iter ++) {
            LOG(INFO) << "Iteration: " << iter;

            EstimateDeformationField();
            
            ComputeRigidWeights();
            
            //todo 修改优化模块接口 
            lost_ids = CameraPoseAndDeformationEstimation(); 
            
            //todo 迭代收敛条件
            // Sophus::SE3f curr_pose = current_frame_->CameraTransformationWorld();
            // Sophus::SE3f delta = prev_pose.inverse() * curr_pose;
            // float pose_change = delta.translation().norm() + delta.so3().log().norm();
    
            // float deformation_change = 0.0f;
            // const auto& curr_deformations = current_frame_->Deformations();
            // const auto& tracked3d_indices = current_frame_->GetIndexWithStatus({TRACKED_WITH_3D});
            // for (int idx : tracked3d_indices) {
            //     deformation_change += (curr_deformations[idx] - prev_deformations[idx]).norm();
            // }
            // deformation_change /= static_cast<float>(tracked3d_indices.size());
    
            // LOG(INFO) << "Iter " << iter << ": pose_change = " << pose_change 
            //         << ", deformation_change = " << deformation_change;
    
    
            // // 收敛判断
            // if (pose_change < pose_epsilon && deformation_change < deformation_epsilon) {
            //     LOG(INFO) << "Converged at iteration " << iter;
            //     break;
            // }
            // prev_pose = curr_pose;
        }
    }

    // Deformation + camera pose estimation.
    return CameraPoseAndDeformationEstimationFinal(); 
}

//! 06.03 start
void Tracking::EstimateDeformationField() {
    const Frame& prev_frame = map_->GetLastFrame();
    const Sophus::SE3f Tcw_prev = prev_frame.CameraTransformationWorld();
    const Sophus::SE3f Tcw_curr = current_frame_->CameraTransformationWorld();

    Sophus::SE3f Tcoarse = Tcw_curr * Tcw_prev.inverse();  // coarse rigid motion

    current_frame_->MutableDeformations().resize(current_frame_->Keypoints().size(), Eigen::Vector3f::Zero());

    std::vector<int> tracked_indices = current_frame_->GetIndexWithStatus({TRACKED_WITH_3D});
    for (int idx : tracked_indices) {
        ID mappoint_id = current_frame_->IndexToMapPointId().at(idx);
        if (!prev_frame.MapPointIdToIndex().contains(mappoint_id))
            continue;

        // 正确获取 3D 点的世界坐标（统一来源）
        Eigen::Vector3f Pw = map_->GetMapPoint(mappoint_id)->GetLastWorldPosition();

        // 在 prev 相机坐标系下的投影
        Eigen::Vector3f Pc_prev = Tcw_prev.inverse() * Pw;

        // 使用刚性变换预测在 curr 相机坐标下的位置
        Eigen::Vector3f Pc_rigid = Tcoarse * Pc_prev;

        // 当前帧实际的相机坐标
        Eigen::Vector3f Pc_curr = Tcw_curr.inverse() * Pw;

        // 非刚性残差
        Eigen::Vector3f dP = Pc_curr - Pc_rigid;

        // 转到世界坐标系下
        Eigen::Vector3f dP_world = Tcw_curr.rotationMatrix() * dP;

        current_frame_->MutableDeformations()[idx] = dP_world;
    }
}


float ComputeMedian(std::vector<float>& data) {
    if (data.empty()) return 0.0f;
    std::sort(data.begin(), data.end());
    size_t n = data.size();
    if (n % 2 == 0)
        return (data[n / 2 - 1] + data[n / 2]) / 2.0f;
    else
        return data[n / 2];
}

float ComputeIQR(std::vector<float>& data) {
    if (data.empty()) return 0.0f;
    std::sort(data.begin(), data.end());
    size_t n = data.size();
    float q1 = data[n / 4];
    float q3 = data[(3 * n) / 4];
    return q3 - q1;
}

void Tracking::ComputeRigidWeights() {
    std::vector<int> tracked3d_indices = current_frame_->GetIndexWithStatus({TRACKED_WITH_3D});

    std::vector<float> deformation_norms;
    deformation_norms.reserve(tracked3d_indices.size());
    for (int idx : tracked3d_indices) {
        deformation_norms.push_back(current_frame_->Deformations()[idx].norm());
    }

    float median = ComputeMedian(deformation_norms);
    float iqr = ComputeIQR(deformation_norms);
    float T = median + 1.5f * iqr;

    current_frame_->MutableRigidWeights().resize(current_frame_->Keypoints().size(), 1.0f);

    for (int i = 0; i < tracked3d_indices.size(); ++i) {
        int idx = tracked3d_indices[i];
        float d_norm = deformation_norms[i];

        float w = 1.0f;
        if (d_norm > 0)
            // w = std::max(0.f, 1.0f - d_norm / T); 
            w = std::exp(-d_norm * d_norm / (2 * T * T));

        w = std::min(0.95f, std::max(0.05f, w));
        current_frame_->MutableRigidWeights()[idx] = w;
    }

    LOG(INFO) << "Deformation Field Computed. Median: " << median << ", IQR: " << iqr << ", T: " << T;
}

void Tracking::DataAssociation(const cv::Mat &im, const cv::Mat &mask) {
    klt_tracker_.Track(im, current_frame_->Keypoints(), current_frame_->LandmarkStatuses(),
                       true, options_.klt_min_SSIM, mask);
}

void Tracking::CameraPoseEstimation() {
    // Apply motion model to get a first seed of the current camera pose.
    current_frame_->MutableCameraTransformationWorld() = motion_model_ *
            current_frame_->CameraTransformationWorld();

    previous_camera_transform_world_ = current_frame_->CameraTransformationWorld();

    // Do optimization.
    CameraPoseOptimization(*current_frame_, previous_camera_transform_world_);
}

absl::flat_hash_set<ID> Tracking::CameraPoseAndDeformationEstimation() {
    // Do optimization.
    auto lost_mappoint_ids = OptimizeReprojectionOnly(*current_frame_,
                                         map_,previous_camera_transform_world_,
                                         map_->GetMapScale());

    return lost_mappoint_ids;
}

absl::flat_hash_set<ID> Tracking::CameraPoseAndDeformationEstimationFinal() {
    // Do optimization.
    auto lost_mappoint_ids = CameraPoseAndDeformationOptimization(*current_frame_,
                                         map_,previous_camera_transform_world_,
                                         map_->GetMapScale());
    // Update motion model.
    motion_model_ = current_frame_->CameraTransformationWorld() *
                    map_->GetLastFrame().CameraTransformationWorld().inverse();

    return lost_mappoint_ids;
}

void Tracking::KeyFrameInsertion(const cv::Mat& im, const cv::Mat& im_color,
                                 const absl::flat_hash_map<std::string, cv::Mat>& masks) {
    if (NeedNewKeyFrame()) {
        CreateNewKeyFrame(im, im_color, masks);
    }
}

bool Tracking::NeedNewKeyFrame() {
    if(n_images_from_last_keyframe_ >= options_.images_to_insert_keyframe){
        n_images_from_last_keyframe_ = 0;
        return true;
    }
    else{
        n_images_from_last_keyframe_++;
        return false;
    }
}

void Tracking::CreateNewKeyFrame(const cv::Mat& im, const cv::Mat& im_color,
                                 const absl::flat_hash_map<std::string, cv::Mat>& masks) {
    // Extract new features.
    ExtractFeaturesInFrame(im, masks.at("Global"), *current_frame_);

    // Create Keyframe from the current frame.
    auto keyframe = make_shared<KeyFrame>(*current_frame_);

    // Insert KeyFrame in the map.
    map_->InsertKeyFrame(keyframe);

    // Update current frame.
    current_frame_->SetFromKeyFrame(keyframe);

    // Set new klt reference.
    // Depending on the type of sequence, the mask type used can be different.
    // SetKLTReference(im, *current_frame_, masks.at("BorderFilter"));
    SetKLTReference(im, *current_frame_, masks.at("Global"));
    // SetKLTReference(im, *current_frame_, cv::Mat());
    // SetKLTReference(im, *current_frame_, masks.at("PredefinedFilter"));    
}

void Tracking::ExtractFeaturesInFrame(const cv::Mat& im, const cv::Mat& mask, Frame &frame) {
    vector<cv::KeyPoint> tracked_keypoints = frame.GetKeypointsWithStatus({TRACKED_WITH_3D, TRACKED});

    ExtractFeatures(im, mask, tracked_keypoints);

    for (int idx = 0; idx < tracked_keypoints.size(); idx++) {
        frame.InsertObservation(tracked_keypoints[idx], Eigen::Vector3f::Zero(), 0, TRACKED);
    }
}

void Tracking::SetKLTReference(const cv::Mat& im, Frame& frame, const cv::Mat& mask) {
    klt_tracker_.SetReferenceImage(im, frame.Keypoints(), mask);

    // Update MapPoints photometric information.
    for (const auto &[mappoint_id, idx] : frame.MapPointIdToIndex()) {
        LucasKanadeTracker::PhotometricInformation photometric_information =
                klt_tracker_.GetPhotometricInformationOfPoint(idx);

        map_->GetMapPoint(mappoint_id)->SetPhotometricInformation(photometric_information);
    }
}

void Tracking::PointReuse(const cv::Mat& im, const cv::Mat& mask,
                          absl::flat_hash_set<ID> lost_mappoint_ids) {
    auto all_mappoints = map_->GetMapPoints();
    for (const auto &[mappoint_id, mappoint] : all_mappoints) {
        if (!current_frame_->LandmarkPosition(mappoint_id).ok()) {
            // Project mappoint into the camera and check if it lies inside the image.
            Eigen::Vector3f landmark_position_seed = mappoint->GetLastWorldPosition();
            Eigen::Vector3f landmark_camera_position = current_frame_->CameraTransformationWorld() * landmark_position_seed;

            if (landmark_camera_position.z() < 0) {
                continue;
            }

            cv::Point2f projected_landmark = calibration_->Project(landmark_camera_position);

            if (projected_landmark.x >= 0 && projected_landmark.x < im.cols &&
                projected_landmark.y >= 0 && projected_landmark.y < im.rows) {
                lost_mappoint_ids.insert(mappoint_id);
            }
        }
    }

    if (lost_mappoint_ids.empty()) {
        return;
    }

    // Project candidates into the image.
    Frame frame_with_only_candidates;

    LucasKanadeTracker klt(cv::Size(options_.klt_window_size, options_.klt_window_size),
                           1, options_.klt_max_iters,
                           options_.klt_epsilon, options_.klt_min_eig_th);

    int candidates_in_image = 0;
    vector<cv::KeyPoint> keypoint_seeds;
    for (const auto& mappoint_id : lost_mappoint_ids) {
        auto mappoint = map_->GetMapPoint(mappoint_id);
        Eigen::Vector3f landmark_position_seed = mappoint->GetLastWorldPosition();
        Eigen::Vector3f landmark_camera_position = current_frame_->CameraTransformationWorld() * landmark_position_seed;
        cv::Point2f projected_landmark = calibration_->Project(landmark_camera_position);

        if (isnan(projected_landmark.x) || isnan(projected_landmark.y)) {
            LOG(FATAL) << "NaN found!";
        }

        if (projected_landmark.x >= 0 && projected_landmark.x < im.cols &&
            projected_landmark.y >= 0 && projected_landmark.y < im.rows) {
            cv::KeyPoint keypoint(projected_landmark, 1);
            frame_with_only_candidates.InsertObservation(keypoint, landmark_position_seed, mappoint_id, TRACKED_WITH_3D);

            // Set photometric information in the KLT.
            LucasKanadeTracker::PhotometricInformation photometric_information =
                    map_->GetMapPoint(mappoint_id)->GetPhotometricInformation();
            klt.InsertPhotometricInformation(keypoint, photometric_information);

            keypoint_seeds.push_back(keypoint);

            candidates_in_image++;
        }
    }

    if (candidates_in_image == 0) {
        return;
    }

    // Track candidates with KLT
    klt.Track(im, frame_with_only_candidates.Keypoints(), frame_with_only_candidates.LandmarkStatuses(),
              true, 0.75, mask);

    // Insert tracked candidates into the current frame
    vector<cv::KeyPoint> tracked_candidate_keypoints =
            frame_with_only_candidates.GetKeypointsWithStatus({TRACKED_WITH_3D});
    vector<Eigen::Vector3f> tracked_candidate_landmarks =
            frame_with_only_candidates.GetLandmarkPositionsWithStatus({TRACKED_WITH_3D});
    vector<ID> tracked_candidate_mappoint_ids =
            frame_with_only_candidates.GetMapPointsIdsWithStatus({TRACKED_WITH_3D});

    int reused_landmarks = 0;

    for (int idx = 0; idx < tracked_candidate_keypoints.size(); idx++) {
        cv::KeyPoint keypoint = tracked_candidate_keypoints[idx];
        Eigen::Vector3f landmark_position = tracked_candidate_landmarks[idx];
        ID mappoint_id = tracked_candidate_mappoint_ids[idx];

        keypoint.class_id = map_->GetMapPoint(mappoint_id)->GetKeyPointId();

        Eigen::Vector3f landmark_camera_position = current_frame_->CameraTransformationWorld() * landmark_position;
        cv::Point2f projected_landmark = calibration_->Project(landmark_camera_position);

        if (SquaredReprojectionError(projected_landmark, keypoint.pt) > 5.99) {
            continue;
        }

        if (current_frame_->MapPointIdToIndex().contains(mappoint_id)) {
            const int idx_in_frame = current_frame_->MapPointIdToIndex().at(mappoint_id);

            current_frame_->Keypoints()[idx_in_frame] = keypoint;
            current_frame_->LandmarkPositions()[idx_in_frame] = landmark_position;
            current_frame_->LandmarkStatuses()[idx_in_frame] = TRACKED_WITH_3D;

        } else {
            current_frame_->InsertObservation(keypoint, landmark_position, mappoint_id, TRACKED_WITH_3D);

            LucasKanadeTracker::PhotometricInformation photometric_information =
                    map_->GetMapPoint(mappoint_id)->GetPhotometricInformation();
            klt_tracker_.InsertPhotometricInformation(keypoint, photometric_information);
        }

        reused_landmarks++;
    }

    LOG(INFO) << "Reused landmarks: " << reused_landmarks;
}

void Tracking::UpdateTriangulatedPoints() {
    auto indices = current_frame_->GetIndexWithStatus({JUST_TRIANGULATED});

    for (auto index : indices) {
        LucasKanadeTracker::PhotometricInformation photometric_information =
                klt_tracker_.GetPhotometricInformationOfPoint(index);

        ID mappoint_id = current_frame_->IndexToMapPointId().at(index);
        map_->GetMapPoint(mappoint_id)->SetPhotometricInformation(photometric_information);

        // Update landmark status.
        current_frame_->LandmarkStatuses()[index] = TRACKED_WITH_3D;
    }
}
