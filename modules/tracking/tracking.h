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

#ifndef NRSLAM_TRACKING_H
#define NRSLAM_TRACKING_H

#include "calibration/camera_model.h"
#include "features/feature.h"
#include "map/frame.h"
#include "map/map.h"
#include "matching/lucas_kanade_tracker.h"
#include "stereo/stereo_lucas_kanade.h"
#include "stereo/stereo_pattern_matching.h"
#include "tracking/monocular_map_initializer.h"
#include "utilities/time_profiler.h"
#include "visualization/image_visualizer.h"

#include "map/surfel_map.h"
#include "sdf/sdf_builder.h"

#include "absl/container/flat_hash_set.h"

#include <opencv2/opencv.hpp>

#include <string>

class Tracking {
public:
    enum class NnAlignMode {
        SUCCESS = 0,
        REUSED_LAST = 1,
        FAILED_EMPTY = 2
    };

    struct NnAlignStats {
        uint64_t frame_id = 0;
        NnAlignMode mode = NnAlignMode::FAILED_EMPTY;
        float scale_a = 1.0f;
        float shift_b = 0.0f;
        int correspondences_total = 0;
        int inliers = 0;
        float inlier_ratio = 0.0f;
        bool has_aligned_depth = false;
    };

    struct RuntimeMetrics {
        uint64_t frame_id = 0;
        NnAlignStats nn_align;
        SDFBuilder::Stats sdf;
        size_t surfel_active = 0;
        size_t surfel_finalized = 0;
        SurfelPhiStats surfel_phi;  // Phi constraint statistics
        SurfelFusionStats surfel_fusion;
    };

    struct Options {
        int klt_window_size = 21;
        int klt_max_level = 3;
        int klt_max_iters = 50;
        float klt_epsilon = 0.01;
        float klt_min_eig_th = 1e-4;
        float klt_min_SSIM = 0.7;

        int images_to_insert_keyframe = 5;

        float radians_per_pixel;

        // SDF integration options.
        float sdf_voxel_size = 0.0025f;
        float sdf_truncation_distance = 0.0100f;
        int sdf_dense_pixel_stride = 2;
        float sdf_dense_weight = 1.0f;
        float sdf_sparse_weight = 5.0f;
        int sdf_max_voxel_updates_per_frame = 300000;
        float sdf_min_depth = 0.01f;
        float sdf_max_depth = 8.00f;
        int sdf_sparse_min_points = 20;

        bool surfel_sdf_constraint_enabled = false;
        float surfel_sdf_phi_abs_threshold = 0.20f;
        float surfel_sdf_normal_cos_threshold = 0.30f;
        bool surfel_sdf_project_to_surface = false;

        int surfel_window_size = 8;
        float surfel_min_depth = 0.02f;
        float surfel_max_depth = 1.00f;
        float surfel_default_radius = 0.003f;
        bool surfel_use_adaptive_radius = true;
        int surfel_sample_step = 4;
        size_t surfel_max_new_per_frame = 10000;
        float surfel_fusion_distance_threshold = 0.01f;
        float surfel_fusion_normal_threshold = 0.85f;
        float surfel_spatial_search_radius = 0.02f;
        float surfel_active_min_depth = 0.02f;
        float surfel_active_max_depth = 0.10f;
        float surfel_confidence_decay_rate = 0.10f;
        float surfel_min_confidence = 0.30f;
        float surfel_warp_sigma = 0.03f;
        float surfel_warp_max_disp = 0.10f;
        std::string surfel_quality_mode = "realtime";
        float surfel_min_radius = 0.001f;
        float surfel_max_radius = 0.006f;
        float surfel_radius_scale = 1.5f;
        float surfel_max_depth_gradient = 0.030f;
        int surfel_mask_border_pixels = 1;
        size_t surfel_max_fused_per_frame = 0;
        size_t surfel_max_added_per_frame = 0;
        float surfel_color_update_max_residual = 0.35f;
        float surfel_color_update_soft_residual = 0.15f;
    };

    enum TrackingStatus {
        NOT_INITIALIZED,
        TRACKING,
        LOST
    };

    Tracking() = delete;

    Tracking(const Options options, std::shared_ptr<Map> map,
             std::shared_ptr<CameraModel> calibration,
             std::shared_ptr<StereoLucasKanade> stereo_matcher,
             std::shared_ptr<ImageVisualizer> image_visualizer,
             TimeProfiler* time_profiler);

    void TrackImage(const cv::Mat& im, const absl::flat_hash_map<std::string, cv::Mat>& masks,
                    const cv::Mat& additional_im = cv::Mat(), const cv::Mat& im_clahe = cv::Mat());

    TrackingStatus GetTrackingStatus() const;

    std::vector<Sophus::SE3f> GetCameraPoses();

    //! Edited 06.03
    void TrackImageNN(const cv::Mat& im, const cv::Mat& im_color, const absl::flat_hash_map<std::string, cv::Mat>& masks,
                      const cv::Mat& nn_im, const cv::Mat& im_clahe = cv::Mat());
    void EstimateDeformationField();
    void ComputeRigidWeights();

    SurfMap* GetSurfMap() { return surf_map_.get(); }
    SDFBuilder* GetSDFBuilder() { return sdf_builder_.get(); }
    bool ExportSDF(const std::string& filename, float max_abs_tsdf = 0.10f);
    RuntimeMetrics GetRuntimeMetrics() const;

private:
    void ExtractFeatures(const cv::Mat& im, const cv::Mat& mask,
                         std::vector<cv::KeyPoint>& keypoints);

    void MonocularMapInitializationNN(const cv::Mat& im_left, const cv::Mat& im_nn,
                                 const cv::Mat& mask, const cv::Mat& im_clahe);
    
    void MonocularMapInitialization(const cv::Mat& im_left,
                                 const cv::Mat& mask, const cv::Mat& im_clahe);

    void StereoMapInitialization(const cv::Mat& im_left, const cv::Mat& im_right,
                                 const cv::Mat& mask, const cv::Mat& im_clahe);

    absl::flat_hash_set<ID> TrackCameraAndDeformation(const cv::Mat& im, const cv::Mat& mask);
    absl::flat_hash_set<ID> TrackCameraAndDeformationNN(const cv::Mat& im, const cv::Mat& im_nn, const cv::Mat& mask);

    void DataAssociation(const cv::Mat& im, const cv::Mat& mask);

    void CameraPoseEstimation();

    absl::flat_hash_set<ID> CameraPoseAndDeformationEstimation();
    absl::flat_hash_set<ID> CameraPoseAndDeformationEstimationFinal();

    void KeyFrameInsertion(const cv::Mat& im, const cv::Mat& im_color, const absl::flat_hash_map<std::string, cv::Mat>& masks);

    bool NeedNewKeyFrame();

    void CreateNewKeyFrame(const cv::Mat& im, const cv::Mat& im_color, const absl::flat_hash_map<std::string, cv::Mat>& masks);

    void ExtractFeaturesInFrame(const cv::Mat& im, const cv::Mat& mask, Frame& frame);

    void SetKLTReference(const cv::Mat& im, Frame& frame, const cv::Mat& mask);

    void PointReuse(const cv::Mat& im, const cv::Mat& mask,
                    absl::flat_hash_set<ID> lost_mappoint_ids);

    void UpdateTriangulatedPoints();

    // Unified NN scale-shift result handler (replaces duplicated 3-branch logic).
    void ApplyNnAlignResult(bool fit_ok, float a, float b,
                            int num_inliers, const cv::Mat& nn,
                            int correspondences_total);

    Options options_;

    std::shared_ptr<Map> map_;

    std::unique_ptr<SurfMap> surf_map_;
    std::unique_ptr<SDFBuilder> sdf_builder_;
    NnAlignStats last_nn_align_stats_;
    bool has_last_nn_scale_shift_ = false;
    float last_nn_scale_ = 1.0f;
    float last_nn_shift_ = 0.0f;
    // 缓存一个内参矩阵
    Eigen::Matrix3f K_eigen_;
    bool K_initialized_ = false;

    std::shared_ptr<CameraModel> calibration_;

    std::shared_ptr<Feature> feature_extractor_;

    LucasKanadeTracker klt_tracker_;

    std::shared_ptr<Frame> current_frame_;

    //std::shared_ptr<StereoPatternMatching> stereo_matcher_;
    std::shared_ptr<StereoLucasKanade> stereo_matcher_;

    Sophus::SE3f motion_model_;

    std::shared_ptr<ImageVisualizer> image_visualizer_;

    int n_images_from_last_keyframe_ = 0;

    std::unique_ptr<MonocularMapInitializer> monocular_map_initializer_;

    TrackingStatus tracking_status_;

    Sophus::SE3f previous_camera_transform_world_;

    TimeProfiler* time_profiler_;

    std::vector<Sophus::SE3f> poses;

};


#endif //NRSLAM_TRACKING_H
