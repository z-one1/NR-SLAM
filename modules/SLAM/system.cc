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

#include "system.h"

#include "absl/log/log.h"

#include <algorithm>
#include <sstream>
#include <iomanip>

using namespace std;

System::System(const string settings_file_path) {
    // Output welcome message
    LOG(INFO).NoPrefix() << "NR-SLAM Copyright (C) Copyright (C) 2022-2023 Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.";
    LOG(INFO).NoPrefix() << "This program comes with ABSOLUTELY NO WARRANTY;";
    LOG(INFO).NoPrefix() << "This is free software, and you are welcome to redistribute it";
    LOG(INFO).NoPrefix() << "under certain conditions. See LICENSE.txt.";

    settings_ = make_unique<Settings>(settings_file_path);
    LOG(INFO) << *settings_;

    // Initialize image processing stuff
    clahe_ = cv::createCLAHE(3.0, cv::Size(8, 8));
    masker_ = settings_->getMasker();

    // Create map
    Map::Options map_options;
    map_options.max_temporal_buffer_size = 20;
    map_ = make_shared<Map>(map_options);

    StereoLucasKanade::Options stereo_matcher_options;
    stereo_matcher_options.klt_window_size = 21;
    stereo_matcher_options.klt_max_level = 4;
    stereo_matcher_options.klt_max_iters = 10;
    stereo_matcher_options.klt_epsilon = 0.0001;
    stereo_matcher_options.klt_min_eig_th = 0.0001;
    stereo_matcher_options.klt_min_SSIM = 0.5;

    stereo_matcher_ = make_shared<StereoLucasKanade>(stereo_matcher_options, settings_->getCalibration(),
                                                     settings_->getBf());

    stereo_pattern_matcher_ = make_shared<StereoPatternMatching>(settings_->getCalibration(),
                                                                 settings_->getBf());

    // Initialize map visualizer and launch it in a different thread
    MapVisualizer::Options map_visualizer_options;
    map_visualizer_options.camera_size_ = settings_->GetCameraSize();
    map_visualizer_options.initial_left_view_ = settings_->GetLeftMapVisualizationView().matrix();
    map_visualizer_options.initial_right_view_ = settings_->GetRightMapVisualizationView().matrix();
    map_visualizer_options.render_save_path = settings_->GetMapVisualizerPath();

    map_visualizer_ = make_unique<MapVisualizer>(map_visualizer_options, map_);
    map_visualizer_thread_ = make_unique<thread>(&MapVisualizer::Run, map_visualizer_.get());

    //Initialize image visualizer.
    ImageVisualizer::Options image_visualizer_options;
    image_visualizer_options.wait_for_user_button = !settings_->GetAutoplay();
    image_visualizer_options.image_save_path = settings_->GetImageVisualizerPath();
    image_visualizer_ = make_shared<ImageVisualizer>(image_visualizer_options);

    // Initialize Tracking.
    Tracking::Options tracking_options;
    tracking_options.klt_window_size = 21;
    tracking_options.klt_max_level = 4;
    tracking_options.klt_max_iters = 10;
    tracking_options.klt_epsilon = 0.0001;
    tracking_options.klt_min_eig_th = 0.0001;
    tracking_options.klt_min_SSIM = 0.7;
    tracking_options.radians_per_pixel = settings_->getRadPerPixel();
    tracking_options.sdf_voxel_size = settings_->GetSDFVoxelSize();
    tracking_options.sdf_truncation_distance = settings_->GetSDFTruncationDistance();
    tracking_options.sdf_dense_pixel_stride = settings_->GetSDFDensePixelStride();
    tracking_options.sdf_dense_weight = settings_->GetSDFDenseWeight();
    tracking_options.sdf_sparse_weight = settings_->GetSDFSparseWeight();
    tracking_options.sdf_max_voxel_updates_per_frame = settings_->GetSDFMaxVoxelUpdatesPerFrame();
    tracking_options.sdf_min_depth = settings_->GetSDFMinDepth();
    tracking_options.sdf_max_depth = settings_->GetSDFMaxDepth();
    tracking_options.sdf_sparse_min_points = settings_->GetSDFSparseMinPoints();
    tracking_options.surfel_sdf_constraint_enabled = settings_->GetSurfelSDFConstraintEnabled();
    tracking_options.surfel_sdf_phi_abs_threshold = settings_->GetSurfelSDFPhiAbsThreshold();
    tracking_options.surfel_sdf_normal_cos_threshold = settings_->GetSurfelSDFNormalCosThreshold();
    tracking_options.surfel_sdf_project_to_surface = settings_->GetSurfelSDFProjectToSurface();
    const int surfmap_window_size = settings_->GetSurfMapWindowSize();
    const int surfmap_max_new_per_frame = settings_->GetSurfMapMaxNewPerFrame();
    tracking_options.surfel_window_size = surfmap_window_size > 0 ? surfmap_window_size : 1;
    tracking_options.surfel_min_depth = settings_->GetSurfMapMinDepth();
    tracking_options.surfel_max_depth = settings_->GetSurfMapMaxDepth();
    tracking_options.surfel_default_radius = settings_->GetSurfMapDefaultRadius();
    tracking_options.surfel_use_adaptive_radius = settings_->GetSurfMapUseAdaptiveRadius();
    tracking_options.surfel_sample_step = settings_->GetSurfMapSampleStep();
    tracking_options.surfel_max_new_per_frame = static_cast<size_t>(
        surfmap_max_new_per_frame > 0 ? surfmap_max_new_per_frame : 0);
    tracking_options.surfel_fusion_distance_threshold = settings_->GetSurfMapFusionDistanceThreshold();
    tracking_options.surfel_fusion_normal_threshold = settings_->GetSurfMapFusionNormalThreshold();
    tracking_options.surfel_spatial_search_radius = settings_->GetSurfMapSpatialSearchRadius();
    tracking_options.surfel_active_min_depth = settings_->GetSurfMapActiveMinDepth();
    tracking_options.surfel_active_max_depth = settings_->GetSurfMapActiveMaxDepth();
    tracking_options.surfel_confidence_decay_rate = settings_->GetSurfMapConfidenceDecayRate();
    tracking_options.surfel_min_confidence = settings_->GetSurfMapMinConfidence();
    tracking_options.surfel_warp_sigma = settings_->GetSurfMapWarpSigma();
    tracking_options.surfel_warp_max_disp = settings_->GetSurfMapWarpMaxDisp();
    tracking_options.surfel_quality_mode = settings_->GetSurfMapQualityMode();
    tracking_options.surfel_min_radius = settings_->GetSurfMapMinRadius();
    tracking_options.surfel_max_radius = settings_->GetSurfMapMaxRadius();
    tracking_options.surfel_radius_scale = settings_->GetSurfMapRadiusScale();
    tracking_options.surfel_max_depth_gradient = settings_->GetSurfMapMaxDepthGradient();
    tracking_options.surfel_mask_border_pixels = std::max(0, settings_->GetSurfMapMaskBorderPixels());
    const int surfmap_max_fused_per_frame = settings_->GetSurfMapMaxFusedPerFrame();
    const int surfmap_max_added_per_frame = settings_->GetSurfMapMaxAddedPerFrame();
    tracking_options.surfel_max_fused_per_frame = static_cast<size_t>(
        surfmap_max_fused_per_frame > 0 ? surfmap_max_fused_per_frame : 0);
    tracking_options.surfel_max_added_per_frame = static_cast<size_t>(
        surfmap_max_added_per_frame > 0 ? surfmap_max_added_per_frame : 0);
    tracking_options.surfel_color_update_max_residual = settings_->GetSurfMapColorUpdateMaxResidual();
    tracking_options.surfel_color_update_soft_residual = settings_->GetSurfMapColorUpdateSoftResidual();

    // Time profiler.
    time_profiler_ = make_unique<TimeProfiler>();

    tracker_ = make_unique<Tracking>(tracking_options, map_, settings_->getCalibration(),
                                     stereo_matcher_, image_visualizer_, time_profiler_.get());

    // Initialize Mapping.
    Mapping::Options mapping_options;
    mapping_options.rad_per_pixel = settings_->getRadPerPixel();
    mapper_ = make_unique<Mapping>(map_, settings_->getCalibration(), mapping_options, time_profiler_.get());

    // Initialize frame evaluator.
    FrameEvaluator::Options frame_evaluator_options;
    frame_evaluator_options.results_file_path = settings_->GetEvaluationPath();
    frame_evaluator_options.precomputed_depth_ = true;
    frame_evaluator_ = make_unique<FrameEvaluator>(frame_evaluator_options, stereo_pattern_matcher_,
                                                   map_visualizer_.get()); 
    
    cout << "Evaluation path: " << settings_->GetEvaluationPath() << endl;
}

System::~System() {
    // Send signal to the visualizer to finish
    map_visualizer_->SetFinish();

    // Wait until is done
    map_visualizer_thread_->join();
}


vector<Eigen::Vector3f> System::GetTraj() {
    auto latest_frames = map_->GetTemporalBuffer()->GetLatestCameraPoses();

    vector<Eigen::Vector3f> trajectory;
    for (auto& pose : latest_frames) {
        trajectory.push_back(pose.inverse().translation());
    }
    return trajectory;
}


void System::SaveTraj() {
    // Get the latest camera poses directly (not just translation)
    // auto latest_frames = map_->GetTemporalBuffer()->GetLatestCameraPoses();
    auto poses = tracker_->GetCameraPoses();

    // Open file for writing
    std::ofstream traj_file("trajectory.tum", std::ios::out);
    if (!traj_file.is_open()) {
        std::cerr << "Failed to open trajectory file!" << std::endl;
        return;
    }

    // Set precision for floating-point output
    traj_file << std::fixed << std::setprecision(6);

    // Iterate over poses and write in TUM format
    double time_step = 0.033; // Assume 30 FPS; adjust as needed
    int frame_idx = 0;
    for (const auto& pose : poses) {
        // Assuming pose is Sophus::SE3f or Sophus::SE3d
        Eigen::Vector3f t = pose.inverse().translation(); // Translation (x, y, z)
        Eigen::Quaternionf q(pose.inverse().unit_quaternion()); // Quaternion (qx, qy, qz, qw)

        // Synthetic timestamp (adjust if real timestamps are available)
        double timestamp = frame_idx * time_step;

        // Write to file: timestamp tx ty tz qx qy qz qw
        traj_file << timestamp << " "
                  << t.x() << " " << t.y() << " " << t.z() << " "
                  << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << "\n";

        frame_idx++;
    }

    // Close the file
    traj_file.close();
    std::cout << "Trajectory saved to trajectory.tum" << std::endl;
}

void System::TrackImage(const cv::Mat &im) {
    // Preprocess image.
    cv::Mat im_gray;
    cv::Mat processed_image = ImageProcessing(im, im_gray);

    // Insert image in the image visualizer.
    image_visualizer_->SetCurrentImage(im, processed_image);

    // Generate image mask.
    auto masks = masker_->GetAllMasks(im_gray);

    // Perform tracking.
    tracker_->TrackImage(im_gray, masks, cv::Mat(), processed_image);

    // Perform mapping.
    mapper_->DoMapping();

    // Draw images.
    image_visualizer_->UpdateWindows();
}

void System::TrackImageWithStereo(const cv::Mat &im_left, const cv::Mat &im_right) {
    // Preprocess images.
    cv::Mat im_gray_left, im_gray_right;
    cv::Mat processed_image_left = ImageProcessing(im_left, im_gray_left);
    cv::Mat processed_image_right = ImageProcessing(im_right, im_gray_right);

    // Insert image in the image visualizer.
    image_visualizer_->SetCurrentImage(im_left, processed_image_left);

    // Generate image mask.
    auto masks = masker_->GetAllMasks(im_gray_left);

    // Perform tracking.
    tracker_->TrackImage(im_gray_left, masks, im_gray_right, processed_image_left);

    // Perform mapping.
    mapper_->DoMapping();

    // Evaluate reconstruction.
    if (false && tracker_->GetTrackingStatus() == Tracking::TRACKING) {
        frame_evaluator_->EvaluateFrameReconstruction(*(map_->GetMutableLastFrame()), im_gray_left, im_gray_right);
        frame_evaluator_->SaveResultsToFile();
    }

    // Draw images.
    image_visualizer_->UpdateWindows();
}


void System::TrackImageWithNN(const cv::Mat &im_left, const cv::Mat &im_depth, const cv::Mat &im_nn) {
    // Preprocess images.
    cv::Mat im_gray_left;
    cv::Mat processed_image_left = ImageProcessing(im_left, im_gray_left);

    // Insert image in the image visualizer.
    image_visualizer_->SetCurrentImage(im_left, processed_image_left);

    // Generate image mask.
    auto masks = masker_->GetAllMasks(im_gray_left);

    time_profiler_->Tic("Tracking & Mapping");

    if(im_gray_left.size() != im_nn.size()) {
        LOG(ERROR) << "nn_image size did not match image_org";
        return;
    }
    // Perform tracking.
    tracker_->TrackImageNN(im_gray_left, im_left, masks, im_nn, processed_image_left);

    // Push latest surfel data to visualizer (thread-safe copy).
    map_visualizer_->UpdateSurfelData(tracker_->GetSurfMap());

    // Perform mapping.
    mapper_->DoMapping();

    time_profiler_->Toc("Tracking & Mapping");
    time_profiler_->PrintStatisticsForIdentifier("Tracking & Mapping");

    // Evaluate reconstruction.
    if (true && tracker_->GetTrackingStatus() == Tracking::TRACKING) {
        frame_evaluator_->EvaluateFrameReconstruction(*(map_->GetMutableLastFrame()), im_gray_left, im_depth);
        frame_evaluator_->SaveResultsToFile();
    }

    // Draw images.
    image_visualizer_->UpdateWindows();

    // cv::Mat nn_vis;
    // double minv, maxv;
    // cv::minMaxLoc(im_nn, &minv, &maxv);

    // cv::Mat scaled;
    // im_nn.convertTo(scaled, CV_8UC1, 255.0 / (maxv - minv), -255.0 * minv / (maxv - minv));
    // cv::applyColorMap(scaled, nn_vis, cv::COLORMAP_JET);

    // cv::imshow("NN Depth", nn_vis);
    // cv::waitKey(1);


    // // 每帧/每 N 帧 导出当前 surf_map
    // if (surf_export_enabled_ && (surf_export_frame_idx_ % surf_export_step_ == 0)) {

    //     auto surf_map = tracker_->GetSurfMap();
    //     if (surf_map) {
    //         // 自动生成文件名：surfmap_0000.ply, surfmap_0001.ply, ...
    //         std::ostringstream oss;
    //         oss << "surfmap_"
    //             << std::setfill('0') << std::setw(4)
    //             << surf_export_frame_idx_
    //             << ".ply";

    //         std::string filename = oss.str();

    //         // false = 导出 active+finalized 一起；true = 只导出 finalized
    //         surf_map->ExportToPLY(filename, /*only_finalized=*/false);

    //         LOG(INFO) << "Exported SurfMap to " << filename;
    //     } else {
    //         LOG(WARNING) << "SurfMap is null, skip export.";
    //     }
    // }

    // // 3) 帧计数 +1
    // surf_export_frame_idx_++;
}

void System::TrackImageWithDepth(const cv::Mat &im_left, const cv::Mat &im_depth) {
    // Preprocess images.
    cv::Mat im_gray_left, im_gray_right;
    cv::Mat processed_image_left = ImageProcessing(im_left, im_gray_left);

    // Insert image in the image visualizer.
    image_visualizer_->SetCurrentImage(im_left, processed_image_left);

    // Generate image mask.
    auto masks = masker_->GetAllMasks(im_gray_left);

    time_profiler_->Tic("Tracking & Mapping");

    // Perform tracking.
    tracker_->TrackImage(im_gray_left, masks, im_gray_right, processed_image_left);

    // Perform mapping.
    mapper_->DoMapping();

    time_profiler_->Toc("Tracking & Mapping");
    time_profiler_->PrintStatisticsForIdentifier("Tracking & Mapping");

    // Evaluate reconstruction.
    if (true && tracker_->GetTrackingStatus() == Tracking::TRACKING) {
        frame_evaluator_->EvaluateFrameReconstruction(*(map_->GetMutableLastFrame()), im_gray_left, im_depth);
        frame_evaluator_->SaveResultsToFile();
    }

    // Draw images.
    image_visualizer_->UpdateWindows();
}

cv::Mat System::ImageProcessing(const cv::Mat &im, cv::Mat& im_gray) {
    cv::Mat processed_image;

    // Convert to grayscale.
    cv::cvtColor(im, processed_image, cv::COLOR_RGB2GRAY);

    im_gray = processed_image.clone();

    // Apply Clahe to the image.
    clahe_->apply(processed_image, processed_image);

    return processed_image;
}

bool System::ExportSurfMap(const std::string& filename,
                           bool only_finalized) {

    if (!tracker_) {
        LOG(WARNING) << "No tracker, cannot export surfel map.";
        return false;
    }

    auto surf_map = tracker_->GetSurfMap();  // <-- 你已有的接口
    if (!surf_map) {
        LOG(WARNING) << "Tracking has no surf_map_ instance.";
        return false;
    }

    if (!surf_map->ExportToPLY(filename, only_finalized)) {
        return false;
    }
    LOG(INFO) << "SurfMap exported to " << filename 
              << ", only_finalized = " << only_finalized;
    return true;
}

bool System::ExportSDF(const std::string& filename,
                       float max_abs_tsdf) {
    if (!tracker_) {
        LOG(WARNING) << "No tracker, cannot export SDF.";
        return false;
    }

    if (!tracker_->ExportSDF(filename, max_abs_tsdf)) {
        return false;
    }
    LOG(INFO) << "SDF exported to " << filename
              << ", max_abs_tsdf = " << max_abs_tsdf;
    return true;
}

Tracking::RuntimeMetrics System::GetRuntimeMetrics() const {
    if (!tracker_) {
        return Tracking::RuntimeMetrics{};
    }
    return tracker_->GetRuntimeMetrics();
}
