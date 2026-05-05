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

#include "datasets/simulation.h"
#include "SLAM/system.h"

#include "absl/flags/parse.h"
#include "absl/flags/flag.h"
#include "absl/log/log.h"
#include "absl/log/check.h"
#include "absl/log/initialize.h"

#include <fstream>


using namespace std;

ABSL_FLAG(std::string, dataset_path, "/home/zy/Dataset/EndoMapper/Simulated_Sequences/Seq_0", "Path to the video dataset");
ABSL_FLAG(std::string, settings_path, "/home/zy/NR-SLAM/data/simulation/settings.yaml", "Path to the settings file");
ABSL_FLAG(int, starting_frame, 50, "First frame of the dataset to process");
ABSL_FLAG(int, end_frame, 280, "Last frame of the dataset to process");
ABSL_FLAG(std::string, sdf_export_path, "", "Optional output PLY path for exported SDF surface points");
ABSL_FLAG(double, sdf_export_max_abs_tsdf, 0.10, "Max |tsdf| threshold used when exporting SDF surface points");
ABSL_FLAG(std::string, surfel_export_path, "", "Optional output PLY path for exported SurfMap");
ABSL_FLAG(bool, surfel_export_only_finalized, false, "Export only finalized surfels when writing SurfMap PLY");
ABSL_FLAG(std::string, metrics_csv_path, "", "Optional CSV path for per-frame runtime metrics");

int main(int argc, char **argv) {
    // Parse command line argumemnts.
    absl::ParseCommandLine(argc, argv);

    // Process command arguments.
    string dataset_path = absl::GetFlag(FLAGS_dataset_path);
    if(dataset_path.empty()){
        LOG(ERROR) << "Must specify an input dataset path." << endl;
        return -1;
    }
    string settings_path = absl::GetFlag(FLAGS_settings_path);
    if(settings_path.empty()){
        LOG(ERROR) << "Must specify an input settings file." << endl;
        return -1;
    }

    int starting_frame = absl::GetFlag(FLAGS_starting_frame);
    int end_frame = absl::GetFlag(FLAGS_end_frame);

    const string metrics_csv_path = absl::GetFlag(FLAGS_metrics_csv_path);
    std::ofstream metrics_csv;
    if (!metrics_csv_path.empty()) {
        metrics_csv.open(metrics_csv_path, std::ios::out);
        if (!metrics_csv.is_open()) {
            LOG(ERROR) << "Cannot open metrics csv path: " << metrics_csv_path;
            return -1;
        }
        metrics_csv << "input_frame,tracking_frame,nn_mode,nn_a,nn_b,nn_pairs,nn_inliers,nn_inlier_ratio,"
                       "sdf_voxels,sdf_dense_pts,sdf_sparse_pts,sdf_updates,sdf_dense_only_fallback,"
                       "surfel_active,surfel_finalized,"
                       "surfel_generated,surfel_accepted,surfel_added,surfel_fused,"
                       "surfel_rejected_by_depth,surfel_rejected_by_gradient,"
                       "surfel_rejected_by_normal,surfel_rejected_by_mask,"
                       "surfel_rejected_by_sdf,surfel_rejected_by_color,"
                       "surfel_warped,surfel_finalized_this_frame,"
                       "surfel_mean_radius,surfel_mean_confidence,surfel_mean_color_residual,"
                       "phi_checked,phi_accepted,phi_rejected,phi_projected,"
                       "phi_before_min,phi_before_max,phi_after_min,phi_after_max\n";
    }

    Simulation dataset(dataset_path);

    // Create SLAM system
    System SLAM(settings_path);

    for (int idx = starting_frame; idx < end_frame; idx++) {
        LOG(INFO) << "Processing image " << idx;
        auto image = dataset.GetImage(idx);
        CHECK_OK(image);

        auto depth_image = dataset.GetDepthImage(idx);
        CHECK_OK(depth_image);

        auto nn_image = dataset.GetNNImage(idx);

        cv::Size new_size((*image).cols/2.0f, (*image).rows/2.0f);
        cv::Mat image_resized, depth_image_resized;
        cv::resize((*image), image_resized, new_size);
        cv::resize((*depth_image), depth_image_resized, new_size);

        // SLAM.TrackImageWithDepth((*image), (*depth_image));
        SLAM.TrackImageWithNN((*image), (*depth_image), (*nn_image));

        if (metrics_csv.is_open()) {
            const auto metrics = SLAM.GetRuntimeMetrics();
            metrics_csv << idx << ","
                        << metrics.frame_id << ","
                        << static_cast<int>(metrics.nn_align.mode) << ","
                        << metrics.nn_align.scale_a << ","
                        << metrics.nn_align.shift_b << ","
                        << metrics.nn_align.correspondences_total << ","
                        << metrics.nn_align.inliers << ","
                        << metrics.nn_align.inlier_ratio << ","
                        << metrics.sdf.active_voxels << ","
                        << metrics.sdf.dense_points_used << ","
                        << metrics.sdf.sparse_points_used << ","
                        << metrics.sdf.voxel_updates << ","
                        << static_cast<int>(metrics.sdf.dense_only_fallback) << ","
                        << metrics.surfel_active << ","
                        << metrics.surfel_finalized << ","
                        << metrics.surfel_fusion.generated << ","
                        << metrics.surfel_fusion.accepted << ","
                        << metrics.surfel_fusion.added << ","
                        << metrics.surfel_fusion.fused << ","
                        << metrics.surfel_fusion.rejected_by_depth << ","
                        << metrics.surfel_fusion.rejected_by_gradient << ","
                        << metrics.surfel_fusion.rejected_by_normal << ","
                        << metrics.surfel_fusion.rejected_by_mask << ","
                        << metrics.surfel_fusion.rejected_by_sdf << ","
                        << metrics.surfel_fusion.rejected_by_color << ","
                        << metrics.surfel_fusion.warped << ","
                        << metrics.surfel_fusion.finalized << ","
                        << metrics.surfel_fusion.mean_radius << ","
                        << metrics.surfel_fusion.mean_confidence << ","
                        << metrics.surfel_fusion.mean_color_residual << ","
                        << metrics.surfel_phi.constraint_checked << ","
                        << metrics.surfel_phi.constraint_accepted << ","
                        << metrics.surfel_phi.constraint_rejected << ","
                        << metrics.surfel_phi.projected_count << ","
                        << metrics.surfel_phi.phi_before_min << ","
                        << metrics.surfel_phi.phi_before_max << ","
                        << metrics.surfel_phi.phi_after_min << ","
                        << metrics.surfel_phi.phi_after_max << "\n";
        }
    }
    SLAM.SaveTraj();

    if (metrics_csv.is_open()) {
        metrics_csv.close();
        LOG(INFO) << "Runtime metrics csv written to: " << metrics_csv_path;
    }

    const string sdf_export_path = absl::GetFlag(FLAGS_sdf_export_path);
    if (!sdf_export_path.empty()) {
        const float max_abs_tsdf = static_cast<float>(absl::GetFlag(FLAGS_sdf_export_max_abs_tsdf));
        LOG(INFO) << "Exporting SDF to: " << sdf_export_path << " (max_abs_tsdf=" << max_abs_tsdf << ")";
        if (!SLAM.ExportSDF(sdf_export_path, max_abs_tsdf)) {
            LOG(ERROR) << "Failed to export SDF to: " << sdf_export_path;
            return -1;
        }
    }

    const string surfel_export_path = absl::GetFlag(FLAGS_surfel_export_path);
    if (!surfel_export_path.empty()) {
        const bool only_finalized = absl::GetFlag(FLAGS_surfel_export_only_finalized);
        LOG(INFO) << "Exporting SurfMap to: " << surfel_export_path
                  << " (only_finalized=" << only_finalized << ")";
        if (!SLAM.ExportSurfMap(surfel_export_path, only_finalized)) {
            LOG(ERROR) << "Failed to export SurfMap to: " << surfel_export_path;
            return -1;
        }
    }

    return 0;
}
