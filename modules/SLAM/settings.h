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

#ifndef NRSLAM_SETTINGS_H
#define NRSLAM_SETTINGS_H

#include "calibration/camera_model.h"

#include "masking/masker.h"

#include <memory>
#include <string>

#include "sophus/se3.hpp"

class Settings {
public:
    /*
     * Default constructor: sets everything to default values
     */
    Settings();

    /*
     * Constructor reading parameters from file
     */
    Settings(const std::string& configFile);

    /*
     * Ostream operator overloading to dump settings to the terminal
     */
    friend std::ostream& operator<<(std::ostream& output, const Settings& D);

    //Getter methods
    std::shared_ptr<CameraModel> getCalibration();
    float getBf();

    std::shared_ptr<Masker> getMasker();

    float getRadPerPixel();

    Sophus::SE3f GetLeftMapVisualizationView();
    Sophus::SE3f GetRightMapVisualizationView();

    float GetCameraSize();

    bool GetAutoplay();

    std::string GetMapVisualizerPath();

    std::string GetImageVisualizerPath();

    std::string GetEvaluationPath();

    float GetSDFVoxelSize();
    float GetSDFTruncationDistance();
    int GetSDFDensePixelStride();
    float GetSDFDenseWeight();
    float GetSDFSparseWeight();
    int GetSDFMaxVoxelUpdatesPerFrame();
    float GetSDFMinDepth();
    float GetSDFMaxDepth();
    int GetSDFSparseMinPoints();
    bool GetSurfelSDFConstraintEnabled();
    float GetSurfelSDFPhiAbsThreshold();
    float GetSurfelSDFNormalCosThreshold();
    bool GetSurfelSDFProjectToSurface();
    int GetSurfMapWindowSize();
    float GetSurfMapMinDepth();
    float GetSurfMapMaxDepth();
    float GetSurfMapDefaultRadius();
    bool GetSurfMapUseAdaptiveRadius();
    int GetSurfMapSampleStep();
    int GetSurfMapMaxNewPerFrame();
    float GetSurfMapFusionDistanceThreshold();
    float GetSurfMapFusionNormalThreshold();
    float GetSurfMapSpatialSearchRadius();
    float GetSurfMapActiveMinDepth();
    float GetSurfMapActiveMaxDepth();
    float GetSurfMapConfidenceDecayRate();
    float GetSurfMapMinConfidence();
    float GetSurfMapWarpSigma();
    float GetSurfMapWarpMaxDisp();
    std::string GetSurfMapQualityMode();
    float GetSurfMapMinRadius();
    float GetSurfMapMaxRadius();
    float GetSurfMapRadiusScale();
    float GetSurfMapMaxDepthGradient();
    int GetSurfMapMaskBorderPixels();
    int GetSurfMapMaxFusedPerFrame();
    int GetSurfMapMaxAddedPerFrame();
    float GetSurfMapColorUpdateMaxResidual();
    float GetSurfMapColorUpdateSoftResidual();

private:
    //Camera parameters
    std::shared_ptr<CameraModel> calibration_;      //Geometric calibration with projection and unprojection functions
    float bf_;                                      //baseline times fx

    std::shared_ptr<Masker> masker_;

    float radPerPixel_;

    Sophus::SE3f left_map_view_, right_map_view_;

    float camera_size_;

    bool autoplay_;

    std::string map_visualizer_save_path_;

    std::string image_visualizer_save_path_;

    std::string evaluation_save_path_;

    // Optional SDF settings.
    float sdf_voxel_size_ = 0.0025f;
    float sdf_truncation_distance_ = 0.0100f;
    int sdf_dense_pixel_stride_ = 2;
    float sdf_dense_weight_ = 1.0f;
    float sdf_sparse_weight_ = 5.0f;
    int sdf_max_voxel_updates_per_frame_ = 300000;
    float sdf_min_depth_ = 0.01f;
    float sdf_max_depth_ = 8.00f;
    int sdf_sparse_min_points_ = 20;

    bool surfel_sdf_constraint_enabled_ = false;
    float surfel_sdf_phi_abs_threshold_ = 0.20f;
    float surfel_sdf_normal_cos_threshold_ = 0.30f;
    bool surfel_sdf_project_to_surface_ = false;

    int surfmap_window_size_ = 8;
    float surfmap_min_depth_ = 0.02f;
    float surfmap_max_depth_ = 1.00f;
    float surfmap_default_radius_ = 0.003f;
    bool surfmap_use_adaptive_radius_ = true;
    int surfmap_sample_step_ = 4;
    int surfmap_max_new_per_frame_ = 10000;
    float surfmap_fusion_distance_threshold_ = 0.01f;
    float surfmap_fusion_normal_threshold_ = 0.85f;
    float surfmap_spatial_search_radius_ = 0.02f;
    float surfmap_active_min_depth_ = 0.02f;
    float surfmap_active_max_depth_ = 0.10f;
    float surfmap_confidence_decay_rate_ = 0.10f;
    float surfmap_min_confidence_ = 0.30f;
    float surfmap_warp_sigma_ = 0.03f;
    float surfmap_warp_max_disp_ = 0.10f;
    std::string surfmap_quality_mode_ = "realtime";
    float surfmap_min_radius_ = 0.001f;
    float surfmap_max_radius_ = 0.006f;
    float surfmap_radius_scale_ = 1.5f;
    float surfmap_max_depth_gradient_ = 0.030f;
    int surfmap_mask_border_pixels_ = 1;
    int surfmap_max_fused_per_frame_ = 0;
    int surfmap_max_added_per_frame_ = 0;
    float surfmap_color_update_max_residual_ = 0.35f;
    float surfmap_color_update_soft_residual_ = 0.15f;

    template<typename T>
    T readParameter(cv::FileStorage& fSettings, const std::string& name, bool& found,const bool required = true){
        cv::FileNode node = fSettings[name];
        if(node.empty()){
            if(required){
                std::cerr << name << " required parameter does not exist, aborting..." << std::endl;
                exit(-1);
            }
            else{
                std::cerr << name << " optional parameter does not exist..." << std::endl;
                found = false;
                return T();
            }

        }
        else{
            found = true;
            return (T) node;
        }
    }

    template<typename T>
    T readOptional(cv::FileStorage& fSettings, const std::string& name, T default_val) {
        bool found;
        T v = readParameter<T>(fSettings, name, found, false);
        return found ? v : default_val;
    }
};


#endif //NRSLAM_SETTINGS_H
