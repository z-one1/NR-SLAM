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

#include "settings.h"

#include "calibration/kannala_brandt_8.h"
#include "calibration/pin_hole.h"
#include "utilities/types_conversions.h"

#include "absl/log/log.h"

using namespace std;

template<>
Sophus::SE3f Settings::readParameter<Sophus::SE3f>(cv::FileStorage& fSettings, const std::string& name, bool& found, const bool required){
    cv::FileNode node = fSettings[name];
    if(node.empty()){
        if(required){
            LOG(ERROR) << name << " required parameter does not exist, aborting...";
            exit(-1);
        }
        else{
            LOG(WARNING) << name << " optional parameter does not exist, aborting...";
            found = false;
            return Sophus::SE3f();
        }
    }
    else{
        found = true;
        cv::Mat cvT = node.mat();

        //Convert to Sophus
        Sophus::SE3f sT = cvToSophus(cvT);

        return sT;
    }
}

template<>
bool Settings::readParameter<bool>(cv::FileStorage& fSettings, const std::string& name, bool& found, const bool required){
    cv::FileNode node = fSettings[name];
    if(node.empty()){
        if(required){
            LOG(ERROR) << name << " required parameter does not exist, aborting...";
            exit(-1);
        }
        else{
            LOG(WARNING) << name << " optional parameter does not exist, aborting...";
            found = false;
            return false;
        }
    }
    else{
        found = true;
        int value = (int) node;

        //Convert to bool
        bool return_value = (bool) value;

        return return_value;
    }
}

Settings::Settings(){}

Settings::Settings(const std::string& configFile) {
    //Open settings file
    cv::FileStorage fSettings(configFile, cv::FileStorage::READ);
    if(!fSettings.isOpened()){
        LOG(ERROR) << "[ERROR]: could not open configuration file at: " << configFile;
        exit(-1);
    }

    //Read camera model
    string cameraModel = (string) fSettings["Camera.model"];
    vector<float> vCalibration;
    if(cameraModel == "PinHole"){
        //Read camera calibration
        float fx = fSettings["Camera.fx"];
        float fy = fSettings["Camera.fy"];
        float cx = fSettings["Camera.cx"];
        float cy = fSettings["Camera.cy"];

        vCalibration = {fx,fy,cx,cy};

        calibration_ = shared_ptr<CameraModel>(new PinHole(vCalibration));
    }
    else if(cameraModel == "KannalaBrandt8"){
        float fx = fSettings["Camera.fx"];
        float fy = fSettings["Camera.fy"];
        float cx = fSettings["Camera.cx"];
        float cy = fSettings["Camera.cy"];

        float k0 = fSettings["Camera.k0"];
        float k1 = fSettings["Camera.k1"];
        float k2 = fSettings["Camera.k2"];
        float k3 = fSettings["Camera.k3"];

        vCalibration = {fx,fy,cx,cy,k0,k1,k2,k3};

        calibration_ = shared_ptr<CameraModel>(new KannalaBrandt8(vCalibration));
    }
    else{
        LOG(ERROR) << "Error: " << cameraModel << " not known";
        exit(-1);
    }

    bf_ = fSettings["Stereo.bf"];

    // Camera drawing size.
    camera_size_ = (float) fSettings["Visualization.cameraSize"];

    //Read filter for mask generation
    string filterFile = (string) fSettings["Masking.filterFile"];
    masker_ = std::make_shared<Masker>();
    masker_->loadFromTxt(filterFile);
    LOG(INFO) << masker_->printFilters();

    radPerPixel_ = (float) fSettings["Camera.radiansPerPixel"];

    //Read Map Visualization initial view points
    bool found;
    left_map_view_ = readParameter<Sophus::SE3f>(fSettings,"MapVisualizer.left_view",found,true);
    if(!found){
        LOG(ERROR) << "Parameter MapVisualizer.left_view not found";
        exit(-1);
    }

    right_map_view_ = readParameter<Sophus::SE3f>(fSettings,"MapVisualizer.right_view",found,true);
    if(!found){
        LOG(ERROR) << "Parameter MapVisualizer.right_view not found";
        exit(-1);
    }

    autoplay_ = readParameter<bool>(fSettings, "System.autoplay", found, true);
    if(!found){
        LOG(ERROR) << "Parameter System.autoplay not found";
        exit(-1);
    }

    map_visualizer_save_path_ = readParameter<string>(fSettings, "MapVisualizer.save_path", found, true);
    if(!found){
        LOG(ERROR) << "Parameter MapVisualizer.save_path not found";
        exit(-1);
    }

    image_visualizer_save_path_ = readParameter<string>(fSettings, "ImageVisualizer.save_path", found, true);
    if(!found){
        LOG(ERROR) << "Parameter ImageVisualizer.save_path not found";
        exit(-1);
    }

    evaluation_save_path_ = readParameter<string>(fSettings, "Evaluation.save_path", found, true);
    if(!found){
        LOG(ERROR) << "Parameter Evaluation.save_path not found";
        exit(-1);
    }

    sdf_voxel_size_                  = readOptional<float>(fSettings, "SDF.voxel_size", 0.0025f);
    sdf_truncation_distance_         = readOptional<float>(fSettings, "SDF.truncation_distance", 0.0100f);
    sdf_dense_pixel_stride_          = readOptional<int>(fSettings, "SDF.dense_pixel_stride", 2);
    sdf_dense_weight_                = readOptional<float>(fSettings, "SDF.dense_weight", 1.0f);
    sdf_sparse_weight_               = readOptional<float>(fSettings, "SDF.sparse_weight", 5.0f);
    sdf_max_voxel_updates_per_frame_ = readOptional<int>(fSettings, "SDF.max_voxel_updates_per_frame", 300000);
    sdf_min_depth_                   = readOptional<float>(fSettings, "SDF.min_depth", 0.01f);
    sdf_max_depth_                   = readOptional<float>(fSettings, "SDF.max_depth", 8.00f);
    sdf_sparse_min_points_           = readOptional<int>(fSettings, "SDF.sparse_min_points", 20);

    surfel_sdf_constraint_enabled_   = readOptional<bool>(fSettings, "SurfelSDF.constraint_enabled", false);
    surfel_sdf_phi_abs_threshold_    = readOptional<float>(fSettings, "SurfelSDF.phi_abs_threshold", 0.20f);
    surfel_sdf_normal_cos_threshold_ = readOptional<float>(fSettings, "SurfelSDF.normal_cos_threshold", 0.30f);
    surfel_sdf_project_to_surface_   = readOptional<bool>(fSettings, "SurfelSDF.project_to_surface", false);

    surfmap_window_size_ = readOptional<int>(fSettings, "SurfMap.window_size", 8);
    surfmap_min_depth_ = readOptional<float>(fSettings, "SurfMap.min_depth", 0.02f);
    surfmap_max_depth_ = readOptional<float>(fSettings, "SurfMap.max_depth", 1.00f);
    surfmap_default_radius_ = readOptional<float>(fSettings, "SurfMap.default_surfel_radius", 0.003f);
    surfmap_use_adaptive_radius_ = readOptional<bool>(fSettings, "SurfMap.use_adaptive_radius", true);
    surfmap_sample_step_ = readOptional<int>(fSettings, "SurfMap.surfel_sample_step", 4);
    surfmap_max_new_per_frame_ = readOptional<int>(fSettings, "SurfMap.max_new_surfels_per_frame", 10000);
    surfmap_fusion_distance_threshold_ = readOptional<float>(fSettings, "SurfMap.fusion_distance_threshold", 0.01f);
    surfmap_fusion_normal_threshold_ = readOptional<float>(fSettings, "SurfMap.fusion_normal_threshold", 0.85f);
    surfmap_spatial_search_radius_ = readOptional<float>(fSettings, "SurfMap.spatial_search_radius", 0.02f);
    surfmap_active_min_depth_ = readOptional<float>(fSettings, "SurfMap.active_min_depth", 0.02f);
    surfmap_active_max_depth_ = readOptional<float>(fSettings, "SurfMap.active_max_depth", 0.10f);
    surfmap_confidence_decay_rate_ = readOptional<float>(fSettings, "SurfMap.confidence_decay_rate", 0.10f);
    surfmap_min_confidence_ = readOptional<float>(fSettings, "SurfMap.min_surfel_confidence", 0.30f);
    surfmap_warp_sigma_ = readOptional<float>(fSettings, "SurfMap.warp_sigma", 0.03f);
    surfmap_warp_max_disp_ = readOptional<float>(fSettings, "SurfMap.warp_max_disp", 0.10f);
    surfmap_quality_mode_ = readOptional<string>(fSettings, "SurfMap.quality_mode", string("realtime"));
    surfmap_min_radius_ = readOptional<float>(fSettings, "SurfMap.min_surfel_radius", 0.001f);
    surfmap_max_radius_ = readOptional<float>(fSettings, "SurfMap.max_surfel_radius", 0.006f);
    surfmap_radius_scale_ = readOptional<float>(fSettings, "SurfMap.radius_scale", 1.5f);
    surfmap_max_depth_gradient_ = readOptional<float>(fSettings, "SurfMap.max_depth_gradient", 0.030f);
    surfmap_mask_border_pixels_ = readOptional<int>(fSettings, "SurfMap.mask_border_pixels", 1);
    surfmap_max_fused_per_frame_ = readOptional<int>(fSettings, "SurfMap.max_fused_surfels_per_frame", 0);
    surfmap_max_added_per_frame_ = readOptional<int>(fSettings, "SurfMap.max_added_surfels_per_frame", 0);
    surfmap_color_update_max_residual_ = readOptional<float>(fSettings, "SurfMap.color_update_max_residual", 0.35f);
    surfmap_color_update_soft_residual_ = readOptional<float>(fSettings, "SurfMap.color_update_soft_residual", 0.15f);
}

ostream &operator<<(std::ostream& output, const Settings& settings){
    output << "SLAM settings: " << endl;

    output << "\t-Camera parameters: [ ";
    output << settings.calibration_->GetParameter(0) << " , " << settings.calibration_->GetParameter(1) << " , ";
    output << settings.calibration_->GetParameter(2) << " , " << settings.calibration_->GetParameter(3) << " ]" << endl;

    output << "\t-Visualization settings:" << endl;
    output << "\t\t-[MapVisualizer] camera size: " << settings.camera_size_ << endl;

    return output;
}

std::shared_ptr<CameraModel> Settings::getCalibration() {
    return calibration_;
}

float Settings::getBf() {
    return bf_;
}

std::shared_ptr<Masker> Settings::getMasker() {
    return masker_;
}

float Settings::getRadPerPixel() {
    return radPerPixel_;
}

Sophus::SE3f Settings::GetLeftMapVisualizationView() {
    return left_map_view_;
}

Sophus::SE3f Settings::GetRightMapVisualizationView() {
    return right_map_view_;
}

float Settings::GetCameraSize() {
    return camera_size_;
}

bool Settings::GetAutoplay() {
    return autoplay_;
}

std::string Settings::GetMapVisualizerPath() {
    return map_visualizer_save_path_;
}

std::string Settings::GetImageVisualizerPath() {
    return image_visualizer_save_path_;
}

std::string Settings::GetEvaluationPath() {
    return evaluation_save_path_;
}

float Settings::GetSDFVoxelSize() {
    return sdf_voxel_size_;
}

float Settings::GetSDFTruncationDistance() {
    return sdf_truncation_distance_;
}

int Settings::GetSDFDensePixelStride() {
    return sdf_dense_pixel_stride_;
}

float Settings::GetSDFDenseWeight() {
    return sdf_dense_weight_;
}

float Settings::GetSDFSparseWeight() {
    return sdf_sparse_weight_;
}

int Settings::GetSDFMaxVoxelUpdatesPerFrame() {
    return sdf_max_voxel_updates_per_frame_;
}

float Settings::GetSDFMinDepth() {
    return sdf_min_depth_;
}

float Settings::GetSDFMaxDepth() {
    return sdf_max_depth_;
}

int Settings::GetSDFSparseMinPoints() {
    return sdf_sparse_min_points_;
}

bool Settings::GetSurfelSDFConstraintEnabled() {
    return surfel_sdf_constraint_enabled_;
}

float Settings::GetSurfelSDFPhiAbsThreshold() {
    return surfel_sdf_phi_abs_threshold_;
}

float Settings::GetSurfelSDFNormalCosThreshold() {
    return surfel_sdf_normal_cos_threshold_;
}

bool Settings::GetSurfelSDFProjectToSurface() {
    return surfel_sdf_project_to_surface_;
}

int Settings::GetSurfMapWindowSize() {
    return surfmap_window_size_;
}

float Settings::GetSurfMapMinDepth() {
    return surfmap_min_depth_;
}

float Settings::GetSurfMapMaxDepth() {
    return surfmap_max_depth_;
}

float Settings::GetSurfMapDefaultRadius() {
    return surfmap_default_radius_;
}

bool Settings::GetSurfMapUseAdaptiveRadius() {
    return surfmap_use_adaptive_radius_;
}

int Settings::GetSurfMapSampleStep() {
    return surfmap_sample_step_;
}

int Settings::GetSurfMapMaxNewPerFrame() {
    return surfmap_max_new_per_frame_;
}

float Settings::GetSurfMapFusionDistanceThreshold() {
    return surfmap_fusion_distance_threshold_;
}

float Settings::GetSurfMapFusionNormalThreshold() {
    return surfmap_fusion_normal_threshold_;
}

float Settings::GetSurfMapSpatialSearchRadius() {
    return surfmap_spatial_search_radius_;
}

float Settings::GetSurfMapActiveMinDepth() {
    return surfmap_active_min_depth_;
}

float Settings::GetSurfMapActiveMaxDepth() {
    return surfmap_active_max_depth_;
}

float Settings::GetSurfMapConfidenceDecayRate() {
    return surfmap_confidence_decay_rate_;
}

float Settings::GetSurfMapMinConfidence() {
    return surfmap_min_confidence_;
}

float Settings::GetSurfMapWarpSigma() {
    return surfmap_warp_sigma_;
}

float Settings::GetSurfMapWarpMaxDisp() {
    return surfmap_warp_max_disp_;
}

std::string Settings::GetSurfMapQualityMode() {
    return surfmap_quality_mode_;
}

float Settings::GetSurfMapMinRadius() {
    return surfmap_min_radius_;
}

float Settings::GetSurfMapMaxRadius() {
    return surfmap_max_radius_;
}

float Settings::GetSurfMapRadiusScale() {
    return surfmap_radius_scale_;
}

float Settings::GetSurfMapMaxDepthGradient() {
    return surfmap_max_depth_gradient_;
}

int Settings::GetSurfMapMaskBorderPixels() {
    return surfmap_mask_border_pixels_;
}

int Settings::GetSurfMapMaxFusedPerFrame() {
    return surfmap_max_fused_per_frame_;
}

int Settings::GetSurfMapMaxAddedPerFrame() {
    return surfmap_max_added_per_frame_;
}

float Settings::GetSurfMapColorUpdateMaxResidual() {
    return surfmap_color_update_max_residual_;
}

float Settings::GetSurfMapColorUpdateSoftResidual() {
    return surfmap_color_update_soft_residual_;
}
