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

#ifndef NRSLAM_SYSTEM_H
#define NRSLAM_SYSTEM_H


#include <memory>
#include <thread>

#include <fstream>
#include <iomanip>
#include <Eigen/Geometry>

#include "map/map.h"
#include "mapping/mapping.h"
#include "SLAM/settings.h"
#include "stereo/stereo_lucas_kanade.h"
#include "stereo/stereo_pattern_matching.h"
#include "tracking/tracking.h"
#include "utilities/frame_evaluator.h"
#include "utilities/time_profiler.h"
#include "visualization/image_visualizer.h"
#include "visualization/map_visualizer.h"

class System {
public:
    System() = delete;

    System(const std::string settings_file_path);

    ~System();

    // Tracks an image using the monocular pipeline
    void TrackImage(const cv::Mat& im);

    // Tracks the next image using some stereo information:
    //  - For stereo map initialization
    //  - Reconstruction evaluation
    // This is controlled by the settings passed to the system
    // TODO: implement stereo options
    void TrackImageWithStereo(const cv::Mat& im_left, const cv::Mat& im_right);

    // Tracks the next image using a precomputed depth image.
    void TrackImageWithDepth(const cv::Mat& im_left, const cv::Mat& im_depth);

    // Dense Tracking using Monocular Depth Estimation NN
    void TrackImageWithNN(const cv::Mat& im_left, const cv::Mat& im_depth, const cv::Mat& im_nn);

    vector<Eigen::Vector3f> GetTraj();

    void SaveTraj();

    bool ExportSurfMap(const std::string& filename,
                       bool only_finalized = false);

    bool ExportSDF(const std::string& filename,
                   float max_abs_tsdf = 0.10f);

    Tracking::RuntimeMetrics GetRuntimeMetrics() const;

private:
    // Applies preprocessing to the input image (CLAHE, etc).
    cv::Mat ImageProcessing(const cv::Mat& im, cv::Mat& im_gray);

    std::shared_ptr<Map> map_;

    std::unique_ptr<Tracking> tracker_;

    std::unique_ptr<Mapping> mapper_;

    std::unique_ptr<Settings> settings_;

    cv::Ptr<cv::CLAHE> clahe_;
    std::shared_ptr<Masker> masker_;

    // std::shared_ptr<StereoPatternMatching> stereo_matcher_;
    std::shared_ptr<StereoLucasKanade> stereo_matcher_;
    std::shared_ptr<StereoPatternMatching> stereo_pattern_matcher_;

    std::unique_ptr<MapVisualizer> map_visualizer_;
    std::unique_ptr<std::thread> map_visualizer_thread_;

    std::shared_ptr<ImageVisualizer> image_visualizer_;

    std::unique_ptr<FrameEvaluator> frame_evaluator_;

    std::unique_ptr<TimeProfiler> time_profiler_;

    // === 新增：用来控制每帧 surfmap 导出 ===
    size_t surf_export_frame_idx_ = 0;   // 当前是第几帧
    int    surf_export_step_      = 1;   // 每多少帧导出一次：1=每帧, 5=每5帧
    bool   surf_export_enabled_   = true;
};

#endif //NRSLAM_SYSTEM_H
