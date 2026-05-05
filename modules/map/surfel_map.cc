#include "surfel_map.h"
#include "sdf/sdf_builder.h"
#include <algorithm>
#include <cmath>
#include <unordered_set>
#include "absl/log/log.h"

#include <fstream>   // 确保已经有这个

namespace {

// 把一个 surfel 向量导出成 PLY 文件（只点+法线+颜色）
void ExportSurfelVectorToPLY(const std::vector<std::shared_ptr<Surfel>>& surfels,
                             const std::string& filename)
{
    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        LOG(WARNING) << "Cannot open file " << filename << " to export single-frame surfels.";
        return;
    }

    const size_t N = surfels.size();

    ofs << "ply\n";
    ofs << "format ascii 1.0\n";
    ofs << "element vertex " << N << "\n";
    ofs << "property float x\n";
    ofs << "property float y\n";
    ofs << "property float z\n";
    ofs << "property float nx\n";
    ofs << "property float ny\n";
    ofs << "property float nz\n";
    ofs << "property uchar red\n";
    ofs << "property uchar green\n";
    ofs << "property uchar blue\n";
    ofs << "end_header\n";

    for (const auto& s : surfels) {
        const Eigen::Vector3f& p = s->position;
        const Eigen::Vector3f& n = s->normal;
        Eigen::Vector3f c = s->color.cwiseMax(0.0f).cwiseMin(1.0f);
        int r = static_cast<int>(std::round(c.x() * 255.0f));
        int g = static_cast<int>(std::round(c.y() * 255.0f));
        int b = static_cast<int>(std::round(c.z() * 255.0f));

        ofs << p.x() << " " << p.y() << " " << p.z() << " "
            << n.x() << " " << n.y() << " " << n.z() << " "
            << r << " " << g << " " << b << "\n";
    }

    ofs.close();
    LOG(INFO) << "Exported single-frame surfels to " << filename
              << " with " << N << " points.";
}

} // namespace


// ============ 工具函数 ============ //

namespace {

inline bool IsValidDepth(float z, float min_z, float max_z) {
    return std::isfinite(z) && z > min_z && z < max_z;
}

} // namespace

// ============ SurfMap 实现 ============ //

SurfMap::SurfMap(const SurfConfig& config)
    : config_(config), grid_cell_size_(config.spatial_search_radius) {
}

SurfMap::~SurfMap() = default;

void SurfMap::UpdateConfig(const SurfConfig& new_config) {
    if (std::abs(grid_cell_size_ - new_config.spatial_search_radius) > 1e-6f) {
        grid_cell_size_ = new_config.spatial_search_radius;
        // cell 尺寸变了，旧索引全部失效，从 active_surfels_ 重建
        spatial_grid_.clear();
        for (const auto& kv : active_surfels_) {
            InsertSurfelToGrid(kv.first, kv.second->position);
        }
    }
    config_ = new_config;
}

SurfConfig SurfMap::GetConfig() const {
    return config_;
}

void SurfMap::SetSDFBuilder(const SDFBuilder* sdf_builder) {
    sdf_builder_ = sdf_builder;
}

void SurfMap::BeginFrame(uint64_t frame_id) {
    phi_stats_ = SurfelPhiStats();
    phi_stats_.frame_id = frame_id;
    const int previous_warped = fusion_stats_.warped;
    const float previous_mean_radius = fusion_stats_.mean_radius;
    const float previous_mean_confidence = fusion_stats_.mean_confidence;
    fusion_stats_ = SurfelFusionStats();
    fusion_stats_.frame_id = frame_id;
    fusion_stats_.warped = previous_warped;
    fusion_stats_.active = active_surfels_.size();
    fusion_stats_.finalized_total = finalized_surfels_.size();
    fusion_stats_.mean_radius = previous_mean_radius;
    fusion_stats_.mean_confidence = previous_mean_confidence;
}

void SurfMap::UpdateMeanMapStats() {
    if (active_surfels_.empty()) {
        fusion_stats_.mean_radius = 0.0f;
        fusion_stats_.mean_confidence = 0.0f;
        fusion_stats_.active = 0;
        fusion_stats_.finalized_total = finalized_surfels_.size();
        return;
    }

    double radius_sum = 0.0;
    double confidence_sum = 0.0;
    for (const auto& kv : active_surfels_) {
        radius_sum += kv.second->radius;
        confidence_sum += kv.second->confidence;
    }

    fusion_stats_.active = active_surfels_.size();
    fusion_stats_.finalized_total = finalized_surfels_.size();
    fusion_stats_.mean_radius = static_cast<float>(radius_sum / active_surfels_.size());
    fusion_stats_.mean_confidence = static_cast<float>(confidence_sum / active_surfels_.size());
}

void SurfMap::IntegrateFrame(const cv::Mat& color_image,
                             const cv::Mat& estimated_depth,
                             const Eigen::Matrix4f& camera_pose,
                             const Eigen::Matrix3f& camera_intrinsics,
                             uint64_t frame_id,
                             const cv::Mat& valid_mask)
{
    if (fusion_stats_.frame_id != frame_id) {
        BeginFrame(frame_id);
    }

    if (color_image.empty() || estimated_depth.empty()) return;
    if (color_image.size() != estimated_depth.size()) return;

    // 1. 记录一下这个 frame（可选，仅用于 debug / 可视化）
    auto kf = std::make_shared<SurfKeyFrame>();
    kf->id              = frame_id;
    kf->color_image     = color_image.clone();
    kf->estimated_depth = estimated_depth.clone();
    kf->pose            = camera_pose;
    keyframe_window_.push_back(kf);
    if ((int)keyframe_window_.size() > config_.window_size) {
        RemoveOldestKeyFrame();
    }

    // 2. 从当前帧生成 surfel 候选
    auto new_surfels = GenerateSurfelsFromKeyFrame(
        color_image,
        estimated_depth,
        valid_mask,
        camera_pose,
        camera_intrinsics,
        frame_id);

    LOG(INFO) << "GenerateSurfelsFromKeyFrame: " << new_surfels.size()
              << " surfels generated from frame " << frame_id;

    // 3. 融合到 active，限制每帧最多新建 N 个（防止太 dense 爆炸）
    FuseNewSurfelsToActiveMap(new_surfels, frame_id, frame_id);

    // 4. 更新生命周期：把已经被甩在身后的 surfel finalize 掉
    UpdateSurfelLifecycle(frame_id, camera_pose);

    LOG(INFO) << "SurfMap::IntegrateFrame: active=" << active_surfels_.size()
              << ", finalized=" << finalized_surfels_.size();
    fusion_stats_.active = active_surfels_.size();
    fusion_stats_.finalized_total = finalized_surfels_.size();
}



// ============ Surfel 生成 ============ //

float SurfMap::ComputeSurfelRadius(float depth,
                                   const Eigen::Vector3f& normal_cam,
                                   const Eigen::Matrix3f& intrinsics) const {
    const float fx = intrinsics(0, 0);
    const float fy = intrinsics(1, 1);
    const float focal = std::max(1e-6f, std::min(fx, fy));
    const float step = static_cast<float>(std::max(1, config_.surfel_sample_step));
    const float pixel_footprint = depth * step / focal;

    float angle_factor = 1.0f;
    const float nz = std::abs(normal_cam.z());
    if (std::isfinite(nz) && nz > 1e-3f) {
        angle_factor = std::min(2.0f, 1.0f / nz);
    }

    const float unclamped = config_.surfel_radius_scale * pixel_footprint * angle_factor;
    const float min_r = std::max(1e-6f, config_.min_surfel_radius);
    const float max_r = std::max(min_r, config_.max_surfel_radius);
    return std::min(max_r, std::max(min_r, unclamped));
}

bool SurfMap::IsMaskBorder(const cv::Mat& valid_mask, int u, int v) const {
    if (valid_mask.empty() || valid_mask.type() != CV_8UC1) return false;
    const int border = std::max(0, config_.mask_border_pixels);

    for (int dy = -border; dy <= border; ++dy) {
        for (int dx = -border; dx <= border; ++dx) {
            const int x = u + dx;
            const int y = v + dy;
            if (x < 0 || x >= valid_mask.cols || y < 0 || y >= valid_mask.rows) {
                return true;
            }
            if (valid_mask.at<uint8_t>(y, x) == 0) {
                return true;
            }
        }
    }
    return false;
}

bool SurfMap::PassDepthGradientFilter(const cv::Mat& depth_map,
                                      int u,
                                      int v,
                                      float center_depth) const {
    if (config_.max_depth_gradient <= 0.0f) return true;
    if (u <= 0 || u >= depth_map.cols - 1 || v <= 0 || v >= depth_map.rows - 1) return false;

    const float zx = depth_map.at<float>(v, u + 1);
    const float zy = depth_map.at<float>(v + 1, u);
    if (!IsValidDepth(zx, config_.min_depth, config_.max_depth) ||
        !IsValidDepth(zy, config_.min_depth, config_.max_depth)) {
        return false;
    }

    const float dzx = std::abs(zx - center_depth);
    const float dzy = std::abs(zy - center_depth);
    if (!std::isfinite(dzx) || !std::isfinite(dzy)) return false;
    return dzx <= config_.max_depth_gradient && dzy <= config_.max_depth_gradient;
}

std::vector<std::shared_ptr<Surfel>> SurfMap::GenerateSurfelsFromKeyFrame(
    const cv::Mat& color_image,
    const cv::Mat& depth_map,
    const cv::Mat& valid_mask,
    const Eigen::Matrix4f& pose,
    const Eigen::Matrix3f& K,
    uint64_t frame_id) {

    std::vector<std::shared_ptr<Surfel>> result;
    const int W = depth_map.cols;
    const int H = depth_map.rows;

    const float fx = K(0,0);
    const float fy = K(1,1);
    const float cx = K(0,2);
    const float cy = K(1,2);

    // 世界坐标变换
    Eigen::Matrix3f R_wc = pose.block<3,3>(0,0);
    Eigen::Vector3f t_wc = pose.block<3,1>(0,3);

    auto backproject = [&](int u, int v, float z) -> Eigen::Vector3f {
        float x = (u - cx) * z / fx;
        float y = (v - cy) * z / fy;
        return Eigen::Vector3f(x, y, z);
    };

    // 法线估计：使用深度邻域梯度（相机坐标系）
    auto estimate_normal_cam = [&](int u, int v) -> Eigen::Vector3f {
        if (u <= 0 || u >= W-1 || v <= 0 || v >= H-1) {
            return Eigen::Vector3f::Zero();
        }

        float zc = depth_map.at<float>(v,   u);
        float zx = depth_map.at<float>(v,   u+1);
        float zy = depth_map.at<float>(v+1, u);

        // ★ 不再用 config_.min_depth / max_depth，只要 >0 且 finite
        auto valid = [](float z) {
            return std::isfinite(z) && z > 0.0f;
        };

        if (!valid(zc) || !valid(zx) || !valid(zy)) {
            return Eigen::Vector3f::Zero();
        }

        Eigen::Vector3f pc = backproject(u,   v,   zc);
        Eigen::Vector3f px = backproject(u+1, v,   zx);
        Eigen::Vector3f py = backproject(u,   v+1, zy);

        Eigen::Vector3f vx = px - pc;
        Eigen::Vector3f vy = py - pc;
        Eigen::Vector3f n  = vx.cross(vy);
        if (n.norm() < 1e-6f) return Eigen::Vector3f::Zero();
        n.normalize();

        // 朝向相机
        if (n.z() > 0) n = -n;
        return n;
    };


    const int step = std::max(1, config_.surfel_sample_step);
    const bool use_valid_mask = !valid_mask.empty() &&
                                valid_mask.type() == CV_8UC1 &&
                                valid_mask.size() == depth_map.size();

    for (int v = 0; v < H; v += step) {
        const float* depth_row = depth_map.ptr<float>(v);
        for (int u = 0; u < W; u += step) {
            float depth = depth_row[u];
            fusion_stats_.generated++;

            if (use_valid_mask && valid_mask.at<uint8_t>(v, u) == 0) {
                fusion_stats_.rejected_by_mask++;
                continue;
            }

            if (use_valid_mask && IsMaskBorder(valid_mask, u, v)) {
                fusion_stats_.rejected_by_mask++;
                continue;
            }

            if (!IsValidDepth(depth, config_.min_depth, config_.max_depth)) {
                fusion_stats_.rejected_by_depth++;
                continue;
            }

            if (!PassDepthGradientFilter(depth_map, u, v, depth)) {
                fusion_stats_.rejected_by_gradient++;
                continue;
            }

            // 相机坐标
            Eigen::Vector3f pc = backproject(u, v, depth);
            // 世界坐标
            Eigen::Vector3f pw = R_wc * pc + t_wc;

            // 法线（相机 -> 世界）
            Eigen::Vector3f nc = estimate_normal_cam(u, v);
            if (nc.norm() < config_.min_normal_norm) {
                fusion_stats_.rejected_by_normal++;
                continue;
            }
            Eigen::Vector3f nw = R_wc * nc;

            // 颜色（OpenCV默认BGR）
            Eigen::Vector3f color(0.5f, 0.5f, 0.5f);
            if (!color_image.empty() && color_image.type() == CV_8UC3) {
                const cv::Vec3b& pix = color_image.at<cv::Vec3b>(v, u);
                color = Eigen::Vector3f(pix[2], pix[1], pix[0]) / 255.0f;
            }

            // 半径
            float radius = config_.default_surfel_radius;
            if (config_.use_adaptive_radius) {
                radius = ComputeSurfelRadius(depth, nc, K);
            }

            // 注意：这里用临时 id=0，真正插入 active_surfels_ 时再赋 id
            auto surfel = std::make_shared<Surfel>(0, frame_id);
            surfel->position  = pw;
            surfel->normal    = nw;
            surfel->color     = color;
            surfel->radius    = radius;
            surfel->confidence= 1.0f;
            surfel->observation_count   = 1;
            surfel->first_observed_frame= frame_id;
            surfel->last_observed_frame = frame_id;
            surfel->state     = Surfel::ACTIVE;

            fusion_stats_.accepted++;
            result.push_back(surfel);
        }
    }

    return result;
}

// ============ 融合到 active 地图 ============ //

void SurfMap::FuseNewSurfelsToActiveMap(
    const std::vector<std::shared_ptr<Surfel>>& new_surfels,
    uint64_t source_keyframe_id,
    uint64_t frame_id)
{
    size_t added = 0;
    size_t fused = 0;
    size_t rejected_by_sdf = 0;
    const size_t legacy_total_budget = config_.max_new_surfels_per_frame;
    const size_t max_fused = config_.max_fused_surfels_per_frame > 0
        ? config_.max_fused_surfels_per_frame
        : legacy_total_budget;
    const size_t max_added = config_.max_added_surfels_per_frame > 0
        ? config_.max_added_surfels_per_frame
        : legacy_total_budget;

    for (const auto& s_new : new_surfels) {
        if (legacy_total_budget > 0 && (added + fused) >= legacy_total_budget) break;

        auto constrained = s_new;
        float phi_before = 0.0f, phi_after = 0.0f;
        if (!ApplySDFConstraint(constrained, &phi_before, &phi_after)) {
            rejected_by_sdf++;
            continue;  // SDF约束拒绝此候选
        }

        // 记录来源帧
        constrained->observed_keyframes.push_back(source_keyframe_id);

        // ★ 融合优先工作流：尝试与邻居融合
        auto neighbors = FindNeighbors(constrained->position);
        auto best_match = FindBestMatch(*constrained, neighbors);

        if (best_match != nullptr) {
            if (max_fused == 0 || fused < max_fused) {
                FuseSurfelPair(best_match, constrained, frame_id);
                fused++;
            }
            continue;
        }

        if (max_added > 0 && added >= max_added) {
            continue;
        }

        constrained->id = next_surfel_id_++;
        constrained->state = Surfel::ACTIVE;
        constrained->last_observed_frame = frame_id;
        active_surfels_[constrained->id] = constrained;
        InsertSurfelToGrid(constrained->id, constrained->position);
        added++;
    }

    LOG(INFO) << "FuseNewSurfelsToActiveMap (fusion-first): added " << added
              << " new, fused " << fused << ", total active=" << active_surfels_.size()
              << " [phi_stats: checked=" << phi_stats_.constraint_checked
              << " accepted=" << phi_stats_.constraint_accepted
              << " rejected=" << phi_stats_.constraint_rejected
              << " projected=" << phi_stats_.projected_count << "]";
    fusion_stats_.added = static_cast<int>(added);
    fusion_stats_.fused = static_cast<int>(fused);
    if (fusion_stats_.fused > 0) {
        fusion_stats_.mean_color_residual /= static_cast<float>(fusion_stats_.fused);
    } else {
        fusion_stats_.mean_color_residual = 0.0f;
    }
    fusion_stats_.rejected_by_sdf = static_cast<int>(rejected_by_sdf);
    fusion_stats_.active = active_surfels_.size();
    fusion_stats_.finalized_total = finalized_surfels_.size();
    UpdateMeanMapStats();
}

bool SurfMap::ApplySDFConstraint(std::shared_ptr<Surfel>& surfel, float* phi_before, float* phi_after) const {
    if (!config_.sdf_constraint_enabled || sdf_builder_ == nullptr) {
        return true;
    }

    float phi = 0.0f;
    if (!sdf_builder_->QueryPhi(surfel->position, phi)) {
        // If local TSDF neighborhood is unavailable, do not hard reject.
        if (phi_before) *phi_before = 0.0f;
        if (phi_after) *phi_after = 0.0f;
        return true;
    }

    if (phi_before) *phi_before = phi;
    phi_stats_.constraint_checked++;

    // phi_before 分布统计：在 reject 之前记录，覆盖所有被检查的 surfel
    const float abs_phi = std::abs(phi);
    if (phi_stats_.constraint_checked == 1) {
        phi_stats_.phi_before_min = phi_stats_.phi_before_max = abs_phi;
    } else {
        phi_stats_.phi_before_min = std::min(phi_stats_.phi_before_min, abs_phi);
        phi_stats_.phi_before_max = std::max(phi_stats_.phi_before_max, abs_phi);
    }

    if (abs_phi > config_.sdf_phi_abs_threshold) {
        phi_stats_.constraint_rejected++;
        return false;
    }

    Eigen::Vector3f gradient = Eigen::Vector3f::Zero();
    if (!sdf_builder_->QueryGradient(surfel->position, gradient)) {
        if (phi_after) *phi_after = phi;
        phi_stats_.constraint_accepted++;
        return true;
    }

    const float gnorm = gradient.norm();
    if (gnorm < 1e-6f) {
        if (phi_after) *phi_after = phi;
        phi_stats_.constraint_accepted++;
        return true;
    }

    const Eigen::Vector3f normal_from_sdf = gradient / gnorm;
    const float cos_sim = std::abs(normal_from_sdf.dot(surfel->normal.normalized()));
    if (cos_sim < config_.sdf_normal_cos_threshold) {
        phi_stats_.constraint_rejected++;
        return false;
    }

    const int accepted_so_far = phi_stats_.constraint_accepted + phi_stats_.projected_count;
    if (config_.sdf_project_to_surface) {
        surfel->position -= phi * normal_from_sdf;
        surfel->normal = normal_from_sdf;
        phi_stats_.projected_count++;
        if (phi_after) *phi_after = 0.0f;
        // 投影后 phi ≈ 0，phi_after 统计归零
        if (accepted_so_far == 0) {
            phi_stats_.phi_after_min = phi_stats_.phi_after_max = 0.0f;
        }
        // phi_after_min 维持 0，phi_after_max 维持 0（投影后理论准确）
    } else {
        if (phi_after) *phi_after = phi;
        if (accepted_so_far == 0) {
            phi_stats_.phi_after_min = phi_stats_.phi_after_max = abs_phi;
        } else {
            phi_stats_.phi_after_min = std::min(phi_stats_.phi_after_min, abs_phi);
            phi_stats_.phi_after_max = std::max(phi_stats_.phi_after_max, abs_phi);
        }
    }

    phi_stats_.constraint_accepted++;
    return true;
}

// ============ 滑动窗口管理 ============ //

void SurfMap::RemoveOldestKeyFrame() {
    if (keyframe_window_.empty()) return;
    keyframe_window_.pop_front();
}

// ============ 生命周期管理 ============ //

void SurfMap::UpdateSurfelLifecycle(uint64_t current_frame_id,
                                    const Eigen::Matrix4f& current_pose)
{
    Eigen::Matrix4f T_cw = current_pose.inverse();
    float active_min_z   = config_.active_min_depth;
    float active_max_z   = config_.active_max_depth;
    uint64_t max_age     = static_cast<uint64_t>(config_.window_size);

    std::vector<uint64_t> to_remove;
    to_remove.reserve(active_surfels_.size());

    for (auto& kv : active_surfels_) {
        auto& s = kv.second;

        // 当前这一帧刚观测到，不做衰减
        if (s->last_observed_frame == current_frame_id) {
            continue;
        }

        // 1) 年龄
        uint64_t age = current_frame_id - s->last_observed_frame;

        // 2) 在当前相机下的深度
        Eigen::Vector4f pw(s->position.x(), s->position.y(), s->position.z(), 1.0f);
        Eigen::Vector4f pc = T_cw * pw;
        float z = pc.z();  // 相机前为正，后为负

        // 3) 置信度衰减（仅作软指标）
        s->confidence *= (1.0f - config_.confidence_decay_rate);
        if (s->confidence < 0.0f) s->confidence = 0.0f;

        bool behind_camera = (z < 0.0f);                  // 已在相机后面
        bool out_of_view   = (z < active_min_z) || (z > active_max_z);
        bool too_old       = (age > max_age);

        if ( (behind_camera || out_of_view) && too_old ) {
            // 已经走远 & 很久没看到了 -> 冻结成历史管腔
            RemoveSurfelFromGrid(s->id, s->position);
            s->state = Surfel::FINALIZED;
            finalized_surfels_[s->id] = s;
            to_remove.push_back(s->id);
        }
    }

    for (auto id : to_remove) {
        active_surfels_.erase(id);
    }
    fusion_stats_.finalized = static_cast<int>(to_remove.size());
    UpdateMeanMapStats();
}

// ============ 查询 & 邻域搜索 ============ //

std::vector<std::shared_ptr<Surfel>> SurfMap::GetActiveSurfels() const {
    std::vector<std::shared_ptr<Surfel>> result;
    result.reserve(active_surfels_.size());
    for (const auto& kv : active_surfels_) {
        result.push_back(kv.second);
    }
    return result;
}

std::vector<std::shared_ptr<Surfel>> SurfMap::GetFinalizedSurfels() const {
    std::vector<std::shared_ptr<Surfel>> result;
    result.reserve(finalized_surfels_.size());
    for (const auto& kv : finalized_surfels_) {
        result.push_back(kv.second);
    }
    return result;
}

std::vector<std::shared_ptr<Surfel>> SurfMap::GetAllSurfelsForRendering() const {
    auto active    = GetActiveSurfels();
    auto finalized = GetFinalizedSurfels();
    active.insert(active.end(), finalized.begin(), finalized.end());
    return active;
}

// ============ 空间哈希网格辅助 ============ //

SurfMap::GridCellIndex SurfMap::PositionToCell(const Eigen::Vector3f& pos) const {
    const float inv = 1.0f / grid_cell_size_;
    return {static_cast<int>(std::floor(pos.x() * inv)),
            static_cast<int>(std::floor(pos.y() * inv)),
            static_cast<int>(std::floor(pos.z() * inv))};
}

void SurfMap::InsertSurfelToGrid(uint64_t id, const Eigen::Vector3f& pos) {
    spatial_grid_[PositionToCell(pos)].push_back(id);
}

void SurfMap::RemoveSurfelFromGrid(uint64_t id, const Eigen::Vector3f& pos) {
    auto it = spatial_grid_.find(PositionToCell(pos));
    if (it == spatial_grid_.end()) return;
    auto& vec = it->second;
    vec.erase(std::remove(vec.begin(), vec.end(), id), vec.end());
    if (vec.empty()) spatial_grid_.erase(it);
}

// ============ 查询 & 邻域搜索 ============ //

std::vector<std::shared_ptr<Surfel>> SurfMap::FindNeighbors(
    const Eigen::Vector3f& point) const {

    std::vector<std::shared_ptr<Surfel>> neighbors;
    const float r = config_.spatial_search_radius;
    // cell_size = spatial_search_radius，所以只需检查相邻 ±1 层，共 27 个 cell
    const int cell_r = static_cast<int>(std::ceil(r / grid_cell_size_));
    const GridCellIndex center = PositionToCell(point);

    for (int dx = -cell_r; dx <= cell_r; ++dx) {
        for (int dy = -cell_r; dy <= cell_r; ++dy) {
            for (int dz = -cell_r; dz <= cell_r; ++dz) {
                const GridCellIndex cell{center.x + dx, center.y + dy, center.z + dz};
                auto it = spatial_grid_.find(cell);
                if (it == spatial_grid_.end()) continue;
                for (uint64_t id : it->second) {
                    auto sit = active_surfels_.find(id);
                    if (sit == active_surfels_.end()) continue;
                    if ((sit->second->position - point).norm() <= r) {
                        neighbors.push_back(sit->second);
                    }
                }
            }
        }
    }
    return neighbors;
}

std::shared_ptr<Surfel> SurfMap::FindBestMatch(
    const Surfel& candidate,
    const std::vector<std::shared_ptr<Surfel>>& neighbors) const {

    std::shared_ptr<Surfel> best_match = nullptr;
    float best_score = -1.0f;

    for (const auto& neighbor : neighbors) {
        if (!neighbor->IsActive()) continue;

        // 空间距离
        float distance = (neighbor->position - candidate.position).norm();
        if (distance > config_.fusion_distance_threshold) continue;

        // 法线相似度
        float normal_similarity = neighbor->normal.dot(candidate.normal);
        if (normal_similarity < config_.fusion_normal_threshold) continue;

        // score 越大越好
        float score = normal_similarity / (distance + 1e-3f);

        if (score > best_score) {
            best_score = score;
            best_match = neighbor;
        }
    }

    return best_match;
}

// ============ Surfel 融合 ============ //

void SurfMap::FuseSurfelPair(std::shared_ptr<Surfel> target,
                             const std::shared_ptr<Surfel>& source,
                             uint64_t current_frame_id) {
    const Eigen::Vector3f old_position = target->position;

    // 权重：用观测次数和置信度一起考虑
    float w1 = std::max(1.0f, target->confidence * target->observation_count);
    float w2 = std::max(1.0f, source->confidence);
    float W  = w1 + w2;

    // 位置加权平均（简单版本，全向，未来可以改成沿法线方向融合）
    target->position = (w1 * old_position + w2 * source->position) / W;

    // 法线加权平均 + 归一化
    Eigen::Vector3f n = w1 * target->normal + w2 * source->normal;
    if (n.norm() > 1e-6f) {
        n.normalize();
        target->normal = n;
    }

    const float color_residual = (target->color - source->color).norm();
    float color_weight = w2;
    if (color_residual > config_.color_update_max_residual) {
        color_weight = 0.0f;
        fusion_stats_.rejected_by_color++;
    } else if (color_residual > config_.color_update_soft_residual) {
        const float denom = std::max(1e-6f, config_.color_update_max_residual - config_.color_update_soft_residual);
        const float t = (config_.color_update_max_residual - color_residual) / denom;
        color_weight = w2 * std::max(0.0f, std::min(1.0f, t));
    }

    if (color_weight > 0.0f) {
        target->color = (w1 * target->color + color_weight * source->color) / (w1 + color_weight);
    }
    fusion_stats_.mean_color_residual += color_residual;

    // 半径适度平滑
    target->radius = (w1 * target->radius + w2 * source->radius) / W;

    // 统计信息
    target->observation_count += source->observation_count;
    target->last_observed_frame = current_frame_id;
    target->confidence = std::min(1.0f, 0.5f * (target->confidence + source->confidence));

    // 合并观测关键帧列表
    target->observed_keyframes.insert(
        target->observed_keyframes.end(),
        source->observed_keyframes.begin(),
        source->observed_keyframes.end());

    // 融合可能使 surfel 跨越网格 cell 边界，需同步更新空间索引
    if (!(PositionToCell(old_position) == PositionToCell(target->position))) {
        RemoveSurfelFromGrid(target->id, old_position);
        InsertSurfelToGrid(target->id, target->position);
    }
}

void SurfMap::WarpSurfels(const std::vector<Eigen::Vector3f>& ctrl_pos,
                          const std::vector<Eigen::Vector3f>& ctrl_deform)
{
    if (!config_.warp_enabled || config_.max_warp_surfels_per_frame == 0) return;
    if (ctrl_pos.empty() || ctrl_pos.size() != ctrl_deform.size()) return;
    if (active_surfels_.empty()) return;

    const float sigma        = config_.warp_sigma;
    const float max_disp     = config_.warp_max_disp;
    if (sigma <= 0.0f || max_disp <= 0.0f) return;

    std::vector<size_t> control_indices;
    control_indices.reserve(std::min(ctrl_pos.size(), config_.max_warp_control_points));
    if (config_.max_warp_control_points > 0 &&
        ctrl_pos.size() > config_.max_warp_control_points) {
        const size_t stride =
            (ctrl_pos.size() + config_.max_warp_control_points - 1) /
            config_.max_warp_control_points;
        for (size_t i = 0; i < ctrl_pos.size() &&
             control_indices.size() < config_.max_warp_control_points; i += stride) {
            control_indices.push_back(i);
        }
    } else {
        for (size_t i = 0; i < ctrl_pos.size(); ++i) {
            control_indices.push_back(i);
        }
    }
    if (control_indices.empty()) return;

    const float inv_2sigma2  = 1.0f / (2.0f * sigma * sigma);
    // 超出 3sigma 的控制点贡献接近 0，截断可大幅减少计算
    const float cutoff_dist2 = (3.0f * sigma) * (3.0f * sigma);
    const float support_radius = 3.0f * sigma;
    const int cell_r = std::max(1, static_cast<int>(std::ceil(support_radius / grid_cell_size_)));

    std::vector<uint64_t> candidate_ids;
    candidate_ids.reserve(std::min(active_surfels_.size(), config_.max_warp_surfels_per_frame));
    std::unordered_set<uint64_t> seen_ids;
    seen_ids.reserve(candidate_ids.capacity());

    for (size_t ctrl_idx : control_indices) {
        const GridCellIndex center = PositionToCell(ctrl_pos[ctrl_idx]);
        for (int dx = -cell_r; dx <= cell_r; ++dx) {
            for (int dy = -cell_r; dy <= cell_r; ++dy) {
                for (int dz = -cell_r; dz <= cell_r; ++dz) {
                    const GridCellIndex cell{center.x + dx, center.y + dy, center.z + dz};
                    auto grid_it = spatial_grid_.find(cell);
                    if (grid_it == spatial_grid_.end()) continue;

                    for (uint64_t id : grid_it->second) {
                        if (seen_ids.find(id) != seen_ids.end()) continue;
                        auto surfel_it = active_surfels_.find(id);
                        if (surfel_it == active_surfels_.end()) continue;
                        if ((surfel_it->second->position - ctrl_pos[ctrl_idx]).squaredNorm() >
                            cutoff_dist2) {
                            continue;
                        }

                        seen_ids.insert(id);
                        candidate_ids.push_back(id);
                        if (candidate_ids.size() >= config_.max_warp_surfels_per_frame) break;
                    }
                    if (candidate_ids.size() >= config_.max_warp_surfels_per_frame) break;
                }
                if (candidate_ids.size() >= config_.max_warp_surfels_per_frame) break;
            }
            if (candidate_ids.size() >= config_.max_warp_surfels_per_frame) break;
        }
        if (candidate_ids.size() >= config_.max_warp_surfels_per_frame) break;
    }

    size_t warped_count = 0;

    for (uint64_t id : candidate_ids) {
        auto it = active_surfels_.find(id);
        if (it == active_surfels_.end()) continue;
        auto& s = it->second;
        const Eigen::Vector3f old_pos = s->position;

        float             total_w    = 0.0f;
        Eigen::Vector3f   weighted_d = Eigen::Vector3f::Zero();

        for (size_t i : control_indices) {
            const float dist2 = (old_pos - ctrl_pos[i]).squaredNorm();
            if (dist2 > cutoff_dist2) continue;
            const float w = std::exp(-dist2 * inv_2sigma2);
            weighted_d += w * ctrl_deform[i];
            total_w    += w;
        }

        if (total_w < 1e-8f) continue;

        const Eigen::Vector3f d = weighted_d / total_w;
        if (d.norm() > max_disp) continue;   // 异常大形变，跳过保护
        if (d.squaredNorm() < 1e-12f) continue;

        const Eigen::Vector3f new_pos = old_pos + d;

        // 同步更新空间索引（可能跨 cell）
        if (!(PositionToCell(old_pos) == PositionToCell(new_pos))) {
            RemoveSurfelFromGrid(s->id, old_pos);
            s->position = new_pos;
            InsertSurfelToGrid(s->id, new_pos);
        } else {
            s->position = new_pos;
        }
        ++warped_count;
    }

    LOG(INFO) << "SurfMap::WarpSurfels: warped " << warped_count
              << "/" << active_surfels_.size()
              << " active surfels (candidates=" << candidate_ids.size()
              << ", sigma=" << sigma
              << ", ctrl_pts=" << ctrl_pos.size()
              << ", used_ctrl_pts=" << control_indices.size() << ")";
    fusion_stats_.warped = static_cast<int>(warped_count);
}

bool SurfMap::ExportToPLY(const std::string& filename, bool only_finalized) const {
    std::vector<std::shared_ptr<Surfel>> surfels;

    if (only_finalized) {
        surfels.reserve(finalized_surfels_.size());
        for (const auto& kv : finalized_surfels_)
            surfels.push_back(kv.second);
    } else {
        surfels.reserve(active_surfels_.size() + finalized_surfels_.size());
        for (const auto& kv : active_surfels_)
            surfels.push_back(kv.second);
        for (const auto& kv : finalized_surfels_)
            surfels.push_back(kv.second);
    }

    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        std::cerr << "Cannot open output PLY file: " << filename << std::endl;
        return false;
    }

    // Header
    ofs << "ply\n";
    ofs << "format ascii 1.0\n";
    ofs << "element vertex " << surfels.size() << "\n";
    ofs << "property float x\n";
    ofs << "property float y\n";
    ofs << "property float z\n";
    ofs << "property float nx\n";
    ofs << "property float ny\n";
    ofs << "property float nz\n";
    ofs << "property uchar red\n";
    ofs << "property uchar green\n";
    ofs << "property uchar blue\n";
    ofs << "property float confidence\n";
    ofs << "end_header\n";

    for (const auto& s : surfels) {
        const Eigen::Vector3f& p = s->position;
        const Eigen::Vector3f& n = s->normal;
        Eigen::Vector3f color = s->color;

        // Clamp color
        int r = std::clamp(int(color[0] * 255.f), 0, 255);
        int g = std::clamp(int(color[1] * 255.f), 0, 255);
        int b = std::clamp(int(color[2] * 255.f), 0, 255);

        ofs << p.x() << " " << p.y() << " " << p.z() << " "
            << n.x() << " " << n.y() << " " << n.z() << " "
            << r << " " << g << " " << b << " "
            << s->confidence << "\n";
    }

    ofs.close();
    std::cout << "[SurfMap] Exported " << surfels.size()
              << " surfels to " << filename << std::endl;
    return true;
}
