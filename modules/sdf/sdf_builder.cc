#include "sdf/sdf_builder.h"

#include "absl/log/log.h"

#include <algorithm>
#include <cmath>
#include <fstream>

namespace {
inline bool IsFinitePositive(float x) {
    return std::isfinite(x) && x > 0.0f;
}
} // namespace

SDFBuilder::SDFBuilder()
    : SDFBuilder(Options{}) {
}

SDFBuilder::SDFBuilder(const SDFBuilder::Options& options)
    : options_(options) {
}

SDFBuilder::VoxelIndex SDFBuilder::WorldToVoxel(const Eigen::Vector3f& p) const {
    return VoxelIndex{
        static_cast<int>(std::floor(p.x() / options_.voxel_size)),
        static_cast<int>(std::floor(p.y() / options_.voxel_size)),
        static_cast<int>(std::floor(p.z() / options_.voxel_size))};
}

Eigen::Vector3f SDFBuilder::VoxelCenter(const SDFBuilder::VoxelIndex& idx) const {
    const float s = options_.voxel_size;
    return Eigen::Vector3f((static_cast<float>(idx.x) + 0.5f) * s,
                           (static_cast<float>(idx.y) + 0.5f) * s,
                           (static_cast<float>(idx.z) + 0.5f) * s);
}

bool SDFBuilder::TrilinearInterpolate(const Eigen::Vector3f& p_world, float& phi) const {
    const float s = options_.voxel_size;
    const float gx = p_world.x() / s;
    const float gy = p_world.y() / s;
    const float gz = p_world.z() / s;

    const int x0 = static_cast<int>(std::floor(gx));
    const int y0 = static_cast<int>(std::floor(gy));
    const int z0 = static_cast<int>(std::floor(gz));

    const float tx = gx - static_cast<float>(x0);
    const float ty = gy - static_cast<float>(y0);
    const float tz = gz - static_cast<float>(z0);

    float corners[2][2][2];

    for (int dx = 0; dx <= 1; ++dx) {
        for (int dy = 0; dy <= 1; ++dy) {
            for (int dz = 0; dz <= 1; ++dz) {
                const VoxelIndex idx{x0 + dx, y0 + dy, z0 + dz};
                const auto it = voxels_.find(idx);
                if (it == voxels_.end() || it->second.weight <= 0.0f) {
                    return false;
                }
                corners[dx][dy][dz] = it->second.tsdf;
            }
        }
    }

    const float c00 = corners[0][0][0] * (1.0f - tx) + corners[1][0][0] * tx;
    const float c10 = corners[0][1][0] * (1.0f - tx) + corners[1][1][0] * tx;
    const float c01 = corners[0][0][1] * (1.0f - tx) + corners[1][0][1] * tx;
    const float c11 = corners[0][1][1] * (1.0f - tx) + corners[1][1][1] * tx;

    const float c0 = c00 * (1.0f - ty) + c10 * ty;
    const float c1 = c01 * (1.0f - ty) + c11 * ty;

    phi = c0 * (1.0f - tz) + c1 * tz;
    return true;
}

bool SDFBuilder::QueryPhi(const Eigen::Vector3f& p_world, float& phi) const {
    if (TrilinearInterpolate(p_world, phi)) return true;
    // 三线性插值要求周围 8 个角体素全部存在，在 SDF 边界或稀疏区域常常失败。
    // 回退到最近邻体素：只要当前体素有效即可返回近似 phi，扩大有效覆盖范围。
    const VoxelIndex idx = WorldToVoxel(p_world);
    const auto it = voxels_.find(idx);
    if (it != voxels_.end() && it->second.weight > 0.0f) {
        phi = it->second.tsdf;
        return true;
    }
    return false;
}

bool SDFBuilder::QueryGradient(const Eigen::Vector3f& p_world, Eigen::Vector3f& gradient) const {
    const float h = options_.voxel_size;
    const Eigen::Vector3f ex(h, 0.0f, 0.0f);
    const Eigen::Vector3f ey(0.0f, h, 0.0f);
    const Eigen::Vector3f ez(0.0f, 0.0f, h);

    float phi_px = 0.0f, phi_nx = 0.0f;
    float phi_py = 0.0f, phi_ny = 0.0f;
    float phi_pz = 0.0f, phi_nz = 0.0f;

    if (!TrilinearInterpolate(p_world + ex, phi_px) ||
        !TrilinearInterpolate(p_world - ex, phi_nx) ||
        !TrilinearInterpolate(p_world + ey, phi_py) ||
        !TrilinearInterpolate(p_world - ey, phi_ny) ||
        !TrilinearInterpolate(p_world + ez, phi_pz) ||
        !TrilinearInterpolate(p_world - ez, phi_nz)) {
        return false;
    }

    const float inv_2h = 0.5f / h;
    gradient.x() = (phi_px - phi_nx) * inv_2h;
    gradient.y() = (phi_py - phi_ny) * inv_2h;
    gradient.z() = (phi_pz - phi_nz) * inv_2h;
    return true;
}

void SDFBuilder::IntegrateSurfacePoint(const Eigen::Vector3f& surface_world,
                                       const Eigen::Vector3f& view_dir_world,
                                       float sample_weight,
                                       uint8_t source_mask,
                                       uint64_t frame_id,
                                       int& remaining_updates,
                                       SDFBuilder::Stats& stats) {
    if (remaining_updates <= 0) return;
    if (view_dir_world.squaredNorm() < 1e-12f) return;

    const float trunc = options_.truncation_distance;
    const int r = static_cast<int>(std::ceil(trunc / options_.voxel_size));
    const Eigen::Vector3f dir = view_dir_world.normalized();
    const VoxelIndex center_idx = WorldToVoxel(surface_world);

    for (int dx = -r; dx <= r && remaining_updates > 0; ++dx) {
        for (int dy = -r; dy <= r && remaining_updates > 0; ++dy) {
            for (int dz = -r; dz <= r && remaining_updates > 0; ++dz) {
                VoxelIndex idx{center_idx.x + dx, center_idx.y + dy, center_idx.z + dz};
                Eigen::Vector3f c = VoxelCenter(idx);

                const float signed_dist = (c - surface_world).dot(dir);
                if (std::abs(signed_dist) > trunc) {
                    continue;
                }

                const float tsdf = std::max(-1.0f, std::min(1.0f, signed_dist / trunc));

                VoxelData& voxel = voxels_[idx];
                const float w_old = voxel.weight;
                const float w_new = sample_weight;
                const float w_sum = w_old + w_new;

                voxel.tsdf = (w_old * voxel.tsdf + w_new * tsdf) / std::max(1e-6f, w_sum);
                voxel.weight = std::min(1000.0f, w_sum);
                voxel.last_updated_frame = frame_id;
                voxel.source_mask |= source_mask;

                --remaining_updates;
                ++stats.voxel_updates;
            }
        }
    }
}

void SDFBuilder::IntegrateFrame(const cv::Mat& aligned_depth,
                                const std::vector<Eigen::Vector3f>& sparse_world_points,
                                const Eigen::Matrix4f& T_wc,
                                const Eigen::Matrix3f& K,
                                const cv::Mat& valid_mask,
                                uint64_t frame_id) {
    last_stats_ = Stats{};
    last_stats_.frame_id = frame_id;

    if (aligned_depth.empty()) {
        LOG(WARNING) << "SDFBuilder::IntegrateFrame skipped: aligned depth is empty.";
        last_stats_.active_voxels = voxels_.size();
        return;
    }

    cv::Mat depth_32f;
    if (aligned_depth.type() == CV_32FC1) {
        depth_32f = aligned_depth;
    } else {
        aligned_depth.convertTo(depth_32f, CV_32FC1);
    }

    const bool has_mask = !valid_mask.empty() && valid_mask.type() == CV_8UC1 &&
                          valid_mask.size() == depth_32f.size();

    const float fx = K(0, 0);
    const float fy = K(1, 1);
    const float cx = K(0, 2);
    const float cy = K(1, 2);

    const Eigen::Matrix3f R_wc = T_wc.block<3, 3>(0, 0);
    const Eigen::Vector3f t_wc = T_wc.block<3, 1>(0, 3);
    const Eigen::Vector3f cam_world = t_wc;

    auto backproject = [&](int u, int v, float z) {
        return Eigen::Vector3f((static_cast<float>(u) - cx) * z / fx,
                               (static_cast<float>(v) - cy) * z / fy,
                               z);
    };

    const bool use_sparse =
        static_cast<int>(sparse_world_points.size()) >= options_.sparse_min_points;

    const int total_budget = std::max(1, options_.max_voxel_updates_per_frame);
    const int sparse_reserved_budget = use_sparse ? std::max(1, total_budget / 5) : 0;
    int remaining_dense_updates = total_budget - sparse_reserved_budget;

    const int stride = std::max(1, options_.dense_pixel_stride);

    for (int v = 0; v < depth_32f.rows && remaining_dense_updates > 0; v += stride) {
        const float* depth_row = depth_32f.ptr<float>(v);
        for (int u = 0; u < depth_32f.cols && remaining_dense_updates > 0; u += stride) {
            if (has_mask && valid_mask.at<uint8_t>(v, u) == 0) {
                continue;
            }

            const float z = depth_row[u];
            if (!IsFinitePositive(z)) continue;
            if (z < options_.min_depth || z > options_.max_depth) continue;

            const Eigen::Vector3f p_cam = backproject(u, v, z);
            const Eigen::Vector3f p_world = R_wc * p_cam + t_wc;
            Eigen::Vector3f view_dir_world = p_world - cam_world;
            if (view_dir_world.squaredNorm() < 1e-12f) continue;

            IntegrateSurfacePoint(p_world, view_dir_world, options_.dense_weight, 0x1,
                                  frame_id, remaining_dense_updates, last_stats_);
            ++last_stats_.dense_points_used;
        }
    }

    int remaining_sparse_updates = sparse_reserved_budget + remaining_dense_updates;

    if (!use_sparse) {
        last_stats_.dense_only_fallback = true;
    } else {
        for (const Eigen::Vector3f& p_world : sparse_world_points) {
            if (remaining_sparse_updates <= 0) break;
            if (!p_world.allFinite()) continue;

            Eigen::Vector3f view_dir_world = p_world - cam_world;
            if (view_dir_world.squaredNorm() < 1e-12f) continue;

            IntegrateSurfacePoint(p_world, view_dir_world, options_.sparse_weight, 0x2,
                                  frame_id, remaining_sparse_updates, last_stats_);
            ++last_stats_.sparse_points_used;
        }
    }

    last_stats_.active_voxels = voxels_.size();
}

bool SDFBuilder::ExportToPLY(const std::string& filename, float max_abs_tsdf) const {
    std::vector<Eigen::Vector3f> points;
    points.reserve(voxels_.size());

    for (const auto& kv : voxels_) {
        const VoxelData& v = kv.second;
        if (v.weight <= 0.0f) continue;
        if (std::abs(v.tsdf) > max_abs_tsdf) continue;
        points.push_back(VoxelCenter(kv.first));
    }

    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        LOG(WARNING) << "Cannot open output PLY file: " << filename;
        return false;
    }

    ofs << "ply\n";
    ofs << "format ascii 1.0\n";
    ofs << "element vertex " << points.size() << "\n";
    ofs << "property float x\n";
    ofs << "property float y\n";
    ofs << "property float z\n";
    ofs << "end_header\n";

    for (const auto& p : points) {
        ofs << p.x() << " " << p.y() << " " << p.z() << "\n";
    }

    ofs.close();
    LOG(INFO) << "SDF point cloud exported to " << filename
              << " with " << points.size() << " points.";
    return true;
}
