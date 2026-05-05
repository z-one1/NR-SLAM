#ifndef NRSLAM_SDF_BUILDER_H
#define NRSLAM_SDF_BUILDER_H

#include <Eigen/Core>
#include <opencv2/core.hpp>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

class SDFBuilder {
public:
    struct Options {
        float voxel_size = 0.0025f;
        float truncation_distance = 0.0100f;
        int dense_pixel_stride = 2;
        float dense_weight = 1.0f;
        float sparse_weight = 5.0f;
        int max_voxel_updates_per_frame = 400000;
        float min_depth = 0.01f;
        float max_depth = 8.00f;
        int sparse_min_points = 20;
    };

    struct Stats {
        uint64_t frame_id = 0;
        size_t active_voxels = 0;
        int dense_points_used = 0;
        int sparse_points_used = 0;
        int voxel_updates = 0;
        bool dense_only_fallback = false;
    };

    SDFBuilder();
    explicit SDFBuilder(const Options& options);

    void IntegrateFrame(const cv::Mat& aligned_depth,
                        const std::vector<Eigen::Vector3f>& sparse_world_points,
                        const Eigen::Matrix4f& T_wc,
                        const Eigen::Matrix3f& K,
                        const cv::Mat& valid_mask,
                        uint64_t frame_id);

    const Stats& GetLastStats() const { return last_stats_; }

    bool QueryPhi(const Eigen::Vector3f& p_world, float& phi) const;
    bool QueryGradient(const Eigen::Vector3f& p_world, Eigen::Vector3f& gradient) const;

    bool ExportToPLY(const std::string& filename,
                     float max_abs_tsdf = 0.10f) const;

private:
    struct VoxelIndex {
        int x;
        int y;
        int z;

        bool operator==(const VoxelIndex& other) const {
            return x == other.x && y == other.y && z == other.z;
        }
    };

    struct VoxelIndexHash {
        std::size_t operator()(const VoxelIndex& idx) const {
            const std::size_t h1 = std::hash<int>()(idx.x);
            const std::size_t h2 = std::hash<int>()(idx.y);
            const std::size_t h3 = std::hash<int>()(idx.z);
            return h1 ^ (h2 << 1) ^ (h3 << 2);
        }
    };

    struct VoxelData {
        float tsdf = 1.0f;
        float weight = 0.0f;
        uint64_t last_updated_frame = 0;
        uint8_t source_mask = 0;
    };

    VoxelIndex WorldToVoxel(const Eigen::Vector3f& p) const;
    Eigen::Vector3f VoxelCenter(const VoxelIndex& idx) const;
    bool TrilinearInterpolate(const Eigen::Vector3f& p_world, float& phi) const;

    void IntegrateSurfacePoint(const Eigen::Vector3f& surface_world,
                               const Eigen::Vector3f& view_dir_world,
                               float sample_weight,
                               uint8_t source_mask,
                               uint64_t frame_id,
                               int& remaining_updates,
                               Stats& stats);

    Options options_;
    std::unordered_map<VoxelIndex, VoxelData, VoxelIndexHash> voxels_;
    Stats last_stats_;
};

#endif // NRSLAM_SDF_BUILDER_H
