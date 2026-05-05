#ifndef NRSLAM_SURFELMAP_H
#define NRSLAM_SURFELMAP_H

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <vector>
#include <deque>
#include <unordered_map>
#include <memory>
#include <cstdint>
#include <fstream>
#include <string>

#include "calibration/camera_model.h"

class SDFBuilder;

// ==================== Surfel-SDF 约束统计 ==================== //

struct SurfelPhiStats {
    uint64_t frame_id = 0;
    int constraint_checked = 0;        // 检查的 surfel 总数
    int constraint_rejected = 0;       // 被约束过滤掉的 surfel 数
    int constraint_accepted = 0;       // 通过约束的 surfel 数
    int projected_count = 0;           // 被投影到零level-set的 surfel 数
    float phi_before_min = 0.0f;       // 投影前 |phi| 最小值
    float phi_before_max = 0.0f;       // 投影前 |phi| 最大值
    float phi_after_min = 0.0f;        // 投影后 |phi| 最小值
    float phi_after_max = 0.0f;        // 投影后 |phi| 最大值
};

struct SurfelFusionStats {
    uint64_t frame_id = 0;
    int generated = 0;
    int accepted = 0;
    int added = 0;
    int fused = 0;
    int rejected_by_depth = 0;
    int rejected_by_gradient = 0;
    int rejected_by_normal = 0;
    int rejected_by_mask = 0;
    int rejected_by_sdf = 0;
    int rejected_by_color = 0;
    int finalized = 0;
    int warped = 0;
    size_t active = 0;
    size_t finalized_total = 0;
    float mean_radius = 0.0f;
    float mean_confidence = 0.0f;
    float mean_color_residual = 0.0f;
};

// ==================== 配置 ==================== //

struct SurfConfig {
    // ====== 集成频率 & 生命周期 ======
    int  window_size              = 10;    // 用作最大 age（多少帧没看到就 finalize）
    int  min_observations         = 1;     // 没有跨帧融合前先放宽

    float confidence_decay_rate   = 0.10f;
    float min_surfel_confidence   = 0.10f;

    // “当前视野内”的深度范围（active 区域）
    float active_min_depth        = 0.02f; // 2cm
    float active_max_depth        = 0.10f; // 10cm

    // 新增：每帧最多新建多少个 surfel，防止爆炸
    size_t max_new_surfels_per_frame = 10000;

    // ====== 融合 & 邻域搜索 ======
    float fusion_distance_threshold = 0.01f;
    float fusion_normal_threshold   = 0.85f;
    float spatial_search_radius     = 0.02f;

    // ====== SDF consistency constraint ======
    bool  sdf_constraint_enabled    = false;
    float sdf_phi_abs_threshold     = 0.10f;
    float sdf_normal_cos_threshold  = 0.50f;
    bool  sdf_project_to_surface    = false;

    // ====== Surfel 生成参数 ======
    float min_depth              = 0.01f;  // 生成时允许的最小深度
    float max_depth              = 0.30f;  // 生成时允许的最大深度（比 active_max_depth 稍大）
    float default_surfel_radius  = 0.003f;
    bool  use_adaptive_radius    = true;
    int   surfel_sample_step     = 4;    // 深度图采样步长（1=每像素，推荐4）
    std::string quality_mode       = "realtime";
    float min_surfel_radius        = 0.001f;
    float max_surfel_radius        = 0.006f;
    float surfel_radius_scale      = 1.5f;
    float max_depth_gradient       = 0.030f;
    float min_normal_norm          = 1e-6f;
    int   mask_border_pixels       = 1;
    size_t max_fused_surfels_per_frame = 0;
    size_t max_added_surfels_per_frame = 0;
    float color_update_max_residual = 0.35f;
    float color_update_soft_residual = 0.15f;

    // 形变场传播参数
    bool  warp_enabled           = true;
    float warp_sigma             = 0.03f; // RBF 核宽度（米），控制形变影响半径
    float warp_max_disp          = 0.10f; // 单帧最大形变量（米），超出则跳过该 surfel
    size_t max_warp_surfels_per_frame = 3000;
    size_t max_warp_control_points = 128;
};


// ==================== 数据结构 ==================== //

struct Surfel {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    enum State { ACTIVE, FINALIZED };

    uint64_t id            = 0;
    State    state         = ACTIVE;

    Eigen::Vector3f position = Eigen::Vector3f::Zero();   // 世界坐标
    Eigen::Vector3f normal   = Eigen::Vector3f::UnitZ();  // 世界坐标系下法线
    Eigen::Vector3f color    = Eigen::Vector3f::Zero();   // 0..1 RGB
    float radius             = 0.01f;                     // 支持半径（近似 footprint）
    float confidence         = 1.0f;                      // 0..1 置信度

    // 观测统计
    int      observation_count   = 1;
    uint64_t first_observed_frame= 0;
    uint64_t last_observed_frame = 0;
    std::vector<uint64_t> observed_keyframes;

    Surfel(uint64_t surfel_id, uint64_t frame_id)
        : id(surfel_id),
          first_observed_frame(frame_id),
          last_observed_frame(frame_id) {}

    inline bool IsActive() const   { return state == ACTIVE; }
    inline bool IsFinalized() const{ return state == FINALIZED; }
};

struct SurfKeyFrame {
    uint64_t id = 0;
    cv::Mat  color_image;     // CV_8UC3
    cv::Mat  estimated_depth; // CV_32FC1 (meter)
    Eigen::Matrix4f pose;     // T_wc
};

// ==================== SurfMap ==================== //

class SurfMap {
public:
    explicit SurfMap(const SurfConfig& config = SurfConfig());
    ~SurfMap();

    void UpdateConfig(const SurfConfig& new_config);
    SurfConfig GetConfig() const;
    void SetSDFBuilder(const SDFBuilder* sdf_builder);
    SurfelPhiStats GetPhiStats() const { return phi_stats_; }
    SurfelFusionStats GetLastFusionStats() const { return fusion_stats_; }
    void BeginFrame(uint64_t frame_id);

    // === 核心功能 === //

    void IntegrateFrame(const cv::Mat& color_image,
                        const cv::Mat& estimated_depth,
                        const Eigen::Matrix4f& camera_pose,
                        const Eigen::Matrix3f& camera_intrinsics,
                        uint64_t frame_id,
                        const cv::Mat& valid_mask = cv::Mat());

    // 导出 Surfels，用于可视化/后处理
    std::vector<std::shared_ptr<Surfel>> GetActiveSurfels() const;
    std::vector<std::shared_ptr<Surfel>> GetFinalizedSurfels() const;
    std::vector<std::shared_ptr<Surfel>> GetAllSurfelsForRendering() const;

    size_t NumActiveSurfels() const {
        return active_surfels_.size();
    }

    size_t NumFinalizedSurfels() const {
        return finalized_surfels_.size();
    }

    bool ExportToPLY(const std::string& filename, bool only_finalized = false) const;

    // P0: 将稀疏形变场（RBF）传播到所有 active surfel
    // ctrl_pos    : 控制点的形变前世界坐标（即路标在形变前的位置）
    // ctrl_deform : 对应的形变向量 d_i（形变后位置 = ctrl_pos[i] + ctrl_deform[i]）
    void WarpSurfels(const std::vector<Eigen::Vector3f>& ctrl_pos,
                     const std::vector<Eigen::Vector3f>& ctrl_deform);


private:
    std::vector<std::shared_ptr<Surfel>> GenerateSurfelsFromKeyFrame(
        const cv::Mat& color_image,
        const cv::Mat& depth_map,
        const cv::Mat& valid_mask,
        const Eigen::Matrix4f& pose,
        const Eigen::Matrix3f& intrinsics,
        uint64_t frame_id);

    void FuseNewSurfelsToActiveMap(
        const std::vector<std::shared_ptr<Surfel>>& new_surfels,
        uint64_t source_keyframe_id,
        uint64_t frame_id);

    void RemoveOldestKeyFrame();

    // ★ 生命周期：每帧 IntegrateFrame 之后调用一次
    void UpdateSurfelLifecycle(uint64_t current_frame_id,
                               const Eigen::Matrix4f& current_pose);

    // 匹配查找 / 邻域搜索 / 融合
    std::vector<std::shared_ptr<Surfel>> FindNeighbors(
        const Eigen::Vector3f& point) const;

    std::shared_ptr<Surfel> FindBestMatch(
        const Surfel& candidate,
        const std::vector<std::shared_ptr<Surfel>>& neighbors) const;

    void FuseSurfelPair(std::shared_ptr<Surfel> target,
                        const std::shared_ptr<Surfel>& source,
                        uint64_t current_frame_id);

    bool ApplySDFConstraint(std::shared_ptr<Surfel>& surfel, float* phi_before = nullptr, float* phi_after = nullptr) const;

    float ComputeSurfelRadius(float depth,
                              const Eigen::Vector3f& normal_cam,
                              const Eigen::Matrix3f& intrinsics) const;

    bool IsMaskBorder(const cv::Mat& valid_mask, int u, int v) const;

    bool PassDepthGradientFilter(const cv::Mat& depth_map,
                                 int u,
                                 int v,
                                 float center_depth) const;

    void UpdateMeanMapStats();

    // 空间哈希网格辅助（加速邻域搜索）
    struct GridCellIndex {
        int x, y, z;
        bool operator==(const GridCellIndex& o) const {
            return x == o.x && y == o.y && z == o.z;
        }
    };
    struct GridCellHash {
        size_t operator()(const GridCellIndex& idx) const {
            // FNV-1a 风格，比简单 XOR-shift 分布更均匀
            size_t h = 2166136261u;
            h ^= std::hash<int>()(idx.x); h *= 16777619u;
            h ^= std::hash<int>()(idx.y); h *= 16777619u;
            h ^= std::hash<int>()(idx.z); h *= 16777619u;
            return h;
        }
    };
    GridCellIndex PositionToCell(const Eigen::Vector3f& pos) const;
    void InsertSurfelToGrid(uint64_t id, const Eigen::Vector3f& pos);
    void RemoveSurfelFromGrid(uint64_t id, const Eigen::Vector3f& pos);

private:
    SurfConfig config_;
    const SDFBuilder* sdf_builder_ = nullptr;
    mutable SurfelPhiStats phi_stats_;
    SurfelFusionStats fusion_stats_;

    // 最近的关键帧窗口（主要用于调试和潜在扩展，本版本生命周期只依赖 frame_id 和位姿）
    std::deque<std::shared_ptr<SurfKeyFrame>> keyframe_window_;

    // 活跃（仍可能被当前/未来帧看到并继续融合）的 Surfel
    std::unordered_map<uint64_t, std::shared_ptr<Surfel>> active_surfels_;

    // 已经被甩在身后且稳定的 Surfel（不再改变）
    std::unordered_map<uint64_t, std::shared_ptr<Surfel>> finalized_surfels_;

    float grid_cell_size_ = 0.02f;
    std::unordered_map<GridCellIndex, std::vector<uint64_t>, GridCellHash> spatial_grid_;

    uint64_t next_surfel_id_ = 0;
};

#endif // NRSLAM_SURFELMAP_H
