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

#ifndef NRSLAM_REGULARIZATION_GRAPH_H
#define NRSLAM_REGULARIZATION_GRAPH_H

#include "map.h"

#include "absl/container/flat_hash_map.h"
#include "absl/container/btree_map.h"

#include "eigen3/Eigen/Core"

typedef long unsigned int ID;

class Map;

class RegularizationGraph {
public:
    struct Options {
        float weight_sigma;
        float streching_th;
    };

    enum Status {
        VERIFIED,
        NEIGHBOR,
        NEUTRAL,
        BAD
    };

    struct Edge {
        ID vertex_id_1;
        ID vertex_id_2; // 连接的两个地图点的ID 
        float distance; // 当前两个地图点的距离
        float first_distance; // 初始距离
        float weight; // 权重：根据距离和sigma计算出
        Status status; 
        float max_distance;
        float min_distance;
        Eigen::Vector3f last_relative_position; // 上次的相对位置
    };

    RegularizationGraph() = delete;

    RegularizationGraph(Options& options, Map* map);

    void SetSigma(const float sigma);

    // 给定两个地图点的 ID 和它们的当前相对位置，向图中添加一条边。
    void AddEdge(ID mappoint_id, ID mappoint_id_other, Eigen::Vector3f& relative_position);

    std::shared_ptr<Edge> GetEdge(ID mappoint_id, ID mappoint_id_other);

    std::vector<std::pair<ID, std::shared_ptr<Edge>>> GetEdges(ID mappoint_id) const;

    // 当地图点的位姿或者观测信息发生变化时，更新正则化图中对应的边或顶点的状态
    bool UpdateConnection(ID mappoint_id_1, ID mappoint_id_2, Eigen::Vector3f& landmark_position_1,
                          Eigen::Vector3f& landmark_position_2);

    int UpdateVertex(ID mappoint_id);

    std::string ToString();

    typedef absl::btree_map<ID, std::shared_ptr<Edge>> VertexConnections;

    // 根据给定的地图点集合和所需的邻居数量，提取零阶、一级及二级邻居关系信息
    void GetOptimizationNeighbours(const std::vector<ID>& mappoints_ids, const int connections_per_point,
                                   absl::flat_hash_map<ID, absl::flat_hash_set<ID>>& zero_order_connections,
                                   absl::flat_hash_map<ID, absl::flat_hash_set<ID>>& first_order_connections,
                                   absl::flat_hash_set<ID>& second_order_connections);

    float GetMinWeightAllowed();

private:
    absl::btree_map<ID, VertexConnections> graph_;

    Map* map_;

    float min_weight_;

    Options options_;
};

#endif //NRSLAM_REGULARIZATION_GRAPH_H
