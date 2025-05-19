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

#ifndef NRSLAM_POSITION_REGULARIZER_WITH_DEFORMATION_H
#define NRSLAM_POSITION_REGULARIZER_WITH_DEFORMATION_H

#include "optimization/landmark_vertex.h"

#include "g2o/core/base_binary_edge.h"

// 继承自 g2o::BaseBinaryEdge，误差维度为1（标量误差），连接两个 LandmarkVertex
class PositionRegularizerWithDeformation : public g2o::BaseBinaryEdge<1, double,
        LandmarkVertex, LandmarkVertex> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW // Eigen 宏，内存对齐

    PositionRegularizerWithDeformation();

    bool read(std::istream& is);
    bool write(std::ostream& os) const;

    void computeError(); // 计算当前误差

    virtual void linearizeOplus(); // 计算 误差对顶点（Landmark）的雅可比矩阵（Jacobian）

    Eigen::Vector3d rest_position_1_, rest_position_2_; // 两个点的“静止位置”（未变形时的位置）
    double k_; // 弹性系数
};


#endif //NRSLAM_POSITION_REGULARIZER_WITH_DEFORMATION_H
/*
    边（Edge）必须实现的函数
    1. computeError() 作用：计算当前误差（优化目标是最小化该误差）
    2. linearizeOplus() 作用：计算误差对顶点的雅可比矩阵（用于优化求解）
*/
