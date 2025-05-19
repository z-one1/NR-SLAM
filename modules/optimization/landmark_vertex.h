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

#ifndef NRSLAM_LANDMARK_VERTEX_H
#define NRSLAM_LANDMARK_VERTEX_H

#include "g2o/core/base_vertex.h"

// 储存landmark地图点坐标：继承自g2o顶点对象(3维 - Eigen::Vector3d)
class LandmarkVertex : public g2o::BaseVertex<3, Eigen::Vector3d> { 
public:
    /*
        Eigen 库中的某些对象（如 Eigen::Matrix4f、Eigen::Vector3d）需要 内存对齐（比如 16/32 字节对齐）
        以支持 SIMD（如 SSE/AVX）加速运算
        如果用 new 动态分配内存，默认的内存分配可能 不满足对齐要求，导致程序崩溃或性能下降
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW 重载了 operator new
        确保用 new 创建对象时，内存始终按 Eigen 要求的对齐方式分配
        如果类里有固定大小的 Eigen 对象，必须加这个宏！
    */
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW // Eigen 宏定义，用于解决 动态内存对齐 问题
    LandmarkVertex();

    // g2o要求的序列化函数，用于将图结构保存为.g2o文件/从中加载
    bool read(std::istream& is);
    bool write(std::ostream& os) const;

    void setToOriginImpl(); // 设置初始值

    void oplusImpl(const double * update); // 更新顶点
};


#endif //NRSLAM_LANDMARK_VERTEX_H
/*
    顶点（Vertex）必须实现的函数
    1. setToOriginImpl() 作用：初始化顶点的状态
    2. oplusImpl(const double* update) 作用：更新顶点状态（优化时调用）
*/