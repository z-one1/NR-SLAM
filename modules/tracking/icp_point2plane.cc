#include "icp_point2plane.h"
#include <algorithm>
#include <cmath>

static inline float deg2rad(float d){ return d*static_cast<float>(M_PI/180.0); }

PointToPlaneICP::PointToPlaneICP(const Intrinsics& K, const IcpOptions& opt)
  : K_(K), opt_(opt) {}

cv::Mat PointToPlaneICP::buildVertexMap(const cv::Mat& Z, const Intrinsics& K) {
  CV_Assert(Z.type()==CV_32FC1);
  cv::Mat V(Z.size(), CV_32FC3, cv::Scalar(0,0,0));
  for (int y=0; y<Z.rows; ++y) {
    const float* zr = Z.ptr<float>(y);
    cv::Vec3f* vr = V.ptr<cv::Vec3f>(y);
    for (int x=0; x<Z.cols; ++x) {
      float z = zr[x];
      if (z > 0 && std::isfinite(z)) {
        float X = (x - K.cx) * z / K.fx;
        float Y = (y - K.cy) * z / K.fy;
        vr[x] = cv::Vec3f(X, Y, z);
      } else {
        vr[x] = cv::Vec3f(0,0,0);
      }
    }
  }
  return V;
}

cv::Mat PointToPlaneICP::buildNormalMap(const cv::Mat& V) {
  CV_Assert(V.type()==CV_32FC3);
  cv::Mat N(V.size(), CV_32FC3, cv::Scalar(0,0,0));
  auto get = [&](int x,int y)->cv::Vec3f {
    x = std::clamp(x,0,V.cols-1); y = std::clamp(y,0,V.rows-1);
    return V.at<cv::Vec3f>(y,x);
  };
  for (int y=1; y<V.rows-1; ++y) {
    cv::Vec3f* nr = N.ptr<cv::Vec3f>(y);
    for (int x=1; x<V.cols-1; ++x) {
      cv::Vec3f v  = V.at<cv::Vec3f>(y,x);
      if (!(v[2] > 0)) { nr[x]=cv::Vec3f(0,0,0); continue; }
      cv::Vec3f vx1 = get(x+1,y), vx0 = get(x-1,y);
      cv::Vec3f vy1 = get(x,y+1), vy0 = get(x,y-1);
      cv::Vec3f tx = vx1 - vx0;
      cv::Vec3f ty = vy1 - vy0;
      cv::Vec3f n  = cv::Vec3f(
        tx[1]*ty[2] - tx[2]*ty[1],
        tx[2]*ty[0] - tx[0]*ty[2],
        tx[0]*ty[1] - tx[1]*ty[0]
      );
      float nn = std::sqrt(n.dot(n));
      if (nn > 1e-6f) n *= (1.0f/nn); else n = cv::Vec3f(0,0,0);
      nr[x] = n;
    }
  }
  return N;
}

void PointToPlaneICP::buildPyramid(const cv::Mat& Z_ref, const cv::Mat& Z_live,
                                   const Intrinsics& K, int L,
                                   std::vector<Level>& out,
                                   const cv::Mat& M_ref, const cv::Mat& M_live)
{
  out.resize(L);
  cv::Mat zr = Z_ref.clone(), zl = Z_live.clone();
  Intrinsics k = K;
  cv::Mat mr = M_ref, ml = M_live;

  for (int lvl = 0; lvl < L; ++lvl) {
    out[lvl].Z_ref = zr;
    out[lvl].Z_live = zl;
    out[lvl].K = k;

    // 顶点/法线
    out[lvl].V_ref = buildVertexMap(zr, k);
    out[lvl].N_ref = buildNormalMap(out[lvl].V_ref);
    out[lvl].V_live = buildVertexMap(zl, k);

    if (lvl < L-1) {
      cv::Mat zr2, zl2;
      cv::pyrDown(zr, zr2); cv::pyrDown(zl, zl2);
      zr = zr2; zl = zl2;
      // 内参除以2
      k.fx *= 0.5f; k.fy *= 0.5f; k.cx *= 0.5f; k.cy *= 0.5f;
    }
  }
}

void PointToPlaneICP::gnIterationOne(const Level& L, const Sophus::SE3f& T_ref_from_live,
                                     Eigen::Matrix<float,6,6>& H, Eigen::Matrix<float,6,1>& b,
                                     int& inliers, double& sqerr_sum) const
{
  H.setZero(); b.setZero();
  inliers = 0; sqerr_sum = 0.0;
  const float cos_thr = std::cos(deg2rad(opt_.normal_angle_thresh_deg));

  const int stride = std::max(1, opt_.sample_stride);

  for (int y=0; y<L.Z_live.rows; y+=stride) {
    const cv::Vec3f* vrow = L.V_live.ptr<cv::Vec3f>(y);
    for (int x=0; x<L.Z_live.cols; x+=stride) {
      cv::Vec3f v_live = vrow[x];
      if (!(v_live[2] > 0)) continue;

      // live 点 -> ref 相机坐标
      Eigen::Vector3f Vl(v_live[0], v_live[1], v_live[2]);
      Eigen::Vector3f Vr = T_ref_from_live * Vl;

      if (Vr.z() <= 0) continue;

      // 投影到 ref
      float u = L.K.fx * (Vr.x()/Vr.z()) + L.K.cx;
      float v = L.K.fy * (Vr.y()/Vr.z()) + L.K.cy;
      int uu = static_cast<int>(std::round(u));
      int vv = static_cast<int>(std::round(v));
      if (uu<=1 || uu>=L.Z_ref.cols-2 || vv<=1 || vv>=L.Z_ref.rows-2) continue;

      cv::Vec3f v_ref = L.V_ref.at<cv::Vec3f>(vv,uu);
      cv::Vec3f n_ref = L.N_ref.at<cv::Vec3f>(vv,uu);
      if (!(v_ref[2] > 0)) continue;
      if (n_ref == cv::Vec3f(0,0,0)) continue;

      // 法向一致性（朝向相机）
      Eigen::Vector3f n(n_ref[0], n_ref[1], n_ref[2]);
      if (n.norm() < 1e-6f) continue;

      // 深度一致（沿光线）
      float dz = std::abs(v_ref[2] - Vr.z());
      if (dz > opt_.depth_diff_thresh) continue;

      // 法线角度阈值（与观测视线或与 Vr-v_ref 的一致性均可）
      Eigen::Vector3f diff(Vr.x()-v_ref[0], Vr.y()-v_ref[1], Vr.z()-v_ref[2]);
      Eigen::Vector3f view_dir = -Eigen::Vector3f(v_ref[0], v_ref[1], v_ref[2]).normalized();
      if (n.dot(view_dir) < 0) n = -n; // 朝向相机
      if (n.dot(diff.normalized()) < cos_thr) ; // 可选，不强制

      // 点-面残差
      float r = n.dot(diff);

      // Huber 权重
      float w = 1.f;
      if (opt_.use_huber) {
        float a = std::abs(r);
        if (a > opt_.huber_delta) w = opt_.huber_delta / a;
      }

      // 雅可比： n^T [ I | -[Vr]_x ] δξ
      Eigen::Matrix3f Xx;
      Xx << 0, -Vr.z(), Vr.y(),
            Vr.z(), 0, -Vr.x(),
           -Vr.y(), Vr.x(), 0;
      Eigen::Matrix<float,3,6> JT3;
      JT3.block<3,3>(0,0) = Eigen::Matrix3f::Identity();
      JT3.block<3,3>(0,3) = -Xx;
      Eigen::Matrix<float,1,6> J = n.transpose() * JT3;

      H.noalias() += (w * J.transpose() * J);
      b.noalias() += (w * J.transpose() * r);

      ++inliers;
      sqerr_sum += double(w) * double(r) * double(r);
    }
  }
}

IcpResult PointToPlaneICP::Align(const cv::Mat& Z_ref, const cv::Mat& Z_live,
                                 const cv::Mat& mask_ref, const cv::Mat& mask_live,
                                 const Sophus::SE3f& T_init)
{
  IcpResult R; R.T_ref_from_live = T_init; R.ok = true;
  if (Z_ref.empty() || Z_live.empty()) {
    R.ok=false; R.fail_reason="empty depth"; return R;
  }

  // 构建金字塔
  buildPyramid(Z_ref, Z_live, K_, opt_.pyr_levels, lv_, mask_ref, mask_live);

  // 自粗到细
  for (int lvl = opt_.pyr_levels-1; lvl>=0; --lvl) {
    const auto& L = lv_[lvl];
    for (int it=0; it<opt_.iters_per_level; ++it) {
      Eigen::Matrix<float,6,6> H;
      Eigen::Matrix<float,6,1> b;
      int inl=0; double sq=0.0;
      gnIterationOne(L, R.T_ref_from_live, H, b, inl, sq);
      if (inl < 200) { R.ok=false; R.fail_reason="too few correspondences"; break; }

      Eigen::Matrix<float,6,1> dx = - H.ldlt().solve(b);
      if (!dx.allFinite()) { R.ok=false; R.fail_reason="solve failed"; break; }

      // 左乘更新：T_new = exp(dx) * T
      R.T_ref_from_live = Sophus::SE3f::exp(dx) * R.T_ref_from_live;

      R.stats.num_corr  = inl;
      R.stats.rmse      = (inl>0) ? std::sqrt(float(sq / std::max(1,inl))) : 0.f;
      R.stats.iters_done++;
      // 小步终止条件（可选）
      if (dx.head<3>().norm() < 1e-6 && dx.tail<3>().norm() < 1e-6) break;
    }
    if (!R.ok) break;
    // 阶段性放宽/收紧阈值（可选）
  }
  return R;
}
