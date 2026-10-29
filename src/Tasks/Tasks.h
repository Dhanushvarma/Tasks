/*
 * Copyright 2012-2019 CNRS-UM LIRMM, CNRS-AIST JRL
 */

#pragma once

// includes
// Eigen
#include <Eigen/Core>
#include <Eigen/SVD>

// RBDyn
#include <tasks/config.hh>

#include <RBDyn/CoM.h>
#include <RBDyn/Jacobian.h>
#include <RBDyn/Momentum.h>
#include <RBDyn/VisServo.h>

#include <memory>

// forward declarations
// RBDyn
namespace rbd
{
class MultiBody;
struct MultiBodyConfig;
} // namespace rbd

namespace tasks
{

class TASKS_DLLAPI PositionTask
{
public:
  PositionTask(const rbd::MultiBody & mb,
               const std::string & bodyName,
               const Eigen::Vector3d & pos,
               const Eigen::Vector3d & bodyPoint = Eigen::Vector3d::Zero());

  void position(const Eigen::Vector3d & pos);
  const Eigen::Vector3d & position() const;

  void bodyPoint(const Eigen::Vector3d & point);
  const Eigen::Vector3d & bodyPoint() const;

  void update(const rbd::MultiBody & mb, const rbd::MultiBodyConfig & mbc);
  void update(const rbd::MultiBody & mb,
              const rbd::MultiBodyConfig & mbc,
              const std::vector<sva::MotionVecd> & normalAccB);
  void updateDot(const rbd::MultiBody & mb, const rbd::MultiBodyConfig & mbc);

  const Eigen::VectorXd & eval() const;
  const Eigen::VectorXd & speed() const;
  const Eigen::VectorXd & normalAcc() const;

  const Eigen::MatrixXd & jac() const;
  const Eigen::MatrixXd & jacDot() const;

private:
  Eigen::Vector3d pos_;
  sva::PTransformd point_;
  int bodyIndex_;
  rbd::Jacobian jac_;

  Eigen::VectorXd eval_;
  Eigen::VectorXd speed_;
  Eigen::VectorXd normalAcc_;
  Eigen::MatrixXd jacMat_;
  Eigen::MatrixXd jacDotMat_;
};

class TASKS_DLLAPI OrientationTask
{
public:
  OrientationTask(const rbd::MultiBody & mb, const std::string & bodyName, const Eigen::Quaterniond & ori);
  OrientationTask(const rbd::MultiBody & mb, const std::string & bodyName, const Eigen::Matrix3d & ori);

  void orientation(const Eigen::Quaterniond & ori);
  void orientation(const Eigen::Matrix3d & ori);
  const Eigen::Matrix3d & orientation() const;

  void update(const rbd::MultiBody & mb, const rbd::MultiBodyConfig & mbc);
  void update(const rbd::MultiBody & mb,
              const rbd::MultiBodyConfig & mbc,
              const std::vector<sva::MotionVecd> & normalAccB);
  void updateDot(const rbd::MultiBody & mb, const rbd::MultiBodyConfig & mbc);

  const Eigen::VectorXd & eval() const;
  const Eigen::VectorXd & speed() const;
  const Eigen::VectorXd & normalAcc() const;

  const Eigen::MatrixXd & jac() const;
  const Eigen::MatrixXd & jacDot() const;

private:
  Eigen::Matrix3d ori_;
  int bodyIndex_;
  rbd::Jacobian jac_;

  Eigen::VectorXd eval_;
  Eigen::VectorXd speed_;
  Eigen::VectorXd normalAcc_;
  Eigen::MatrixXd jacMat_;
  Eigen::MatrixXd jacDotMat_;
};

class TASKS_DLLAPI TransformTaskCommon
{
public:
  TransformTaskCommon(const rbd::MultiBody & mb,
                      const std::string & bodyName,
                      const sva::PTransformd & X_0_t,
                      const sva::PTransformd & X_b_p);

  void target(const sva::PTransformd & X_0_t);
  const sva::PTransformd & target() const;

  void X_b_p(const sva::PTransformd & X_b_p);
  const sva::PTransformd & X_b_p() const;

  const Eigen::VectorXd & eval() const;
  const Eigen::VectorXd & speed() const;
  const Eigen::VectorXd & normalAcc() const;

  const Eigen::MatrixXd & jac() const;

protected:
  sva::PTransformd X_0_t_;
  sva::PTransformd X_b_p_;
  int bodyIndex_;
  rbd::Jacobian jac_;

  Eigen::VectorXd eval_;
  Eigen::VectorXd speed_;
  Eigen::VectorXd normalAcc_;
  Eigen::MatrixXd jacMat_;
};

class TASKS_DLLAPI SurfaceTransformTask : public TransformTaskCommon
{
public:
  /**
   * Compute eval, speed, normalAcc and jac in moving 'p' frame.
   */
  SurfaceTransformTask(const rbd::MultiBody & mb,
                       const std::string & bodyName,
                       const sva::PTransformd & X_0_t,
                       const sva::PTransformd & X_b_p = sva::PTransformd::Identity());

  void update(const rbd::MultiBody & mb,
              const rbd::MultiBodyConfig & mbc,
              const std::vector<sva::MotionVecd> & normalAccB);

protected:
  Eigen::MatrixXd jacMatTmp_;
};

class TASKS_DLLAPI TransformTask : public TransformTaskCommon
{
public:
  /**
   * Compute eval, speed, normalAcc and jac in the world frame (0) or in the
   * user defined static frame 'c'.
   * @param E_0_c Optional rotation between the world farme and a user
   * defined frame 'c' to change the frame of the task.
   */
  TransformTask(const rbd::MultiBody & mb,
                const std::string & bodyName,
                const sva::PTransformd & X_0_t,
                const sva::PTransformd & X_b_p = sva::PTransformd::Identity(),
                const Eigen::Matrix3d & E_0_c = Eigen::Matrix3d::Identity());

  void E_0_c(const Eigen::Matrix3d & E_0_c);
  const Eigen::Matrix3d & E_0_c() const;

  void update(const rbd::MultiBody & mb,
              const rbd::MultiBodyConfig & mbc,
              const std::vector<sva::MotionVecd> & normalAccB);

private:
  Eigen::Matrix3d E_0_c_;
};

class TASKS_DLLAPI MultiRobotTransformTask
{
public:
  MultiRobotTransformTask(const std::vector<rbd::MultiBody> & mbs,
                          int r1Index,
                          int r2Index,
                          const std::string & r1BodyName,
                          const std::string & r2BodyName,
                          const sva::PTransformd & X_r1b_r1s,
                          const sva::PTransformd & X_r2b_r2s);

  int r1Index() const;
  int r2Index() const;

  void X_r1b_r1s(const sva::PTransformd & X_r1b_r1s);
  const sva::PTransformd & X_r1b_r1s() const;

  void X_r2b_r2s(const sva::PTransformd & X_r2b_r2s);
  const sva::PTransformd & X_r2b_r2s() const;

  void update(const std::vector<rbd::MultiBody> & mbs,
              const std::vector<rbd::MultiBodyConfig> & mbcs,
              const std::vector<std::vector<sva::MotionVecd>> & normalAccB);

  const Eigen::VectorXd & eval() const;
  const Eigen::VectorXd & speed() const;
  const Eigen::VectorXd & normalAcc() const;

  const Eigen::MatrixXd & jac(int index) const;

private:
  int r1Index_, r2Index_;
  int r1BodyIndex_, r2BodyIndex_;
  sva::PTransformd X_r1b_r1s_, X_r2b_r2s_;
  rbd::Jacobian jacR1B_, jacR2B_;

  Eigen::VectorXd eval_;
  Eigen::VectorXd speed_;
  Eigen::VectorXd normalAcc_;
  Eigen::MatrixXd jacMat1_, jacMat2_;
  std::vector<Eigen::MatrixXd> fullJacMat_;
};

class TASKS_DLLAPI SurfaceOrientationTask
{
public:
  SurfaceOrientationTask(const rbd::MultiBody & mb,
                         const std::string & bodyName,
                         const Eigen::Quaterniond & ori,
                         const sva::PTransformd & X_b_s);
  SurfaceOrientationTask(const rbd::MultiBody & mb,
                         const std::string & bodyName,
                         const Eigen::Matrix3d & ori,
                         const sva::PTransformd & X_b_s);

  void orientation(const Eigen::Quaterniond & ori);
  void orientation(const Eigen::Matrix3d & ori);
  const Eigen::Matrix3d & orientation() const;

  void update(const rbd::MultiBody & mb, const rbd::MultiBodyConfig & mbc);
  void update(const rbd::MultiBody & mb,
              const rbd::MultiBodyConfig & mbc,
              const std::vector<sva::MotionVecd> & normalAccB);
  void updateDot(const rbd::MultiBody & mb, const rbd::MultiBodyConfig & mbc);

  const Eigen::VectorXd & eval() const;
  const Eigen::VectorXd & speed() const;
  const Eigen::VectorXd & normalAcc() const;

  const Eigen::MatrixXd & jac() const;
  const Eigen::MatrixXd & jacDot() const;

private:
  Eigen::Matrix3d ori_;
  int bodyIndex_;
  rbd::Jacobian jac_;
  sva::PTransformd X_b_s_;

  Eigen::VectorXd eval_;
  Eigen::VectorXd speed_;
  Eigen::VectorXd normalAcc_;
  Eigen::MatrixXd jacMat_;
  Eigen::MatrixXd jacDotMat_;
};

class TASKS_DLLAPI GazeTask
{
public:
  GazeTask(const rbd::MultiBody & mb,
           const std::string & bodyName,
           const Eigen::Vector2d & point2d,
           double depthEstimate,
           const sva::PTransformd & X_b_gaze,
           const Eigen::Vector2d & point2d_ref = Eigen::Vector2d::Zero());
  GazeTask(const rbd::MultiBody & mb,
           const std::string & bodyName,
           const Eigen::Vector3d & point3d,
           const sva::PTransformd & X_b_gaze,
           const Eigen::Vector2d & point2d_ref = Eigen::Vector2d::Zero());

  GazeTask(const GazeTask & rhs);

  GazeTask(GazeTask &&) = default;

  GazeTask & operator=(const GazeTask & rhs);

  GazeTask & operator=(GazeTask &&) = default;

  void error(const Eigen::Vector2d & point2d, const Eigen::Vector2d & point2d_ref = Eigen::Vector2d::Zero());
  void error(const Eigen::Vector3d & point3d, const Eigen::Vector2d & point2d_ref = Eigen::Vector2d::Zero());

  void update(const rbd::MultiBody & mb,
              const rbd::MultiBodyConfig & mbc,
              const std::vector<sva::MotionVecd> & normalAccB);

  const Eigen::VectorXd & eval() const;
  const Eigen::VectorXd & speed() const;
  const Eigen::VectorXd & normalAcc() const;

  const Eigen::MatrixXd & jac() const;
  const Eigen::MatrixXd & jacDot() const;

private:
  std::unique_ptr<Eigen::Vector2d> point2d_;
  std::unique_ptr<Eigen::Vector2d> point2d_ref_;
  double depthEstimate_;
  int bodyIndex_;
  rbd::Jacobian jac_;
  sva::PTransformd X_b_gaze_;
  std::unique_ptr<Eigen::Matrix<double, 2, 6>> L_img_;
  std::unique_ptr<Eigen::Matrix<double, 6, 1>> surfaceVelocity_;
  std::unique_ptr<Eigen::Matrix<double, 1, 6>> L_Z_dot_;
  std::unique_ptr<Eigen::Matrix<double, 2, 6>> L_img_dot_;

  Eigen::VectorXd eval_;
  Eigen::VectorXd speed_;
  Eigen::VectorXd normalAcc_;
  Eigen::MatrixXd jacMat_;
  Eigen::MatrixXd jacDotMat_;
};

class TASKS_DLLAPI PositionBasedVisServoTask
{
public:
  PositionBasedVisServoTask(const rbd::MultiBody & mb,
                            const std::string & bodyName,
                            const sva::PTransformd & X_t_s,
                            const sva::PTransformd & X_b_s);

  PositionBasedVisServoTask(const PositionBasedVisServoTask & rhs);

  PositionBasedVisServoTask(PositionBasedVisServoTask &&) = default;

  PositionBasedVisServoTask & operator=(const PositionBasedVisServoTask & rhs);

  PositionBasedVisServoTask & operator=(PositionBasedVisServoTask &&) = default;

  void error(const sva::PTransformd & X_t_s);

  void update(const rbd::MultiBody & mb,
              const rbd::MultiBodyConfig & mbc,
              const std::vector<sva::MotionVecd> & normalAccB);

  const Eigen::VectorXd & eval() const;
  const Eigen::VectorXd & speed() const;
  const Eigen::VectorXd & normalAcc() const;

  const Eigen::MatrixXd & jac() const;
  const Eigen::MatrixXd & jacDot() const;

private:
  sva::PTransformd X_t_s_;
  sva::PTransformd X_b_s_;
  double angle_;
  Eigen::Vector3d axis_;
  int bodyIndex_;
  rbd::Jacobian jac_;
  std::unique_ptr<Eigen::Matrix<double, 6, 6>> L_pbvs_;
  std::unique_ptr<Eigen::Matrix<double, 6, 1>> surfaceVelocity_;
  Eigen::Matrix3d omegaSkew_;
  std::unique_ptr<Eigen::Matrix<double, 6, 6>> L_pbvs_dot_;

  Eigen::VectorXd eval_;
  Eigen::VectorXd speed_;
  Eigen::VectorXd normalAcc_;
  Eigen::MatrixXd jacMat_;
  Eigen::MatrixXd jacDotMat_;
};

class TASKS_DLLAPI PostureTask
{
public:
  PostureTask(const rbd::MultiBody & mb, std::vector<std::vector<double>> q);

  void posture(std::vector<std::vector<double>> q);
  const std::vector<std::vector<double>> posture() const;

  void update(const rbd::MultiBody & mb, const rbd::MultiBodyConfig & mbc);
  void updateDot(const rbd::MultiBody & mb, const rbd::MultiBodyConfig & mbc);

  const Eigen::VectorXd & eval() const;
  const Eigen::MatrixXd & jac() const;
  const Eigen::MatrixXd & jacDot() const;

private:
  std::vector<std::vector<double>> q_;

  Eigen::VectorXd eval_;
  Eigen::MatrixXd jacMat_;
  Eigen::MatrixXd jacDotMat_;
};

class TASKS_DLLAPI CoMTask
{
public:
  CoMTask(const rbd::MultiBody & mb, const Eigen::Vector3d & com);
  CoMTask(const rbd::MultiBody & mb, const Eigen::Vector3d & com, std::vector<double> weight);

  void com(const Eigen::Vector3d & com);
  const Eigen::Vector3d & com() const;
  const Eigen::Vector3d & actual() const;

  void updateInertialParameters(const rbd::MultiBody & mb);

  void update(const rbd::MultiBody & mb, const rbd::MultiBodyConfig & mbc);
  void update(const rbd::MultiBody & mb,
              const rbd::MultiBodyConfig & mbc,
              const Eigen::Vector3d & com,
              const std::vector<sva::MotionVecd> & normalAccB);
  void updateDot(const rbd::MultiBody & mb, const rbd::MultiBodyConfig & mbc);

  const Eigen::VectorXd & eval() const;
  const Eigen::VectorXd & speed() const;
  const Eigen::VectorXd & normalAcc() const;

  const Eigen::MatrixXd & jac() const;
  const Eigen::MatrixXd & jacDot() const;

private:
  Eigen::Vector3d com_;
  Eigen::Vector3d actual_;
  rbd::CoMJacobian jac_;

  Eigen::VectorXd eval_;
  Eigen::VectorXd speed_;
  Eigen::VectorXd normalAcc_;
  Eigen::MatrixXd jacMat_;
  Eigen::MatrixXd jacDotMat_;
};

class TASKS_DLLAPI MultiCoMTask
{
public:
  MultiCoMTask(const std::vector<rbd::MultiBody> & mbs, std::vector<int> robotIndexes, const Eigen::Vector3d & com);

  const std::vector<int> & robotIndexes() const;

  void com(const Eigen::Vector3d & com);
  const Eigen::Vector3d com() const;

  void updateInertialParameters(const std::vector<rbd::MultiBody> & mbs);

  void update(const std::vector<rbd::MultiBody> & mbs, const std::vector<rbd::MultiBodyConfig> & mbcs);
  void update(const std::vector<rbd::MultiBody> & mbs,
              const std::vector<rbd::MultiBodyConfig> & mbcs,
              const std::vector<std::vector<sva::MotionVecd>> & normalAccB);
  void update(const std::vector<rbd::MultiBody> & mbs,
              const std::vector<rbd::MultiBodyConfig> & mbcs,
              const std::vector<Eigen::Vector3d> & coms,
              const std::vector<std::vector<sva::MotionVecd>> & normalAccB);

  const Eigen::VectorXd & eval() const;
  const Eigen::VectorXd & speed() const;
  const Eigen::VectorXd & normalAcc() const;

  const Eigen::MatrixXd & jac(int index) const;

private:
  void update(const std::vector<rbd::MultiBody> & mbs,
              const std::vector<rbd::MultiBodyConfig> & mbcs,
              const std::vector<std::vector<sva::MotionVecd>> & normalAccB,
              const Eigen::Vector3d & multiCoM);

  void computeRobotsWeight(const std::vector<rbd::MultiBody> & mbs);

private:
  Eigen::Vector3d com_;
  std::vector<int> robotIndexes_;
  std::vector<double> robotsWeight_;
  std::vector<rbd::CoMJacobian> jac_;

  Eigen::VectorXd eval_;
  Eigen::VectorXd speed_;
  Eigen::VectorXd normalAcc_;
  std::vector<Eigen::MatrixXd> jacMat_;
};

class TASKS_DLLAPI MomentumTask
{
public:
  MomentumTask(const rbd::MultiBody & mb, const sva::ForceVecd mom);

  void momentum(const sva::ForceVecd & mom);
  const sva::ForceVecd momentum() const;

  void update(const rbd::MultiBody & mb, const rbd::MultiBodyConfig & mbc);
  void update(const rbd::MultiBody & mb,
              const rbd::MultiBodyConfig & mbc,
              const std::vector<sva::MotionVecd> & normalAccB);
  void updateDot(const rbd::MultiBody & mb, const rbd::MultiBodyConfig & mbc);

  const Eigen::VectorXd & eval() const;
  const Eigen::VectorXd & speed() const;
  const Eigen::VectorXd & normalAcc() const;

  const Eigen::MatrixXd & jac() const;
  const Eigen::MatrixXd & jacDot() const;

private:
  sva::ForceVecd momentum_;
  rbd::CentroidalMomentumMatrix momentumMatrix_;
  Eigen::VectorXd eval_;
  Eigen::VectorXd speed_;
  Eigen::VectorXd normalAcc_;
  Eigen::MatrixXd jacMat_;
  Eigen::MatrixXd jacDotMat_;
};

class TASKS_DLLAPI LinVelocityTask
{
public:
  LinVelocityTask(const rbd::MultiBody & mb,
                  const std::string & bodyName,
                  const Eigen::Vector3d & vel,
                  const Eigen::Vector3d & bodyPoint = Eigen::Vector3d::Zero());

  void velocity(const Eigen::Vector3d & s);
  const Eigen::Vector3d & velocity() const;

  void bodyPoint(const Eigen::Vector3d & point);
  const Eigen::Vector3d & bodyPoint() const;

  void update(const rbd::MultiBody & mb, const rbd::MultiBodyConfig & mbc);
  void update(const rbd::MultiBody & mb,
              const rbd::MultiBodyConfig & mbc,
              const std::vector<sva::MotionVecd> & normalAccB);
  void updateDot(const rbd::MultiBody & mb, const rbd::MultiBodyConfig & mbc);

  const Eigen::VectorXd & eval() const;
  const Eigen::VectorXd & speed() const;
  const Eigen::VectorXd & normalAcc() const;

  const Eigen::MatrixXd & jac() const;
  const Eigen::MatrixXd & jacDot() const;

private:
  Eigen::Vector3d vel_;
  sva::PTransformd point_;
  int bodyIndex_;
  rbd::Jacobian jac_;

  Eigen::VectorXd eval_;
  Eigen::VectorXd speed_;
  Eigen::VectorXd normalAcc_;
  Eigen::MatrixXd jacMat_;
  Eigen::MatrixXd jacDotMat_;
};

class TASKS_DLLAPI OrientationTrackingTask
{
public:
  OrientationTrackingTask(const rbd::MultiBody & mb,
                          const std::string & bodyName,
                          const Eigen::Vector3d & bodyPoint,
                          const Eigen::Vector3d & bodyAxis,
                          const std::vector<std::string> & trackingJointsName,
                          const Eigen::Vector3d & trackedPoint);

  void trackedPoint(const Eigen::Vector3d & tp);
  const Eigen::Vector3d & trackedPoint() const;

  void bodyPoint(const Eigen::Vector3d & bp);
  const Eigen::Vector3d & bodyPoint() const;

  void bodyAxis(const Eigen::Vector3d & ba);
  const Eigen::Vector3d & bodyAxis() const;

  void update(const rbd::MultiBody & mb, const rbd::MultiBodyConfig & mbc);
  void updateDot(const rbd::MultiBody & mb, const rbd::MultiBodyConfig & mbc);

  virtual const Eigen::MatrixXd & jac() const;
  virtual const Eigen::MatrixXd & jacDot() const;
  virtual const Eigen::VectorXd & eval() const;

private:
  void zeroJacobian(Eigen::MatrixXd & jac) const;

private:
  int bodyIndex_;
  sva::PTransformd bodyPoint_;
  Eigen::Vector3d bodyAxis_;
  std::vector<int> zeroJacIndex_;
  Eigen::Vector3d trackedPoint_;
  rbd::Jacobian jac_;

  Eigen::VectorXd eval_;
  Eigen::MatrixXd shortJacMat_;
  Eigen::MatrixXd jacMat_;
  Eigen::MatrixXd jacDotMat_;
};

class TASKS_DLLAPI RelativeDistTask
{
public:
  // Give information on 1 body (bodyName, r_b1_p, r_0_b2p)
  typedef std::tuple<std::string, Eigen::Vector3d, Eigen::Vector3d> rbInfo;

public:
  RelativeDistTask(const rbd::MultiBody & mb,
                   const double timestep,
                   const rbInfo & rbi1,
                   const rbInfo & rbi2,
                   const Eigen::Vector3d & u1 = Eigen::Vector3d::Zero(),
                   const Eigen::Vector3d & u2 = Eigen::Vector3d::Zero());

  void robotPoint(const int bIndex, const Eigen::Vector3d & point);
  void envPoint(const int bIndex, const Eigen::Vector3d & point);
  void vector(const int bIndex, const Eigen::Vector3d & u);

  void update(const rbd::MultiBody & mb,
              const rbd::MultiBodyConfig & mbc,
              const std::vector<sva::MotionVecd> & normalAccB);

  const Eigen::VectorXd & eval() const;
  const Eigen::VectorXd & speed() const;
  const Eigen::VectorXd & normalAcc() const;

  const Eigen::MatrixXd & jac() const;

private:
  struct RelativeBodiesInfo
  {
    RelativeBodiesInfo() {}

    RelativeBodiesInfo(const rbd::MultiBody & mb, const rbInfo & rbi, const Eigen::Vector3d & fixedVector);

    int b1Index;
    Eigen::Vector3d r_b1_p, r_0_b2p, n;
    rbd::Jacobian jac;
    Eigen::Vector3d u;
    Eigen::Vector3d offn;
  };

private:
  double timestep_;
  bool isVectorFixed_;

  Eigen::VectorXd eval_;
  Eigen::VectorXd speed_;
  Eigen::VectorXd normalAcc_;
  Eigen::MatrixXd jacMat_;

  RelativeBodiesInfo rbInfo_[2];
};

class TASKS_DLLAPI VectorOrientationTask
{
public:
  VectorOrientationTask(const rbd::MultiBody & mb,
                        const std::string & bodyName,
                        const Eigen::Vector3d & bodyVector,
                        const Eigen::Vector3d & targetVector);

  void update(const rbd::MultiBody & mb,
              const rbd::MultiBodyConfig & mbc,
              const std::vector<sva::MotionVecd> & normalAccB);

  void bodyVector(const Eigen::Vector3d & vector);
  const Eigen::Vector3d & bodyVector() const;
  void target(const Eigen::Vector3d & vector);
  const Eigen::Vector3d & target() const;
  const Eigen::Vector3d & actual() const;

  const Eigen::VectorXd & eval() const;
  const Eigen::VectorXd & speed() const;
  const Eigen::VectorXd & normalAcc() const;

  const Eigen::MatrixXd & jac() const;

private:
  Eigen::Matrix3d skewMatrix(const Eigen::Vector3d & v);

private:
  Eigen::Vector3d actualVector_, bodyVector_, targetVector_;
  int bodyIndex_;
  rbd::Jacobian jac_;

  Eigen::VectorXd eval_;
  Eigen::VectorXd speed_;
  Eigen::VectorXd normalAcc_;
  Eigen::MatrixXd jacMat_;
};

/** Jacobian of a frame in its own coordinates, over the joints between the root and the frame body, with its time
 * derivatives. It also holds the measure joints (Jacobian columns) and the selected axes (Jacobian rows) of a
 * manipulability task.
 *
 * The column of a path degree of freedom c only depends on the joints after its own joint,
 * \f$ \partial J_c / \partial q_k = J_c \times J_k \f$ (motion cross product) for k after c, so the time derivatives
 * come from one backward pass over the path.
 *
 * With a reference frame r, the Jacobian is the one of the frame f relative to r, still in frame coordinates:
 * \f$ J = J_f - X_{r \to f} J_r \f$, where \f$ J_r \f$ is the Jacobian of r in its own coordinates and
 * \f$ X_{r \to f} \f$ transforms motion vectors from r to f coordinates. This is the relative Jacobian of Jamisola and
 * Roberts ("A more compact expression of relative Jacobian based on individual manipulator Jacobians", RAS 2015), in
 * frame coordinates. The path is then made of the joints between the two bodies: the joints between the root and the
 * reference body that are not between the root and the frame body (ordered from the root to the reference body), then
 * the joints between the root and the frame body that are not between the root and the reference body. The joints
 * shared by both paths move both frames together, they do not change J.
 */
class TASKS_DLLAPI FrameJacobianDerivatives
{
public:
  /**
   * @param mb Robot
   * @param bodyName Body the frame is attached to
   * @param X_b_f Frame pose in body coordinates
   * @param measureJoints Measure joints. If empty, every joint of the path, excluding a free root joint (floating base)
   * @param axes Selected axes, ordered as sva::MotionVecd (rotation then translation). A non-zero entry selects the
   * axis
   * @param owner Name used in error messages
   * @param refBodyName Body the reference frame is attached to. If empty, the Jacobian is the one of the frame
   * (relative to the world)
   * @param X_rb_r Reference frame pose in the coordinates of its body
   * @throw std::domain_error if a measure joint is not on the path, appears twice or has no degree of freedom, or if
   * more axes are selected than measure joints degrees of freedom
   */
  FrameJacobianDerivatives(const rbd::MultiBody & mb,
                           const std::string & bodyName,
                           const sva::PTransformd & X_b_f,
                           const std::vector<std::string> & measureJoints,
                           const Eigen::Vector6d & axes,
                           const std::string & owner,
                           const std::string & refBodyName = "",
                           const sva::PTransformd & X_rb_r = sva::PTransformd::Identity());

  /** Compute the frame pose, the frame Jacobian, its time derivative and its second time derivative at zero joint
   * acceleration */
  void update(const rbd::MultiBody & mb, const rbd::MultiBodyConfig & mbc);

  /** Frame Jacobian in frame coordinates, one column per path degree of freedom (relative to the reference frame if
   * there is one) */
  const Eigen::MatrixXd & jacobian() const { return J_; }
  /** Time derivative of jacobian() */
  const Eigen::MatrixXd & jacobianDot() const { return jacDot_; }
  /** Second time derivative of jacobian() at zero joint acceleration */
  const Eigen::MatrixXd & jacobianDDot() const { return jacDDot_; }
  /** Velocity of the path degrees of freedom */
  const Eigen::VectorXd & alpha() const { return alpha_; }
  /** Frame pose in world coordinates */
  const sva::PTransformd & X_0_f() const { return X_0_f_; }
  /** Frame pose in reference frame coordinates, in world coordinates without a reference frame */
  const sva::PTransformd & X_r_f() const { return relative_ ? X_r_f_ : X_0_f_; }
  /** True if the Jacobian is relative to a reference frame */
  bool relative() const { return relative_; }

  /** Partial derivative of the path column c of jacobian() with respect to the path degree of freedom k, a motion cross
   * product of the two columns or zero. Without a reference frame it is not zero when the joint of k is after the joint
   * of c. With a reference frame the path from the reference body to the frame body acts as one chain: a column on the
   * frame side depends on the frame side joints after its own joint, a column on the reference side depends on the
   * other degrees of freedom of its own joint, on the reference side joints closer to the root and on every frame side
   * joint.
   *
   * @return false if the derivative is zero, d is then unchanged
   */
  bool columnDerivative(int c, int k, Eigen::Vector6d & d) const;

  /** Number of degrees of freedom of each joint of the path */
  const std::vector<int> & jointDof() const { return jointDof_; }
  /** First path column of each joint of the path */
  const std::vector<int> & jointCol() const { return jointCol_; }
  /** Index in the path of the joint of each path column */
  const std::vector<int> & colJoint() const { return colJoint_; }
  /** Selected axes */
  const std::vector<int> & rows() const { return rows_; }
  /** Path columns of the measure joints */
  const std::vector<int> & cols() const { return cols_; }
  /** Measure joints, ordered as the path */
  const std::vector<std::string> & measureJoints() const { return measureJoints_; }
  const Eigen::Vector6d & axes() const { return axes_; }

  /** Copy a matrix with one column per path degree of freedom into a matrix with one column per robot degree of
   * freedom */
  void fullJacobian(const rbd::MultiBody & mb, const Eigen::MatrixXd & pathMat, Eigen::MatrixXd & fullMat) const;

private:
  /** Jacobian of a frame in its own coordinates, over the joints between the root and its body, with its time
   * derivatives */
  struct BodyPath
  {
    std::vector<int> jointDof;
    std::vector<int> jointCol;
    Eigen::MatrixXd J;
    Eigen::MatrixXd jacDot;
    Eigen::MatrixXd jacDDot;
    Eigen::VectorXd alpha;
  };

  static void initBodyPath(const rbd::MultiBody & mb, const rbd::Jacobian & jac, BodyPath & path);
  static void updateBodyPath(const rbd::MultiBody & mb,
                             const rbd::MultiBodyConfig & mbc,
                             const sva::PTransformd & X_0_p,
                             rbd::Jacobian & jac,
                             BodyPath & path);
  void updateRelative(const rbd::MultiBody & mb, const rbd::MultiBodyConfig & mbc);

private:
  int bodyIndex_;
  sva::PTransformd X_b_f_;
  rbd::Jacobian jac_;
  Eigen::Vector6d axes_;
  std::vector<std::string> measureJoints_;

  /** Joint index of each joint of the path */
  std::vector<int> path_;
  std::vector<int> jointDof_;
  std::vector<int> jointCol_;
  std::vector<int> colJoint_;
  std::vector<int> rows_;
  std::vector<int> cols_;

  sva::PTransformd X_0_f_;
  Eigen::MatrixXd J_;
  Eigen::MatrixXd jacDot_;
  Eigen::MatrixXd jacDDot_;
  Eigen::VectorXd alpha_;

  /** Reference frame */
  bool relative_;
  int refBodyIndex_;
  sva::PTransformd X_rb_r_;
  rbd::Jacobian refJac_;
  sva::PTransformd X_r_f_;
  /** Paths from the root to the frame body and to the reference body */
  BodyPath framePath_;
  BodyPath refPath_;
  /** For each path column: true on the reference side, its column in framePath_ or refPath_, and the index of its joint
   * in the path from the root to its body */
  std::vector<bool> colRef_;
  std::vector<int> colSide_;
  std::vector<int> colSideJoint_;
  /** \f$ X_{r \to f} J_r \f$ and its time derivatives on the path columns (zero on the frame side) */
  Eigen::MatrixXd XJr_;
  Eigen::MatrixXd XJrDot_;
  Eigen::MatrixXd XJrDDot_;
};

/** Manipulability measures supported by ManipulabilityTask */
enum class ManipulabilityMeasure
{
  /** Yoshikawa's velocity manipulability \f$ w = \sqrt{\det(\tilde{J}\tilde{J}^T)} \f$, the product of the singular
   * values of \f$ \tilde{J} \f$ (Yoshikawa, "Manipulability of Robotic Mechanisms", IJRR 1985) */
  Yoshikawa
};

/** Manipulability of a frame as a scalar function \f$ w(q) \f$ of the robot configuration.
 *
 * The measure is computed on the normalized Jacobian
 * \f[ \tilde{J} = T_r P J_f S T_\theta^{-1} \f]
 * where:
 * - \f$ J_f \f$ is the Jacobian of the frame, expressed in the frame coordinates (rows ordered as sva::MotionVecd)
 * - \f$ P \f$ selects the frame axes and \f$ S \f$ the columns of the measure joints
 * - \f$ T_r = \mathrm{diag}(1/\dot{r}_0) \f$ and \f$ T_\theta = \mathrm{diag}(1/\dot{\theta}_0) \f$ normalize the
 *   frame and joint velocities by their maximum values (both are identity by default)
 *
 * The task error is \f$ w^* - w \f$. jac() is the analytic gradient of \f$ w \f$ and normalAcc() the analytic
 * \f$ \dot{J} \alpha \f$ term, they are defined for every joint of the robot, including the joints that are not
 * measure joints.
 */
class TASKS_DLLAPI ManipulabilityTask
{
public:
  /**
   * @param mb Robot
   * @param bodyName Body the frame is attached to
   * @param X_b_f Frame pose in body coordinates
   * @param measureJoints Joints whose Jacobian columns define the measure. If empty, every joint between the root and
   * the body, excluding a free root joint (floating base)
   * @param axes Frame axes used by the measure, ordered as sva::MotionVecd (rotation then translation). A non-zero
   * entry selects the axis
   * @param measure Manipulability measure
   * @throw std::domain_error if a measure joint is not between the root and the body, appears twice or has no degree of
   * freedom, or if more axes are selected than measure joints degrees of freedom
   */
  ManipulabilityTask(const rbd::MultiBody & mb,
                     const std::string & bodyName,
                     const sva::PTransformd & X_b_f = sva::PTransformd::Identity(),
                     const std::vector<std::string> & measureJoints = {},
                     const Eigen::Vector6d & axes = Eigen::Vector6d::Ones(),
                     ManipulabilityMeasure measure = ManipulabilityMeasure::Yoshikawa);

  void update(const rbd::MultiBody & mb, const rbd::MultiBodyConfig & mbc);

  /** Set the target manipulability \f$ w^* \f$ */
  void target(double target);
  double target() const;

  /** Manipulability computed by the last call to update */
  double manipulability() const;

  /** Set the maximum frame velocity \f$ \dot{r}_0 \f$ along each axis (entries of non-selected axes are unused)
   *
   * @throw std::domain_error if a selected entry is not strictly positive
   */
  void maxTaskVelocity(const Eigen::Vector6d & velocity);
  const Eigen::Vector6d & maxTaskVelocity() const;

  /** Set the maximum velocity \f$ \dot{\theta}_0 \f$ of each measure joint degree of freedom, ordered as
   * measureJoints()
   *
   * @throw std::domain_error if the size is not measureDof() or an entry is not strictly positive
   */
  void maxJointVelocity(const Eigen::VectorXd & velocity);
  const Eigen::VectorXd & maxJointVelocity() const;

  ManipulabilityMeasure measure() const;
  const Eigen::Vector6d & axes() const;
  /** Measure joints, ordered from the root to the body */
  const std::vector<std::string> & measureJoints() const;
  /** Number of degrees of freedom of the measure joints */
  int measureDof() const;

  const Eigen::VectorXd & eval() const;
  const Eigen::VectorXd & speed() const;
  const Eigen::VectorXd & normalAcc() const;

  const Eigen::MatrixXd & jac() const;

private:
  /** Compute w_ and its gradient with respect to Jt_ (Gt_), return the second derivative of w along Dt_ */
  double computeYoshikawa();

private:
  FrameJacobianDerivatives frameJac_;
  ManipulabilityMeasure measure_;

  Eigen::Vector6d maxTaskVelocity_;
  Eigen::VectorXd maxJointVelocity_;
  double target_;
  double w_;

  /** Gradient of w with respect to the frame Jacobian and to the path degrees of freedom */
  Eigen::MatrixXd G_;
  Eigen::MatrixXd grad_;
  /** Normalized Jacobian, its time derivative and the gradient of w with respect to it */
  Eigen::MatrixXd Jt_;
  Eigen::MatrixXd Dt_;
  Eigen::MatrixXd Gt_;
  Eigen::JacobiSVD<Eigen::MatrixXd> svd_;
  /** U^T Dt_ V */
  Eigen::MatrixXd Dh_;

  Eigen::VectorXd eval_;
  Eigen::VectorXd speed_;
  Eigen::VectorXd normalAcc_;
  Eigen::MatrixXd jacMat_;
};

/** Manipulability ellipsoids supported by ManipulabilityEllipsoidTask */
enum class ManipulabilityEllipsoidType
{
  /** Velocity manipulability ellipsoid \f$ M = \tilde{J}\tilde{J}^T \f$ */
  Velocity,
  /** Force manipulability ellipsoid \f$ M = (\tilde{J}\tilde{J}^T)^{-1} \f$, regularized at singular configurations */
  Force
};

/** Errors between the manipulability ellipsoid M and its target \f$ \hat{M} \f$ */
enum class ManipulabilityEllipsoidError
{
  /** Logarithmic map \f$ \mathrm{Log}_M(\hat{M}) \f$ on the manifold of SPD matrices with the affine-invariant metric
   * (geometry-aware tracking) */
  GeometryAware,
  /** Euclidean difference \f$ \hat{M} - M \f$, which ignores that ellipsoids are SPD matrices */
  Euclidean
};

/** Manipulability ellipsoid of a frame, a symmetric positive definite (SPD) matrix \f$ M(q) \f$, tracked towards a
 * target ellipsoid \f$ \hat{M} \f$ (Jaquier et al., "Geometry-aware manipulability learning, tracking, and transfer",
 * IJRR 2021).
 *
 * The ellipsoid is computed on
 * \f[ \tilde{J} = P R J_f S \Theta \f]
 * where:
 * - \f$ J_f \f$ is the Jacobian of the frame in frame coordinates, and \f$ R \f$ rotates its rows to world axes: the
 *   rows of \f$ R J_f \f$ are the angular and linear velocities of the frame origin in world coordinates
 * - \f$ P \f$ selects the world axes and \f$ S \f$ the columns of the measure joints
 * - \f$ \Theta \f$ scales the columns: \f$ \Theta = \mathrm{diag}(\dot{\theta}_0) \f$ with the maximum joint velocities
 *   for the velocity ellipsoid, \f$ \Theta = \mathrm{diag}(1 / \tau_0) \f$ with the maximum joint torques for the force
 *   ellipsoid (identity by default)
 *
 * For D selected axes, \f$ N = \tilde{J}\tilde{J}^T \f$ is a D x D matrix. The velocity ellipsoid is \f$ M = N \f$.
 * The force ellipsoid is its inverse (Jaquier et al., eq. 35, and Sec. 4.3 for the torque scaling). With the
 * translation axes, it holds the forces at the frame origin, with zero moment, that the measure joints balance with
 * torques \f$ \|\mathrm{diag}(1 / \tau_0) \, \tau\| \le 1 \f$. The axes are selected before the inversion, so this is
 * not a block of the inverse of the 6D N. The force ellipsoid is regularized so that it stays finite at singular
 * configurations, \f$ M = (N + \epsilon I)^{-1} \f$ with \f$ \epsilon = 10^{-6} \, \mathrm{tr}(N) / D \f$, which
 * changes it by a relative 1e-6 times the condition number of N at most.
 *
 * With a reference frame r of the same robot, the ellipsoid is the one of the frame relative to r: \f$ J_f \f$ is the
 * Jacobian of the frame relative to r (see FrameJacobianDerivatives) and \f$ R \f$ rotates its rows to the axes of r,
 * so the rows of \f$ R J_f \f$ are the angular and linear velocities of the frame origin relative to r, in r
 * coordinates, and the selected axes are axes of r. The joints shared by the paths from the root to both frames do not
 * move one frame relative to the other, they are not measure joints. With a frame on each hand, this is the bimanual
 * relative manipulability of Li et al. ("ManiDP: Manipulability-Aware Diffusion Policy for Posture-Dependent Bimanual
 * Manipulation", 2025, eqs. 2 to 4): the velocity ellipsoid describes how fast one hand can move relative to the
 * other, the force ellipsoid the force one hand can apply against the other, which takes the reaction at the same
 * point: a squeeze or a pull along the line between the hands, and also a moment across it.
 *
 * The task works on the Mandel vectorization vec() of symmetric matrices (see tasks::spd::vec): eval() is
 * \f$ \mathrm{vec}(\mathrm{Log}_M(\hat{M})) \f$ (or \f$ \mathrm{vec}(\hat{M} - M) \f$ with the Euclidean error),
 * speed() is \f$ \mathrm{vec}(\dot{M}) \f$, normalAcc() is \f$ \mathrm{vec}(\ddot{M}) \f$ at zero joint acceleration
 * and jac() is \f$ \partial \mathrm{vec}(M) / \partial q \f$, defined for every joint of the robot. In a trajectory
 * task this gives the acceleration-based tracking controller of Jaquier et al. (eqs. 46 and 47).
 */
class TASKS_DLLAPI ManipulabilityEllipsoidTask
{
public:
  /**
   * @param mb Robot
   * @param bodyName Body the frame is attached to
   * @param X_b_f Frame pose in body coordinates
   * @param measureJoints Joints whose Jacobian columns define the ellipsoid. If empty, every joint between the root and
   * the body (between the reference body and the body with a reference frame), excluding a free root joint (floating
   * base)
   * @param axes World axes of the ellipsoid (axes of the reference frame with a reference frame), ordered as
   * sva::MotionVecd (rotation then translation). A non-zero entry selects the axis. Translation by default
   * @param type Manipulability ellipsoid
   * @param refBodyName Body the reference frame is attached to, for the ellipsoid of the frame relative to the
   * reference frame. If empty, the ellipsoid is the one of the frame in world axes
   * @param X_rb_r Reference frame pose in the coordinates of its body
   * @throw std::domain_error if a measure joint is not between the root (or the reference body) and the body, appears
   * twice or has no degree of freedom, or if more axes are selected than measure joints degrees of freedom
   */
  ManipulabilityEllipsoidTask(const rbd::MultiBody & mb,
                              const std::string & bodyName,
                              const sva::PTransformd & X_b_f = sva::PTransformd::Identity(),
                              const std::vector<std::string> & measureJoints = {},
                              const Eigen::Vector6d & axes = (Eigen::Vector6d() << 0., 0., 0., 1., 1., 1.).finished(),
                              ManipulabilityEllipsoidType type = ManipulabilityEllipsoidType::Velocity,
                              const std::string & refBodyName = "",
                              const sva::PTransformd & X_rb_r = sva::PTransformd::Identity());

  void update(const rbd::MultiBody & mb, const rbd::MultiBodyConfig & mbc);

  /** Set the target ellipsoid \f$ \hat{M} \f$ (identity at construction)
   *
   * @throw std::domain_error if the target is not a size() x size() symmetric positive definite matrix
   */
  void target(const Eigen::MatrixXd & target);
  const Eigen::MatrixXd & target() const;

  /** Set the target to the ellipsoid computed by the last call to update. Its eigenvalues are bounded below by 1e-6
   * times the largest one (and 1e-12), so the target is positive definite at singular configurations */
  void resetTarget();

  /** Ellipsoid computed by the last call to update */
  const Eigen::MatrixXd & ellipsoid() const;

  /** Affine-invariant distance between the ellipsoid and the target, computed by the last call to update (whatever the
   * error) */
  double distance() const;

  void error(ManipulabilityEllipsoidError error);
  ManipulabilityEllipsoidError error() const;

  /** Set the maximum velocity \f$ \dot{\theta}_0 \f$ of each measure joint degree of freedom, ordered as
   * measureJoints(), for the velocity ellipsoid
   *
   * @throw std::domain_error if the ellipsoid is not the velocity ellipsoid, the size is not measureDof() or an entry
   * is not strictly positive and finite
   */
  void maxJointVelocity(const Eigen::VectorXd & velocity);
  const Eigen::VectorXd & maxJointVelocity() const;

  /** Set the maximum torque \f$ \tau_0 \f$ of each measure joint degree of freedom, ordered as measureJoints(), for the
   * force ellipsoid
   *
   * @throw std::domain_error if the ellipsoid is not the force ellipsoid, the size is not measureDof() or an entry is
   * not strictly positive and finite
   */
  void maxJointTorque(const Eigen::VectorXd & torque);
  const Eigen::VectorXd & maxJointTorque() const;

  ManipulabilityEllipsoidType type() const;
  const Eigen::Vector6d & axes() const;
  /** Measure joints, ordered from the root to the body (with a reference frame, the joints on the reference body side
   * from the root to the reference body, then the joints on the body side) */
  const std::vector<std::string> & measureJoints() const;
  /** Number of degrees of freedom of the measure joints */
  int measureDof() const;
  /** Size D of the ellipsoid matrix, the number of selected axes */
  int size() const;
  /** True if the ellipsoid is relative to a reference frame */
  bool relative() const;

  const Eigen::VectorXd & eval() const;
  const Eigen::VectorXd & speed() const;
  const Eigen::VectorXd & normalAcc() const;

  const Eigen::MatrixXd & jac() const;

private:
  FrameJacobianDerivatives frameJac_;
  ManipulabilityEllipsoidType type_;
  ManipulabilityEllipsoidError error_;
  Eigen::VectorXd maxJointVelocity_;
  Eigen::VectorXd maxJointTorque_;
  /** Diagonal of \f$ \Theta \f$ */
  Eigen::VectorXd scale_;

  Eigen::MatrixXd target_;
  Eigen::MatrixXd M_;
  double distance_;

  /** Measure columns of the frame Jacobian, scaled, and their time derivatives in frame coordinates */
  Eigen::MatrixXd Jh_;
  Eigen::MatrixXd JhDot_;
  Eigen::MatrixXd JhDDot_;
  /** Ellipsoid Jacobian \f$ \tilde{J} \f$ and its time derivatives */
  Eigen::MatrixXd Jt_;
  Eigen::MatrixXd JtDot_;
  Eigen::MatrixXd JtDDot_;
  /** Partial derivative of \f$ \tilde{J} \f$ with respect to one degree of freedom, in frame then in world (or
   * reference frame) coordinates */
  Eigen::MatrixXd A_;
  Eigen::MatrixXd At_;
  /** Jacobian of vec(M) with respect to the path degrees of freedom */
  Eigen::MatrixXd pathJac_;

  Eigen::VectorXd eval_;
  Eigen::VectorXd speed_;
  Eigen::VectorXd normalAcc_;
  Eigen::MatrixXd jacMat_;
};

} // namespace tasks
