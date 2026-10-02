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
  int bodyIndex_;
  sva::PTransformd X_b_f_;
  rbd::Jacobian jac_;
  ManipulabilityMeasure measure_;
  Eigen::Vector6d axes_;
  std::vector<std::string> measureJoints_;

  /** Number of dof of each joint of the path and index of its first column in the path Jacobian */
  std::vector<int> pathDof_;
  std::vector<int> pathCol_;
  /** Selected rows of the frame Jacobian and path Jacobian columns of the measure joints */
  std::vector<int> rows_;
  std::vector<int> cols_;

  Eigen::Vector6d maxTaskVelocity_;
  Eigen::VectorXd maxJointVelocity_;
  double target_;
  double w_;

  /** Path quantities: velocity, Jacobian time derivatives and gradient of w */
  Eigen::VectorXd alpha_;
  Eigen::MatrixXd jacDot_;
  Eigen::MatrixXd jacDDot_;
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

} // namespace tasks
