/*
 * Copyright 2012-2019 CNRS-UM LIRMM, CNRS-AIST JRL
 */

// includes
// std
#include <cmath>
#include <fstream>
#include <iostream>
#include <tuple>

// boost
#define BOOST_TEST_MODULE TasksTest
#include <boost/math/constants/constants.hpp>
#include <boost/test/unit_test.hpp>

// SpaceVecAlg
#include <SpaceVecAlg/SpaceVecAlg>

// RBDyn
#include <RBDyn/FK.h>
#include <RBDyn/FV.h>
#include <RBDyn/NumericalIntegration.h>

// Tasks
#include "Tasks/Tasks.h"

// Arms
#include "arms.h"

template<typename Task>
struct TanAccel
{
  Eigen::VectorXd tanAcc(Task & task, const std::vector<Eigen::VectorXd> & alphaD)
  { return Eigen::VectorXd(task.jac() * alphaD[0]); }
};

template<typename Task>
struct MRTanAccel
{
  MRTanAccel(int taskDim) : tanAccV(taskDim) {}

  Eigen::VectorXd tanAcc(Task & task, const std::vector<Eigen::VectorXd> & alphaD)
  {
    tanAccV.setZero();
    for(std::size_t i = 0; i < alphaD.size(); ++i) { tanAccV += task.jac(int(i)) * alphaD[i]; }
    return tanAccV;
  }

  Eigen::VectorXd tanAccV;
};

/// run the task update(mb, mbc) method
template<typename Task>
struct ClassicUpdater : public TanAccel<Task>
{
  void operator()(Task & task, const std::vector<rbd::MultiBody> & mbs, const std::vector<rbd::MultiBodyConfig> & mbcs)
  { task.update(mbs[0], mbcs[0]); }
};

/// run the task update(mbs, mbcs) method
template<typename Task>
struct MRClassicUpdater : public MRTanAccel<Task>
{
  MRClassicUpdater(int taskDim) : MRTanAccel<Task>(taskDim) {}

  void operator()(Task & task, const std::vector<rbd::MultiBody> & mbs, const std::vector<rbd::MultiBodyConfig> & mbcs)
  { task.update(mbs, mbcs); }
};

/// Compute normal acceleration (like QPSolverData::computeNormalAccB)
void computeNormalAccB(const rbd::MultiBody & mb,
                       const rbd::MultiBodyConfig & mbc,
                       std::vector<sva::MotionVecd> & normalAccB)
{
  const std::vector<int> & pred = mb.predecessors();
  const std::vector<int> & succ = mb.successors();

  for(size_t i = 0; i < static_cast<size_t>(mb.nrJoints()); ++i)
  {
    const sva::PTransformd & X_p_i = mbc.parentToSon[i];
    const sva::MotionVecd & vj_i = mbc.jointVelocity[i];
    const sva::MotionVecd & vb_i = mbc.bodyVelB[i];

    size_t succ_i = static_cast<size_t>(succ[i]);
    size_t pred_i = static_cast<size_t>(pred[i]);
    if(pred[i] != -1)
      normalAccB[succ_i] = X_p_i * normalAccB[pred_i] + vb_i.cross(vj_i);
    else
      normalAccB[succ_i] = vb_i.cross(vj_i);
  }
}

/// Multi-robot version of computeNormalAccB
void computeNormalAccB(const std::vector<rbd::MultiBody> & mbs,
                       const std::vector<rbd::MultiBodyConfig> & mbcs,
                       std::vector<std::vector<sva::MotionVecd>> & normalAccB)
{
  for(std::size_t i = 0; i < mbs.size(); ++i) { computeNormalAccB(mbs[i], mbcs[i], normalAccB[i]); }
}

/// run the task update(mb, mbc, bodyNormalAcc) method
template<typename Task>
struct NormalAccUpdater : public TanAccel<Task>
{
  NormalAccUpdater(const rbd::MultiBody & mb) : normalAccB(static_cast<size_t>(mb.nrBodies())) {}

  void operator()(Task & task, const std::vector<rbd::MultiBody> & mbs, const std::vector<rbd::MultiBodyConfig> & mbcs)
  {
    computeNormalAccB(mbs[0], mbcs[0], normalAccB);
    task.update(mbs[0], mbcs[0], normalAccB);
  }

  std::vector<sva::MotionVecd> normalAccB;
};

/// run the task update(mbs, mbcs, bodyNormalAccs) method
template<typename Task>
struct MRNormalAccUpdater : public MRTanAccel<Task>
{
  MRNormalAccUpdater(const std::vector<rbd::MultiBody> & mbs, int taskDim)
  : MRTanAccel<Task>(taskDim), normalAccBs(mbs.size())
  {
    for(std::size_t i = 0; i < mbs.size(); ++i) { normalAccBs[i].resize(static_cast<size_t>(mbs[i].nrBodies())); }
  }

  void operator()(Task & task, const std::vector<rbd::MultiBody> & mbs, const std::vector<rbd::MultiBodyConfig> & mbcs)
  {
    computeNormalAccB(mbs, mbcs, normalAccBs);
    task.update(mbs, mbcs, normalAccBs);
  }

  std::vector<std::vector<sva::MotionVecd>> normalAccBs;
};

/// run the task update(mb, mbc, com, bodyNormalAcc) method
template<typename Task>
struct NormalAccCoMUpdater : public TanAccel<Task>
{
  NormalAccCoMUpdater(const rbd::MultiBody & mb) : normalAccB(static_cast<size_t>(mb.nrBodies())) {}

  void operator()(Task & task, const std::vector<rbd::MultiBody> & mbs, const std::vector<rbd::MultiBodyConfig> & mbcs)
  {
    computeNormalAccB(mbs[0], mbcs[0], normalAccB);
    Eigen::Vector3d com = rbd::computeCoM(mbs[0], mbcs[0]);
    task.update(mbs[0], mbcs[0], com, normalAccB);
  }

  std::vector<sva::MotionVecd> normalAccB;
};

/// run the task update(mbs, mbcs, coms, bodyNormalAccs) method
template<typename Task>
struct MRNormalAccCoMUpdater : public MRTanAccel<Task>
{
  MRNormalAccCoMUpdater(const std::vector<rbd::MultiBody> & mbs, int taskDim)
  : MRTanAccel<Task>(taskDim), normalAccBs(mbs.size()), coms(mbs.size())
  {
    for(std::size_t i = 0; i < mbs.size(); ++i) { normalAccBs[i].resize(static_cast<size_t>(mbs[i].nrBodies())); }
  }

  void operator()(Task & task, const std::vector<rbd::MultiBody> & mbs, const std::vector<rbd::MultiBodyConfig> & mbcs)
  {
    computeNormalAccB(mbs, mbcs, normalAccBs);
    for(std::size_t i = 0; i < mbs.size(); ++i) { coms[i] = rbd::computeCoM(mbs[i], mbcs[i]); }
    task.update(mbs, mbcs, coms, normalAccBs);
  }

  std::vector<std::vector<sva::MotionVecd>> normalAccBs;
  std::vector<Eigen::Vector3d> coms;
};

/// Test position task (eval, speed and acc are defined)
struct PosTester
{
  void operator()(const Eigen::VectorXd & speedCur,
                  const Eigen::VectorXd & accCur,
                  const Eigen::VectorXd & speedDiff,
                  const Eigen::VectorXd & accDiff,
                  double tol)
  {
    BOOST_CHECK_SMALL((speedCur - speedDiff).norm(), tol);
    BOOST_CHECK_SMALL((accCur - accDiff).norm(), tol);
  }
};

/// Test TransformTask, position is reliable but only rotational acceleration
/// is reliable
struct PosTTTester
{
  void operator()(const Eigen::VectorXd & speedCur,
                  const Eigen::VectorXd & accCur,
                  const Eigen::VectorXd & speedDiff,
                  const Eigen::VectorXd & accDiff,
                  double tol)
  {
    BOOST_CHECK_SMALL((speedCur.tail<3>() - speedDiff.tail<3>()).norm(), tol);
    BOOST_CHECK_SMALL((accCur - accDiff).norm(), tol);
  }
};

/// Test MultiRobotTransformTask, the rotation part is not reliable since
/// he use rotationVelocity
struct PosMRTTTester
{
  void operator()(const Eigen::VectorXd & speedCur,
                  const Eigen::VectorXd & accCur,
                  const Eigen::VectorXd & speedDiff,
                  const Eigen::VectorXd & accDiff,
                  double tol)
  {
    BOOST_CHECK_SMALL((speedCur.tail<3>() - speedDiff.tail<3>()).norm(), tol);
    BOOST_CHECK_SMALL((accCur.tail<3>() - accDiff.tail<3>()).norm(), tol);
  }
};

/// Test position task with orientation error (speedDiff is not realiable)
struct OriTaskTester
{
  void operator()(const Eigen::VectorXd & /* speedCur */,
                  const Eigen::VectorXd & accCur,
                  const Eigen::VectorXd & /* speedDiff */,
                  const Eigen::VectorXd & accDiff,
                  double tol)
  { BOOST_CHECK_SMALL((accCur - accDiff).norm(), tol); }
};

/// Test velocity task (eval, and acc are defined)
struct VelTester
{
  void operator()(const Eigen::VectorXd & /* speedCur */,
                  const Eigen::VectorXd & accCur,
                  const Eigen::VectorXd & speedDiff,
                  const Eigen::VectorXd & /* accDiff */,
                  double tol)
  { BOOST_CHECK_SMALL((accCur - speedDiff).norm(), tol); }
};

/// Test position task (eval, speed and acc are defined)
struct VectOriTester
{
  void operator()(const Eigen::VectorXd & speedCur,
                  const Eigen::VectorXd & accCur,
                  const Eigen::VectorXd & speedDiff,
                  const Eigen::VectorXd & accDiff,
                  double tol)
  {
    BOOST_CHECK_SMALL((speedCur - speedDiff).norm(), tol);
    BOOST_CHECK_SMALL((accCur - accDiff).norm(), tol);
  }
};

/**
 * Use finite difference to test a task.
 * Task is the task to test, Updater is a function to call the Task::update
 * method, Tester test the task speed and acceleration against the finite
 * difference speed and accelartion.
 */
template<typename Task, typename Updater, typename Tester>
void testTaskNumDiff(const std::vector<rbd::MultiBody> & mbs,
                     const std::vector<rbd::MultiBodyConfig> & mbcs,
                     Task & task,
                     Updater updater,
                     Tester tester,
                     int nrIter = 100,
                     double diffStep = 1e-6,
                     double tol = 1e-4)
{
  using namespace Eigen;
  using namespace sva;
  using namespace rbd;

  std::vector<MultiBodyConfig> mbcsPost(mbcs), mbcsCur(mbcs);

  std::vector<Eigen::VectorXd> q(mbs.size());
  std::vector<Eigen::VectorXd> alpha(mbs.size());
  std::vector<Eigen::VectorXd> alphaD(mbs.size());

  for(int i = 0; i < nrIter; ++i)
  {
    for(std::size_t r = 0; r < mbs.size(); ++r)
    {
      const rbd::MultiBody & mb = mbs[r];
      const rbd::MultiBodyConfig & mbc = mbcs[r];
      rbd::MultiBodyConfig & mbcPost = mbcsPost[r];
      rbd::MultiBodyConfig & mbcCur = mbcsCur[r];

      q[r].setRandom(mb.nrParams());
      alpha[r].setRandom(mb.nrDof());
      alphaD[r].setRandom(mb.nrDof());

      mbcCur = mbc;
      vectorToParam(q[r], mbcCur.q);
      vectorToParam(alpha[r], mbcCur.alpha);
      vectorToParam(alphaD[r], mbcCur.alphaD);

      mbcPost = mbcCur;

      integration(mb, mbcPost, diffStep);

      forwardKinematics(mb, mbcCur);
      forwardKinematics(mb, mbcPost);
      forwardVelocity(mb, mbcCur);
      forwardVelocity(mb, mbcPost);
    }

    updater(task, mbs, mbcsCur);
    VectorXd evalCur = task.eval();
    VectorXd speedCur = -task.speed();
    VectorXd accCur = -task.normalAcc();
    accCur -= updater.tanAcc(task, alphaD);

    updater(task, mbs, mbcsPost);
    VectorXd evalPost = task.eval();
    VectorXd speedPost = -task.speed();

    VectorXd speedDiff = (evalPost - evalCur) / diffStep;
    VectorXd accDiff = (speedPost - speedCur) / diffStep;

    tester(speedCur, accCur, speedDiff, accDiff, tol);
  }
}

template<typename Task, typename Updater, typename Tester>
void testTaskNumDiff(const rbd::MultiBody & mb,
                     const rbd::MultiBodyConfig & mbc,
                     Task & task,
                     Updater updater,
                     Tester tester,
                     int nrIter = 100,
                     double diffStep = 1e-6,
                     double tol = 1e-4)
{
  testTaskNumDiff(std::vector<rbd::MultiBody>{mb}, std::vector<rbd::MultiBodyConfig>{mbc}, task, updater, tester,
                  nrIter, diffStep, tol);
}

BOOST_AUTO_TEST_CASE(PositionTaskTest)
{
  using namespace Eigen;
  using namespace rbd;

  MultiBody mb;
  MultiBodyConfig mbc;

  std::tie(mb, mbc) = makeZXZArm();

  tasks::PositionTask pt(mb, "b3", Vector3d::Random(), Vector3d::Random());

  testTaskNumDiff(mb, mbc, pt, ClassicUpdater<tasks::PositionTask>(), PosTester());
  testTaskNumDiff(mb, mbc, pt, NormalAccUpdater<tasks::PositionTask>(mb), PosTester());
}

BOOST_AUTO_TEST_CASE(OrientationTaskTest)
{
  using namespace Eigen;
  using namespace rbd;

  MultiBody mb;
  MultiBodyConfig mbc;

  std::tie(mb, mbc) = makeZXZArm();

  tasks::OrientationTask ot(mb, "b3", Quaterniond(Vector4d::Random().normalized()));

  testTaskNumDiff(mb, mbc, ot, ClassicUpdater<tasks::OrientationTask>(), OriTaskTester());
  testTaskNumDiff(mb, mbc, ot, NormalAccUpdater<tasks::OrientationTask>(mb), OriTaskTester());
}

BOOST_AUTO_TEST_CASE(TransformTaskTest)
{
  using namespace Eigen;
  using namespace rbd;

  MultiBody mb;
  MultiBodyConfig mbc;

  std::tie(mb, mbc) = makeZXZArm();

  sva::PTransformd X_b_s(Quaterniond(Vector4d::Random().normalized()), Vector3d::Random());
  sva::PTransformd X_0_t(Quaterniond(Vector4d::Random().normalized()), Vector3d::Random());
  Eigen::Quaterniond E_0_c(Vector4d::Random().normalized());

  tasks::TransformTask tt(mb, "b3", X_0_t, X_b_s, E_0_c.matrix());

  testTaskNumDiff(mb, mbc, tt, NormalAccUpdater<tasks::TransformTask>(mb), PosTTTester(), 100);
}

BOOST_AUTO_TEST_CASE(SurfaceTransformTaskTest)
{
  using namespace Eigen;
  using namespace rbd;

  MultiBody mb;
  MultiBodyConfig mbc;

  std::tie(mb, mbc) = makeZXZArm();

  sva::PTransformd X_b_s(Quaterniond(Vector4d::Random().normalized()), Vector3d::Random());
  sva::PTransformd X_0_t(Quaterniond(Vector4d::Random().normalized()), Vector3d::Random());

  tasks::SurfaceTransformTask tt(mb, "b3", X_0_t, X_b_s);

  testTaskNumDiff(mb, mbc, tt, NormalAccUpdater<tasks::SurfaceTransformTask>(mb), PosMRTTTester(), 100);
}

BOOST_AUTO_TEST_CASE(MultiRobotTransformTaskTest)
{
  using namespace Eigen;
  using namespace rbd;

  MultiBody mb1, mb2;
  MultiBodyConfig mbc1, mbc2;

  std::tie(mb1, mbc1) = makeZXZArm();
  std::tie(mb2, mbc2) = makeZXZArm();

  std::vector<MultiBody> mbs{mb1, mb2};
  std::vector<MultiBodyConfig> mbcs{mbc1, mbc2};

  sva::PTransformd X_r1b_r1s(Quaterniond(Vector4d::Random().normalized()), Vector3d::Random());
  sva::PTransformd X_r2b_r2s(Quaterniond(Vector4d::Random().normalized()), Vector3d::Random());

  tasks::MultiRobotTransformTask mrtt(mbs, 0, 1, "b3", "b3", X_r1b_r1s, X_r2b_r2s);

  testTaskNumDiff(mbs, mbcs, mrtt, MRNormalAccUpdater<tasks::MultiRobotTransformTask>(mbs, 6), PosMRTTTester());
}

BOOST_AUTO_TEST_CASE(SurfaceOrientationTaskTest)
{
  using namespace Eigen;
  using namespace rbd;

  MultiBody mb;
  MultiBodyConfig mbc;

  std::tie(mb, mbc) = makeZXZArm();

  tasks::SurfaceOrientationTask sot(mb, "b3", Quaterniond(Vector4d::Random().normalized()),
                                    sva::PTransformd(Quaterniond(Vector4d::Random().normalized()), Vector3d::Random()));

  testTaskNumDiff(mb, mbc, sot, ClassicUpdater<tasks::SurfaceOrientationTask>(), OriTaskTester());
  testTaskNumDiff(mb, mbc, sot, NormalAccUpdater<tasks::SurfaceOrientationTask>(mb), OriTaskTester());
}

BOOST_AUTO_TEST_CASE(CoMTaskTest)
{
  using namespace Eigen;
  using namespace rbd;

  MultiBody mb;
  MultiBodyConfig mbc;

  std::tie(mb, mbc) = makeZXZArm();

  tasks::CoMTask ct(mb, Vector3d::Random());

  testTaskNumDiff(mb, mbc, ct, ClassicUpdater<tasks::CoMTask>(), PosTester());
  testTaskNumDiff(mb, mbc, ct, NormalAccCoMUpdater<tasks::CoMTask>(mb), PosTester());
}

BOOST_AUTO_TEST_CASE(MultiCoMTaskTest)
{
  using namespace Eigen;
  using namespace rbd;

  MultiBody mb1, mb2;
  MultiBodyConfig mbc1, mbc2;

  std::tie(mb1, mbc1) = makeZXZArm();
  std::tie(mb2, mbc2) = makeZXZArm();

  std::vector<MultiBody> mbs{mb1, mb2};
  std::vector<MultiBodyConfig> mbcs{mbc1, mbc2};

  tasks::MultiCoMTask mct(mbs, {0, 1}, Vector3d::Random());

  testTaskNumDiff(mbs, mbcs, mct, MRClassicUpdater<tasks::MultiCoMTask>(3), PosTester());
  testTaskNumDiff(mbs, mbcs, mct, MRNormalAccUpdater<tasks::MultiCoMTask>(mbs, 3), PosTester());
  testTaskNumDiff(mbs, mbcs, mct, MRNormalAccCoMUpdater<tasks::MultiCoMTask>(mbs, 3), PosTester());
}

BOOST_AUTO_TEST_CASE(MomentumTaskTest)
{
  using namespace Eigen;
  using namespace rbd;

  MultiBody mb;
  MultiBodyConfig mbc;

  std::tie(mb, mbc) = makeZXZArm();

  tasks::MomentumTask mt(mb, sva::ForceVecd(Vector6d::Random()));

  testTaskNumDiff(mb, mbc, mt, ClassicUpdater<tasks::MomentumTask>(), VelTester());
  testTaskNumDiff(mb, mbc, mt, NormalAccUpdater<tasks::MomentumTask>(mb), VelTester());
}

BOOST_AUTO_TEST_CASE(LinVelocityTaskTest)
{
  using namespace Eigen;
  using namespace rbd;

  MultiBody mb;
  MultiBodyConfig mbc;

  std::tie(mb, mbc) = makeZXZArm();

  tasks::LinVelocityTask lvt(mb, "b3", Vector3d::Random());

  testTaskNumDiff(mb, mbc, lvt, ClassicUpdater<tasks::LinVelocityTask>(), VelTester());
  testTaskNumDiff(mb, mbc, lvt, NormalAccUpdater<tasks::LinVelocityTask>(mb), VelTester());
}

BOOST_AUTO_TEST_CASE(VectorOrientationTaskTest)
{
  using namespace Eigen;
  using namespace rbd;

  MultiBody mb;
  MultiBodyConfig mbc;

  std::tie(mb, mbc) = makeZXZArm();

  tasks::VectorOrientationTask vot(mb, "b3", Vector3d::Random(), Vector3d::Random());

  testTaskNumDiff(mb, mbc, vot, NormalAccUpdater<tasks::VectorOrientationTask>(mb), VectOriTester());
}

/// @return A branched arm with every RBDyn joint type on the path to "b7" and a side branch "b8"
std::tuple<rbd::MultiBody, rbd::MultiBodyConfig> makeManipulabilityArm(bool isFixed)
{
  using namespace Eigen;
  using namespace sva;
  using namespace rbd;

  MultiBodyGraph mbg;
  RBInertiad rbi(1., Vector3d::Zero(), Matrix3d::Identity());
  for(int i = 0; i < 9; ++i) { mbg.addBody(Body(rbi, "b" + std::to_string(i))); }

  mbg.addJoint(Joint(Joint::Rev, Vector3d(0., 0., 1.), true, "j1"));
  mbg.addJoint(Joint(Joint::Spherical, true, "j2"));
  mbg.addJoint(Joint(Joint::Prism, Vector3d(1., 0., 0.), true, "j3"));
  mbg.addJoint(Joint(Joint::Rev, Vector3d(0., 1., 0.), false, "j4"));
  mbg.addJoint(Joint(Joint::Planar, true, "j5"));
  mbg.addJoint(Joint(Joint::Cylindrical, Vector3d(1., 1., 0.).normalized(), true, "j6"));
  mbg.addJoint(Joint(Joint::Rev, Vector3d(1., 0., 0.), true, "j7"));
  mbg.addJoint(Joint(Joint::Rev, Vector3d(0., 0., 1.), true, "j8"));

  auto offset = [](double rx, double rz, const Vector3d & t) { return PTransformd(RotX(rx) * RotZ(rz), t); };
  mbg.linkBodies("b0", offset(0.1, 0.2, Vector3d(0., 0., 0.3)), "b1", PTransformd::Identity(), "j1");
  mbg.linkBodies("b1", offset(-0.3, 0.5, Vector3d(0.1, 0.2, 0.4)), "b2", PTransformd::Identity(), "j2");
  mbg.linkBodies("b2", offset(0.7, -0.2, Vector3d(0.3, -0.1, 0.)), "b3", PTransformd::Identity(), "j3");
  mbg.linkBodies("b3", offset(0.2, 0.9, Vector3d(0., 0.4, 0.1)), "b4", PTransformd::Identity(), "j4");
  mbg.linkBodies("b4", offset(-0.6, 0.1, Vector3d(0.2, 0., 0.3)), "b5", PTransformd::Identity(), "j5");
  mbg.linkBodies("b5", offset(0.4, -0.7, Vector3d(0.1, 0.3, 0.)), "b6", PTransformd::Identity(), "j6");
  mbg.linkBodies("b6", offset(-0.2, 0.3, Vector3d(0.25, 0., 0.15)), "b7", PTransformd::Identity(), "j7");
  mbg.linkBodies("b3", offset(0., 0., Vector3d(0., -0.3, 0.)), "b8", PTransformd::Identity(), "j8");

  MultiBody mb = mbg.makeMultiBody("b0", isFixed);
  MultiBodyConfig mbc(mb);
  mbc.zero(mb);
  return std::make_tuple(mb, mbc);
}

/// Random valid configuration (unit quaternions) with random velocity and acceleration
void randomManipulabilityState(const rbd::MultiBody & mb, rbd::MultiBodyConfig & mbc)
{
  mbc.zero(mb);
  rbd::vectorToParam(Eigen::VectorXd::Random(mb.nrDof()), mbc.alpha);
  rbd::integration(mb, mbc, 1.);
  rbd::vectorToParam(Eigen::VectorXd::Random(mb.nrDof()), mbc.alpha);
  rbd::vectorToParam(Eigen::VectorXd::Random(mb.nrDof()), mbc.alphaD);
  rbd::forwardKinematics(mb, mbc);
  rbd::forwardVelocity(mb, mbc);
}

/// Update the task after integrating mbc over step (backward in time if step is negative)
void updateAfterStep(const rbd::MultiBody & mb,
                     const rbd::MultiBodyConfig & mbc,
                     double step,
                     tasks::ManipulabilityTask & task)
{
  rbd::MultiBodyConfig mbcStep(mbc);
  if(step < 0) { rbd::vectorToParam(-rbd::dofToVector(mb, mbcStep.alpha), mbcStep.alpha); }
  rbd::integration(mb, mbcStep, std::abs(step));
  if(step < 0) { rbd::vectorToParam(-rbd::dofToVector(mb, mbcStep.alpha), mbcStep.alpha); }
  rbd::forwardKinematics(mb, mbcStep);
  rbd::forwardVelocity(mb, mbcStep);
  task.update(mb, mbcStep);
}

/// Check jac, speed and normalAcc against central differences
///
/// testTaskNumDiff uses forward differences and random joint parameters, which are not valid for free and spherical
/// joints, and the manipulability of the larger arm is not small enough for the forward difference error to be below
/// tolerance
void testManipulabilityNumDiff(const rbd::MultiBody & mb,
                               const rbd::MultiBodyConfig & mbcInit,
                               tasks::ManipulabilityTask & task,
                               int nrIter = 50)
{
  const double diffStep = 1e-5;
  const double tol = 1e-4;
  rbd::MultiBodyConfig mbcCur(mbcInit);
  for(int iter = 0; iter < nrIter; ++iter)
  {
    randomManipulabilityState(mb, mbcCur);
    task.update(mb, mbcCur);
    BOOST_REQUIRE(std::isfinite(task.manipulability()));
    const Eigen::MatrixXd jac = task.jac();
    Eigen::VectorXd speedCur = -task.speed();
    Eigen::VectorXd alphaD = rbd::dofToVector(mb, mbcCur.alphaD);
    Eigen::VectorXd accCur = -task.normalAcc() - task.jac() * alphaD;

    // Gradient, moving along each degree of freedom
    rbd::MultiBodyConfig mbcDof(mbcCur);
    rbd::vectorToParam(Eigen::VectorXd::Zero(mb.nrDof()), mbcDof.alphaD);
    Eigen::VectorXd gradDiff(mb.nrDof());
    for(int k = 0; k < mb.nrDof(); ++k)
    {
      rbd::vectorToParam(Eigen::VectorXd::Unit(mb.nrDof(), k), mbcDof.alpha);
      updateAfterStep(mb, mbcDof, diffStep, task);
      double wPost = task.manipulability();
      updateAfterStep(mb, mbcDof, -diffStep, task);
      gradDiff(k) = (wPost - task.manipulability()) / (2 * diffStep);
    }
    BOOST_CHECK_SMALL((jac.row(0).transpose() - gradDiff).norm(), tol);

    // Speed and acceleration, moving along the current velocity and acceleration
    updateAfterStep(mb, mbcCur, diffStep, task);
    Eigen::VectorXd evalPost = task.eval();
    Eigen::VectorXd speedPost = -task.speed();
    updateAfterStep(mb, mbcCur, -diffStep, task);
    Eigen::VectorXd speedDiff = (evalPost - task.eval()) / (2 * diffStep);
    Eigen::VectorXd accDiff = (speedPost + task.speed()) / (2 * diffStep);

    BOOST_CHECK_SMALL((speedCur - speedDiff).norm(), tol);
    BOOST_CHECK_SMALL((accCur - accDiff).norm(), tol);
  }
}

BOOST_AUTO_TEST_CASE(ManipulabilityTaskTest)
{
  using namespace Eigen;
  using namespace rbd;

  MultiBody mb;
  MultiBodyConfig mbc;
  const sva::PTransformd X_b_f(sva::RotY(0.4) * sva::RotZ(-0.3), Vector3d(0.1, -0.05, 0.2));

  // Revolute arm, translation only
  std::tie(mb, mbc) = makeZXZArm();
  Vector6d translation;
  translation << 0., 0., 0., 1., 1., 1.;
  tasks::ManipulabilityTask zxz(mb, "b3", X_b_f, {}, translation);
  BOOST_CHECK_EQUAL(zxz.measureDof(), 3);
  testManipulabilityNumDiff(mb, mbc, zxz);

  for(bool isFixed : {true, false})
  {
    std::tie(mb, mbc) = makeManipulabilityArm(isFixed);

    // Default measure joints: the path to b7 without the free root joint, all axes
    tasks::ManipulabilityTask all(mb, "b7", X_b_f);
    BOOST_CHECK_EQUAL(all.measureJoints().size(), 7);
    BOOST_CHECK_EQUAL(all.measureDof(), 12);
    testManipulabilityNumDiff(mb, mbc, all);

    // Translation only, and an [rz x y z] selection
    tasks::ManipulabilityTask trans(mb, "b7", X_b_f, {}, translation);
    testManipulabilityNumDiff(mb, mbc, trans);
    Vector6d scara;
    scara << 0., 0., 1., 1., 1., 1.;
    tasks::ManipulabilityTask scaraTask(mb, "b7", X_b_f, {}, scara);
    testManipulabilityNumDiff(mb, mbc, scaraTask);

    // Measure joints with non measure joints in between and after, and velocity normalization
    tasks::ManipulabilityTask subset(mb, "b7", X_b_f, {"j6", "j2", "j3", "j1"});
    BOOST_CHECK_EQUAL(subset.measureJoints().front(), "j1");
    BOOST_CHECK_EQUAL(subset.measureJoints().back(), "j6");
    BOOST_CHECK_EQUAL(subset.measureDof(), 7);
    Vector6d maxTaskVelocity;
    maxTaskVelocity << 2., 2., 2., 0.5, 0.5, 0.5;
    subset.maxTaskVelocity(maxTaskVelocity);
    subset.maxJointVelocity(VectorXd::LinSpaced(7, 0.5, 2.));
    testManipulabilityNumDiff(mb, mbc, subset);

    if(!isFixed)
    {
      // The measure is computed in frame coordinates: moving the floating base does not change it
      randomManipulabilityState(mb, mbc);
      all.update(mb, mbc);
      BOOST_CHECK_SMALL(all.jac().block(0, 0, 1, 6).norm(), 1e-10);
    }
  }
}

BOOST_AUTO_TEST_CASE(ManipulabilityTaskSingularTest)
{
  using namespace Eigen;
  using namespace rbd;

  MultiBody mb;
  MultiBodyConfig mbc;
  std::tie(mb, mbc) = makeManipulabilityArm(true);
  forwardKinematics(mb, mbc);
  forwardVelocity(mb, mbc);

  // j1 and j7 alone cannot rotate the frame about three axes: w is 0 in every configuration
  Vector6d rotation;
  rotation << 1., 1., 1., 0., 0., 0.;
  tasks::ManipulabilityTask singular(mb, "b7", sva::PTransformd::Identity(), {"j1", "j3", "j7"}, rotation);
  for(int i = 0; i < 10; ++i)
  {
    randomManipulabilityState(mb, mbc);
    singular.update(mb, mbc);
    BOOST_CHECK_SMALL(singular.manipulability(), 1e-10);
    BOOST_CHECK(singular.jac().allFinite());
    BOOST_CHECK(singular.speed().allFinite());
    BOOST_CHECK(singular.normalAcc().allFinite());
  }

  // Invalid definitions
  BOOST_CHECK_THROW(tasks::ManipulabilityTask(mb, "b7", sva::PTransformd::Identity(), {"j8"}), std::domain_error);
  BOOST_CHECK_THROW(tasks::ManipulabilityTask(mb, "b7", sva::PTransformd::Identity(), {"j1", "j1"}), std::domain_error);
  BOOST_CHECK_THROW(tasks::ManipulabilityTask(mb, "b7", sva::PTransformd::Identity(), {"j1", "j7"}), std::domain_error);
  tasks::ManipulabilityTask task(mb, "b7");
  BOOST_CHECK_THROW(task.maxJointVelocity(VectorXd::Ones(3)), std::domain_error);
  BOOST_CHECK_THROW(task.maxTaskVelocity(Vector6d::Zero()), std::domain_error);
}
