/*
 * Copyright 2012-2019 CNRS-UM LIRMM, CNRS-AIST JRL
 */

#pragma once

// includes
// std
#include <tuple>

// RBDyn
#include <RBDyn/Body.h>
#include <RBDyn/Joint.h>
#include <RBDyn/MultiBody.h>
#include <RBDyn/MultiBodyConfig.h>
#include <RBDyn/MultiBodyGraph.h>

/// @return An simple ZXZ arm with Y as up axis.
std::tuple<rbd::MultiBody, rbd::MultiBodyConfig> makeZXZArm(bool isFixed = true,
                                                            const sva::PTransformd X_base = sva::PTransformd::Identity())
{
  using namespace Eigen;
  using namespace sva;
  using namespace rbd;

  MultiBodyGraph mbg;

  double mass = 1.;
  Matrix3d I = Matrix3d::Identity();
  Vector3d h = Vector3d::Zero();

  RBInertiad rbi(mass, h, I);

  Body b0(rbi, "b0");
  Body b1(rbi, "b1");
  Body b2(rbi, "b2");
  Body b3(rbi, "b3");

  mbg.addBody(b0);
  mbg.addBody(b1);
  mbg.addBody(b2);
  mbg.addBody(b3);

  Joint j0(Joint::RevZ, true, "j0");
  Joint j1(Joint::RevX, true, "j1");
  Joint j2(Joint::RevZ, true, "j2");

  mbg.addJoint(j0);
  mbg.addJoint(j1);
  mbg.addJoint(j2);

  //  Root     j0       j1     j2
  //  ---- b0 ---- b1 ---- b2 ----b3
  //  Fixed    Z       X       Z

  PTransformd to(Vector3d(0., 0.5, 0.));
  PTransformd from(Vector3d(0., 0., 0.));

  mbg.linkBodies("b0", PTransformd::Identity(), "b1", from, "j0");
  mbg.linkBodies("b1", to, "b2", from, "j1");
  mbg.linkBodies("b2", to, "b3", from, "j2");

  MultiBody mb = mbg.makeMultiBody("b0", isFixed, X_base);

  MultiBodyConfig mbc(mb);
  mbc.zero(mb);

  return std::make_tuple(mb, mbc);
}

/// @return Two ZXZ arms on a fixed base b0 with Y as up axis: the left arm l1-l3 (joints jl1-jl3) and the right arm
/// r1-r3 (joints jr1-jr3), mounted 0.3 m to each side of the base along X.
std::tuple<rbd::MultiBody, rbd::MultiBodyConfig> makeTwoArms()
{
  using namespace Eigen;
  using namespace sva;
  using namespace rbd;

  MultiBodyGraph mbg;
  RBInertiad rbi(1., Vector3d::Zero(), Matrix3d::Identity());
  for(const std::string name : {"b0", "l1", "l2", "l3", "r1", "r2", "r3"}) { mbg.addBody(Body(rbi, name)); }

  const PTransformd to(Vector3d(0., 0.5, 0.));
  for(const std::string side : {"l", "r"})
  {
    mbg.addJoint(Joint(Joint::RevZ, true, "j" + side + "1"));
    mbg.addJoint(Joint(Joint::RevX, true, "j" + side + "2"));
    mbg.addJoint(Joint(Joint::RevZ, true, "j" + side + "3"));
    const double x = side == "l" ? -0.3 : 0.3;
    mbg.linkBodies("b0", PTransformd(Vector3d(x, 0., 0.)), side + "1", PTransformd::Identity(), "j" + side + "1");
    mbg.linkBodies(side + "1", to, side + "2", PTransformd::Identity(), "j" + side + "2");
    mbg.linkBodies(side + "2", to, side + "3", PTransformd::Identity(), "j" + side + "3");
  }

  MultiBody mb = mbg.makeMultiBody("b0", true);
  MultiBodyConfig mbc(mb);
  mbc.zero(mb);
  return std::make_tuple(mb, mbc);
}

/// @return A one body robot for the environnment.
std::tuple<rbd::MultiBody, rbd::MultiBodyConfig> makeEnv()
{
  using namespace Eigen;
  using namespace sva;
  using namespace rbd;

  MultiBodyGraph mbg;

  double mass = 1.;
  Matrix3d I = Matrix3d::Identity();
  Vector3d h = Vector3d::Zero();

  RBInertiad rbi(mass, h, I);

  Body b0(rbi, "b0");

  mbg.addBody(b0);

  MultiBody mb = mbg.makeMultiBody("b0", true);

  MultiBodyConfig mbc(mb);
  mbc.zero(mb);

  return std::make_tuple(mb, mbc);
}
