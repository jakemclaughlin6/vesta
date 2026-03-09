#pragma once

/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2019, Locus Robotics
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the copyright holder nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 */

#include <vesta_core/fuse_macros.h>
#include <vesta_core/manifold.h>

#include <ceres/internal/autodiff.h>

#include <memory>

namespace vesta_core
{

/**
 * @brief Create a manifold with the Jacobians computed via automatic
 * differentiation.
 *
 * To get an auto differentiated manifold, you must define two classes with a
 * templated operator() (a.k.a. a functor).
 *
 * The first functor should compute:
 *
 *   Plus(x, delta) -> x_plus_delta
 *
 * And the second functor should compute the inverse operation:
 *
 *   Minus(x1, x2) -> delta
 *
 * Minus() should be defined such that if Plus(x1, delta) -> x2, then Minus(x1,
 * x2) -> delta
 *
 * The autodiff framework substitutes appropriate "Jet" objects for the template
 * parameter T in order to compute the derivative when necessary, but this is
 * hidden, and you should write the function as if T were a scalar type (e.g. a
 * double-precision floating point number).
 *
 * Additionally the AmbientSize and TangentSize must be specified as template
 * parameters.
 * - AmbientSize is the size of the variables x1 and x2. If this is a
 * quaternion, the AmbientSize would be 4.
 * - TangentSize is the size of delta, and may be different from AmbientSize.
 * For quaternions, there are only 3 degrees of freedom, so the TangentSize
 * is 3.
 *
 * For more information on manifolds, see vesta_core::Manifold
 */
template <typename PlusFunctor, typename MinusFunctor, int kAmbientSize, int kTangentSize>
class AutoDiffManifold : public Manifold
{
public:
  VESTA_SMART_PTR_DEFINITIONS(AutoDiffManifold<PlusFunctor, MinusFunctor, kAmbientSize, kTangentSize>);

  /**
   * @brief Constructs new PlusFunctor and MinusFunctor instances
   */
  AutoDiffManifold();

  /**
   * @brief Takes ownership of the provided PlusFunctor and MinusFunctor
   * instances
   */
  AutoDiffManifold(PlusFunctor* plus_functor, MinusFunctor* minus_functor);

  /**
   * @brief Generalization of the addition operation, implemented by the
   * provided PlusFunctor
   *
   * @param[in]  x            The starting variable value, of size \p
   * AmbientSize()
   * @param[in]  delta        The variable increment to apply, of size \p
   * TangentSize()
   * @param[out] x_plus_delta The final variable value, of size \p AmbientSize()
   * @return True if successful, false otherwise
   */
  bool Plus(const double* x, const double* delta, double* x_plus_delta) const override;

  /**
   * @brief The Jacobian of Plus(x, delta) w.r.t delta at delta = 0, computed
   * using automatic differentiation
   *
   * @param[in]  x        The value used to evaluate the Jacobian, of size \p
   * AmbientSize()
   * @param[out] jacobian The Jacobian in row-major order, of size \p
   * AmbientSize() x \p TangentSize()
   */
  bool PlusJacobian(const double* x, double* jacobian) const override;

  /**
   * @brief Generalization of the subtraction operation, implemented by the
   * provided MinusFunctor
   *
   * @param[in]  x1    The value of the first variable, of size \p AmbientSize()
   * @param[in]  x2    The value of the second variable, of size \p
   * AmbientSize()
   * @param[out] delta The difference between the second variable and the first,
   * of size \p TangentSize()
   * @return True if successful, false otherwise
   */
  bool Minus(const double* x1, const double* x2, double* delta) const override;

  /**
   * @brief The Jacobian of Minus(x1, x2) w.r.t x2 evaluated at x1 = x2 = x,
   * computed using automatic differentiation
   *
   * @param[in]  x        The value used to evaluate the Jacobian, of size \p
   * AmbientSize()
   * @param[out] jacobian The Jacobian in row-major order, of size \p
   * TangentSize() x \p AmbientSize()
   */
  bool MinusJacobian(const double* x, double* jacobian) const override;

  /**
   * @brief The size of the variable parameterization in the nonlinear manifold
   */
  int AmbientSize() const override
  {
    return kAmbientSize;
  }

  /**
   * @brief The size of a delta vector in the linear tangent space to the
   * nonlinear manifold
   */
  int TangentSize() const override
  {
    return kTangentSize;
  }

private:
  std::unique_ptr<PlusFunctor> plus_functor_;
  std::unique_ptr<MinusFunctor> minus_functor_;
};

template <typename PlusFunctor, typename MinusFunctor, int kAmbientSize, int kTangentSize>
AutoDiffManifold<PlusFunctor, MinusFunctor, kAmbientSize, kTangentSize>::AutoDiffManifold()
  : plus_functor_(new PlusFunctor()), minus_functor_(new MinusFunctor())
{
}

template <typename PlusFunctor, typename MinusFunctor, int kAmbientSize, int kTangentSize>
AutoDiffManifold<PlusFunctor, MinusFunctor, kAmbientSize, kTangentSize>::AutoDiffManifold(PlusFunctor* plus_functor,
                                                                                          MinusFunctor* minus_functor)
  : plus_functor_(plus_functor), minus_functor_(minus_functor)
{
}

template <typename PlusFunctor, typename MinusFunctor, int kAmbientSize, int kTangentSize>
bool AutoDiffManifold<PlusFunctor, MinusFunctor, kAmbientSize, kTangentSize>::Plus(const double* x, const double* delta,
                                                                                   double* x_plus_delta) const
{
  return (*plus_functor_)(x, delta, x_plus_delta);
}

template <typename PlusFunctor, typename MinusFunctor, int kAmbientSize, int kTangentSize>
bool AutoDiffManifold<PlusFunctor, MinusFunctor, kAmbientSize, kTangentSize>::PlusJacobian(const double* x,
                                                                                           double* jacobian) const
{
  double zero_delta[kTangentSize] = {};  // zero-initialize
  double x_plus_delta[kAmbientSize];

  const double* parameter_ptrs[2] = { x, zero_delta };
  double* jacobian_ptrs[2] = { NULL, jacobian };
  ceres::internal::AutoDifferentiate<kAmbientSize, ceres::internal::StaticParameterDims<kAmbientSize, kTangentSize>>(
      *plus_functor_, parameter_ptrs, kAmbientSize, x_plus_delta, jacobian_ptrs);
  return true;
}

template <typename PlusFunctor, typename MinusFunctor, int kAmbientSize, int kTangentSize>
bool AutoDiffManifold<PlusFunctor, MinusFunctor, kAmbientSize, kTangentSize>::Minus(const double* x1, const double* x2,
                                                                                    double* delta) const
{
  return (*minus_functor_)(x1, x2, delta);
}

template <typename PlusFunctor, typename MinusFunctor, int kAmbientSize, int kTangentSize>
bool AutoDiffManifold<PlusFunctor, MinusFunctor, kAmbientSize, kTangentSize>::MinusJacobian(const double* x,
                                                                                            double* jacobian) const
{
  double delta[kTangentSize] = {};  // zero-initialize

  const double* parameter_ptrs[2] = { x, x };
  double* jacobian_ptrs[2] = { NULL, jacobian };
  using StaticParameters = ceres::internal::StaticParameterDims<kAmbientSize, kAmbientSize>;
  ceres::internal::AutoDifferentiate<kTangentSize, StaticParameters>(*minus_functor_, parameter_ptrs, kTangentSize,
                                                                     delta, jacobian_ptrs);
  return true;
}

}  // namespace vesta_core
