/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2024, Locus Robotics
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
#include <vesta_variables/vision/stereo_camera.h>

#include <vesta_core/uuid.h>
#include <vesta_core/variable.h>
#include <vesta_variables/common/fixed_size_variable.h>

#include <boost/serialization/export.hpp>

#include <ostream>

namespace vesta_variables
{
StereoCamera::StereoCamera(const vesta_core::UUID& uuid, uint64_t camera_id) : BaseCamera(uuid, camera_id)
{
}

StereoCamera::StereoCamera(uint64_t camera_id)
  : StereoCamera(vesta_core::uuid::generate(detail::type(), camera_id), camera_id)
{
}

StereoCamera::StereoCamera(const vesta_core::UUID& uuid, uint64_t camera_id, double fx, double fy, double cx, double cy,
                           double baseline)
  : StereoCamera(uuid, camera_id)
{
  data_[FX] = fx;
  data_[FY] = fy;
  data_[CX] = cx;
  data_[CY] = cy;
  data_[BASELINE] = baseline;
}

void StereoCamera::print(std::ostream& stream) const
{
  stream << type() << ":\n"
         << "  uuid: " << uuid() << "\n"
         << "  size: " << size() << "\n"
         << "  camera id: " << id() << "\n"
         << "  data:\n"
         << "  - fx: " << fx() << "\n"
         << "  - fy: " << fy() << "\n"
         << "  - cx: " << cx() << "\n"
         << "  - cy: " << cy() << "\n"
         << "  - baseline: " << baseline() << "\n";
}

}  // namespace vesta_variables

BOOST_CLASS_EXPORT_IMPLEMENT(vesta_variables::StereoCamera);
