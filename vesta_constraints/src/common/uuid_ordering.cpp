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
#include <vesta_constraints/common/uuid_ordering.h>

#include <vesta_core/uuid.h>


namespace vesta_constraints
{
UuidOrdering::UuidOrdering(std::initializer_list<vesta_core::UUID> uuid_list) :
  UuidOrdering(uuid_list.begin(), uuid_list.end())
{
}

bool UuidOrdering::empty() const
{
  return index_to_uuid_.empty();
}

size_t UuidOrdering::size() const
{
  return index_to_uuid_.size();
}

bool UuidOrdering::exists(const unsigned int index) const
{
  return (index < index_to_uuid_.size());
}

bool UuidOrdering::exists(const vesta_core::UUID& uuid) const
{
  return (uuid_to_index_.count(uuid) > 0);
}

bool UuidOrdering::push_back(const vesta_core::UUID& uuid)
{
  if (uuid_to_index_.count(uuid) > 0)
  {
    return false;
  }
  uuid_to_index_.emplace(uuid, static_cast<unsigned int>(index_to_uuid_.size()));
  index_to_uuid_.push_back(uuid);
  return true;
}

const vesta_core::UUID& UuidOrdering::operator[](const unsigned int index) const
{
  return index_to_uuid_[index];
}

unsigned int UuidOrdering::operator[](const vesta_core::UUID& uuid)
{
  push_back(uuid);
  return uuid_to_index_[uuid];
}

const vesta_core::UUID& UuidOrdering::at(const unsigned int index) const
{
  return index_to_uuid_.at(index);
}

unsigned int UuidOrdering::at(const vesta_core::UUID& uuid) const
{
  return uuid_to_index_.at(uuid);
}

}  // namespace vesta_constraints
