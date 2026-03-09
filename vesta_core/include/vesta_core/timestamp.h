#pragma once

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

#include <boost/serialization/access.hpp>

#include <chrono>
#include <compare>
#include <cstdint>
#include <ostream>

namespace vesta_core
{

/**
 * @brief A simple timestamp type that replaces ros::Time
 *
 * Represents time as nanoseconds since epoch. Provides conversion to/from
 * seconds and interoperability with std::chrono.
 */
struct Timestamp
{
  int64_t nanoseconds{ 0 };

  Timestamp() = default;

  explicit Timestamp(int64_t ns) : nanoseconds(ns)
  {
  }

  /**
   * @brief Construct from seconds and nanoseconds (matching ros::Time(sec,
   * nsec) pattern)
   */
  Timestamp(uint32_t sec, uint32_t nsec)
    : nanoseconds(static_cast<int64_t>(sec) * 1000000000LL + static_cast<int64_t>(nsec))
  {
  }

  /**
   * @brief Get the current time
   */
  static Timestamp now()
  {
    auto now = std::chrono::system_clock::now();
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch());
    return Timestamp(ns.count());
  }

  /**
   * @brief Create a Timestamp from a floating-point seconds value
   */
  static Timestamp fromSec(double sec)
  {
    return Timestamp(static_cast<int64_t>(sec * 1e9));
  }

  /**
   * @brief Convert to a floating-point seconds value
   */
  double toSec() const
  {
    return static_cast<double>(nanoseconds) / 1e9;
  }

  /**
   * @brief Check if the timestamp is zero (uninitialized)
   */
  bool isZero() const
  {
    return nanoseconds == 0;
  }

  auto operator<=>(const Timestamp&) const = default;

  friend std::ostream& operator<<(std::ostream& os, const Timestamp& t)
  {
    os << t.toSec();
    return os;
  }

private:
  friend class boost::serialization::access;

  template <class Archive>
  void serialize(Archive& ar, const unsigned int /* version */)
  {
    ar & nanoseconds;
  }
};

/**
 * @brief A simple duration type that replaces ros::Duration
 *
 * Represents a time duration as nanoseconds. Provides conversion to/from
 * seconds and interoperability with std::chrono.
 */
struct Duration
{
  int64_t nanoseconds{ 0 };

  Duration() = default;

  explicit Duration(int64_t ns) : nanoseconds(ns)
  {
  }

  /**
   * @brief Construct from seconds and nanoseconds
   */
  Duration(int32_t sec, int32_t nsec)
    : nanoseconds(static_cast<int64_t>(sec) * 1000000000LL + static_cast<int64_t>(nsec))
  {
  }

  /**
   * @brief Create a Duration from a floating-point seconds value
   */
  static Duration fromSec(double sec)
  {
    return Duration(static_cast<int64_t>(sec * 1e9));
  }

  /**
   * @brief Convert to a floating-point seconds value
   */
  double toSec() const
  {
    return static_cast<double>(nanoseconds) / 1e9;
  }

  /**
   * @brief Check if the duration is zero
   */
  bool isZero() const
  {
    return nanoseconds == 0;
  }

  /**
   * @brief Maximum representable duration (replaces ros::DURATION_MAX)
   */
  static const Duration MAX;

  auto operator<=>(const Duration&) const = default;

  friend std::ostream& operator<<(std::ostream& os, const Duration& d)
  {
    os << d.toSec();
    return os;
  }

private:
  friend class boost::serialization::access;

  template <class Archive>
  void serialize(Archive& ar, const unsigned int /* version */)
  {
    ar & nanoseconds;
  }
};

/**
 * @brief Arithmetic operators for Timestamp and Duration
 */
inline Timestamp operator+(const Timestamp& t, const Duration& d)
{
  return Timestamp(t.nanoseconds + d.nanoseconds);
}

inline Timestamp operator-(const Timestamp& t, const Duration& d)
{
  return Timestamp(t.nanoseconds - d.nanoseconds);
}

inline Duration operator-(const Timestamp& a, const Timestamp& b)
{
  return Duration(a.nanoseconds - b.nanoseconds);
}

inline Duration operator+(const Duration& a, const Duration& b)
{
  return Duration(a.nanoseconds + b.nanoseconds);
}

inline Duration operator-(const Duration& a, const Duration& b)
{
  return Duration(a.nanoseconds - b.nanoseconds);
}

}  // namespace vesta_core
