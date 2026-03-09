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

#include <vesta_core/graph.h>

#include <vector>

namespace vesta_core {

/**
 * @brief Serialize a graph into a byte buffer
 *
 * @param[in]  graph  The graph to serialize
 * @param[out] data   The output byte buffer
 */
void serializeGraph(const vesta_core::Graph &graph,
                    std::vector<unsigned char> &data);

/**
 * @brief Deserialize a graph from a byte buffer
 *
 * Uses Boost.Serialization with BOOST_CLASS_EXPORT for polymorphic
 * deserialization. The appropriate derived types must have been registered via
 * BOOST_CLASS_EXPORT in their respective compilation units.
 *
 * @param[in] data        The serialized byte buffer
 * @param[in] plugin_name The fully-qualified type name of the Graph
 * implementation
 * @return A unique_ptr to the deserialized Graph object
 */
vesta_core::Graph::UniquePtr
deserializeGraph(const std::vector<unsigned char> &data,
                 const std::string &plugin_name);

} // namespace vesta_core
