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
#include <fuse_core/graph_deserializer.h>

#include <fuse_core/serialization.h>

#include <boost/iostreams/stream.hpp>

#include <stdexcept>
#include <vector>


namespace fuse_core
{

void serializeGraph(const fuse_core::Graph& graph, std::vector<unsigned char>& data)
{
  data.clear();
  boost::iostreams::stream<fuse_core::MessageBufferStreamSink> stream(data);
  {
    BinaryOutputArchive archive(stream);
    graph.serialize(archive);
  }
}

fuse_core::Graph::UniquePtr deserializeGraph(const std::vector<unsigned char>& data,
                                             const std::string& /*plugin_name*/)
{
  // Boost.Serialization with BOOST_CLASS_EXPORT handles polymorphic deserialization.
  // The plugin_name parameter is retained for API compatibility but is not used;
  // the archive contains the type information needed for deserialization.
  fuse_core::Graph* raw_graph = nullptr;
  boost::iostreams::stream<fuse_core::MessageBufferStreamSource> stream(data);
  {
    BinaryInputArchive archive(stream);
    archive >> raw_graph;
  }
  if (!raw_graph)
  {
    throw std::runtime_error("Failed to deserialize graph from byte buffer");
  }
  return fuse_core::Graph::UniquePtr(raw_graph);
}

}  // namespace fuse_core
