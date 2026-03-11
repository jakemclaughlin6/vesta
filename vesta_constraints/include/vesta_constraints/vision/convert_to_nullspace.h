#pragma once

#include <vesta_core/graph.h>
#include <vesta_core/transaction.h>
#include <vesta_core/uuid.h>

#include <string>
#include <vector>

namespace vesta_constraints
{

/**
 * @brief Convert reprojection constraints to nullspace projection constraints.
 *
 * For each specified landmark, finds all ReprojectionErrorConstraint instances
 * connected to it, groups their observations, and creates a single
 * NullspaceProjectionConstraint that constrains the involved camera poses
 * without requiring the landmark variable.
 *
 * Returns a Transaction that:
 *   - Adds a NullspaceProjectionConstraint for each landmark with >= 2 observations
 *   - Removes the original ReprojectionErrorConstraint instances
 *   - Removes the landmark variables
 *
 * Landmarks with fewer than 2 observations are skipped (not enough for triangulation).
 *
 * @param[in] source           The source name to assign to new constraints
 * @param[in] landmark_uuids   The UUIDs of Point3DLandmark variables to eliminate
 * @param[in] graph            The graph containing variables and constraints
 * @return A Transaction that replaces reprojection constraints with nullspace constraints
 */
vesta_core::Transaction convertToNullspaceConstraints(const std::string& source,
                                                      const std::vector<vesta_core::UUID>& landmark_uuids,
                                                      const vesta_core::Graph& graph);

/**
 * @brief Convert stereo reprojection constraints to stereo nullspace projection constraints.
 *
 * For each specified landmark, finds all StereoReprojectionErrorConstraint instances
 * connected to it, groups their observations, and creates a single
 * StereoNullspaceProjectionConstraint that constrains the involved camera poses
 * without requiring the landmark variable.
 *
 * Returns a Transaction that:
 *   - Adds a StereoNullspaceProjectionConstraint for each landmark with >= 2 observations
 *   - Removes the original StereoReprojectionErrorConstraint instances
 *   - Removes the landmark variables
 *
 * Landmarks with fewer than 2 observations are skipped (not enough for triangulation).
 *
 * @param[in] source           The source name to assign to new constraints
 * @param[in] landmark_uuids   The UUIDs of Point3DLandmark variables to eliminate
 * @param[in] graph            The graph containing variables and constraints
 * @return A Transaction that replaces stereo reprojection constraints with stereo nullspace constraints
 */
vesta_core::Transaction convertToStereoNullspaceConstraints(const std::string& source,
                                                            const std::vector<vesta_core::UUID>& landmark_uuids,
                                                            const vesta_core::Graph& graph);

}  // namespace vesta_constraints
