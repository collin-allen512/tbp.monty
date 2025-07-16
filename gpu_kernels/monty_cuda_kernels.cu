#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// ============================================================================
// HYPOTHESIS UPDATE KERNELS
// ============================================================================

// Displacement calculation kernel
template<typename scalar_t>
__global__ void displacement_kernel(
    const scalar_t* __restrict__ poses,
    const scalar_t* __restrict__ displacement,
    const scalar_t* __restrict__ locations,
    scalar_t* __restrict__ output,
    int N) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    __shared__ scalar_t s_disp[3];
    if (threadIdx.x < 3) {
        s_disp[threadIdx.x] = displacement[threadIdx.x];
    }
    __syncthreads();

    scalar_t rotated_disp[3] = {scalar_t(0), scalar_t(0), scalar_t(0)};

    rotated_disp[0] = poses[idx * 9 + 0] * s_disp[0] +
                      poses[idx * 9 + 1] * s_disp[1] +
                      poses[idx * 9 + 2] * s_disp[2];

    rotated_disp[1] = poses[idx * 9 + 3] * s_disp[0] +
                      poses[idx * 9 + 4] * s_disp[1] +
                      poses[idx * 9 + 5] * s_disp[2];

    rotated_disp[2] = poses[idx * 9 + 6] * s_disp[0] +
                      poses[idx * 9 + 7] * s_disp[1] +
                      poses[idx * 9 + 8] * s_disp[2];

    output[idx * 3 + 0] = locations[idx * 3 + 0] + rotated_disp[0];
    output[idx * 3 + 1] = locations[idx * 3 + 1] + rotated_disp[1];
    output[idx * 3 + 2] = locations[idx * 3 + 2] + rotated_disp[2];
}

// Evidence aggregation kernel
template<typename scalar_t>
__global__ void evidence_aggregation_kernel(
    const scalar_t* __restrict__ old_evidence,
    const scalar_t* __restrict__ new_evidence,
    const int64_t* __restrict__ test_indices,
    scalar_t* __restrict__ output_evidence,
    scalar_t min_update,
    scalar_t past_weight,
    scalar_t present_weight,
    int N,
    int M) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    scalar_t update_value = min_update;

    for (int i = 0; i < M; i++) {
        if (test_indices[i] == idx) {
            update_value = new_evidence[i];
            break;
        }
    }

    output_evidence[idx] = old_evidence[idx] * past_weight +
                          update_value * present_weight;
}

// ============================================================================
// EVIDENCE CALCULATION KERNELS
// ============================================================================

// Pose transformation kernel
// template<typename scalar_t>
// __global__ void pose_transformation_kernel(
//     const scalar_t* __restrict__ pose_vectors,
//     const scalar_t* __restrict__ reference_poses,
//     scalar_t* __restrict__ output_vectors,
//     int N) {

//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx >= N) return;

//     scalar_t ref_pose[9];
//     for (int i = 0; i < 9; i++) {
//         ref_pose[i] = reference_poses[idx * 9 + i];
//     }

//     // Temporary storage for the intermediate result
//     scalar_t temp_result[9];

//     for (int vec_idx = 0; vec_idx < 3; vec_idx++) {
//         for (int i = 0; i < 3; i++) {
//             scalar_t result = 0;
//             for (int j = 0; j < 3; j++) {
//                 result += ref_pose[i * 3 + j] * pose_vectors[idx * 9 + vec_idx * 3 + j];
//             }
//             temp_result[vec_idx * 3 + i] = result;  // Store in temp
//         }
//     }

//     // Transpose: (0, 2, 1) - swap last two dimensions
//     for (int i = 0; i < 3; i++) {
//         for (int j = 0; j < 3; j++) {
//             output_vectors[idx * 9 + i * 3 + j] = temp_result[j * 3 + i];
//         }
//     }
// }

// __global__ void pose_transformation_kernel(
//     const scalar_t* __restrict__ pose_vectors,
//     const scalar_t* __restrict__ reference_poses,
//     scalar_t* __restrict__ output_vectors,
//     int N) {

//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx >= N) return;

//     scalar_t ref_pose[9];
//     for (int i = 0; i < 9; i++) {
//         ref_pose[i] = reference_poses[idx * 9 + i];
//     }

//     // Directly compute the transposed result
//     for (int i = 0; i < 3; i++) {          // Output row
//         for (int j = 0; j < 3; j++) {      // Output column
//             scalar_t result = 0;
//             for (int k = 0; k < 3; k++) {
//                 // This computes: output[i,j] = sum_k(ref_pose[k,i] * pose_vector[k,j])
//                 // Which is equivalent to (ref_pose @ pose_vector.T).T
//                 result += ref_pose[k * 3 + i] * pose_vectors[idx * 9 + j * 3 + k];
//             }
//             output_vectors[idx * 9 + i * 3 + j] = result;
//         }
//     }
// }
template<typename scalar_t>
__global__ void pose_transformation_kernel(
    const scalar_t* __restrict__ pose_vectors,
    const scalar_t* __restrict__ reference_poses,
    scalar_t* __restrict__ output_vectors,
    int N) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Load reference pose matrix for this hypothesis
    scalar_t ref_pose[9];
    for (int i = 0; i < 9; i++) {
        ref_pose[i] = reference_poses[idx * 9 + i];
    }

    // Load pose vectors (shared across all hypotheses)
    scalar_t pose_vec[9];
    for (int i = 0; i < 9; i++) {
        pose_vec[i] = pose_vectors[i];
    }

    // CPU does: rotated_pv = ref_frame_rots.dot(old_pv.T).transpose((0, 2, 1))
    // Which is equivalent to: output = pose_vec @ ref_pose.T

    // Compute pose_vec @ ref_pose.T
    for (int i = 0; i < 3; i++) {        // row of output
        for (int j = 0; j < 3; j++) {    // column of output
            scalar_t sum = 0;
            for (int k = 0; k < 3; k++) {
                // pose_vec[i,k] * ref_pose[j,k] (ref_pose transposed)
                sum += pose_vec[i * 3 + k] * ref_pose[j * 3 + k];
            }
            output_vectors[idx * 9 + i * 3 + j] = sum;
        }
    }
}

// KNN search kernel (brute force, matching scipy.spatial.KDTree behavior)
template<typename scalar_t>
__global__ void knn_search_kernel(
    const scalar_t* __restrict__ graph_locations,
    const scalar_t* __restrict__ query_locations,
    int64_t* __restrict__ nearest_indices,
    int N, int M, int K) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Safety check: K should not exceed graph size
    int safe_K = min(K, M);

    scalar_t query_loc[3];
    for (int i = 0; i < 3; i++) {
        query_loc[i] = query_locations[idx * 3 + i];
    }

    // Use a simple insertion sort approach to find K smallest distances
    // This exactly matches the behavior of scipy.spatial.KDTree
    scalar_t best_distances[64];  // Assuming K <= 64
    int64_t best_indices[64];

    // Initialize with first K points or all points if M < K
    int init_count = min(safe_K, M);

    // Calculate initial distances
    for (int k = 0; k < init_count; k++) {
        scalar_t dist_sq = 0;
        for (int i = 0; i < 3; i++) {
            scalar_t diff = graph_locations[k * 3 + i] - query_loc[i];
            dist_sq += diff * diff;
        }
        best_distances[k] = dist_sq;
        best_indices[k] = k;
    }

    // Sort the initial set (insertion sort)
    for (int i = 1; i < init_count; i++) {
        scalar_t key_dist = best_distances[i];
        int64_t key_idx = best_indices[i];
        int j = i - 1;

        while (j >= 0 && best_distances[j] > key_dist) {
            best_distances[j + 1] = best_distances[j];
            best_indices[j + 1] = best_indices[j];
            j--;
        }
        best_distances[j + 1] = key_dist;
        best_indices[j + 1] = key_idx;
    }

    // Process remaining points
    for (int node_idx = init_count; node_idx < M; node_idx++) {
        scalar_t dist_sq = 0;
        for (int i = 0; i < 3; i++) {
            scalar_t diff = graph_locations[node_idx * 3 + i] - query_loc[i];
            dist_sq += diff * diff;
        }

        // If this distance is smaller than the largest in our K-best list
        if (dist_sq < best_distances[safe_K - 1]) {
            // Insert this point in the correct position
            int pos = safe_K - 1;
            while (pos > 0 && best_distances[pos - 1] > dist_sq) {
                best_distances[pos] = best_distances[pos - 1];
                best_indices[pos] = best_indices[pos - 1];
                pos--;
            }
            best_distances[pos] = dist_sq;
            best_indices[pos] = node_idx;
        }
    }

    // Copy results to output
    for (int k = 0; k < safe_K; k++) {
        nearest_indices[idx * K + k] = best_indices[k];
    }

    // Pad with -1 if necessary
    for (int k = safe_K; k < K; k++) {
        nearest_indices[idx * K + k] = -1;
    }
}

// ============================================================================
// C++ WRAPPERS AND DISPATCHERS
// ============================================================================

// Hypothesis update wrappers
torch::Tensor displacement_cuda(
    torch::Tensor poses,
    torch::Tensor displacement,
    torch::Tensor locations) {

    const int N = poses.size(0);
    auto output = torch::empty({N, 3}, locations.options());

    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(poses.scalar_type(), "displacement_cuda", ([&] {
        displacement_kernel<scalar_t><<<blocks, threads>>>(
            poses.data_ptr<scalar_t>(),
            displacement.data_ptr<scalar_t>(),
            locations.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            N
        );
    }));

    return output;
}

torch::Tensor evidence_aggregation_cuda(
    torch::Tensor old_evidence,
    torch::Tensor new_evidence,
    torch::Tensor test_indices,
    float min_update,
    float past_weight,
    float present_weight) {

    const int N = old_evidence.size(0);
    const int M = new_evidence.size(0);
    auto output = torch::empty_like(old_evidence);

    if (test_indices.scalar_type() != torch::kLong) {
        test_indices = test_indices.to(torch::kLong);
    }

    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(old_evidence.scalar_type(), "evidence_aggregation_cuda", ([&] {
        evidence_aggregation_kernel<scalar_t><<<blocks, threads>>>(
            old_evidence.data_ptr<scalar_t>(),
            new_evidence.data_ptr<scalar_t>(),
            test_indices.data_ptr<int64_t>(),
            output.data_ptr<scalar_t>(),
            static_cast<scalar_t>(min_update),
            static_cast<scalar_t>(past_weight),
            static_cast<scalar_t>(present_weight),
            N,
            M
        );
    }));

    return output;
}

// Evidence calculation wrappers
torch::Tensor pose_transformation_cuda(
    torch::Tensor pose_vectors,
    torch::Tensor reference_poses) {

    const int N = reference_poses.size(0);  // N is number of reference poses
    auto output = torch::empty({N, 3, 3}, pose_vectors.options());

    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(pose_vectors.scalar_type(), "pose_transformation_cuda", ([&] {
        pose_transformation_kernel<scalar_t><<<blocks, threads>>>(
            pose_vectors.data_ptr<scalar_t>(),
            reference_poses.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            N
        );
    }));

    return output;
}

torch::Tensor knn_search_cuda(
    torch::Tensor graph_locations,
    torch::Tensor query_locations,
    int k_neighbors) {

    const int N = query_locations.size(0);
    const int M = graph_locations.size(0);
    const int K = k_neighbors;

    auto indices = torch::empty({N, K}, torch::TensorOptions()
                                      .dtype(torch::kLong)
                                      .device(query_locations.device()));

    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(query_locations.scalar_type(), "knn_search_cuda", ([&] {
        knn_search_kernel<scalar_t><<<blocks, threads>>>(
            graph_locations.data_ptr<scalar_t>(),
            query_locations.data_ptr<scalar_t>(),
            indices.data_ptr<int64_t>(),
            N, M, K
        );
    }));

    return indices;
}


// Custom distance calculation kernel
template<typename scalar_t>
__global__ void custom_distance_kernel(
    const scalar_t* __restrict__ node_locs,
    const scalar_t* __restrict__ query_locs,
    const scalar_t* __restrict__ pose_normals,
    scalar_t* __restrict__ custom_distances,
    scalar_t max_abs_curvature,
    int N, int M) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    scalar_t query_loc[3];
    scalar_t pose_normal[3];

    // Load query location and pose normal
    for (int i = 0; i < 3; i++) {
        query_loc[i] = query_locs[idx * 3 + i];
        pose_normal[i] = pose_normals[idx * 3 + i];
    }

    // Calculate custom distance for each neighbor
    for (int j = 0; j < M; j++) {
        scalar_t diff[3];
        scalar_t euclidean_dist_sq = 0;
        scalar_t dot_product = 0;

        // Calculate difference vector and dot product
        for (int i = 0; i < 3; i++) {
            diff[i] = node_locs[idx * M * 3 + j * 3 + i] - query_loc[i];
            euclidean_dist_sq += diff[i] * diff[i];
            dot_product += diff[i] * pose_normal[i];
        }

        scalar_t euclidean_dist = sqrt(euclidean_dist_sq);
        scalar_t curvature_factor = 1.0 / (abs(max_abs_curvature) + 0.5);
        scalar_t custom_dist = euclidean_dist + abs(dot_product) * curvature_factor;

        custom_distances[idx * M + j] = custom_dist;
    }
}

// Angle calculation kernel
template<typename scalar_t>
__global__ void angle_calculation_kernel(
    const scalar_t* __restrict__ node_vectors,
    const scalar_t* __restrict__ query_vectors,
    scalar_t* __restrict__ angles,
    int N, int M) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    scalar_t query_vec[3];

    // Load query vector
    for (int i = 0; i < 3; i++) {
        query_vec[i] = query_vectors[idx * 3 + i];
    }

    // Calculate angles for each neighbor
    for (int j = 0; j < M; j++) {
        scalar_t dot_product = 0;

        // Calculate dot product (assumes vectors are already normalized)
        for (int i = 0; i < 3; i++) {
            scalar_t node_val = node_vectors[idx * M * 3 + j * 3 + i];
            dot_product += node_val * query_vec[i];
        }

        // Clamp and calculate angle (no normalization needed)
        dot_product = fmax(-1.0, fmin(1.0, dot_product));  // Clamp to [-1, 1]
        angles[idx * M + j] = acos(dot_product);
    }
}

// Pose evidence calculation kernel
template<typename scalar_t>
__global__ void pose_evidence_kernel(
    const scalar_t* __restrict__ pn_angles,
    const scalar_t* __restrict__ cd1_angles,
    const int* __restrict__ use_cd,
    scalar_t* __restrict__ pose_evidence,
    scalar_t pn_weight,
    scalar_t cd1_weight,
    int N, int M) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * M) return;

    int hyp_idx = idx / M;
    int neighbor_idx = idx % M;

    // Calculate PN evidence: -(sin(angle/2) - 0.5)
    scalar_t pn_angle = pn_angles[idx];
    scalar_t pn_evidence = -(sin(pn_angle / 2.0) - 0.5);

    // Calculate CD1 evidence if available
    scalar_t cd1_evidence = 0;

    // Only process CD1 if cd1_weight > 0 (query pose is fully defined)
    if (cd1_weight > 0) {
        if (use_cd[idx]) {
            scalar_t cd1_angle = cd1_angles[idx];
            scalar_t cd1_error = M_PI / 2.0 - abs(cd1_angle - M_PI / 2.0);
            cd1_evidence = -(sin(cd1_error) - 0.5);
        } else {
            // Double PN evidence if CD1 not available (only when pose is fully defined)
            pn_evidence *= 2.0;
        }
    }

    // Combine weighted evidence
    pose_evidence[idx] = pn_evidence * pn_weight + cd1_evidence * cd1_weight;
}

// CPU fallback implementations
torch::Tensor displacement_cpu(
    torch::Tensor poses,
    torch::Tensor displacement,
    torch::Tensor locations) {
    torch::Tensor rotated_displacement = torch::matmul(poses, displacement);
    return locations + rotated_displacement;
}

torch::Tensor evidence_aggregation_cpu(
    torch::Tensor old_evidence,
    torch::Tensor new_evidence,
    torch::Tensor test_indices,
    float min_update,
    float past_weight,
    float present_weight) {

    if (test_indices.scalar_type() != torch::kLong) {
        test_indices = test_indices.to(torch::kLong);
    }

    torch::Tensor evidence_to_add = torch::full_like(old_evidence, min_update);
    evidence_to_add.index_put_({test_indices}, new_evidence);

    return old_evidence * past_weight + evidence_to_add * present_weight;
}

torch::Tensor pose_transformation_cpu(
    torch::Tensor pose_vectors,
    torch::Tensor reference_poses) {
    // CPU implementation matching spatial_arithmetics.py:
    // rotated_pv = ref_frame_rots.dot(old_pv.T)
    // rotated_pv = rotated_pv.transpose((0, 2, 1))

    // First transpose pose_vectors
    auto pose_vectors_T = pose_vectors.t();

    // Then do ref_frame_rots @ pose_vectors_T
    auto result = torch::matmul(reference_poses, pose_vectors_T);

    // Transpose last two dimensions (0, 2, 1)
    return result.transpose(1, 2);
}

torch::Tensor knn_search_cpu(
    torch::Tensor graph_locations,
    torch::Tensor query_locations,
    int k_neighbors) {

    const int N = query_locations.size(0);
    const int M = graph_locations.size(0);
    const int K = std::min(k_neighbors, M);  // Can't find more neighbors than graph points

    auto indices = torch::empty({N, k_neighbors}, torch::TensorOptions()
                                      .dtype(torch::kLong)
                                      .device(query_locations.device()));

    // Simple PyTorch-based implementation without problematic indexing
    for (int i = 0; i < N; i++) {
        auto query = query_locations[i].unsqueeze(0);
        auto diffs = graph_locations - query;
        auto distances = torch::norm(diffs, /*dim=*/1);
        auto sorted_indices = torch::argsort(distances);

        // Copy the first K indices, pad with -1 if needed
        for (int k = 0; k < k_neighbors; k++) {
            if (k < K) {
                auto idx = sorted_indices[k];
                indices.index_put_({i, k}, idx);
            } else {
                indices.index_put_({i, k}, -1);
            }
        }
    }

    return indices;
}


// C++ wrapper implementations
torch::Tensor custom_distance(
    torch::Tensor node_locs,
    torch::Tensor query_locs,
    torch::Tensor pose_normals,
    float max_abs_curvature) {

    const int N = query_locs.size(0);
    const int M = node_locs.size(1);

    auto custom_distances = torch::empty({N, M}, query_locs.options());

    if (query_locs.device().is_cuda()) {
        const int threads = 256;
        const int blocks = (N + threads - 1) / threads;

        AT_DISPATCH_FLOATING_TYPES(query_locs.scalar_type(), "custom_distance_cuda", ([&] {
            custom_distance_kernel<scalar_t><<<blocks, threads>>>(
                node_locs.data_ptr<scalar_t>(),
                query_locs.data_ptr<scalar_t>(),
                pose_normals.data_ptr<scalar_t>(),
                custom_distances.data_ptr<scalar_t>(),
                static_cast<scalar_t>(max_abs_curvature),
                N, M
            );
        }));
    } else {
        // CPU implementation
        for (int i = 0; i < N; i++) {
            auto query_loc = query_locs[i];
            auto pose_normal = pose_normals[i];

            for (int j = 0; j < M; j++) {
                auto node_loc = node_locs[i][j];
                auto diff = node_loc - query_loc;
                auto euclidean_dist = torch::norm(diff);
                auto dot_product = torch::dot(diff, pose_normal);
                auto curvature_factor = 1.0 / (std::abs(max_abs_curvature) + 0.5);
                auto custom_dist = euclidean_dist + torch::abs(dot_product) * curvature_factor;
                custom_distances[i][j] = custom_dist;
            }
        }
    }

    return custom_distances;
}

torch::Tensor angle_calculation(
    torch::Tensor node_vectors,
    torch::Tensor query_vectors) {

    const int N = query_vectors.size(0);
    const int M = node_vectors.size(1);

    auto angles = torch::empty({N, M}, query_vectors.options());

    if (query_vectors.device().is_cuda()) {
        const int threads = 256;
        const int blocks = (N + threads - 1) / threads;

        AT_DISPATCH_FLOATING_TYPES(query_vectors.scalar_type(), "angle_calculation_cuda", ([&] {
            angle_calculation_kernel<scalar_t><<<blocks, threads>>>(
                node_vectors.data_ptr<scalar_t>(),
                query_vectors.data_ptr<scalar_t>(),
                angles.data_ptr<scalar_t>(),
                N, M
            );
        }));
    } else {
        // CPU implementation
        for (int i = 0; i < N; i++) {
            auto query_vec = query_vectors[i];

            for (int j = 0; j < M; j++) {
                auto node_vec = node_vectors[i][j];
                auto dot_product = torch::dot(node_vec, query_vec);
                // Assume vectors are already normalized, no extra normalization
                dot_product = torch::clamp(dot_product, -1.0, 1.0);
                angles[i][j] = torch::acos(dot_product);
            }
        }
    }

    return angles;
}

torch::Tensor pose_evidence(
    torch::Tensor pn_angles,
    torch::Tensor cd1_angles,
    torch::Tensor use_cd,
    float pn_weight,
    float cd1_weight) {

    const int N = pn_angles.size(0);
    const int M = pn_angles.size(1);

    auto pose_evidence_result = torch::empty_like(pn_angles);

    if (pn_angles.device().is_cuda()) {
        const int threads = 256;
        const int blocks = (N * M + threads - 1) / threads;

        AT_DISPATCH_FLOATING_TYPES(pn_angles.scalar_type(), "pose_evidence_cuda", ([&] {
            pose_evidence_kernel<scalar_t><<<blocks, threads>>>(
                pn_angles.data_ptr<scalar_t>(),
                cd1_angles.data_ptr<scalar_t>(),
                use_cd.data_ptr<int>(),
                pose_evidence_result.data_ptr<scalar_t>(),
                static_cast<scalar_t>(pn_weight),
                static_cast<scalar_t>(cd1_weight),
                N, M
            );
        }));
    } else {
        // CPU implementation
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < M; j++) {
                auto pn_angle = pn_angles[i][j];
                auto pn_evidence = -(torch::sin(pn_angle / 2.0) - 0.5);

                auto cd1_evidence = torch::zeros_like(pn_evidence);

                // Only process CD1 if cd1_weight > 0 (query pose is fully defined)
                if (cd1_weight > 0) {
                    if (use_cd[i][j].item<int>()) {
                        auto cd1_angle = cd1_angles[i][j];
                        auto cd1_error = M_PI / 2.0 - torch::abs(cd1_angle - M_PI / 2.0);
                        cd1_evidence = -(torch::sin(cd1_error) - 0.5);
                    } else {
                        pn_evidence *= 2.0;
                    }
                }

                pose_evidence_result[i][j] = pn_evidence * pn_weight + cd1_evidence * cd1_weight;
            }
        }
    }

    return pose_evidence_result;
}

torch::Tensor final_aggregation(
    torch::Tensor evidence_matrix) {
    return std::get<0>(torch::max(evidence_matrix, /*dim=*/1));
}

// Main dispatcher functions
torch::Tensor displacement(
    torch::Tensor poses,
    torch::Tensor displacement,
    torch::Tensor locations) {
    if (poses.device().is_cuda()) {
        return displacement_cuda(poses, displacement, locations);
    } else {
        return displacement_cpu(poses, displacement, locations);
    }
}

torch::Tensor evidence_aggregation(
    torch::Tensor old_evidence,
    torch::Tensor new_evidence,
    torch::Tensor test_indices,
    float min_update,
    float past_weight,
    float present_weight) {
    if (old_evidence.device().is_cuda()) {
        return evidence_aggregation_cuda(
            old_evidence, new_evidence, test_indices,
            min_update, past_weight, present_weight);
    } else {
        return evidence_aggregation_cpu(
            old_evidence, new_evidence, test_indices,
            min_update, past_weight, present_weight);
    }
}

torch::Tensor pose_transformation(
    torch::Tensor pose_vectors,
    torch::Tensor reference_poses) {
    if (pose_vectors.device().is_cuda()) {
        return pose_transformation_cuda(pose_vectors, reference_poses);
    } else {
        return pose_transformation_cpu(pose_vectors, reference_poses);
    }
}

torch::Tensor knn_search(
    torch::Tensor graph_locations,
    torch::Tensor query_locations,
    int k_neighbors) {
    if (query_locations.device().is_cuda()) {
        return knn_search_cuda(graph_locations, query_locations, k_neighbors);
    } else {
        return knn_search_cpu(graph_locations, query_locations, k_neighbors);
    }
}

// ============================================================================
// PYTHON BINDINGS
// ============================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Hypothesis update functions
    m.def("displacement", &displacement,
          "Displacement calculation with CUDA acceleration");
    m.def("evidence_aggregation", &evidence_aggregation,
          "Evidence aggregation with CUDA acceleration");

    // Evidence calculation functions
    m.def("pose_transformation", &pose_transformation,
          "Pose transformation with CUDA acceleration");
    m.def("knn_search", &knn_search,
          "KNN search with CUDA acceleration");
    m.def("custom_distance", &custom_distance,
          "Custom distance calculation");
    m.def("angle_calculation", &angle_calculation,
          "Angle calculation");
    m.def("pose_evidence", &pose_evidence,
          "Pose evidence calculation");
    m.def("final_aggregation", &final_aggregation,
          "Final aggregation");
}