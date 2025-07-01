# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Profiled version of HypothesesDisplacer to measure computational bottlenecks."""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Type

import numpy as np

from tbp.monty.frameworks.models.evidence_matching.feature_evidence.calculator import (
    DefaultFeatureEvidenceCalculator,
    FeatureEvidenceCalculator,
)
from tbp.monty.frameworks.models.evidence_matching.graph_memory import (
    EvidenceGraphMemory,
)
from tbp.monty.frameworks.models.evidence_matching.hypotheses import ChannelHypotheses
from tbp.monty.frameworks.models.evidence_matching.hypotheses_displacer import (
    DefaultHypothesesDisplacer,
)
from tbp.monty.frameworks.utils.graph_matching_utils import (
    get_custom_distances,
    get_relevant_curvature,
)
from tbp.monty.frameworks.utils.spatial_arithmetics import (
    get_angles_for_all_hypotheses,
    rotate_pose_dependent_features,
)

logger = logging.getLogger(__name__)


class ProfiledHypothesesDisplacer(DefaultHypothesesDisplacer):
    """Profiled version of DefaultHypothesesDisplacer with detailed timing."""

    def __init__(
        self,
        feature_weights: dict,
        graph_memory: EvidenceGraphMemory,
        max_match_distance: float,
        tolerances: dict,
        use_features_for_matching: dict[str, bool],
        feature_evidence_calculator: Type[
            FeatureEvidenceCalculator
        ] = DefaultFeatureEvidenceCalculator,
        feature_evidence_increment: int = 1,
        max_nneighbors: int = 3,
        past_weight: float = 1,
        present_weight: float = 1,
    ):
        super().__init__(
            feature_weights=feature_weights,
            graph_memory=graph_memory,
            max_match_distance=max_match_distance,
            tolerances=tolerances,
            use_features_for_matching=use_features_for_matching,
            feature_evidence_calculator=feature_evidence_calculator,
            feature_evidence_increment=feature_evidence_increment,
            max_nneighbors=max_nneighbors,
            past_weight=past_weight,
            present_weight=present_weight,
        )
        
        # Profiling data storage
        self.timing_data = defaultdict(list)
        self.operation_counts = defaultdict(int)
        self.data_sizes = defaultdict(list)
        
    def displace_hypotheses_and_compute_evidence(
        self,
        channel_displacement: np.ndarray,
        channel_features: dict,
        evidence_update_threshold: float,
        graph_id: str,
        possible_hypotheses: ChannelHypotheses,
        total_hypotheses_count: int,
    ) -> ChannelHypotheses:
        """Profiled version of hypothesis displacement and evidence computation."""
        
        start_total = time.perf_counter()
        
        # Profile displacement calculation
        start_disp = time.perf_counter()
        rotated_displacements = possible_hypotheses.poses.dot(channel_displacement)
        search_locations = possible_hypotheses.locations + rotated_displacements
        time_disp = time.perf_counter() - start_disp
        
        self.timing_data["displacement_calculation"].append(time_disp)
        self.data_sizes["num_hypotheses"].append(len(possible_hypotheses.locations))
        self.operation_counts["displacement_ops"] += 1
        
        # Get indices of hypotheses with evidence > threshold
        hyp_ids_to_test = np.where(
            possible_hypotheses.evidence >= evidence_update_threshold
        )[0]
        num_hypotheses_to_test = hyp_ids_to_test.shape[0]
        
        if num_hypotheses_to_test > 0:
            logger.info(
                f"Testing {num_hypotheses_to_test} out of "
                f"{total_hypotheses_count} hypotheses for {graph_id} "
                f"(evidence > {evidence_update_threshold})"
            )
            
            # Profile evidence calculation
            start_evidence = time.perf_counter()
            new_evidence = self._calculate_evidence_for_new_locations(
                graph_id=graph_id,
                input_channel=possible_hypotheses.input_channel,
                search_locations=search_locations[hyp_ids_to_test],
                channel_possible_poses=possible_hypotheses.poses[hyp_ids_to_test],
                channel_features=channel_features,
            )
            time_evidence = time.perf_counter() - start_evidence
            
            self.timing_data["evidence_calculation"].append(time_evidence)
            self.data_sizes["num_tested_hypotheses"].append(num_hypotheses_to_test)
            
            # Profile evidence aggregation
            start_agg = time.perf_counter()
            min_update = np.clip(np.min(new_evidence), 0, np.inf)
            evidence_to_add = np.ones_like(possible_hypotheses.evidence) * min_update
            evidence_to_add[hyp_ids_to_test] = new_evidence
            
            evidence = (
                possible_hypotheses.evidence * self.past_weight
                + evidence_to_add * self.present_weight
            )
            time_agg = time.perf_counter() - start_agg
            
            self.timing_data["evidence_aggregation"].append(time_agg)
        else:
            evidence = possible_hypotheses.evidence
            
        time_total = time.perf_counter() - start_total
        self.timing_data["total_displace_and_compute"].append(time_total)
        
        return ChannelHypotheses(
            input_channel=possible_hypotheses.input_channel,
            evidence=evidence,
            locations=search_locations,
            poses=possible_hypotheses.poses,
        )
        
    def _calculate_evidence_for_new_locations(
        self,
        graph_id: str,
        input_channel: str,
        search_locations: np.ndarray,
        channel_possible_poses: np.ndarray,
        channel_features: dict,
    ):
        """Profiled version of evidence calculation."""
        
        logger.debug(
            f"Calculating evidence for {graph_id} using input from {input_channel}"
        )
        
        # Profile pose transformation
        start_pose_transform = time.perf_counter()
        pose_transformed_features = rotate_pose_dependent_features(
            channel_features,
            channel_possible_poses,
        )
        time_pose_transform = time.perf_counter() - start_pose_transform
        self.timing_data["pose_transformation"].append(time_pose_transform)
        
        # Profile nearest neighbor search
        start_nn = time.perf_counter()
        nearest_node_ids = self.graph_memory.get_graph(
            graph_id, input_channel
        ).find_nearest_neighbors(
            search_locations,
            num_neighbors=self.max_nneighbors,
        )
        time_nn = time.perf_counter() - start_nn
        self.timing_data["nearest_neighbor_search"].append(time_nn)
        self.data_sizes["nn_search_queries"].append(len(search_locations))
        
        if self.max_nneighbors == 1:
            nearest_node_ids = np.expand_dims(nearest_node_ids, axis=1)
        
        # Profile location retrieval
        start_loc = time.perf_counter()
        nearest_node_locs = self.graph_memory.get_locations_in_graph(
            graph_id, input_channel
        )[nearest_node_ids]
        time_loc = time.perf_counter() - start_loc
        self.timing_data["location_retrieval"].append(time_loc)
        
        # Profile custom distance calculation
        start_dist = time.perf_counter()
        max_abs_curvature = get_relevant_curvature(channel_features)
        custom_nearest_node_dists = get_custom_distances(
            nearest_node_locs,
            search_locations,
            pose_transformed_features["pose_vectors"][:, 0],
            max_abs_curvature,
        )
        node_distance_weights = self._get_node_distance_weights(
            custom_nearest_node_dists
        )
        time_dist = time.perf_counter() - start_dist
        self.timing_data["distance_calculation"].append(time_dist)
        
        mask = node_distance_weights <= 0
        
        # Profile feature retrieval
        start_feat = time.perf_counter()
        new_pos_features = self.graph_memory.get_features_at_node(
            graph_id,
            input_channel,
            nearest_node_ids,
            feature_keys=["pose_vectors", "pose_fully_defined"],
        )
        time_feat = time.perf_counter() - start_feat
        self.timing_data["feature_retrieval"].append(time_feat)
        
        # Profile pose evidence calculation
        start_pose_ev = time.perf_counter()
        radius_evidence = self._get_pose_evidence_matrix(
            pose_transformed_features,
            new_pos_features,
            input_channel,
            node_distance_weights,
        )
        time_pose_ev = time.perf_counter() - start_pose_ev
        self.timing_data["pose_evidence_calculation"].append(time_pose_ev)
        
        radius_evidence[mask] = -1
        node_distance_weights[mask] = 1
        
        # Profile feature evidence calculation (if enabled)
        if self.use_features_for_matching[input_channel]:
            start_feat_ev = time.perf_counter()
            node_feature_evidence = self.feature_evidence_calculator.calculate(
                channel_feature_array=self.graph_memory.get_feature_array(graph_id)[
                    input_channel
                ],
                channel_feature_order=self.graph_memory.get_feature_order(graph_id)[
                    input_channel
                ],
                channel_feature_weights=self.feature_weights[input_channel],
                channel_query_features=channel_features,
                channel_tolerances=self.tolerances[input_channel],
                input_channel=input_channel,
            )
            hypothesis_radius_feature_evidence = node_feature_evidence[nearest_node_ids]
            hypothesis_radius_feature_evidence[mask] = 0
            radius_evidence = (
                radius_evidence
                + hypothesis_radius_feature_evidence * self.feature_evidence_increment
            )
            time_feat_ev = time.perf_counter() - start_feat_ev
            self.timing_data["feature_evidence_calculation"].append(time_feat_ev)
            
        # Profile final aggregation
        start_final = time.perf_counter()
        location_evidence = np.max(radius_evidence, axis=1)
        time_final = time.perf_counter() - start_final
        self.timing_data["final_aggregation"].append(time_final)
        
        return location_evidence
        
    def _get_pose_evidence_matrix(
        self,
        query_features,
        node_features,
        input_channel,
        node_distance_weights,
    ):
        """Profiled version of pose evidence calculation."""
        
        start_total = time.perf_counter()
        
        evidences_shape = node_distance_weights.shape[:2]
        pose_evidence_weighted = np.zeros(evidences_shape)
        
        # Profile angle calculation
        start_angle = time.perf_counter()
        pn_error = get_angles_for_all_hypotheses(
            node_features["pose_vectors"][:, :, :3],
            query_features["pose_vectors"][:, 0],
        )
        time_angle = time.perf_counter() - start_angle
        self.timing_data["angle_calculation"].append(time_angle)
        
        # Profile evidence computation
        start_comp = time.perf_counter()
        pn_evidence = -(np.sin(pn_error / 2) - 0.5)
        pn_weight = self.feature_weights[input_channel]["pose_vectors"][0]
        
        if not query_features["pose_fully_defined"]:
            cd1_weight = 0
            cd1_evidence = np.zeros(pn_error.shape)
        else:
            cd1_weight = self.feature_weights[input_channel]["pose_vectors"][1]
            use_cd = np.array(
                node_features["pose_fully_defined"][:, :, 0],
                dtype=bool,
            )
            cd1_angle = get_angles_for_all_hypotheses(
                node_features["pose_vectors"][:, :, 3:6],
                query_features["pose_vectors"][:, 1],
            )
            cd1_error = np.pi / 2 - np.abs(cd1_angle - np.pi / 2)
            cd1_evidence = -(np.sin(cd1_error) - 0.5)
            cd1_evidence = cd1_evidence * use_cd
            pn_evidence[np.logical_not(use_cd)] * 2
            
        pose_evidence_weighted += pn_evidence * pn_weight + cd1_evidence * cd1_weight
        time_comp = time.perf_counter() - start_comp
        self.timing_data["pose_evidence_computation"].append(time_comp)
        
        time_total = time.perf_counter() - start_total
        self.timing_data["total_pose_evidence"].append(time_total)
        
        return pose_evidence_weighted
        
    def get_profiling_summary(self):
        """Get a summary of the profiling data."""
        summary = {}
        
        # Calculate statistics for each operation
        for operation, times in self.timing_data.items():
            if times:
                summary[operation] = {
                    "count": len(times),
                    "total_time": sum(times),
                    "mean_time": np.mean(times),
                    "std_time": np.std(times),
                    "min_time": min(times),
                    "max_time": max(times),
                }
                
        # Add data size information
        summary["data_sizes"] = {}
        for size_type, sizes in self.data_sizes.items():
            if sizes:
                summary["data_sizes"][size_type] = {
                    "mean": np.mean(sizes),
                    "std": np.std(sizes),
                    "min": min(sizes),
                    "max": max(sizes),
                }
                
        return summary
        
    def save_profiling_data(self, output_dir="."):
        """Save profiling data (called by ProfiledHypothesesUpdater)."""
        # This method is called by the updater, so we don't need to save separately
        # The updater will include our data in its comprehensive summary
        pass
        
    def reset_profiling(self):
        """Reset all profiling data."""
        self.timing_data.clear()
        self.operation_counts.clear()
        self.data_sizes.clear()
        
    def print_profiling_report(self):
        """Print a human-readable profiling report."""
        summary = self.get_profiling_summary()
        
        print("\n=== HYPOTHESES DISPLACER PROFILING REPORT ===\n")
        
        # Timing operations
        print("Operation Timing:")
        for op, stats in summary.items():
            if op == "data_sizes":
                continue
            print(f"  {op}:")
            print(f"    Count: {stats['count']}")
            print(f"    Total: {stats['total_time']:.6f}s")
            print(f"    Mean:  {stats['mean_time']:.6f}s")
            print(f"    Std:   {stats['std_time']:.6f}s")
            
        # Data sizes
        if "data_sizes" in summary:
            print("\nData Sizes:")
            for size_type, stats in summary["data_sizes"].items():
                print(f"  {size_type}:")
                print(f"    Mean: {stats['mean']:.1f}")
                print(f"    Max:  {stats['max']}")
                
        print("\n" + "="*45 + "\n")