# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Profiled version of HypothesesUpdater to measure computational bottlenecks."""

from __future__ import annotations

import logging
import pickle
import time
from collections import defaultdict
from typing import Literal, Type

import numpy as np
from scipy.spatial.transform import Rotation

from tbp.monty.frameworks.models.evidence_matching.feature_evidence.calculator import (
    DefaultFeatureEvidenceCalculator,
    FeatureEvidenceCalculator,
)
from tbp.monty.frameworks.models.evidence_matching.features_for_matching.selector import (
    DefaultFeaturesForMatchingSelector,
    FeaturesForMatchingSelector,
)
from tbp.monty.frameworks.models.evidence_matching.graph_memory import (
    EvidenceGraphMemory,
)
from tbp.monty.frameworks.models.evidence_matching.hypotheses import (
    ChannelHypotheses,
    Hypotheses,
)
from tbp.monty.frameworks.models.evidence_matching.hypotheses_updater import (
    DefaultHypothesesUpdater,
)
from tbp.monty.frameworks.models.evidence_matching.profiled_hypotheses_displacer import (
    ProfiledHypothesesDisplacer,
)
from tbp.monty.frameworks.utils.evidence_matching import ChannelMapper

logger = logging.getLogger(__name__)


class ProfiledHypothesesUpdater(DefaultHypothesesUpdater):
    """Profiled version of DefaultHypothesesUpdater with detailed timing."""

    def __init__(
        self,
        feature_weights: dict,
        graph_memory: EvidenceGraphMemory,
        max_match_distance: float,
        tolerances: dict,
        feature_evidence_calculator: type[FeatureEvidenceCalculator] = (
            DefaultFeatureEvidenceCalculator
        ),
        feature_evidence_increment: int = 1,
        features_for_matching_selector: type[FeaturesForMatchingSelector] = (
            DefaultFeaturesForMatchingSelector
        ),
        initial_possible_poses: Literal["uniform", "informed"]
        | list[Rotation] = "informed",
        max_nneighbors: int = 3,
        past_weight: float = 1,
        present_weight: float = 1,
        umbilical_num_poses: int = 8,
        save_example_data: bool = False,
        example_data_path: str = "hypothesis_update_examples.pkl",
    ):
        super().__init__(
            feature_weights=feature_weights,
            graph_memory=graph_memory,
            max_match_distance=max_match_distance,
            tolerances=tolerances,
            feature_evidence_calculator=feature_evidence_calculator,
            feature_evidence_increment=feature_evidence_increment,
            features_for_matching_selector=features_for_matching_selector,
            initial_possible_poses=initial_possible_poses,
            max_nneighbors=max_nneighbors,
            past_weight=past_weight,
            present_weight=present_weight,
            umbilical_num_poses=umbilical_num_poses,
        )

        # Replace the default displacer with profiled version
        self.hypotheses_displacer = ProfiledHypothesesDisplacer(
            feature_evidence_increment=self.feature_evidence_increment,
            feature_weights=self.feature_weights,
            graph_memory=self.graph_memory,
            max_match_distance=max_match_distance,
            max_nneighbors=max_nneighbors,
            past_weight=past_weight,
            present_weight=present_weight,
            tolerances=self.tolerances,
            use_features_for_matching=self.use_features_for_matching,
        )

        # Profiling data storage
        self.timing_data = defaultdict(list)
        self.operation_counts = defaultdict(int)
        self.save_example_data = save_example_data
        self.example_data_path = example_data_path
        self.saved_examples = []

    def update_hypotheses(
        self,
        hypotheses: Hypotheses,
        features: dict,
        displacements: dict | None,
        graph_id: str,
        mapper: ChannelMapper,
        evidence_update_threshold: float,
    ) -> list[ChannelHypotheses]:
        """Profiled version of hypothesis update."""

        start_total = time.perf_counter()

        # Get all usable input channels
        from tbp.monty.frameworks.models.evidence_matching.hypotheses_updater import (
            all_usable_input_channels,
        )

        input_channels_to_use = all_usable_input_channels(
            features, self.graph_memory.get_input_channels_in_graph(graph_id)
        )

        # Save example data for GPU testing
        if self.save_example_data and len(self.saved_examples) < 10:
            example = {
                "hypotheses": hypotheses,
                "features": features,
                "displacements": displacements,
                "graph_id": graph_id,
                "mapper": mapper,
                "evidence_update_threshold": evidence_update_threshold,
                "input_channels": input_channels_to_use,
            }
            self.saved_examples.append(example)

        hypotheses_updates = []

        for input_channel in input_channels_to_use:
            start_channel = time.perf_counter()

            # Get channel-specific data
            channel_displacement = displacements[input_channel] if displacements else None
            channel_features = features[input_channel]

            initialize_hyp_space = bool(input_channel not in mapper.channels)
            # Initialize a new hypothesis space using graph nodes
            if initialize_hyp_space:
                # TODO H: When initializing a hypothesis for a channel later on (if
                # displacement is not None), include most likely existing hypothesis
                # from other channels?
                channel_possible_hypotheses = self._get_initial_hypothesis_space(
                    channel_features=features[input_channel],
                    graph_id=graph_id,
                    input_channel=input_channel,
                )
            else:
                channel_hypotheses = mapper.extract_hypotheses(
                    hypotheses, input_channel
                )

                # We only displace existing hypotheses since the newly sampled
                # hypotheses should not be affected by the displacement from the last
                # sensory input.
                channel_possible_hypotheses = (
                    self.hypotheses_displacer.displace_hypotheses_and_compute_evidence(
                        channel_displacement=displacements[input_channel],
                        channel_features=features[input_channel],
                        evidence_update_threshold=evidence_update_threshold,
                        graph_id=graph_id,
                        possible_hypotheses=channel_hypotheses,
                        total_hypotheses_count=hypotheses.evidence.shape[0],
                    )
                )
            hypotheses_updates.append(channel_possible_hypotheses)


            time_channel = time.perf_counter() - start_channel
            self.timing_data[f"channel_{input_channel}_update"].append(time_channel)

        time_total = time.perf_counter() - start_total
        self.timing_data["total_hypothesis_update"].append(time_total)
        self.operation_counts["updates"] += 1

        return hypotheses_updates

    def get_profiling_summary(self):
        """Get a combined summary of profiling data."""
        summary = {
            "updater": {},
            "displacer": self.hypotheses_displacer.get_profiling_summary(),
        }

        # Calculate statistics for updater operations
        for operation, times in self.timing_data.items():
            if times:
                summary["updater"][operation] = {
                    "count": len(times),
                    "total_time": sum(times),
                    "mean_time": np.mean(times),
                    "std_time": np.std(times),
                    "min_time": min(times),
                    "max_time": max(times),
                }

        # Add operation counts
        summary["operation_counts"] = dict(self.operation_counts)

        return summary

    def save_profiling_data(self, output_dir="."):
        """Save profiling data as a single JSON file."""
        import json
        import os
        from datetime import datetime

        # Get comprehensive summary
        summary = self.get_profiling_summary()

        # Add metadata
        summary["metadata"] = {
            "timestamp": datetime.now().isoformat(),
            "total_updates": self.operation_counts.get("updates", 0),
            "example_data_saved": len(self.saved_examples),
        }

        # Save as JSON
        output_file = os.path.join(output_dir, "hypothesis_profiling_results.json")
        with open(output_file, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info(f"Saved hypothesis profiling results to {output_file}")

        # Save example data if collected
        if self.saved_examples:
            example_file = os.path.join(output_dir, "hypothesis_update_examples.pkl")
            with open(example_file, "wb") as f:
                pickle.dump(self.saved_examples, f)
            logger.info(f"Saved {len(self.saved_examples)} example updates to {example_file}")

        return output_file

    def reset_profiling(self):
        """Reset all profiling data."""
        self.timing_data.clear()
        self.operation_counts.clear()
        self.hypotheses_displacer.reset_profiling()
        self.saved_examples.clear()

    def print_profiling_report(self):
        """Print a human-readable profiling report."""
        summary = self.get_profiling_summary()

        print("\n=== HYPOTHESIS UPDATE PROFILING REPORT ===\n")

        # Updater timing
        print("Updater Operations:")
        for op, stats in summary["updater"].items():
            print(f"  {op}:")
            print(f"    Count: {stats['count']}")
            print(f"    Total: {stats['total_time']:.6f}s")
            print(f"    Mean:  {stats['mean_time']:.6f}s")
            print(f"    Std:   {stats['std_time']:.6f}s")

        # Displacer timing
        print("\nDisplacer Operations:")
        for op, stats in summary["displacer"].items():
            if op == "data_sizes":
                continue
            print(f"  {op}:")
            print(f"    Count: {stats['count']}")
            print(f"    Total: {stats['total_time']:.6f}s")
            print(f"    Mean:  {stats['mean_time']:.6f}s")
            print(f"    Std:   {stats['std_time']:.6f}s")

        # Data sizes
        if "data_sizes" in summary["displacer"]:
            print("\nData Sizes:")
            for size_type, stats in summary["displacer"]["data_sizes"].items():
                print(f"  {size_type}:")
                print(f"    Mean: {stats['mean']:.1f}")
                print(f"    Max:  {stats['max']}")

        print("\n" + "="*40 + "\n")

    def finalize_profiling(self, output_dir="."):
        """Called at the end of an experiment to save profiling data and print summary."""
        if self.timing_data or self.hypotheses_displacer.timing_data:
            logger.info("Finalizing hypothesis update profiling...")

            # Print summary to console
            self.print_profiling_report()

            # Save data to file
            output_file = self.save_profiling_data(output_dir)

            print(f"\n🔍 Hypothesis profiling complete!")
            print(f"📊 Results saved to: {output_file}")
            if self.saved_examples:
                print(f"💾 {len(self.saved_examples)} examples saved for GPU testing")
            print("")
        else:
            logger.info("No profiling data collected.")