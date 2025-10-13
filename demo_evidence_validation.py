#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demonstration of Evidence Validation Layer

Shows how the validation layer tracks evidence counts per question
across detector stages 1-12, enforcing minimum evidence thresholds
with complete provenance traceability.
"""

from datetime import datetime

from evidence_registry import EvidenceProvenance, EvidenceRegistry


def demo_evidence_validation():
    """Demonstrate evidence validation with 300 questions"""

    print("=" * 80)
    print("EVIDENCE VALIDATION LAYER DEMONSTRATION")
    print("=" * 80)

    # Create registry
    registry = EvidenceRegistry()

    # Generate 300 question IDs (10 decalogos × 30 questions each)
    print("\nGenerating 300 question IDs...")
    all_questions = []
    for d in range(1, 11):
        for q in range(1, 31):
            all_questions.append(f"D{d}-Q{q}")

    print(f"✓ Created {len(all_questions)} question IDs")

    # Simulate evidence collection from 12 detector stages
    print("\nSimulating evidence collection from detector stages 1-12...")

    # Stage configuration: (stages, detector_type, base_confidence)
    detector_configs = [
        (range(1, 4), "monetary", 0.85),  # Stages 1-3
        (range(4, 7), "responsibility", 0.80),  # Stages 4-6
        (range(7, 10), "causal_pattern", 0.75),  # Stages 7-9
        (range(10, 13), "contradiction", 0.70),  # Stages 10-12
    ]

    evidence_count = 0

    # Populate registry with evidence
    for stages, detector_type, base_confidence in detector_configs:
        for stage_num in stages:
            # Each stage produces evidence for ~25% of questions
            for i, qid in enumerate(all_questions):
                if (i + stage_num) % 4 == 0:
                    provenance = EvidenceProvenance(
                        detector_type=detector_type,
                        stage_number=stage_num,
                        source_text_location={
                            "page": (i % 50) + 1,
                            "line": (stage_num * 10) + (i % 30),
                            "char_start": i * 100,
                            "char_end": (i * 100) + 200,
                        },
                        execution_timestamp=datetime.utcnow().isoformat() + "Z",
                        quality_metrics={
                            "precision": base_confidence,
                            "recall": base_confidence - 0.05,
                            "f1": base_confidence - 0.02,
                        },
                    )

                    registry.register(
                        source_component=f"{detector_type}_stage_{stage_num}",
                        evidence_type=f"{detector_type}_evidence",
                        content={
                            "detected_value": f"evidence_from_stage_{stage_num}",
                            "question_id": qid,
                            "detector": detector_type,
                        },
                        confidence=base_confidence,
                        applicable_questions=[qid],
                        provenance=provenance,
                    )
                    evidence_count += 1

    print(f"✓ Registered {evidence_count} evidence items")
    print(f"  Average per question: {evidence_count / len(all_questions):.1f}")

    # Perform validation
    print("\n" + "-" * 80)
    print("PERFORMING POST-EXECUTION VALIDATION")
    print("-" * 80)

    validation_result = registry.validate_evidence_counts(
        all_question_ids=all_questions, min_evidence_threshold=3
    )

    # Display validation summary
    print("\nVALIDATION SUMMARY")
    print("=" * 80)
    print(f"Valid: {validation_result['valid']}")
    print(f"Total Questions: {validation_result['total_questions']}")
    print(
        f"Questions Meeting Threshold: {validation_result['questions_meeting_threshold']}"
    )
    print(
        f"Questions Below Threshold: {len(validation_result['questions_below_threshold'])}"
    )
    print(f"Minimum Evidence Required: {validation_result['min_evidence_threshold']}")
    print(f"Validation Timestamp: {validation_result['validation_timestamp']}")

    # Stage coverage statistics
    print("\n" + "-" * 80)
    print("STAGE COVERAGE ANALYSIS")
    print("-" * 80)

    stage_coverage = validation_result["stage_coverage_summary"]

    print("\nEvidence Count Per Stage:")
    for stage in range(1, 13):
        count = stage_coverage["evidence_count_per_stage"][stage]
        bar = "█" * (count // 10)
        print(f"  Stage {stage:2d}: {count:4d} evidence {bar}")

    print(f"\nStages with evidence: {stage_coverage['stages_with_evidence']}")
    print(f"Stages without evidence: {stage_coverage['stages_without_evidence']}")

    # Sample questions below threshold
    if validation_result["questions_below_threshold"]:
        print("\n" + "-" * 80)
        print("QUESTIONS BELOW THRESHOLD (Sample: first 5)")
        print("-" * 80)

        for qid in validation_result["questions_below_threshold"][:5]:
            summary = validation_result["evidence_summary"][qid]

            print(f"\n{qid}:")
            print(f"  Evidence Count: {summary['evidence_count']}")
            print(
                f"  Contributing Stages: {sorted(summary['stage_contributions'].keys())}"
            )
            print(f"  Missing Stages: {summary['missing_stages'][:8]}...")

            print(f"  Evidence Sources:")
            for source in summary["evidence_sources"]:
                print(
                    f"    • Stage {source['stage_number']}: "
                    f"{source['detector_type']} "
                    f"(confidence: {source['confidence']:.2%})"
                )

    # Sample questions meeting threshold
    meeting_threshold = [
        qid
        for qid in all_questions
        if qid not in validation_result["questions_below_threshold"]
    ]

    if meeting_threshold:
        print("\n" + "-" * 80)
        print("QUESTIONS MEETING THRESHOLD (Sample: first 3)")
        print("-" * 80)

        for qid in meeting_threshold[:3]:
            summary = validation_result["evidence_summary"][qid]

            print(f"\n{qid}:")
            print(f"  Evidence Count: {summary['evidence_count']} ✓")
            print(
                f"  Contributing Stages: {sorted(summary['stage_contributions'].keys())}"
            )

            print(f"  Complete Traceability Chain:")
            for source in summary["evidence_sources"]:
                print(f"    • Evidence ID: {source['evidence_id'][:50]}...")
                print(
                    f"      Detector: {source['detector_type']}, Stage: {source['stage_number']}"
                )
                print(f"      Component: {source['source_component']}")
                print(f"      Confidence: {source['confidence']:.2%}")
                print(f"      Timestamp: {source['execution_timestamp']}")
                if source["quality_metrics"]:
                    metrics = source["quality_metrics"]
                    print(
                        f"      Quality: P={metrics.get('precision', 0):.2%}, "
                        f"R={metrics.get('recall', 0):.2%}, "
                        f"F1={metrics.get('f1', 0):.2%}"
                    )

    # Export validation results
    print("\n" + "-" * 80)
    print("EXPORTING VALIDATION RESULTS")
    print("-" * 80)

    output_file = "evidence_validation_results.json"
    registry.export_validation_results(validation_result, output_file)
    print(f"✓ Validation results exported to: {output_file}")
    print(f"  File contains full traceability for all {len(all_questions)} questions")

    # Summary statistics
    print("\n" + "=" * 80)
    print("KEY METRICS")
    print("=" * 80)

    # Calculate per-stage contribution rate
    total_possible = len(all_questions) * 12  # 300 questions × 12 stages
    actual_contributions = sum(stage_coverage["evidence_count_per_stage"].values())
    coverage_rate = actual_contributions / total_possible

    print(f"Total Possible Evidence (300 Q × 12 stages): {total_possible}")
    print(f"Actual Evidence Items: {evidence_count}")
    print(f"Coverage Rate: {coverage_rate:.1%}")
    print(
        f"\nQuestions with >= 3 evidence: {validation_result['questions_meeting_threshold']} "
        f"({validation_result['questions_meeting_threshold'] / len(all_questions):.1%})"
    )
    print(
        f"Questions with < 3 evidence: {len(validation_result['questions_below_threshold'])} "
        f"({len(validation_result['questions_below_threshold']) / len(all_questions):.1%})"
    )

    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)

    return registry, validation_result


if __name__ == "__main__":
    demo_evidence_validation()
