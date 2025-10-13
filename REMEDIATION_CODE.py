#!/usr/bin/env python3.10
"""
REMEDIATION CODE FOR MISSING EVIDENCE REGISTRATIONS
Apply these changes to miniminimoon_orchestrator.py
"""

# ============================================================================
# ISSUE 1: Missing Contradiction Evidence Registration
# ============================================================================
# Location: _build_evidence_registry method, after line ~1132
# Insert after the monetary registration call


def remediation_issue_1():
    """
    Add this code after the monetary evidence registration:

    register_evidence(
        PipelineStage.MONETARY, all_inputs.get("monetary", []), "money"
    )

    # ← INSERT THIS CODE HERE:
    """
    code = """
    register_evidence(
        PipelineStage.CONTRADICTION, all_inputs.get("contradictions", []), "contra"
    )
    """
    return code


# ============================================================================
# ISSUE 2: Missing DAG Evidence Registration
# ============================================================================
# Location: _build_evidence_registry method, after line ~1167
# Insert after the teoria_cambio industrial metrics registration


def remediation_issue_2():
    """
    Add this code after the teoria industrial metrics registration:

    industrial_validation = teoria_result.get("industrial_validation")
    if isinstance(industrial_validation, dict):
        industrial_metrics = industrial_validation.get("metrics")
        if isinstance(industrial_metrics, list) and industrial_metrics:
            register_evidence(
                PipelineStage.TEORIA, industrial_metrics, "toc_metric"
            )

    # ← INSERT THIS CODE HERE:
    """
    code = """
    # Register DAG validation evidence
    dag_diagnostics_entry = all_inputs.get("dag_diagnostics")
    if isinstance(dag_diagnostics_entry, dict):
        try:
            dag_str = json.dumps(dag_diagnostics_entry, sort_keys=True, default=str)
            dag_evidence_id = f"dag_{hashlib.sha1(dag_str.encode()).hexdigest()[:10]}"
            
            dag_entry = EvidenceEntry(
                evidence_id=dag_evidence_id,
                stage=PipelineStage.DAG.value,
                content=dag_diagnostics_entry,
                source_segment_ids=[],
                confidence=0.9,
                metadata={
                    "p_value": dag_diagnostics_entry.get("p_value"),
                    "acyclic": dag_diagnostics_entry.get("acyclic"),
                }
            )
            
            self.evidence_registry.register(dag_entry)
            
        except (TypeError, AttributeError) as e:
            self.logger.warning(
                "Could not register DAG evidence: %s", e
            )
    """
    return code


# ============================================================================
# COMPLETE PATCHED _build_evidence_registry METHOD
# ============================================================================


def complete_patched_method():
    """
    Complete patched version of _build_evidence_registry with both fixes.
    Replace the entire method with this code.
    """
    return '''
    def _build_evidence_registry(self, all_inputs: Dict[str, Any]):
        """Build evidence registry from detector outputs (Stage 12)."""
        self.logger.info("Building evidence registry...")

        def register_evidence(stage: PipelineStage, items: List[Any], id_prefix: str):
            if not isinstance(items, list):
                return

            for item in items:
                try:
                    item_str = json.dumps(item, sort_keys=True, default=str)
                    evidence_id = f"{id_prefix}_{hashlib.sha1(item_str.encode()).hexdigest()[:10]}"

                    entry = EvidenceEntry(
                        evidence_id=evidence_id,
                        stage=stage.value,
                        content=item,
                        source_segment_ids=[],
                        confidence=item.get("confidence", 0.8)
                        if isinstance(item, dict)
                        else 0.8,
                    )

                    self.evidence_registry.register(entry)
                except (TypeError, AttributeError) as e:
                    self.logger.warning(
                        "Could not process item in stage %s: %s", stage.value, e
                    )

        # Register detector outputs
        register_evidence(
            PipelineStage.RESPONSIBILITY, all_inputs.get("responsibilities", []), "resp"
        )
        
        # FIX #1: Add contradiction evidence registration
        register_evidence(
            PipelineStage.CONTRADICTION, all_inputs.get("contradictions", []), "contra"
        )
        
        register_evidence(
            PipelineStage.MONETARY, all_inputs.get("monetary", []), "money"
        )

        feasibility_report = all_inputs.get("feasibility")
        if isinstance(feasibility_report, dict):
            register_evidence(
                PipelineStage.FEASIBILITY,
                feasibility_report.get("indicators", []),
                "feas",
            )

        causal_report = all_inputs.get("causal_patterns")
        if isinstance(causal_report, dict):
            register_evidence(
                PipelineStage.CAUSAL, causal_report.get("patterns", []), "causal"
            )

        teoria_result = all_inputs.get("toc_graph")
        if isinstance(teoria_result, dict):
            framework_entry = teoria_result.get("toc_graph")
            if isinstance(framework_entry, dict):
                register_evidence(PipelineStage.TEORIA, [framework_entry], "toc")

            industrial_validation = teoria_result.get("industrial_validation")
            if isinstance(industrial_validation, dict):
                industrial_metrics = industrial_validation.get("metrics")
                if isinstance(industrial_metrics, list) and industrial_metrics:
                    register_evidence(
                        PipelineStage.TEORIA, industrial_metrics, "toc_metric"
                    )

        # FIX #2: Add DAG evidence registration
        dag_diagnostics_entry = all_inputs.get("dag_diagnostics")
        if isinstance(dag_diagnostics_entry, dict):
            try:
                dag_str = json.dumps(dag_diagnostics_entry, sort_keys=True, default=str)
                dag_evidence_id = f"dag_{hashlib.sha1(dag_str.encode()).hexdigest()[:10]}"
                
                dag_entry = EvidenceEntry(
                    evidence_id=dag_evidence_id,
                    stage=PipelineStage.DAG.value,
                    content=dag_diagnostics_entry,
                    source_segment_ids=[],
                    confidence=0.9,
                    metadata={
                        "p_value": dag_diagnostics_entry.get("p_value"),
                        "acyclic": dag_diagnostics_entry.get("acyclic"),
                    }
                )
                
                self.evidence_registry.register(dag_entry)
                
            except (TypeError, AttributeError) as e:
                self.logger.warning(
                    "Could not register DAG evidence: %s", e
                )

        self.logger.info(
            "Evidence registry built with %s entries",
            len(self.evidence_registry._evidence),
        )

        return {"status": "built", "entries": len(self.evidence_registry._evidence)}
    '''


if __name__ == "__main__":
    print("=" * 80)
    print("REMEDIATION CODE FOR MINIMINIMOON_ORCHESTRATOR.PY")
    print("=" * 80)
    print()

    print("ISSUE #1: Missing Contradiction Evidence Registration")
    print("-" * 80)
    print(remediation_issue_1())
    print()

    print("ISSUE #2: Missing DAG Evidence Registration")
    print("-" * 80)
    print(remediation_issue_2())
    print()

    print("=" * 80)
    print("See complete_patched_method() for full replacement code")
    print("=" * 80)
