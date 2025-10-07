import json
from output_quality_assessor import validate_output_quality


class TestOutputQualityAssessor:
    
    def test_question_count_validation_success(self, tmp_path):
        """Test that exactly 300 questions passes validation"""
        answers_file = tmp_path / "answers_report.json"
        answers_data = {
            "question_answers": [{"question_id": f"Q{i}"} for i in range(300)]
        }
        with open(answers_file, 'w') as f:
            json.dump(answers_data, f)
        
        results = validate_output_quality(
            answers_path=str(answers_file),
            rubric_path="/nonexistent",
            evidence_registry_path="/nonexistent",
            flow_runtime_path="/nonexistent",
            flow_doc_path="/nonexistent",
            validation_gates_path="/nonexistent",
            output_path=tmp_path / "output.json"
        )
        
        assert results["criteria"]["question_count"]["pass"] is True
        assert results["criteria"]["question_count"]["actual"] == 300
    
    def test_question_count_validation_failure(self, tmp_path):
        """Test that non-300 question count fails validation"""
        answers_file = tmp_path / "answers_report.json"
        answers_data = {
            "question_answers": [{"question_id": f"Q{i}"} for i in range(250)]
        }
        with open(answers_file, 'w') as f:
            json.dump(answers_data, f)
        
        results = validate_output_quality(
            answers_path=str(answers_file),
            rubric_path="/nonexistent",
            evidence_registry_path="/nonexistent",
            flow_runtime_path="/nonexistent",
            flow_doc_path="/nonexistent",
            validation_gates_path="/nonexistent",
            output_path=tmp_path / "output.json"
        )
        
        assert results["criteria"]["question_count"]["pass"] is False
        assert results["criteria"]["question_count"]["actual"] == 250
    
    def test_pipeline_stage_coverage_success(self, tmp_path):
        """Test that all 15 stages contributing passes validation"""
        evidence_file = tmp_path / "evidence_registry.json"
        evidence_data = {
            "evidences": [
                {"evidence_id": f"ev{i}", "pipeline_stage": f"stage_{i}"}
                for i in range(1, 16)
            ]
        }
        with open(evidence_file, 'w') as f:
            json.dump(evidence_data, f)
        
        results = validate_output_quality(
            answers_path="/nonexistent",
            rubric_path="/nonexistent",
            evidence_registry_path=str(evidence_file),
            flow_runtime_path="/nonexistent",
            flow_doc_path="/nonexistent",
            validation_gates_path="/nonexistent",
            output_path=tmp_path / "output.json"
        )
        
        assert results["criteria"]["pipeline_stage_coverage"]["pass"] is True
        assert results["criteria"]["pipeline_stage_coverage"]["actual_stages"] == 15
    
    def test_pipeline_stage_coverage_failure(self, tmp_path):
        """Test that fewer than 15 stages fails validation"""
        evidence_file = tmp_path / "evidence_registry.json"
        evidence_data = {
            "evidences": [
                {"evidence_id": f"ev{i}", "pipeline_stage": f"stage_{i}"}
                for i in range(1, 10)
            ]
        }
        with open(evidence_file, 'w') as f:
            json.dump(evidence_data, f)
        
        results = validate_output_quality(
            answers_path="/nonexistent",
            rubric_path="/nonexistent",
            evidence_registry_path=str(evidence_file),
            flow_runtime_path="/nonexistent",
            flow_doc_path="/nonexistent",
            validation_gates_path="/nonexistent",
            output_path=tmp_path / "output.json"
        )
        
        assert results["criteria"]["pipeline_stage_coverage"]["pass"] is False
        assert results["criteria"]["pipeline_stage_coverage"]["actual_stages"] == 9
    
    def test_flow_order_match_success(self, tmp_path):
        """Test that matching flow order passes validation"""
        flow_runtime = tmp_path / "flow_runtime.json"
        flow_doc = tmp_path / "flow_doc.json"
        
        order = ["stage1", "stage2", "stage3"]
        with open(flow_runtime, 'w') as f:
            json.dump({"stage_order": order}, f)
        with open(flow_doc, 'w') as f:
            json.dump({"canonical_order": order}, f)
        
        results = validate_output_quality(
            answers_path="/nonexistent",
            rubric_path="/nonexistent",
            evidence_registry_path="/nonexistent",
            flow_runtime_path=str(flow_runtime),
            flow_doc_path=str(flow_doc),
            validation_gates_path="/nonexistent",
            output_path=tmp_path / "output.json"
        )
        
        assert results["criteria"]["flow_order_match"]["pass"] is True
        assert len(results["criteria"]["flow_order_match"]["deviations"]) == 0
    
    def test_flow_order_match_failure(self, tmp_path):
        """Test that mismatched flow order fails validation"""
        flow_runtime = tmp_path / "flow_runtime.json"
        flow_doc = tmp_path / "flow_doc.json"
        
        with open(flow_runtime, 'w') as f:
            json.dump({"stage_order": ["stage1", "stage3", "stage2"]}, f)
        with open(flow_doc, 'w') as f:
            json.dump({"canonical_order": ["stage1", "stage2", "stage3"]}, f)
        
        results = validate_output_quality(
            answers_path="/nonexistent",
            rubric_path="/nonexistent",
            evidence_registry_path="/nonexistent",
            flow_runtime_path=str(flow_runtime),
            flow_doc_path=str(flow_doc),
            validation_gates_path="/nonexistent",
            output_path=tmp_path / "output.json"
        )
        
        assert results["criteria"]["flow_order_match"]["pass"] is False
        assert len(results["criteria"]["flow_order_match"]["deviations"]) > 0
    
    def test_validation_gates_all_passing(self, tmp_path):
        """Test that all 6 gates passing succeeds"""
        gates_file = tmp_path / "validation_gates.json"
        gates_data = {
            "immutability_verified": {"status": "pass"},
            "flow_order_match": {"status": "pass"},
            "evidence_deterministic_hash_consistency": {"status": "pass"},
            "coverage_300_300": {"status": "pass"},
            "rubric_alignment": {"status": "pass"},
            "triple_run_determinism": {"status": "pass"}
        }
        with open(gates_file, 'w') as f:
            json.dump(gates_data, f)
        
        results = validate_output_quality(
            answers_path="/nonexistent",
            rubric_path="/nonexistent",
            evidence_registry_path="/nonexistent",
            flow_runtime_path="/nonexistent",
            flow_doc_path="/nonexistent",
            validation_gates_path=str(gates_file),
            output_path=tmp_path / "output.json"
        )
        
        assert results["criteria"]["validation_gates"]["pass"] is True
        assert results["criteria"]["validation_gates"]["passing_gates"] == 6
    
    def test_validation_gates_some_failing(self, tmp_path):
        """Test that some gates failing causes overall failure"""
        gates_file = tmp_path / "validation_gates.json"
        gates_data = {
            "immutability_verified": {"status": "pass"},
            "flow_order_match": {"status": "fail"},
            "evidence_deterministic_hash_consistency": {"status": "pass"},
            "coverage_300_300": {"status": "pass"},
            "rubric_alignment": {"status": "fail"},
            "triple_run_determinism": {"status": "pass"}
        }
        with open(gates_file, 'w') as f:
            json.dump(gates_data, f)
        
        results = validate_output_quality(
            answers_path="/nonexistent",
            rubric_path="/nonexistent",
            evidence_registry_path="/nonexistent",
            flow_runtime_path="/nonexistent",
            flow_doc_path="/nonexistent",
            validation_gates_path=str(gates_file),
            output_path=tmp_path / "output.json"
        )
        
        assert results["criteria"]["validation_gates"]["pass"] is False
        assert results["criteria"]["validation_gates"]["passing_gates"] == 4
    
    def test_confidence_metrics(self, tmp_path):
        """Test confidence score metrics computation"""
        answers_file = tmp_path / "answers_report.json"
        answers_data = {
            "question_answers": [
                {"question_id": "Q1", "confidence": 0.9},
                {"question_id": "Q2", "confidence": 0.8},
                {"question_id": "Q3", "confidence": 0.7},
                {"question_id": "Q4", "confidence": 0.6}
            ]
        }
        with open(answers_file, 'w') as f:
            json.dump(answers_data, f)
        
        results = validate_output_quality(
            answers_path=str(answers_file),
            rubric_path="/nonexistent",
            evidence_registry_path="/nonexistent",
            flow_runtime_path="/nonexistent",
            flow_doc_path="/nonexistent",
            validation_gates_path="/nonexistent",
            output_path=tmp_path / "output.json"
        )
        
        assert "confidence_scores" in results["metrics"]
        assert results["metrics"]["confidence_scores"]["mean"] == 0.75
        assert results["metrics"]["confidence_scores"]["min"] == 0.6
        assert results["metrics"]["confidence_scores"]["max"] == 0.9
    
    def test_evidence_distribution_metrics(self, tmp_path):
        """Test evidence distribution metrics computation"""
        answers_file = tmp_path / "answers_report.json"
        answers_data = {
            "question_answers": [
                {"question_id": "Q1", "evidence_count": 3},
                {"question_id": "Q2", "evidence_count": 2},
                {"question_id": "Q3", "evidence_count": 2},
                {"question_id": "Q4", "evidence_count": 0},
                {"question_id": "Q5", "evidence_count": 1}
            ]
        }
        with open(answers_file, 'w') as f:
            json.dump(answers_data, f)
        
        results = validate_output_quality(
            answers_path=str(answers_file),
            rubric_path="/nonexistent",
            evidence_registry_path="/nonexistent",
            flow_runtime_path="/nonexistent",
            flow_doc_path="/nonexistent",
            validation_gates_path="/nonexistent",
            output_path=tmp_path / "output.json"
        )
        
        assert "evidence_distribution" in results["metrics"]
        assert results["metrics"]["evidence_distribution"]["questions_with_zero_evidence"] == 1
        assert results["metrics"]["evidence_distribution"]["max"] == 3
    
    def test_rationale_completeness_metrics(self, tmp_path):
        """Test rationale completeness metrics computation"""
        answers_file = tmp_path / "answers_report.json"
        answers_data = {
            "question_answers": [
                {"question_id": "Q1", "rationale": "Complete rationale here"},
                {"question_id": "Q2", "rationale": "Another rationale"},
                {"question_id": "Q3", "rationale": ""},
                {"question_id": "Q4"}
            ]
        }
        with open(answers_file, 'w') as f:
            json.dump(answers_data, f)
        
        results = validate_output_quality(
            answers_path=str(answers_file),
            rubric_path="/nonexistent",
            evidence_registry_path="/nonexistent",
            flow_runtime_path="/nonexistent",
            flow_doc_path="/nonexistent",
            validation_gates_path="/nonexistent",
            output_path=tmp_path / "output.json"
        )
        
        assert "rationale_completeness" in results["metrics"]
        assert results["metrics"]["rationale_completeness"]["questions_with_rationale"] == 2
        assert results["metrics"]["rationale_completeness"]["total_questions"] == 4
        assert results["metrics"]["rationale_completeness"]["completeness_percentage"] == 50.0
    
    def test_overall_pass_determination(self, tmp_path):
        """Test overall pass is true only when all criteria pass"""
        # Create minimal passing files
        answers_file = tmp_path / "answers_report.json"
        with open(answers_file, 'w') as f:
            json.dump({
                "question_answers": [{"question_id": f"Q{i}"} for i in range(300)]
            }, f)
        
        evidence_file = tmp_path / "evidence_registry.json"
        with open(evidence_file, 'w') as f:
            json.dump({
                "evidences": [
                    {"evidence_id": f"ev{i}", "pipeline_stage": f"stage_{i}"}
                    for i in range(1, 16)
                ]
            }, f)
        
        flow_runtime = tmp_path / "flow_runtime.json"
        flow_doc = tmp_path / "flow_doc.json"
        order = ["s1", "s2"]
        with open(flow_runtime, 'w') as f:
            json.dump({"stage_order": order}, f)
        with open(flow_doc, 'w') as f:
            json.dump({"canonical_order": order}, f)
        
        gates_file = tmp_path / "validation_gates.json"
        with open(gates_file, 'w') as f:
            json.dump({
                "immutability_verified": {"status": "pass"},
                "flow_order_match": {"status": "pass"},
                "evidence_deterministic_hash_consistency": {"status": "pass"},
                "coverage_300_300": {"status": "pass"},
                "rubric_alignment": {"status": "pass"},
                "triple_run_determinism": {"status": "pass"}
            }, f)
        
        # Rubric check will fail since files don't exist
        results = validate_output_quality(
            answers_path=str(answers_file),
            rubric_path="/nonexistent",
            evidence_registry_path=str(evidence_file),
            flow_runtime_path=str(flow_runtime),
            flow_doc_path=str(flow_doc),
            validation_gates_path=str(gates_file),
            output_path=tmp_path / "output.json"
        )
        
        # Overall should fail because rubric_alignment fails
        assert results["overall_pass"] is False
    
    def test_output_file_creation(self, tmp_path):
        """Test that results are written to output file"""
        output_file = tmp_path / "reports" / "output_quality_assessment.json"
        
        results = validate_output_quality(
            answers_path="/nonexistent",
            rubric_path="/nonexistent",
            evidence_registry_path="/nonexistent",
            flow_runtime_path="/nonexistent",
            flow_doc_path="/nonexistent",
            validation_gates_path="/nonexistent",
            output_path=str(output_file)
        )
        
        assert output_file.exists()
        with open(output_file) as f:
            saved_results = json.load(f)
        assert "overall_pass" in saved_results
        assert "criteria" in saved_results
        assert "metrics" in saved_results
    
    def test_summary_statistics(self, tmp_path):
        """Test that summary statistics are computed correctly"""
        results = validate_output_quality(
            answers_path="/nonexistent",
            rubric_path="/nonexistent",
            evidence_registry_path="/nonexistent",
            flow_runtime_path="/nonexistent",
            flow_doc_path="/nonexistent",
            validation_gates_path="/nonexistent",
            output_path=tmp_path / "output.json"
        )
        
        assert "summary" in results
        assert "total_criteria" in results["summary"]
        assert "passing_criteria" in results["summary"]
        assert "failing_criteria" in results["summary"]
        assert isinstance(results["summary"]["failing_criteria"], list)
