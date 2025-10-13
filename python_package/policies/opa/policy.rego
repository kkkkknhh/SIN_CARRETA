# policy.rego
# OPA Policy for EvidencePacket Validation
# ===========================================
#
# This policy enforces business rules for evidence packets:
# 1. All packets must be signed
# 2. Confidence must be >= 0.2
# 3. Source component must be from allowed list
# 4. Evidence type must be non-empty

package evidence.validation

# Default deny
default allow = false

# Allow if all rules pass
allow {
    is_signed
    has_valid_confidence
    has_valid_source
    has_valid_type
}

# Rule: Packet must have signature
is_signed {
    input.signature != null
    input.signature != ""
}

# Rule: Confidence must be >= 0.2
has_valid_confidence {
    input.confidence >= 0.2
}

# Rule: Source component must be from allowed list
has_valid_source {
    allowed_sources := {
        "sanitization",
        "plan_processor",
        "document_segmenter",
        "embedding_model",
        "responsibility_detector",
        "contradiction_detector",
        "monetary_detector",
        "feasibility_scorer",
        "causal_detector",
        "teoria_cambio",
        "dag_validator",
        "evidence_registry_builder",
        "decalogo_loader",
        "decalogo_evaluator",
        "questionnaire_evaluator",
        "answer_assembler",
    }
    allowed_sources[input.source_component]
}

# Rule: Evidence type must be non-empty
has_valid_type {
    input.evidence_type != null
    input.evidence_type != ""
}

# Violation messages for debugging
violations[msg] {
    not is_signed
    msg := "Evidence packet must be signed"
}

violations[msg] {
    not has_valid_confidence
    msg := sprintf("Confidence %.2f is below minimum 0.2", [input.confidence])
}

violations[msg] {
    not has_valid_source
    msg := sprintf("Source component '%s' is not in allowed list", [input.source_component])
}

violations[msg] {
    not has_valid_type
    msg := "Evidence type cannot be empty"
}

# Helper: Get all violations
get_violations = violations
