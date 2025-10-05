"""
Comprehensive tests for plan name sanitization and JSON key standardization.
"""

import os
import tempfile

import pytest

from plan_sanitizer import (
    PlanSanitizer,
    create_plan_directory,
    sanitize_plan_name,
    standardize_json_keys,
)


class TestPlanSanitizer:
    """Test the PlanSanitizer class functionality."""

    @staticmethod
    def test_sanitize_basic_plan_names():
        """Test basic plan name sanitization."""
        # Normal names should remain unchanged
        assert (
            PlanSanitizer.sanitize_plan_name("Plan Nacional 2024")
            == "Plan Nacional 2024"
        )
        assert (
            PlanSanitizer.sanitize_plan_name("Proyecto Educativo")
            == "Proyecto Educativo"
        )

    @staticmethod
    def test_sanitize_problematic_characters():
        """Test sanitization of problematic characters."""
        test_cases = [
            # Slashes
            ("Plan/Meta/Objetivo", "Plan - Meta - Objetivo"),
            ("Plan\\Desarrollo\\2024", "Plan - Desarrollo - 2024"),
            # Colons
            ("Plan: Desarrollo Social", "Plan - Desarrollo Social"),
            ("Proyecto: Meta 2024", "Proyecto - Meta 2024"),
            # Asterisks and wildcards
            ("Plan*Meta*2024", "PlanMeta2024"),
            ("Proyecto?Meta", "ProyectoMeta"),
            # Quotes
            ('Plan "Importante"', "Plan 'Importante'"),
            ("Plan 'Meta'", "Plan 'Meta'"),
            # Angle brackets
            ("Plan <urgente>", "Plan (urgente)"),
            ("Meta <2024>", "Meta (2024)"),
            # Pipes
            ("Plan|Meta|2024", "Plan - Meta - 2024"),
            # Combined problematic characters
            ("Plan: Meta/Objetivo*2024?", "Plan - Meta - Objetivo2024"),
            ('<Plan>:"Meta"/Objetivo*', "(Plan) - 'Meta' - Objetivo"),
        ]

        for input_name, expected in test_cases:
            result = PlanSanitizer.sanitize_plan_name(input_name)
            assert result == expected, (
                f"Input: {input_name}, Expected: {expected}, Got: {result}"
            )

    @staticmethod
    def test_sanitize_control_characters():
        """Test removal of control characters."""
        # Tab, newline, carriage return
        assert (
            PlanSanitizer.sanitize_plan_name(
                "Plan\tMeta\n2024\r") == "Plan Meta 2024"
        )

        # Null bytes and other control chars
        test_name = "Plan\x00Meta\x01"
        result = PlanSanitizer.sanitize_plan_name(test_name)
        assert "\x00" not in result
        assert "\x01" not in result
        assert result == "PlanMeta"

    @staticmethod
    def test_sanitize_reserved_names():
        """Test handling of Windows reserved names."""
        reserved_cases = [
            ("CON", "plan_CON"),
            ("PRN", "plan_PRN"),
            ("AUX", "plan_AUX"),
            ("NUL", "plan_NUL"),
            ("COM1", "plan_COM1"),
            ("LPT1", "plan_LPT1"),
            ("con", "plan_con"),  # Case insensitive
        ]

        for input_name, expected in reserved_cases:
            result = PlanSanitizer.sanitize_plan_name(input_name)
            assert result == expected, (
                f"Input: {input_name}, Expected: {expected}, Got: {result}"
            )

    @staticmethod
    def test_sanitize_length_limits():
        """Test length truncation."""
        # Very long name
        long_name = "Plan de Desarrollo Nacional Integral Sostenible " * 10
        result = PlanSanitizer.sanitize_plan_name(long_name, max_length=50)
        assert len(result) <= 50
        assert result.endswith(
            "Nacional Integral Sostenible"
        )  # Should cut at word boundary

        # Extremely long single word
        long_word = "PlanDeDesarrolloNacionalIntegralSostenible" * 5
        result = PlanSanitizer.sanitize_plan_name(long_word, max_length=50)
        assert len(result) <= 50

    @staticmethod
    def test_sanitize_edge_cases():
        """Test edge cases."""
        # Empty or None names
        assert PlanSanitizer.sanitize_plan_name("") == "plan_sin_nombre"
        assert PlanSanitizer.sanitize_plan_name("   ") == "plan_sin_nombre"
        assert PlanSanitizer.sanitize_plan_name(None) == "plan_sin_nombre"

        # Only problematic characters
        assert PlanSanitizer.sanitize_plan_name(
            "///**???") == "plan_sin_nombre"
        assert PlanSanitizer.sanitize_plan_name("<<<>>>") == "()"

        # Leading/trailing problematic chars
        assert PlanSanitizer.sanitize_plan_name(
            "...Plan Meta...") == "Plan Meta"
        assert PlanSanitizer.sanitize_plan_name("---Plan---") == "Plan"

    @staticmethod
    def test_create_safe_directory():
        """Test safe directory creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Basic directory creation
            plan_name = "Plan: Meta/2024"
            result_path = PlanSanitizer.create_safe_directory(
                plan_name, temp_dir, create=True
            )
            expected_name = "Plan - Meta - 2024"
            assert os.path.basename(result_path) == expected_name
            assert os.path.exists(result_path)
            assert os.path.isdir(result_path)

            # Test duplicate handling
            result_path2 = PlanSanitizer.create_safe_directory(
                plan_name, temp_dir, create=True
            )
            assert os.path.basename(result_path2) == f"{expected_name}_1"
            assert os.path.exists(result_path2)

            # Third duplicate
            result_path3 = PlanSanitizer.create_safe_directory(
                plan_name, temp_dir, create=True
            )
            assert os.path.basename(result_path3) == f"{expected_name}_2"
            assert os.path.exists(result_path3)

    @staticmethod
    def test_standardize_json_key_basic():
        """Test basic JSON key standardization."""
        test_cases = [
            # Spanish tildes
            ("línea_base", "linea_base"),
            ("número_página", "numero_pagina"),
            ("evaluación", "evaluacion"),
            ("implementación", "implementacion"),
            ("descripción", "descripcion"),
            # CamelCase to snake_case
            ("numeroElementos", "numero_elementos"),
            ("lineaBase", "linea_base"),
            ("tipoDocumento", "tipo_documento"),
            # Mixed cases
            ("líneaBase", "linea_base"),
            ("númeroElementos", "numero_elementos"),
            # Special characters
            ("línea-base", "linea_base"),
            ("número página", "numero_pagina"),
            ("eval@ción", "evalcion"),
        ]

        for input_key, expected in test_cases:
            result = PlanSanitizer.standardize_json_key(input_key)
            assert result == expected, (
                f"Input: {input_key}, Expected: {expected}, Got: {result}"
            )

    @staticmethod
    def test_standardize_json_key_edge_cases():
        """Test edge cases for JSON key standardization."""
        # Empty/None keys
        assert PlanSanitizer.standardize_json_key("") == ""
        assert PlanSanitizer.standardize_json_key(None) == None

        # Multiple underscores/dashes/spaces
        assert PlanSanitizer.standardize_json_key(
            "línea___base") == "linea_base"
        assert PlanSanitizer.standardize_json_key(
            "número---página") == "numero_pagina"
        assert PlanSanitizer.standardize_json_key(
            "línea   base") == "linea_base"

        # Leading/trailing underscores
        assert PlanSanitizer.standardize_json_key(
            "_línea_base_") == "linea_base"

        # Numbers and mixed content
        assert PlanSanitizer.standardize_json_key(
            "página123Meta") == "pagina123_meta"
        assert PlanSanitizer.standardize_json_key(
            "línea2024Base") == "linea2024_base"

    @staticmethod
    def test_standardize_json_object_simple():
        """Test JSON object standardization with simple structure."""
        input_obj = {
            "línea_base": "2023",
            "número_página": 15,
            "evaluación": "completa",
            "tipoDocumento": "plan",
        }

        result = PlanSanitizer.standardize_json_object(input_obj)

        expected_keys = {"linea_base", "numero_pagina",
                         "evaluacion", "tipo_documento"}
        assert set(result.keys()) & expected_keys == expected_keys

        # Values should be preserved
        assert result["linea_base"] == "2023"
        assert result["numero_pagina"] == 15
        assert result["evaluacion"] == "completa"
        assert result["tipo_documento"] == "plan"

    @staticmethod
    def test_standardize_json_object_with_display_keys():
        """Test JSON object standardization preserving display keys."""
        input_obj = {
            "línea_base": "2023",
            "número_página": 15,
        }

        result = PlanSanitizer.standardize_json_object(
            input_obj, preserve_display_keys=True
        )

        # Should have both standardized and display keys
        assert "linea_base" in result
        assert "linea_base_display" in result
        assert "numero_pagina" in result
        assert "numero_pagina_display" in result

        # Display keys should preserve original
        assert result["linea_base_display"] == "línea_base"
        assert result["numero_pagina_display"] == "número_página"

    @staticmethod
    def test_standardize_json_object_nested():
        """Test nested JSON object standardization."""
        input_obj = {
            "información_general": {
                "línea_base": "2023",
                "número_página": 15,
                "datos_técnicos": {
                    "evaluación": "completa",
                    "implementación": "parcial",
                },
            },
            "metas_principales": [
                {"descripción": "Meta 1", "situación": "activa"},
                {"descripción": "Meta 2", "situación": "pendiente"},
            ],
        }

        result = PlanSanitizer.standardize_json_object(
            input_obj, preserve_display_keys=False
        )

        # Check top level
        assert "informacion_general" in result
        assert "metas_principales" in result

        # Check nested object
        info_general = result["informacion_general"]
        assert "linea_base" in info_general
        assert "numero_pagina" in info_general
        assert "datos_tecnicos" in info_general

        # Check deeply nested
        datos_tecnicos = info_general["datos_tecnicos"]
        assert "evaluacion" in datos_tecnicos
        assert "implementacion" in datos_tecnicos

        # Check list items
        metas = result["metas_principales"]
        assert len(metas) == 2
        assert "descripcion" in metas[0]
        assert "situacion" in metas[0]

    @staticmethod
    def test_get_markdown_display_key():
        """Test getting display keys for Markdown."""
        # With display keys available
        json_obj = {
            "linea_base": "2023",
            "linea_base_display": "línea_base",
            "numero_pagina": 15,
            "numero_pagina_display": "número_página",
        }

        assert (
            PlanSanitizer.get_markdown_display_key("linea_base", json_obj)
            == "línea_base"
        )
        assert (
            PlanSanitizer.get_markdown_display_key("numero_pagina", json_obj)
            == "número_página"
        )

        # Fallback to common patterns
        assert PlanSanitizer.get_markdown_display_key(
            "evaluacion", {}) == "evaluación"
        assert (
            PlanSanitizer.get_markdown_display_key("implementacion", {})
            == "implementación"
        )

        # Unknown key fallback
        assert (
            PlanSanitizer.get_markdown_display_key("campo_desconocido", {})
            == "campo desconocido"
        )

    @staticmethod
    def test_convenience_functions():
        """Test convenience functions."""
        # sanitize_plan_name
        assert sanitize_plan_name("Plan: Meta/2024") == "Plan - Meta - 2024"

        # standardize_json_keys
        input_obj = {"línea_base": "2023"}
        result = standardize_json_keys(input_obj)
        assert "linea_base" in result

        # create_plan_directory
        with tempfile.TemporaryDirectory() as temp_dir:
            result_path = create_plan_directory("Plan: Meta/2024", temp_dir)
            assert os.path.exists(result_path)
            assert os.path.basename(result_path) == "Plan - Meta - 2024"


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""

    @staticmethod
    def test_complete_plan_processing():
        """Test complete plan processing workflow."""
        # Simulate a plan with problematic name and data structure
        plan_name = "Plan: Desarrollo/Social*2024 <Urgente>"
        plan_data = {
            "información_básica": {
                "línea_base": "Situación inicial 2023",
                "número_elementos": 25,
                "fecha_creación": "2024-01-15",
            },
            "metas_específicas": [
                {
                    "descripción": "Reducir pobreza al 15%",
                    "línea_temporal": "2024-2026",
                    "población_objetivo": "Rural",
                },
                {
                    "descripción": "Mejorar educación",
                    "evaluación_inicial": "Deficiente",
                    "implementación_esperada": "2025",
                },
            ],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            # 1. Sanitize plan name and create directory
            sanitized_name = PlanSanitizer.sanitize_plan_name(plan_name)
            plan_dir = PlanSanitizer.create_safe_directory(plan_name, temp_dir)

            assert sanitized_name == "Plan - Desarrollo - Social2024 (Urgente)"
            assert os.path.exists(plan_dir)
            assert os.path.basename(plan_dir) == sanitized_name

            # 2. Standardize JSON structure
            standardized_data = PlanSanitizer.standardize_json_object(
                plan_data, preserve_display_keys=True
            )

            # Verify structure is standardized
            assert "informacion_basica" in standardized_data
            assert "metas_especificas" in standardized_data

            # Verify nested standardization
            info_basica = standardized_data["informacion_basica"]
            assert "linea_base" in info_basica
            assert "numero_elementos" in info_basica
            assert "fecha_creacion" in info_basica

            # Verify display keys are preserved
            assert "linea_base_display" in info_basica
            assert info_basica["linea_base_display"] == "línea_base"

            # Verify list items are standardized
            metas = standardized_data["metas_especificas"]
            assert "descripcion" in metas[0]
            assert "linea_temporal" in metas[0]
            assert "poblacion_objetivo" in metas[0]

            assert "evaluacion_inicial" in metas[1]
            assert "implementacion_esperada" in metas[1]

    @staticmethod
    def test_extreme_plan_names():
        """Test with extremely problematic plan names."""
        extreme_cases = [
            # All forbidden characters
            '/<>:"|?*\\',
            # Control characters
            "\x00\x01\x02Plan\x03\x04\x05",
            # Reserved names with extension
            "CON.txt",
            "PRN.doc",
            # Mixed extreme case
            '<Plan>:"Meta"/Obj*2024?\x00\x01',
            # Unicode and special chars
            "Plan 🚀 Meta ⭐ 2024 💫",
            # Very long with problematic chars
            ("Plan: Desarrollo/Nacional*Integral?" + "A" * 300),
        ]

        for extreme_name in extreme_cases:
            result = PlanSanitizer.sanitize_plan_name(extreme_name)

            # Should not contain any forbidden characters
            for forbidden_char in PlanSanitizer.INVALID_CHARS:
                assert forbidden_char not in result, (
                    f"Found forbidden char in: {result}"
                )

            # Should not be empty
            assert len(result) > 0

            # Should be safe for directory creation
            with tempfile.TemporaryDirectory() as temp_dir:
                try:
                    safe_path = os.path.join(temp_dir, result)
                    os.makedirs(safe_path, exist_ok=True)
                    assert os.path.exists(safe_path)
                except (OSError, ValueError) as e:
                    pytest.fail(
                        f"Failed to create directory with name '{result}': {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
