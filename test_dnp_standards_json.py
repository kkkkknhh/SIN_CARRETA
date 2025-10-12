"""Test to verify dnp-standards.latest.clean.json is valid JSON"""

import json


def test_dnp_standards_json_is_valid():
    """Test that dnp-standards.latest.clean.json is valid JSON"""
    # Use central path resolver
    from repo_paths import get_dnp_path

    json_path = get_dnp_path()

    with open(json_path, "r", encoding="utf-8") as f:
        content = f.read()

    # This should not raise an exception
    data = json.loads(content)

    # Verify it's a dict
    assert isinstance(data, dict), "Root element should be a dictionary"

    # Verify basic structure
    assert "version" in data, "Should have version field"
    assert "schema" in data, "Should have schema field"

    print(f"✅ JSON is valid with {len(content)} characters")
    return data


def test_milagros_line_has_escaped_quotes():
    """Test that the 'milagros' line has properly escaped quotes"""
    # Use central path resolver
    from repo_paths import get_dnp_path

    json_path = get_dnp_path()

    with open(json_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Line 1109 (index 1108)
    line_1109 = lines[1108]

    # Should contain escaped quotes, not unescaped ones
    assert '\\"milagros\\"' in line_1109, (
        "Line 1109 should have escaped quotes around milagros"
    )
    assert '"milagros"' not in line_1109 or '\\"milagros\\"' in line_1109, (
        "Line 1109 should not have unescaped quotes around milagros"
    )

    print(f"✅ Line 1109 has properly escaped quotes")


if __name__ == "__main__":
    print("Testing dnp-standards.latest.clean.json...")
    try:
        test_dnp_standards_json_is_valid()
        test_milagros_line_has_escaped_quotes()
        print("\n✅ All tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise
