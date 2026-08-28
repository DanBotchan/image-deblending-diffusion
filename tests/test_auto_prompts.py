from scripts.generate_auto_prompts import parse_two_lines


def test_parse_numbered_auto_prompts():
    assert parse_two_lines("1. A red car on a road.\n2. White flowers in a field.") == (
        "A red car on a road.", "White flowers in a field."
    )


def test_parse_single_line_matches_frozen_fallback():
    assert parse_two_lines("One visible scene") == ("One visible scene", "One visible scene")


def test_parse_empty_response():
    assert parse_two_lines("  \n") == (None, None)
