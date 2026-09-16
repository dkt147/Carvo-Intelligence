from app.api.auth import bearer_matches, extract_bearer_token


def test_extract_bearer_token():
    assert extract_bearer_token(None) == ""
    assert extract_bearer_token("Basic abc") == ""
    assert extract_bearer_token("Bearer abc") == "abc"
    assert extract_bearer_token("bearer  abc  ") == "abc"


def test_bearer_matches_is_constant_time_and_accepts_only_the_expected_token():
    assert bearer_matches("Bearer secret", "secret") is True
    assert bearer_matches("bearer secret", "secret") is True
    assert bearer_matches("Bearer wrong", "secret") is False
    assert bearer_matches("Bearer secret", "longer-secret") is False
    assert bearer_matches(None, "secret") is False
    assert bearer_matches("secret", "secret") is False
