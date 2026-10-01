"""Offline tests for the pieces that do not need a GPU or network.

Run: .venv\\Scripts\\python.exe -m pytest tests -q
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from app.llm import (  # noqa: E402
    LLMClassifier,
    LLMConfig,
    build_prompt,
    load_env_file,
)
from app.metrics import (  # noqa: E402
    accuracy,
    bootstrap_ci,
    compare,
    evaluate_system,
    exact_set_match,
    macro_f1,
    majority_baseline,
    mcnemar_exact,
    normalise,
)

LABELS = [
    "card_lost",
    "card_pin_change",
    "balance_inquiry",
    "transfer_pending",
    "refund_request",
]


def classifier() -> LLMClassifier:
    return LLMClassifier(LLMConfig.from_env())


# --- LLM reply parsing ---------------------------------------------------

@pytest.mark.parametrize(
    "reply,expected_label,expected_valid",
    [
        ("card_lost", "card_lost", True),
        ("   card_lost   ", "card_lost", True),
        ('"card_lost"', "card_lost", True),
        ("'card_lost'", "card_lost", True),
        ("`card_lost`", "card_lost", True),
        ("card_lost.", "card_lost", True),
        ("CARD_LOST", "card_lost", True),
        ("0", "card_lost", True),  # positional
        ("3", "transfer_pending", True),
        ("The answer is card_pin_change.", "card_pin_change", True),
        ("card_lost\nand then some chatter", "card_lost", True),
        ("balance_inquiry", "balance_inquiry", True),
    ],
)
def test_extract_label_accepts(reply: str, expected_label: str, expected_valid: bool) -> None:
    got, valid = classifier()._extract_label(reply, LABELS)
    assert (got, valid) == (expected_label, expected_valid)


@pytest.mark.parametrize(
    "reply,expected_label",
    [
        # The live endpoint answered "Refund_not_showing_up" where banking77's
        # schema says "refund_not_showing_up". That label is not in this small
        # fixture, so the case is reproduced against a schema that has it.
        ("card-lost", "card_lost"),
        ("card lost", "card_lost"),
        ("refund_request.", "refund_request"),
        ("`refund_request`", "refund_request"),
        ('"refund_request"', "refund_request"),
    ],
)
def test_extract_label_tolerates_cosmetic_variants(reply: str, expected_label: str) -> None:
    """Formatting differences are the same answer, not a wrong one.

    A live endpoint answered 'Refund_not_showing_up' where the schema said
    'refund_not_showing_up'. Rejecting that would understate the LLM arm on
    punctuation alone.
    """
    got, valid = classifier()._extract_label(reply, LABELS)
    assert valid is True
    assert got == expected_label


def test_extract_label_normalises_the_observed_capitalisation() -> None:
    """Regression: the real endpoint's exact failure mode.

    Asked for refund_not_showing_up it answered "Refund_not_showing_up" - a
    capitalised variant of a label that does exist in banking77. This must be
    accepted, while an invented label stays rejected.
    """
    schema = ["refund_not_showing_up", "card_lost", "other"]
    got, valid = classifier()._extract_label("Refund_not_showing_up", schema)
    assert (got, valid) == ("refund_not_showing_up", True)

    # Same shape, but not in the schema: still a failure.
    got2, valid2 = classifier()._extract_label("Refund_returned_twice", schema)
    assert (got2, valid2) == (None, False)


def test_extract_label_accepts_a_leading_phrase() -> None:
    """A short lead-in before the label is tolerated; prose is not.

    Models often prefix the answer with a few words. Accepting that is fair.
    The boundary is drawn at whitespace: the label has to be a whole token,
    otherwise 'the customer is unhappy' would be credited to whatever label
    shares its first characters.
    """
    got, valid = classifier()._extract_label("The answer is card_pin_change.", LABELS)
    assert (got, valid) == ("card_pin_change", True)
    # ...but a bare sentence is not an answer, however it is worded.
    got2, valid2 = classifier()._extract_label("this looks like a refund issue", LABELS)
    assert (got2, valid2) == (None, False)


@pytest.mark.parametrize(
    "reply",
    [
        "banana",
        "9",
        "",
        "   ",
        "unknown_intent",
        "the customer is unhappy",
        "definitely_refund_request_here_but_not_a_label",
    ],
)
def test_extract_label_still_rejects_invented_labels(reply: str) -> None:
    """Normalisation must not turn a wrong answer into a right one.

    An invented label, or prose that merely mentions a label somewhere, is a
    genuine failure and stays invalid.
    """
    got, valid = classifier()._extract_label(reply, LABELS)
    assert got is None
    assert valid is False


def test_extract_label_prefix_prefers_longest() -> None:
    """'card' is a prefix of 'card_lost'; the longest match must win."""
    labels = ["card", "card_lost"]
    got, _ = classifier()._extract_label("card_lost", labels)
    assert got == "card_lost"


def test_extract_label_requires_a_label_on_its_own_line() -> None:
    """Only the first line of a reply is considered.

    A model that explains itself across several lines is answering with prose,
    and crediting it to whichever label happens to appear in that prose would
    inflate the arm. Here the label is on line two, so it does not count.
    """
    got, valid = classifier()._extract_label(
        "The customer is unhappy\ntransfer_pending", LABELS
    )
    assert got is None
    assert valid is False


def test_extract_label_ignores_a_label_prefix_as_the_whole_reply() -> None:
    """A lead-in that merely *starts* with label characters is not an answer.

    "the" is a prefix of "theft_report" in many schemas. Matching on prefix
    alone would credit plain prose like "the customer is unhappy" to whatever
    label happens to share its first characters, which is how this rule was
    originally too permissive.
    """
    schema = ["theft_report", "transfer_pending", "other"]
    got, valid = classifier()._extract_label("the customer is unhappy", schema)
    assert (got, valid) == (None, False)
    # The real label still works.
    assert classifier()._extract_label("theft_report", schema) == ("theft_report", True)


def test_prompt_contains_labels_and_text() -> None:
    prompt = build_prompt("my card is missing", LABELS)
    assert "card_lost" in prompt
    assert "my card is missing" in prompt


def test_unset_config_reports_missing_env() -> None:
    cfg = LLMConfig(base_url=None, api_key=None, model=None, input_per_mtok=0.0, output_per_mtok=0.0)
    status = cfg.status()
    assert status["configured"] is False
    assert set(status["missing_env"]) == {"LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL"}
    assert "reason" in status


def test_status_never_leaks_the_key() -> None:
    cfg = LLMConfig(
        base_url="https://example.invalid/v1",
        api_key="sk-do-not-leak-this-value",
        model="some-model",
        input_per_mtok=0.0,
        output_per_mtok=0.0,
    )
    rendered = repr(cfg.status())
    assert "sk-do-not-leak-this-value" not in rendered
    assert cfg.status()["configured"] is True


# --- .env loading --------------------------------------------------------

def test_load_env_file_parses_quotes_comments_and_blanks(tmp_path) -> None:
    env = tmp_path / ".env"
    env.write_text(
        "# a comment\n"
        "\n"
        "LLM_BASE_URL=https://example.invalid/v1\n"
        'LLM_API_KEY="quoted-key"\n'
        "LLM_MODEL=plain-model\n"
        "not a pair\n",
        encoding="utf-8",
    )
    # Isolate from any real credentials in the developer's environment.
    for var in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL"):
        os.environ.pop(var, None)

    assert load_env_file(env) is True
    assert os.environ["LLM_BASE_URL"] == "https://example.invalid/v1"
    assert os.environ["LLM_API_KEY"] == "quoted-key"
    assert os.environ["LLM_MODEL"] == "plain-model"
    for var in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL"):
        os.environ.pop(var, None)


def test_load_env_file_does_not_override_existing_environment(tmp_path) -> None:
    """A shell export must win, so an operator can override a bad .env locally."""
    env = tmp_path / ".env"
    env.write_text("LLM_MODEL=from-file\n", encoding="utf-8")
    os.environ["LLM_MODEL"] = "from-shell"
    try:
        assert load_env_file(env) is True
        assert os.environ["LLM_MODEL"] == "from-shell"
    finally:
        os.environ.pop("LLM_MODEL", None)


def test_load_env_file_missing_file_is_not_an_error(tmp_path) -> None:
    assert load_env_file(tmp_path / "absent.env") is False


# --- metric semantics ----------------------------------------------------

def test_normalise_accepts_every_payload_shape() -> None:
    assert normalise("a") == frozenset({"a"})
    assert normalise(["a", "b"]) == frozenset({"a", "b"})
    assert normalise({"label": "a", "confidence": 0.9}) == frozenset({"a"})
    assert normalise([{"label": "a"}, {"label": "b"}]) == frozenset({"a", "b"})
    assert normalise(None) == frozenset()


def test_exact_set_match_is_not_partial_credit() -> None:
    assert exact_set_match(["a"], [["a"]]) == [True]
    # predicting a superset is wrong, not partially right
    assert exact_set_match([["a", "b"]], [["a"]]) == [False]
    assert exact_set_match([["a"]], [["a", "b"]]) == [False]


def test_exact_set_match_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError):
        exact_set_match(["a"], ["a", "b"])


def test_bootstrap_ci_degenerate_on_constant_input() -> None:
    assert bootstrap_ci([True] * 500) == (1.0, 1.0)
    assert bootstrap_ci([False] * 500) == (0.0, 0.0)


def test_mcnemar_exact_matches_binomial() -> None:
    # all 10 discordant one way: p = 2 * (1/2)^10
    assert mcnemar_exact([True] * 10, [False] * 10)["p_value"] == pytest.approx(2 / 1024)
    # perfect tie is not significant
    assert mcnemar_exact([True, False], [False, True])["p_value"] == 1.0
    # no discordant pairs
    assert mcnemar_exact([True] * 5, [True] * 5)["p_value"] == 1.0


def test_macro_f1_penalises_partial_overlap() -> None:
    assert macro_f1(["a"], ["a"]) == pytest.approx(1.0)
    assert macro_f1(["a"], ["b"]) == pytest.approx(0.0)
    assert macro_f1([["a", "b"]], [["a"]]) < 1.0


def test_compare_flags_a_real_difference() -> None:
    refs = ["a"] * 50
    good = evaluate_system("good", refs, refs)
    poor = evaluate_system("poor", ["b"] * 50, refs)
    result = compare(good, poor)
    assert result["diff"] == pytest.approx(1.0)
    assert result["significant"] is True
    assert result["mcnemar"]["b"] == 50


def test_compare_refuses_unpaired_item_counts() -> None:
    a = evaluate_system("a", ["x"] * 10, ["x"] * 10)
    b = evaluate_system("b", ["x"] * 9, ["x"] * 9)
    with pytest.raises(ValueError):
        compare(a, b)


def test_accuracy_is_plain_mean() -> None:
    assert accuracy([True, False, True, True]) == pytest.approx(0.75)


# --- majority-class floor ------------------------------------------------

def test_majority_baseline_is_the_floor() -> None:
    refs = ["a"] * 7 + ["b"] * 2 + ["c"]
    floor = majority_baseline(refs)
    assert floor["label"] == "a"
    assert floor["accuracy"] == pytest.approx(0.7)
    assert floor["count"] == 7


def test_majority_baseline_catches_a_collapsed_classifier() -> None:
    """The case that motivates the floor row.

    A system that spreads its predictions can lose to the floor on accuracy
    while beating it on macro-F1. On the real hate_speech fixture GLiNER2.5-Decide
    scores 44.4% accuracy against the floor's 77.6%, yet macro-F1 0.336 against
    0.291. Reporting accuracy alone would call that a collapse; reporting both
    shows it is finding the rare classes and mis-ranking them.

    The prediction list is written item-by-item on purpose: macro-F1 depends on
    which prediction lands on which reference, so marginal counts alone do not
    determine it.
    """
    refs = ["a"] * 8 + ["b", "c"]
    floor = majority_baseline(refs)
    assert floor["accuracy"] == pytest.approx(0.8)

    collapsed = ["a"] * len(refs)
    assert accuracy(exact_set_match(collapsed, refs)) == pytest.approx(floor["accuracy"])

    # Recovers both rare classes, but mislabels four "a" items as "b".
    spread = ["a"] * 4 + ["b", "b", "b", "b", "b", "c"]
    assert len(spread) == len(refs)
    assert accuracy(exact_set_match(spread, refs)) < floor["accuracy"]
    assert macro_f1(spread, refs) > floor["macro_f1"]


def test_macro_f1_depends_on_pairing_not_marginals() -> None:
    """Same label counts, different alignment, different macro-F1.

    This is why the panel scores real predictions from the real fixture rather
    than synthesising a prediction vector from summary statistics.
    """
    refs = ["offensive"] * 194 + ["neither"] * 46 + ["hate_speech"] * 10
    blocked = ["hate_speech"] * 88 + ["neither"] * 42 + ["offensive"] * 120
    assert macro_f1(blocked, refs) < macro_f1(blocked, list(reversed(refs))[:250] + refs[250:])


def test_majority_baseline_handles_empty() -> None:
    assert majority_baseline([])["label"] is None