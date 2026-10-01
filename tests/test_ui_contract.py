"""Static checks on the single-file UI.

The UI has no build step and no framework, so there is no compiler to catch a
mistyped element id, a duplicated one, or a renderer that writes to an element
that does not exist. Every one of those fails silently at runtime: the panel
renders an empty line, or a note silently stops appearing, and nothing fails.

These are not browser tests. They parse index.html and assert the wiring holds.
A real DOM test (Playwright) would cover more, but it needs a browser download
and CI must stay fast and dependency-free; the checks below catch the class of
mistake that actually shipped.

Run: .venv\\Scripts\\python.exe -m pytest tests -q
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "app" / "static" / "index.html"


@pytest.fixture(scope="module")
def html() -> str:
    assert INDEX.exists(), f"UI shell missing at {INDEX}"
    return INDEX.read_text(encoding="utf-8")


def declared_ids(html: str) -> list[str]:
    """Ids declared in the markup. Only these can collide with each other."""
    return re.findall(r'\bid="([^"]+)"', html)


def targeted_ids(html: str) -> set[str]:
    """Ids the script looks up. Every one must exist in the markup."""
    return set(re.findall(r"\$\('#([A-Za-z0-9_-]+)'\)", html))


def strip_comments(html: str) -> str:
    """Code only. A comment that mentions a banned pattern is documentation,
    not a violation — and the file is heavily commented on purpose."""
    html = re.sub(r"/\*.*?\*/", "", html, flags=re.S)
    return re.sub(r"^\s*//.*$", "", html, flags=re.M)


def test_every_field_error_slot_exists(html: str) -> None:
    """Validation writes to a slot passed as a string, so id checks miss it.

    The two id guards above only see literal `$('#name')` lookups. A renderer
    that takes the target as an argument — `fieldError('pLimit', 'pLimitErr')`
    — is invisible to them, so a mistyped or missing slot throws a TypeError at
    click time and the error never appears. That shipped during this work: a
    batch field was validated against `#pBatchErr` with no such element, and
    every panel run died before it sent a request.

    So check the pairs themselves. Each entry is the wiring a validation call
    depends on: an input that exists, an error slot that exists, and the
    describedby link between them.
    """
    pairs = re.findall(r"intField\(\s*'([A-Za-z0-9_-]+)'\s*,\s*'([A-Za-z0-9_-]+)'", html)
    assert pairs, "the numeric fields are no longer validated; did the helper go?"
    declared = set(declared_ids(html))
    for input_id, err_id in pairs:
        assert input_id in declared, f"intField validates #{input_id}, which does not exist"
        assert err_id in declared, (
            f"intField writes its error to #{err_id}, which does not exist; "
            "the validation throws instead of reporting"
        )
        # The slot must be linked to its field, or a screen reader never
        # announces the message that fieldError just populated.
        assert re.search(
            rf'id="{input_id}"[^>]*aria-describedby="[^"]*\b{re.escape(err_id)}\b', html
        ), f"#{input_id} does not reference #{err_id} via aria-describedby"


def test_a_forbidden_interactive_dialog(html: str) -> None:
    """No window.confirm/alert/prompt.

    A blocking modal freezes the page, cannot be styled with the rest of the
    console, and its appearance and keyboard behaviour are the browser's
    rather than the app's. The slow-LLM warning used one to ask about an item
    count, which put the decision in a dialog instead of next to the field it
    concerned, and the estimate it showed was wrong by 60x.
    """
    code = strip_comments(html)
    for token in ("confirm(", "alert(", "prompt("):
        assert token not in code, (
            f"{token} blocks the page and puts a decision in a native dialog; "
            "render it in the page next to the field it concerns"
        )


def test_interactive_elements_have_a_focus_ring(html: str) -> None:
    """Keyboard focus must be visible on every interactive control.

    There is no compiler and no browser in CI to notice a `outline:none` or a
    misspelled selector, and the failure is invisible to a mouse user, so it is
    asserted statically.

    Both halves matter and they fail differently. A missing rule leaves the UA
    default, which is a 1px dotted outline that is nearly invisible against
    --panel. A `none` removes the indicator outright. So check that a
    focus-visible rule actually exists as a rule — not merely that the token
    appears somewhere, since the token is also in the comment explaining it.
    """
    code = strip_comments(html)
    # A real rule, and it must cover the controls rather than one incidental
    # element. `:focus-visible` on a single container says nothing about the
    # buttons inside it, so the selector is required to name an interactive
    # type and the block must set an outline.
    m = re.search(
        r"([^{}]*?:focus-visible[^{}]*)\{([^}]*)\}", code, re.S)
    assert m, "no focus-visible rule; keyboard focus is invisible on the dark surfaces"
    assert "outline" in m.group(2), (
        f"the focus-visible rule for {m.group(1).strip()!r} sets no outline"
    )
    selector = m.group(1)
    assert re.search(r"\b(button|input|select|textarea|a)\b|\[tabindex\]", selector), (
        f"the focus-visible rule targets {selector.strip()!r}, which is not a set "
        "of interactive controls; the focus ring must reach the buttons and fields"
    )
    # Nothing may remove an outline without providing its own indicator.
    for m in re.finditer(r"outline\s*:\s*(none|0)\b", code):
        raise AssertionError(
            f"an outline was set to {m.group(1)!r} with no replacement focus "
            "indicator; keyboard users lose the only cue they have"
        )


def relative_luminance(hex_color: str) -> float:
    """WCAG 2.x relative luminance. sRGB channels are linearised first."""
    def channel(value: int) -> float:
        c = value / 255
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = (int(hex_color.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))
    return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b)


def contrast_ratio(fg: str, bg: str) -> float:
    """Contrast between two opaque colours, per WCAG 2.x."""
    hi, lo = sorted((relative_luminance(fg), relative_luminance(bg)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def function_body(html: str, name: str) -> str:
    """The source of one function, found by brace counting.

    A non-greedy regex to the next `\\n}` stops at the first nested block, which
    is not the end of the function — the plot has loops and an arrow callback
    inside it, so a lazy match saw only the opening lines and reported a
    correctly-written function as a hardcoded-hex one.
    """
    m = re.search(rf"\bfunction\s+{re.escape(name)}\s*\(", html)
    if not m:
        return ""
    start = html.index("{", m.end())
    depth = 0
    for i in range(start, len(html)):
        if html[i] == "{":
            depth += 1
        elif html[i] == "}":
            depth -= 1
            if depth == 0:
                return html[start : i + 1]
    return ""


def css_tokens(html: str) -> dict[str, str]:
    """Custom properties declared in :root, as name -> hex value.

    The contrast checks below resolve colours through this rather than
    hardcoding them. A hardcoded pair passes forever: the test keeps asserting
    4.5:1 about colours the file no longer uses, so editing a token to something
    unreadable is invisible. Resolving from the file means the assertion
    follows the palette.
    """
    root = re.search(r":root\s*\{(.*?)\}", html, re.S)
    assert root, "no :root block; the palette tokens are gone"
    tokens: dict[str, str] = {}
    for name, value in re.findall(r"(--[A-Za-z0-9_-]+)\s*:\s*(#[0-9a-fA-F]{3,8})", root.group(1)):
        tokens[name] = value
    return tokens


def test_text_colours_meet_wcag_aa(html: str) -> None:
    """Every text-on-surface pair in the palette clears 4.5:1.

    AA for normal text is 4.5:1. There is no browser in CI, so nothing catches
    a token change that quietly pushes a label under the floor, and the console
    is mostly small grey text on near-black — exactly the combination where a
    regression hides.

    This caught a real one: white on the #2f81f7 accent was 3.75:1, so every
    Run button label failed AA. The accent remains the right blue for a *mark*
    against the panel, which is why the fix was a second token for fills under
    white text rather than darkening the accent and losing the chart contrast.
    """
    t = css_tokens(html)
    for token in ("--text", "--dim", "--panel", "--bg", "--accent", "--accent-solid"):
        assert token in t, f"{token} is not declared in :root"

    pairs = [
        ("body text on card", t["--text"], t["--panel"]),
        ("caveat note on card", t["--dim"], t["--panel"]),
        ("dim text on page background", t["--dim"], t["--bg"]),
        ("run button label on accent fill", "#ffffff", t["--accent-solid"]),
        ("accent mark on card", t["--accent"], t["--panel"]),
    ]
    failures = [
        f"{name}: {contrast_ratio(fg, bg):.2f}:1"
        for name, fg, bg in pairs
        if contrast_ratio(fg, bg) < 4.5
    ]
    assert not failures, "text below the WCAG AA 4.5:1 floor: " + "; ".join(failures)


def test_non_text_indicators_meet_wcag_aa(html: str) -> None:
    """Focus rings and invalid borders clear the 3:1 UI-component floor.

    These are not text, so the bar is 3:1 rather than 4.5:1, but they are still
    the only signal a keyboard user gets. A focus ring below 3:1 is a ring that
    is not there.
    """
    t = css_tokens(html)
    pairs = [
        ("focus ring on card", t["--accent"], t["--panel"]),
        ("focus ring on page background", t["--accent"], t["--bg"]),
        ("invalid field border", t["--bad"], t["--bg"]),
        ("invalid field border on card", t["--bad"], t["--panel"]),
    ]
    failures = [
        f"{name}: {contrast_ratio(fg, bg):.2f}:1"
        for name, fg, bg in pairs
        if contrast_ratio(fg, bg) < 3.0
    ]
    assert not failures, "indicator below the WCAG AA 3:1 floor: " + "; ".join(failures)


def test_chart_gridlines_are_visible(html: str) -> None:
    """A gridline has to be visible to be a gridline.

    The plot originally drew its grid in --line, which is 1.14:1 against
    --panel. That is not a subtle gridline, it is an invisible one: the axis
    labels had nothing to sit against, so the reader lost the only scale the
    chart offers.

    --grid is deliberately kept faint — it must not compete with the data marks
    — but it is resolved from the file and held above the point where it stops
    existing. --line stays available for dividers, which are meant to recede.
    """
    t = css_tokens(html)
    assert "--grid" in t, "no --grid token; the chart has no dedicated gridline colour"
    ratio = contrast_ratio(t["--grid"], t["--panel"])
    assert ratio >= 1.4, (
        f"the chart gridline is {ratio:.2f}:1 against the panel, which reads as "
        "nothing; the axis labels lose their only scale"
    )
    # The plot must *read* the token, not carry its own copy of the colour. A
    # hardcoded hex here passed every check above while the token changed
    # nothing: the palette had two sources of truth and the chart followed the
    # one nobody was testing.
    body = function_body(html, "forest")
    assert body, "the forest plot function is gone"
    # The plot may read the token directly or through a local helper, so accept
    # either shape: what matters is that --grid is resolved at draw time.
    reads_token = re.search(r"getPropertyValue\(\s*['\"]--grid['\"]\s*\)", body) or re.search(
        r"\btoken\(\s*['\"]--grid['\"]\s*\)", body
    )
    assert reads_token, (
        "the forest plot does not resolve --grid from the stylesheet, so the "
        "gridline colour is duplicated in JS and will drift from the token"
    )
    # And it must actually use the resolved value when drawing.
    assert re.search(r"stroke\s*:\s*[A-Za-z_$][\w.$]*\.grid", body), (
        "the gridlines are not drawn with the resolved --grid value"
    )
    # No raw hex in the plot: every colour should come from a token.
    hexes = re.findall(r"#[0-9a-fA-F]{6}\b", body)
    assert not hexes, (
        f"the forest plot hardcodes colours instead of reading tokens: {hexes}. "
        "The palette must have one source of truth."
    )


def test_marks_and_text_use_separate_accent_tokens(html: str) -> None:
    """A fill under white text and a mark on the panel are different jobs.

    One token for both is how white-on-accent text ended up at 3.75:1. Keeping
    them separate lets the chart keep its contrast while the button label stays
    readable.
    """
    t = css_tokens(html)
    assert t["--accent-solid"] != t["--accent"], (
        "--accent-solid and --accent are the same colour, so one of the two "
        "jobs is going to fail its contrast floor"
    )
    assert "var(--accent-solid)" in strip_comments(html), (
        "nothing uses --accent-solid; the button label is still on --accent"
    )


def test_motion_respects_the_reduced_motion_preference(html: str) -> None:
    """Transitions are opt-out.

    A transition on a control is small, but the preference is one media query
    away and costs one line; ignoring it makes the console less usable for
    exactly the readers it is already dense for.
    """
    if "transition" not in html:
        return  # nothing moves, nothing to respect
    assert "prefers-reduced-motion" in html, (
        "transitions were added without a prefers-reduced-motion opt-out"
    )


def test_the_console_survives_a_partial_payload(html: str) -> None:
    """A missing optional key must not destroy already-rendered results.

    `comparisons` and `unavailable` are read after the table is built. A
    shape regression there used to throw into the catch block, which replaced
    a correct table with a red box — losing the run's real numbers over a
    field the table had already rendered.
    """
    code = strip_comments(html)
    for key in ("comparisons", "unavailable"):
        assert re.search(rf"d\.{key}\s*\|\|\s*\[\]", code), (
            f"d.{key} is read without a default; a missing key throws and wipes the results"
        )


def test_the_llm_time_estimate_is_not_sixty_times_wrong(html: str) -> None:
    """The slow-arm estimate must convert seconds to minutes.

    It read `n * 3` and labelled the result minutes, when the measurement
    behind it is ~3 seconds per item. A 300-item run was announced as 900
    minutes; it takes about 15. An estimate off by 60x makes a real ceiling
    look arbitrary and teaches the reader to ignore the warning.
    """
    code = strip_comments(html)
    assert re.search(r"LLM_SECONDS_PER_ITEM\s*=", code), (
        "the estimate should be a named measurement, not a literal in the string"
    )
    # The helper must exist *and* be the thing the warning actually calls, and it
    # must perform the unit conversion. A defined-but-unused helper satisfies a
    # weaker check while the string still shows the wrong number, which is the
    # bug itself. Match a declaration of either form.
    assert re.search(
        r"(?:function\s+llmEstimate\s*\([^)]*\)|llmEstimate\s*=\s*(?:\([^)]*\)|[A-Za-z_$][\w$]*)\s*=>)"
        r"[\s\S]{0,200}?/\s*60",
        code,
    ), "llmEstimate does not convert seconds to minutes"
    assert re.search(r"\$\{\s*llmEstimate\(", code), (
        "the warning does not use llmEstimate, so it is showing an unconverted "
        "seconds-per-item figure labelled as minutes"
    )
    assert not re.search(r"Math\.round\(\s*\w+\s*\*\s*\d+\s*\)", code), (
        "a bare Math.round(n * k) is being presented as a duration; it needs the "
        "per-item measurement and a unit conversion"
    )


def test_no_payload_value_reaches_innerhtml(html: str) -> None:
    """Response data is rendered as text nodes, never interpolated into markup.

    The fixture note and the comparison system names are free text from data
    files and response payloads. A template literal into `innerHTML` is a
    markup sink for a string nobody reviewed, and it is invisible in review
    because the surrounding code looks like ordinary formatting.
    """
    # One template literal per match, scanned from the assignment to its own
    # closing backtick. Matching across literals would let a later statement's
    # interpolation be blamed on an earlier, static one.
    for m in re.finditer(r"\.innerHTML\s*=\s*`([^`]*)`", strip_comments(html)):
        snippet = m.group(1)
        # A literal with no interpolation is fine: the static table headers are
        # markup this file owns, not data from a response.
        assert "${" not in snippet, (
            f"payload data is interpolated into innerHTML: {snippet[:70]!r}. "
            "Build it with textContent or el() instead."
        )


def test_element_ids_are_unique(html: str) -> None:
    """A duplicate id is silent: querySelector returns the first match only.

    This shipped once already. A fixture-detail line was given the same id as
    the fixture <select>, so the text was written into the select and the line
    rendered empty, with no error anywhere.
    """
    counts: dict[str, int] = {}
    for name in declared_ids(html):
        counts[name] = counts.get(name, 0) + 1
    dupes = {k: v for k, v in counts.items() if v > 1}
    assert not dupes, f"duplicate element ids: {dupes}"


def test_every_written_element_exists_in_the_markup(html: str) -> None:
    """A renderer targeting a missing id writes nowhere and shows nothing."""
    missing = sorted(targeted_ids(html) - set(declared_ids(html)))
    assert not missing, f"script writes to ids that do not exist: {missing}"


def test_the_cost_column_does_not_invent_a_price(html: str) -> None:
    """An unpriced token cost must not render as $0.00.

    A null price means two different things: a local encoder genuinely costs
    nothing to run, while the LLM arm is unpriced because no token price is
    configured. Rendering both as $0.00 reads as free, which understates the
    one arm whose cost is the reason it loses.
    """
    code = strip_comments(html)
    assert "$0.00" not in code, "a fabricated zero price is still in the code"
    assert "cost_status" in code, "the UI must branch on the backend's cost state"


def test_the_panel_does_not_invent_a_verdict(html: str) -> None:
    """No client-side quality grade on a row the backend did not judge.

    The column used to label systems strong/usable/weak from hardcoded accuracy
    thresholds. On an imbalanced fixture that inverts the finding: the model was
    marked 'weak' and the majority-class floor, which is not a system, was
    marked 'usable' because it scored higher.
    """
    code = strip_comments(html)
    for token in ("'strong'", "'usable'", "'weak'"):
        assert token not in code, f"client-invented verdict {token} is still present"
    assert not re.search(r"accuracy\s*>\s*0\.", code), (
        "a client-side threshold on accuracy is grading rows the backend did not"
    )


def test_the_independence_note_cannot_be_overwritten(html: str) -> None:
    """The panel's honesty caveat must survive a run.

    It is the sentence that keeps the numbers meaningful: that the model's own
    training distribution is not mixed in. An earlier version replaced this
    element's text with the fixture description, so it survived exactly until
    the first panel run.
    """
    assert "training distribution" in html, "the independence caveat is gone"
    permanent = re.search(r'id="pNote"[^>]*>(.*?)</div>', html, re.S)
    assert permanent, "the permanent note element is gone"
    assert "training distribution" in permanent.group(1)
    # No renderer may assign to it.
    assert not re.search(r"\$\('#pNote'\)\s*\.\s*(textContent|innerHTML)\s*=", html), (
        "a renderer overwrites #pNote; the independence caveat must be permanent"
    )


def test_latency_percentiles_are_both_shown(html: str) -> None:
    """p50 alone hides the tail, and the tail is the LLM arm's story.

    Measured on the live endpoint: p50 9,363 ms against p95 25,321 ms. A
    backend field that the renderer drops is worse than one that was never sent,
    because the table then looks complete.
    """
    for field in ("p50_ms", "p95_ms"):
        assert field in html, f"the panel does not render {field}"


def test_run_level_notes_have_a_home(html: str) -> None:
    """The panel's top-level notes carry facts the table cannot hold.

    Most importantly what the LLM arm actually cost per item, which is the
    number an operator uses to decide how many items to request.
    """
    assert "pRunNotes" in html
    assert re.search(r"d\.notes", html), "the top-level notes payload is never read"


# The four fields that silently stopped rendering while the backend kept sending
# them. tests/AGENTS.md names all four as the reason the shape guards exist, so
# each needs BOTH halves pinned: the producer still emits it, and the renderer
# still reads it. Checking one side passes while the panel renders a blank cell,
# which is the original defect.
#
# Two things make this guard bite, both learned the hard way. The producer is
# pinned per field rather than searched package-wide, because `label_coverage` is
# emitted by two modules and a rename in one of them is invisible to an
# "any module emits it" search. And the reader is pinned to the exact expression,
# not a bare `.notes`: the panel reads `notes` twice, as the run-level `d.notes`
# and per-system `s.notes`, so a generic property search stays satisfied when the
# run-level read is the one that broke.
PANEL_PAYLOAD_FIELDS = (
    # field, the exact payload read in the renderer, modules that must emit it
    ("p95_ms", r"s\.p95_ms", ("app/metrics.py",)),
    ("notes", r"d\.notes", ("app/metrics.py", "app/evaluate.py")),
    ("label_coverage", r"d\.fixture\.label_coverage", ("app/evaluate.py", "app/main.py")),
    ("supervised_trained_on", r"d\.supervised_trained_on", ("app/evaluate.py",)),
)


@pytest.mark.parametrize(
    "field,reader,producers",
    PANEL_PAYLOAD_FIELDS,
    ids=[f[0] for f in PANEL_PAYLOAD_FIELDS],
)
def test_a_panel_field_is_sent_by_the_backend_and_read_by_the_renderer(
    field: str, reader: str, producers: tuple[str, ...], html: str
) -> None:
    assert re.search(reader, strip_comments(html)), (
        f"the UI no longer reads {field} off the payload ({reader})"
    )

    pattern = re.compile(rf"""["']{field}["']""")
    for rel in producers:
        source = (ROOT / rel).read_text(encoding="utf-8")
        assert pattern.search(strip_comments(source)), (
            f"{rel} stopped emitting {field}, so the renderer reads a field that no "
            "longer exists. Either side alone passing means the panel row is blank."
        )


def test_error_handling_covers_the_array_detail_shape(html: str) -> None:
    """FastAPI returns `detail` as a string for HTTPException, as an array of
    objects for a 422. Assigning the array to textContent yields
    "[object Object]", so every validation error looked like a crash."""
    assert "errText" in html
    assert "Array.isArray" in html


def test_the_three_api_surfaces_are_all_wired(html: str) -> None:
    for path in ("/api/route", "/api/panel", "/api/schema-swap", "/api/health"):
        assert path in html, f"{path} is not called by the UI"


def test_tabs_declare_the_aria_tabs_pattern(html: str) -> None:
    """Once aria-selected is used, the tabs pattern is only honest with the
    matching roles: a tablist of tabs pointing at labelled tabpanels."""
    assert 'role="tablist"' in html
    assert html.count('role="tab"') == 3
    assert html.count('role="tabpanel"') == 3
    assert 'aria-labelledby="tabbtn-route"' in html


def test_no_cdn_or_external_asset(html: str) -> None:
    """The UI must work offline once the API is running.

    This is the contract that keeps the console a single file with no build
    step, so an accidental CDN <script> or <link> is a regression, not a
    convenience. XML namespace URIs are identifiers, not fetches, and the
    inline SVG chart needs one.
    """
    for pattern in (r"<script[^>]+src=", r"<link[^>]+href=", r"@import"):
        found = re.findall(pattern, html)
        assert not found, f"external asset reference breaks the offline rule: {found}"
    # Any absolute URL that is not a namespace declaration and not a same-origin
    # API path would be a network dependency at runtime.
    for url in re.findall(r"https?://[^\s\"'()<>]+", html):
        assert url.startswith("http://www.w3.org/"), f"external URL: {url}"


def test_the_ui_is_still_one_file() -> None:
    """A bundler or a split would be a recorded decision, not a side effect."""
    static_dir = INDEX.parent
    sources = [p for p in static_dir.iterdir() if p.suffix in {".html", ".js", ".css", ".ts", ".tsx"}]
    assert [p.name for p in sources] == ["index.html"], (
        f"app/static gained source files: {[p.name for p in sources]}"
    )


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-q"]))
