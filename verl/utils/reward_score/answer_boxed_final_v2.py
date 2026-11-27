"""Terminal math submissions. Extraction never receives a reference answer."""
import re

VERSION = "answer_boxed_final_v2"
REVISION = "r2"
_MARKER = re.compile(
    r"(?im)(?:^|(?<=[.!?])[ \t]+(?=(?:actually|correction|(?:the )?final answer)\b))[ \t]*(?:#{1,6}[ \t]+)?(?:[*_]{2})?"
    r"(?:(?:therefore|thus|hence|so|correction|actually)[,:]?[ \t]+)?"
    r"(?:the[ \t]+)?(?:final[ \t]+)?answer(?:[*_]{2})?[ \t]*(?::|[ \t]+is\b[ \t]*:?)[ \t]*(?:[*_]{2}(?=[ \t]))?"
)
# A box must start a submission or follow an affirmative conclusion in the same sentence.
_TRANSITION = r"(?:therefore|thus|hence|consequently|so|in conclusion)[,:]?[ \t]+"
_RESULT = (r"(?:the|our)[ \t]+(?:(?:required|desired|final|maximum|minimum)[ \t]+)?"
           r"(?:value|result|answer|solution|maximum|minimum|total|number)[ \t]+is[ \t]+")
_BOX = re.compile(
    r"(?im)(?:^|(?<=[.!?])[ \t]+)[ \t]*"
    rf"(?P<intro>(?:\*\*|__)?(?:{_TRANSITION}(?:(?:we[ \t]+(?:obtain|get|find)|this[ \t]+(?:gives|yields))[ \t]+|{_RESULT})?|{_RESULT}))?"
    r"(?P<value>(?:(?:\*\*|__|\x60|\${1,2}|\\[\[(]|\\begin\{(?:equation\*?|displaymath)\})\s*)*\\boxed\{)"
)
_NONFINAL = re.compile(r"\b(?:candidates?|possib(?:le|ility|ilities)|alternativ(?:es?|ely)|examples?|"
                       r"hypothetical|intermediate|for instance|tentative|suppos(?:e|ing)|assum(?:e|ing)|if|either|perhaps|maybe|"
                       r"case\s+\w+)\b|\bor\s*[:;,]?$", re.I)
_CLOSING = re.compile(r"(?:[.!]?\s+(?:I hope (?:this|that) helps|hope this helps|thanks|thank you|that completes the proof|"
                      r"this completes the proof|Q\.?E\.?D)|\s*(?:∎|\\qed))\s*[.!]?\s*$", re.I)
_WRAPS = [(r"\begin{equation}", r"\end{equation}"), (r"\begin{equation*}", r"\end{equation*}"),
          (r"\begin{displaymath}", r"\end{displaymath}"), (r"\[", r"\]"), (r"\(", r"\)"),
          ("$$", "$$"), ("$", "$"), ("**", "**"), ("__", "__"), ("\x60", "\x60")]


def _payload(value):
    value = _CLOSING.sub("", value.strip()).removesuffix(".").strip()
    while value:
        for left, right in _WRAPS:
            if value.startswith(left):
                if len(value) <= len(left) + len(right) or not value.endswith(right):
                    return None
                value = value[len(left):-len(right)].strip()
                break
        else:
            break
    if value.startswith(r"\boxed{"):
        if not value.endswith("}"):
            return None
        value = value[7:-1].strip()
    if not value or re.search(r"\\boxed|\$|\x60|\*\*|\\(?:begin|end)\b|\\[\[\]()]|\b(?:or|alternatively|maybe|if)\b", value, re.I):
        return None
    # A line break within a displayed formula is whitespace, not a change of answer.
    if "\n" in value:
        if re.search(r"\n\s*\n|\n[ \t]*[A-Za-z]{2,}\s+[A-Za-z]{2,}", value):
            return None
        value = " ".join(value.splitlines())
    if re.fullmatch(r"[+-]?[\d, ]+", value) and not re.fullmatch(r"[+-]?(?:\d+|\d{1,3}(?:,\d{3})+)", value):
        return None
    stack = []
    for c in value:
        if c in "{([":
            stack.append((c, 0))
        elif c == "," and stack:
            stack[-1] = (stack[-1][0], stack[-1][1] + 1)
        elif c in "})]":
            if not stack:
                return None
            opening, commas = stack.pop()
            if opening + c not in ("{}", "()", "[]") and not (commas == 1 and opening + c in ("(]", "[)")):
                return None
    if stack or re.search(r"[+\-*/^_=,:\\]$|\\(?:frac|sqrt)$|\\frac\{[^{}]*\}$", value):
        return None
    # Legacy normalization keeps the RHS of an assignment. Only accept a single named LHS.
    if "=" in value and not re.fullmatch(r"[a-zA-Z]\s*=\s*[^=]+", value):
        return None
    if re.search(r"\n\s*\n|[.!?]\s+[A-Za-z]|\b(?:instead|versus)\b", value, re.I):
        return None
    return value


def _normalize(value, normalize_final_answer):
    value = normalize_final_answer(value)
    if re.fullmatch(r"[+-]?[0-9]+", value):
        digits = value.lstrip("+-").lstrip("0") or "0"
        return ("-" if value.startswith("-") and digits != "0" else "") + digits
    return value


def extract(response, normalize_final_answer):
    depth = start = 0
    for tag in re.finditer(r"</?think>", response, re.I):
        if tag.group().lower() == "<think>":
            depth += 1
        else:
            depth = max(0, depth - 1)
            start = tag.end()
    if depth:
        return None, "unclosed_think", False
    text = response[start:].strip()
    marks = [(m.start(), m.end(), "answer") for m in _MARKER.finditer(text)]
    for m in _BOX.finditer(text):
        marks.append((m.start(), m.start("value"), "inline_boxed" if m["intro"] else "boxed"))
    marks.sort()
    # "Final answer:\n\boxed{...}" is one labelled submission.
    submissions = []
    for mark in marks:
        if submissions and submissions[-1][2] == "answer" and mark[2] != "answer":
            pos, end, _ = submissions[-1]
            if end >= mark[0] or not text[end:mark[0]].strip():
                continue
        submissions.append(mark)

    seen = set()
    context_open = ambiguous = False
    last_end, rank = 0, -1
    pred, form = None, "missing"
    for i, (pos, end, form) in enumerate(submissions):
        stop = submissions[i + 1][0] if i + 1 < len(submissions) else len(text)
        part = text[end:stop].strip()
        label = text[pos:end].strip()
        correction = bool(re.search(r"\b(?:correction|actually)\b", label, re.I))
        decisive = correction or bool(re.search(r"\bfinal answer\b", label, re.I))
        # Inspect the governing clause, including line continuations. A list remains open
        # across bare displayed boxes, even when a previous box has a period after it.
        lead = text[last_end:pos].strip()
        clause = re.split(r"(?<=[.!?])\s+|\n[ \t]*\n", lead)[-1]
        resolved = not re.search(r"\b(?:if|not|unless)\b", clause, re.I) and re.search(r"\b(?:we (?:reject|eliminate|discard|rule out)|(?:is|was|were) (?:ruled out|eliminated|rejected))\b", clause, re.I)
        if _NONFINAL.search(clause) and not resolved:
            context_open = True
        elif decisive or resolved:
            context_open = False
        last_end = end
        for wrap in ("**", "__"):
            if label.lstrip("# ").startswith(wrap) and part.rstrip(".").endswith(wrap) and not part.startswith(wrap):
                part = part.rstrip(".")[:-2].rstrip()
        value = _payload(part)
        candidate = _normalize(value, normalize_final_answer) if value else None
        if form == "answer" and r"\boxed{" in part:
            form = "answer_boxed"
        if candidate:
            seen.add(candidate)
        if context_open:
            pred, form = None, "nonfinal_context"
            continue
        if candidate is None:
            pred, form = None, "incomplete"
            continue
        priority = 2 if decisive else int(form != "boxed")
        if correction or priority > rank:
            ambiguous = False
        elif pred is not None and pred != candidate:
            ambiguous = True
        rank = max(rank, priority)
        pred = candidate
        if ambiguous:
            form = "unresolved_conflict"
    # The third field reports disagreement for auditing, including resolved corrections.
    return (None if ambiguous else pred), form, len(seen) > 1


def score(response, ground_truth, normalize_final_answer):
    pred, _, _ = extract(response, normalize_final_answer)
    correct = pred is not None and pred == _normalize(str(ground_truth), normalize_final_answer)
    return {"score": 1.0 if correct else -1.0, "acc": correct, "pred": pred if pred is not None else "[INVALID]"}
