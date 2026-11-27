"""Terminal math submissions. Extraction never receives a reference answer."""
import re

VERSION = "answer_boxed_final_v2"
_MARKER = re.compile(
    r"(?im)(?:^|(?<=[.!?])[ \t]+(?=(?:actually|correction|(?:the )?final answer)\b))[ \t]*(?:#{1,6}[ \t]+)?(?:[*_]{2})?"
    r"(?:(?:therefore|thus|hence|so|correction|actually)[,:]?[ \t]+)?"
    r"(?:the[ \t]+)?(?:final[ \t]+)?answer(?:[*_]{2})?[ \t]*(?::|[ \t]+is\b[ \t]*:?)[ \t]*(?:[*_]{2}(?=[ \t]))?"
)
_BOX = re.compile(r"(?im)^[ \t]*(?:(?:thus|hence|therefore|so)[,:]?[ \t]+)?"
                  r"(?P<value>(?:(?:\*\*|__|\x60|\${1,2}|\\[\[(]|\\begin\{(?:equation\*?|displaymath)\})\s*)*\\boxed\{)")
_WRAPS = [(r"\begin{equation}", r"\end{equation}"), (r"\begin{equation*}", r"\end{equation*}"),
          (r"\begin{displaymath}", r"\end{displaymath}"), (r"\[", r"\]"), (r"\(", r"\)"),
          ("$$", "$$"), ("$", "$"), ("**", "**"), ("__", "__"), ("\x60", "\x60")]


def _payload(value):
    value = value.strip().removesuffix(".").strip()
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
    if re.fullmatch(r"[+-]?[\d, ]+", value) and not re.fullmatch(r"[+-]?(?:\d+|\d{1,3}(?:,\d{3})+)", value):
        return None
    stack = []
    for c in value:
        if c in "{([":
            stack.append({"{": "}", "(": ")", "[": "]"}[c])
        elif c in "})]" and (not stack or stack.pop() != c):
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
    # Segment successive submissions before checking their syntax. Only the last can win.
    marks = [(m.start(), m.end(), "answer") for m in _MARKER.finditer(text)]
    for m in _BOX.finditer(text):
        marks.append((m.start(), m.start("value"), "boxed"))
    marks.sort()
    previous = set()
    pred, form = None, "missing"
    for i, (pos, end, form) in enumerate(marks):
        stop = marks[i + 1][0] if i + 1 < len(marks) else len(text)
        part = text[end:stop].strip()
        preceding_line = text[text.rfind("\n", 0, max(0, pos - 1)) + 1:pos].strip()
        if re.search(r"\b(?:if|suppose|assume|intermediate|example|candidate)\b[^\n]*[:;,]\s*$",
                                        preceding_line, re.I):
            part = ""
        if form == "answer":
            for wrap in ("**", "__"):
                if text[pos:end].lstrip().startswith(wrap) and part.endswith(wrap) and not part.startswith(wrap):
                    part = part[:-2].rstrip()
        value = _payload(part)
        pred = _normalize(value, normalize_final_answer) if value else None
        if pred:
            previous.add(pred)
        if form == "answer" and r"\boxed{" in part:
            form = "answer_boxed"
    return pred or None, form, len(previous) > 1


def score(response, ground_truth, normalize_final_answer):
    pred, _, _ = extract(response, normalize_final_answer)
    correct = pred is not None and pred == _normalize(str(ground_truth), normalize_final_answer)
    return {"score": 1.0 if correct else -1.0, "acc": correct, "pred": pred if pred is not None else "[INVALID]"}
