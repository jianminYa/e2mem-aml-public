"""Search-output privacy filter: masks credentials and payment data in returned evidence.

Stored memories are never modified; only the text returned by Search is masked. Patterns are
deliberately narrow (validated on the 1,020-question development set and the AML smoke store:
1 and 0 matches respectively), so ordinary numbers, tables, code and status codes are kept.
"""
from __future__ import annotations

import re



def _luhn_ok(digits: str) -> bool:
    total, parity = 0, len(digits) % 2
    for i, ch in enumerate(digits):
        d = int(ch)
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


# Card-shaped strings only: 4-4-4-4 / 4-6-5 groups with one consistent separator, or 13-19
# contiguous digits. Free spacing (``16 -93 6 10``) matched numeric tables in CL-bench.
_CARD = re.compile(r"(?<![\d.-])(?:\d{4}([ -])\d{4}\1\d{4}\1\d{4}(?:\1\d{1,3})?|\d{4}([ -])\d{6}\2\d{5}|\d{13,19})(?![\d.-])")
_CARD_CONTEXT = re.compile(r"(?i)\b(card|visa|master ?card|amex|american express|discover|account|acct|cc)\b")
_PATTERNS = [
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S), "[REDACTED PRIVATE KEY]"),
    (re.compile(r"\b(AKIA|ASIA)[0-9A-Z]{16}\b"), "[REDACTED ACCESS KEY]"),
    (re.compile(r"\b(sk|pk|rk)-(live|test|proj)?[-_]?[A-Za-z0-9]{20,}\b"), "[REDACTED API KEY]"),
    (re.compile(r"\b(ghp|gho|ghu|ghs|github_pat|xox[abpr])_[A-Za-z0-9_]{16,}\b"), "[REDACTED TOKEN]"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[REDACTED SSN]"),
    # A secret-looking value (>= 6 chars with a digit or symbol), never a code expression such as
    # ``password = request.form['password']`` or a placeholder like ``"secret"``.
    (re.compile(r"(?i)\b(password|passwd|pwd|passcode)\b(\s*(?:is|was|:|=)\s*['\"]?)"
                r"(?=[^\s'\"\[\](),;]*[\d!@#$%^&*?~+])(?![\w.]*[\[(])([^\s'\"\[\](),;]{6,})"), None),
    (re.compile(r"(?i)\b(cvv|cvc|cvv2|pin)\b(\s*(?:is|was|:|=)\s*)(\d{3,6})\b"), None),
]


def redact_text(text: str) -> tuple[str, int]:
    hits = 0
    for pattern, replacement in _PATTERNS:
        if replacement is None:  # keep the label, mask only the secret value
            text, n = pattern.subn(lambda m: f"{m.group(1)}{m.group(2)}[REDACTED]", text)
        else:
            text, n = pattern.subn(replacement, text)
        hits += n

    def card(m: re.Match) -> str:
        nonlocal hits
        digits = re.sub(r"\D", "", m.group(0))
        context = text[max(0, m.start() - 40):m.start()]
        if not _CARD_CONTEXT.search(context):
            return m.group(0)  # digit runs / year sequences in tables are not card numbers
        if 13 <= len(digits) <= 19 and _luhn_ok(digits):
            hits += 1
            return "[REDACTED CARD]"
        return m.group(0)

    return _CARD.sub(card, text), hits


def redact(items: list[dict]) -> tuple[list[dict], int]:
    out, total = [], 0
    for item in items:
        text, n = redact_text(str(item.get("content") or ""))
        total += n
        out.append({**item, "content": text} if n else item)
    return out, total
