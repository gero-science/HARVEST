"""Detect bioactivity measurements in patent XML.

USPTO bulk archives are mostly patents with no binding data at all. Matching
this pattern against a patent's text first keeps the extraction pipeline from
spending LLM calls on documents that cannot contain anything useful.

The pattern is deliberately HTML-aware: USPTO XML writes subscripts as markup,
so ``IC50`` may appear as ``IC<sub>50</sub>``, ``IC₅₀`` or ``IC 50``.
"""

import re
from html.parser import HTMLParser


class TextExtractor(HTMLParser):
    """Flatten markup to text, keeping block elements on separate lines."""

    _block_end = {'p', 'div', 'section', 'article', 'li', 'tr', 'table',
                  'h1', 'h2', 'h3', 'h4', 'h5', 'h6'}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.buf = []

    def handle_starttag(self, tag, attrs):
        if tag in ('br', 'hr'):
            self.buf.append('\n')
        if tag == 'li':
            self.buf.append('- ')

    def handle_endtag(self, tag):
        if tag in self._block_end:
            self.buf.append('\n')

    def handle_data(self, data):
        self.buf.append(data)


def html_to_text(markup: str) -> str:
    """Return the visible text of an HTML/XML fragment."""
    if not markup:
        return ""
    parser = TextExtractor()
    parser.feed(markup)
    text = ''.join(parser.buf)
    text = re.sub(r'[ \t\u00A0]+', ' ', text)
    text = re.sub(r'[ \t]*\n[ \t]*', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


rx = re.compile(
 r"\b("
        # --- Metrics containing the number 50 (HTML-aware) ---
        r"(?:(?:p|log)?IC(?:<sub>\s*50\s*</sub>|<sup>\s*50\s*</sup>|[\s_\-]*50|₅₀))|"
        r"(?:(?:p|log)?EC(?:<sub>\s*50\s*</sub>|<sup>\s*50\s*</sup>|[\s_\-]*50|₅₀))|"
        r"ic50|ec50|pic50|pec50|"

        # --- Ki, Kd, Ka metrics (HTML-aware, case-insensitive) ---
        r"(?:p?K(?:<sub>\s*i\s*</sub>|ᵢ|[iI])(?!-))|"      # Ki, KI, Ki subscript, pKi... (but not Ki-)
        r"(?:p?K(?:<sub>\s*d\s*</sub>|ₔ|[dD])(?!a\b|-))|" # Kd, KD, Kd subscript, pKd... (but not kDa or Kd-)
        r"(?:p?K(?:<sub>\s*a\s*</sub>|ₐ|a))|"              # Ka, Ka subscript, pKa...

        # --- Lower-case variants ---
        r"ki(?!-)|kd(?!a\b|-)|ka|pki|pkd|"

        # --- Full text names and log variants ---
        r"logIC50|log\s*IC50|log\s*\(\s*IC50\s*\)|-log\s*\(\s*IC50\s*\)|"
        r"logKi|log\s*Ki|log\s*\(\s*Ki\s*\)|-log\s*\(\s*Ki\s*\)|"
        r"logKd|log\s*Kd|log\s*\(\s*Kd\s*\)|-log\s*\(\s*Kd\s*\)|"
        r"logEC50|log\s*EC50|log\s*\(\s*EC50\s*\)|-log\s*\(\s*EC50\s*\)|"
        r"inhibition[\s\-]*constant|dissociation[\s\-]*constant|binding[\s\-]*constant|"
        r"half[\s\-]*maximal[\s\-]*inhibitory[\s\-]*concentration|"
        r"half[\s\-]*maximal[\s\-]*effective[\s\-]*concentration|"
        r"binding[\s\-]*assay"
        r")\b"
)


def find_regexp(s: str) -> bool:
    """True if the markup mentions a binding-affinity metric."""
    return rx.search(html_to_text(s)) is not None


def decode_bytes(b: bytes) -> str:
    """Decode patent XML, which is not consistently UTF-8."""
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            return b.decode(enc)
        except UnicodeDecodeError:
            continue
    return b.decode("utf-8", errors="replace")
