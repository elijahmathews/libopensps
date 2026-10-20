"""Minimal reader of the ADS BibTeX export (data_release.bib) used for the data provenance.

Only what that export contains is supported: @ARTICLE / @INPROCEEDINGS entries whose field values
are in braces or double quotes (nested braces allowed) or bare numbers/macros. References are
rendered from the parsed fields only ("Authors year, Journal, volume, page"); nothing is typed by
hand except the expansion of the AAS journal macros and of LaTeX accents.
"""
import os
import re

BIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_release.bib")

# AAS journal macros as used by ADS exports
JOURNALS = {r"\aap": "A&A", r"\aaps": "A&AS", r"\aj": "AJ", r"\apj": "ApJ", r"\apjs": "ApJS",
            r"\mnras": "MNRAS", r"\pasa": "PASA", r"\rmxaa": "RMxAA", r"\apjl": "ApJL", r"\pasp": "PASP"}
ACCENTS = {"'": "\u0301", "`": "\u0300", '"': "\u0308", "^": "\u0302", "~": "\u0303", "c": "\u0327",
           "v": "\u030c", "=": "\u0304", ".": "\u0307", "u": "\u0306", "H": "\u030b"}
SERIES = {"IAU Symposium": "IAU Symp.", "Astronomical Society of the Pacific Conference Series": "ASP Conf. Ser."}


def _value(text, i):
    """Parses a field value starting at text[i]; returns (value, next index)."""
    if text[i] == "{":
        depth, j = 0, i
        while True:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    return text[i + 1:j], j + 1
            j += 1
    if text[i] == '"':
        depth, j = 0, i + 1
        while True:
            c = text[j]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
            elif c == '"' and depth == 0:
                return text[i + 1:j], j + 1
            j += 1
    m = re.match(r"[^,}\s]+", text[i:])
    return m.group(0), i + m.end()


def parse(path=BIB_PATH):
    """{bibcode: {"type": ..., field: raw value, ...}} for every entry of the file."""
    text = open(path, encoding="utf-8").read()
    entries = {}
    for m in re.finditer(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", text):
        etype, key = m.group(1).lower(), m.group(2)
        i = m.end()
        fields = {"type": etype}
        while True:
            fm = re.compile(r"\s*(\w+)\s*=\s*").match(text, i)
            if not fm:
                break
            val, i = _value(text, fm.end())
            fields[fm.group(1).lower()] = val
            cm = re.compile(r"\s*,").match(text, i)
            if not cm:
                break
            i = cm.end()
        if key in entries:
            raise ValueError(f"duplicate entry {key}")
        entries[key] = fields
    return entries


def latex_to_text(s):
    s = re.sub(r"\\([\'`\"^~=.])\{?(\w)\}?", lambda m: m.group(2) + ACCENTS[m.group(1)], s)
    s = re.sub(r"\\([cvuH])\{(\w)\}", lambda m: m.group(2) + ACCENTS[m.group(1)], s)
    s = s.replace("~", " ").replace("{", "").replace("}", "")
    import unicodedata
    return unicodedata.normalize("NFC", s).strip()


def last_names(author_field):
    names = []
    for a in re.split(r"\s+and\s+", author_field):
        last = a.split(",")[0]
        names.append(latex_to_text(last))
    return names


def authors_short(author_field):
    n = last_names(author_field)
    if len(n) == 1:
        return n[0]
    if len(n) == 2:
        return f"{n[0]} & {n[1]}"
    return f"{n[0]} et al."


def render(entry):
    """'Authors year, Journal, volume, page' from the parsed fields."""
    parts = [f"{authors_short(entry['author'])} {entry['year']}"]
    if entry["type"] == "article":
        parts.append(JOURNALS.get(entry["journal"], latex_to_text(entry["journal"])))
    else:
        parts.append(SERIES.get(entry.get("series", ""), latex_to_text(entry.get("series", entry.get("booktitle", "")))))
    if "volume" in entry:
        parts.append(entry["volume"])
    if "pages" in entry:
        parts.append(entry["pages"].split("-")[0])
    return ", ".join(parts)


def citation(bibcode, entries=None):
    """Provenance record of one bibcode: bibcode, rendered reference, DOI and ADS URL if given."""
    entries = entries if entries is not None else parse()
    if bibcode not in entries:
        raise KeyError(f"{bibcode} is not in {BIB_PATH}")
    e = entries[bibcode]
    rec = {"bibcode": bibcode, "reference": render(e)}
    if "doi" in e:
        rec["doi"] = e["doi"]
    if "adsurl" in e:
        rec["ads"] = e["adsurl"]
    return rec
