"""Negation-aware report utilities and pathology-token weighting.

Report-level abnormal labels are retained for auditing/export. Training uses
``build_pathology_token_weights`` so only explicitly asserted pathology words
receive extra language-model loss, rather than weighting every word in an
abnormal report.
"""

import json
import re
from pathlib import Path


ABNORMAL_TERMS = (
    "abnormality",
    "airspace disease",
    "atelectasis",
    "atherosclerosis",
    "blunting",
    "calcification",
    "cardiomegaly",
    "cephalization",
    "congestion",
    "consolidation",
    "deformity",
    "degenerative",
    "density",
    "edema",
    "effusion",
    "elevated hemidiaphragm",
    "emphysema",
    "enlarged cardiac silhouette",
    "enlarged heart",
    "fibrosis",
    "fracture",
    "granuloma",
    "hernia",
    "hyperinflation",
    "infiltrate",
    "interstitial disease",
    "interstitial markings",
    "kyphosis",
    "lesion",
    "low lung volume",
    "mass",
    "nodule",
    "opacity",
    "osteopenia",
    "pleural thickening",
    "pneumonia",
    "pneumothorax",
    "pulmonary vascular prominence",
    "scar",
    "scoliosis",
    "sternotomy",
    "tortuous aorta",
)

# Stable class order shared by the dataset, model head, loss and metrics.
# Low lung volume and generic degenerative change are intentionally excluded:
# the previous token-weighting experiment over-generated these easy templates
# without improving recognition of clinically important findings.
PATHOLOGY_NAMES = (
    "atelectasis",
    "cardiomegaly",
    "opacity",
    "consolidation",
    "edema",
    "pleural_effusion",
    "pneumothorax",
    "nodule_mass",
    "fracture",
    "emphysema",
    "hernia",
    "fibrosis_scarring",
    "pleural_thickening",
    "hyperinflation",
)

# These six findings have enough positive IU-Xray training examples to support
# a stable thesis-level clinical comparison. The model still predicts all 14
# classes; this subset is only used for primary sampling, model selection and
# headline clinical metrics.
PRIMARY_PATHOLOGY_NAMES = (
    "opacity",
    "nodule_mass",
    "atelectasis",
    "fibrosis_scarring",
    "cardiomegaly",
    "pleural_effusion",
)

PRIMARY_PATHOLOGY_INDICES = tuple(
    PATHOLOGY_NAMES.index(name) for name in PRIMARY_PATHOLOGY_NAMES
)

PATHOLOGY_PATTERNS = {
    "atelectasis": (r"\batelectasis\b", r"\batelectatic\b"),
    "cardiomegaly": (
        r"\bcardiomegaly\b",
        r"\benlarged\s+(?:heart|cardiac silhouette)\b",
    ),
    "opacity": (
        r"\bopacit(?:y|ies)\b",
        r"\bairspace disease\b",
        r"\bdensit(?:y|ies)\b",
        r"\binfiltrates?\b",
    ),
    "consolidation": (r"\bconsolidations?\b", r"\bpneumonia\b"),
    "edema": (
        r"\bedema\b",
        r"\bvascular congestion\b",
        r"\bpulmonary congestion\b",
    ),
    "pleural_effusion": (r"\bpleural\s+effusions?\b", r"\beffusions?\b"),
    "pneumothorax": (r"\bpneumothorax\b", r"\bpneumothoraces\b"),
    "nodule_mass": (
        r"\bnodules?\b",
        r"\bmasses?\b",
        r"\blesions?\b",
        r"\bgranulomas?\b",
    ),
    "fracture": (r"\bfractures?\b",),
    "emphysema": (r"\bemphysema\b", r"\bemphysematous\b"),
    "hernia": (r"\bhernias?\b",),
    "fibrosis_scarring": (
        r"\bfibrosis\b",
        r"\bscars?\b",
        r"\bscarring\b",
    ),
    "pleural_thickening": (r"\bpleural\s+thickening\b",),
    "hyperinflation": (r"\bhyperinflation\b", r"\bhyperinflated\b"),
}

# High-value findings to strengthen during teacher-forced report generation.
# Generic/chronic template terms such as "degenerative" are deliberately
# excluded to avoid teaching the model one easy abnormal template.
PATHOLOGY_TOKEN_PHRASES = (
    ("airspace", "disease"),
    ("atelectasis",),
    ("blunting",),
    ("cardiomegaly",),
    ("cephalization",),
    ("congestion",),
    ("consolidation",),
    ("consolidations",),
    ("density",),
    ("densities",),
    ("edema",),
    ("effusion",),
    ("effusions",),
    ("elevated", "hemidiaphragm"),
    ("emphysema",),
    ("enlarged", "cardiac", "silhouette"),
    ("enlarged", "heart"),
    ("fibrosis",),
    ("fracture",),
    ("fractures",),
    ("granuloma",),
    ("granulomas",),
    ("hernia",),
    ("hernias",),
    ("hyperinflation",),
    ("infiltrate",),
    ("infiltrates",),
    ("interstitial", "disease"),
    ("interstitial", "markings"),
    ("lesion",),
    ("lesions",),
    ("low", "lung", "volume"),
    ("low", "lung", "volumes"),
    ("mass",),
    ("masses",),
    ("nodule",),
    ("nodules",),
    ("opacity",),
    ("opacities",),
    ("pleural", "thickening"),
    ("pneumonia",),
    ("pneumothorax",),
    ("pulmonary", "vascular", "prominence"),
    ("scar",),
    ("scars",),
    ("scarring",),
)

NEGATION_PATTERN = re.compile(
    r"(?:\bno\b|\bnot\b|\bwithout\b|\bnegative\s+for\b|"
    r"\babsence\s+of\b|\bfree\s+of\b|\bclear\s+of\b|"
    r"\bneither\b|\bresolved\b|\bresolution\s+of\b)"
    r"(?:\W+\w+){0,10}\W*$",
    flags=re.IGNORECASE,
)

_TOKEN_NEGATORS = (
    ("no",),
    ("not",),
    ("without",),
    ("negative", "for"),
    ("absence", "of"),
    ("free", "of"),
    ("clear", "of"),
    ("neither",),
    ("resolved",),
    ("resolution", "of"),
)

_POST_NEGATION_PATTERN = re.compile(
    r"^\W*(?:(?:is|are|was|were)\s+)?"
    r"(?:not\s+(?:seen|present|identified|visualized|evident)|"
    r"absent|resolved)\b|"
    r"^\W*(?:has|have|had)\s+resolved\b",
    flags=re.IGNORECASE,
)


def _normalise(text):
    text = str(text or "").lower().replace("xxxx", " ")
    return re.sub(r"\s+", " ", text).strip()


def _is_negated(sentence, match_start, match_end=None):
    prefix = sentence[:match_start]
    # "No change in mild cardiomegaly" still asserts cardiomegaly.
    if re.search(r"\bno\s+(?:significant\s+)?change\s+in\b[^.;:]*$", prefix):
        return False
    if NEGATION_PATTERN.search(prefix) is not None:
        return True
    if match_end is not None:
        return _POST_NEGATION_PATTERN.search(sentence[match_end:]) is not None
    return False


def _token_span_is_negated(tokens, start, end):
    """Return whether a token span is governed by a preceding negator."""
    clause_start = 0
    for index in range(start - 1, -1, -1):
        if tokens[index] == ".":
            clause_start = index + 1
            break

    prefix = tokens[clause_start:start]
    if not prefix:
        return False

    # Preserve findings in phrases such as "no change in cardiomegaly".
    joined_prefix = " ".join(prefix[-12:])
    if re.search(r"\bno (?:significant )?change in\b", joined_prefix):
        return False

    scope = prefix[-12:]
    for negator in _TOKEN_NEGATORS:
        width = len(negator)
        for index in range(0, len(scope) - width + 1):
            if tuple(scope[index:index + width]) == negator:
                return True

    suffix = " ".join(tokens[end:end + 6])
    if _POST_NEGATION_PATTERN.search(suffix) is not None:
        return True
    return False


def build_pathology_token_weights(report, tokenizer, pathology_weight=2.0):
    """Build weights aligned with ``tokenizer(report)``.

    Only non-negated pathology phrase tokens receive ``pathology_weight``.
    The BOS/EOS tokens, punctuation, normal words and negated findings remain
    at 1.0.
    """
    pathology_weight = float(pathology_weight)
    if pathology_weight < 1.0:
        raise ValueError("pathology_token_weight must be >= 1.0")

    tokens = tokenizer.clean_report(report).split()
    token_weights = [1.0] * len(tokens)

    for start in range(len(tokens)):
        if tokens[start] == ".":
            continue
        for phrase in PATHOLOGY_TOKEN_PHRASES:
            end = start + len(phrase)
            if tuple(tokens[start:end]) != phrase:
                continue
            if _token_span_is_negated(tokens, start, end):
                continue
            for index in range(start, end):
                token_weights[index] = max(token_weights[index], pathology_weight)

    # Tokenizer.__call__ adds BOS and EOS token IDs around the cleaned words.
    return [1.0] + token_weights + [1.0]


def is_abnormal_report(report):
    """Return True when a report contains a non-negated abnormal finding."""
    text = _normalise(report)
    if not text:
        return False

    for sentence in re.split(r"[.;:\n]+", text):
        for term in ABNORMAL_TERMS:
            pattern = re.compile(r"\b" + re.escape(term) + r"s?\b")
            for match in pattern.finditer(sentence):
                if not _is_negated(sentence, match.start(), match.end()):
                    return True
    return False


def extract_pathology_labels(report):
    """Return a reproducible multi-hot pathology vector for one report.

    Labels are derived from non-negated report findings and are intended as a
    fallback when no externally validated CheXpert/CheXbert labels are
    available. The vector order is exactly ``PATHOLOGY_NAMES``.
    """
    text = _normalise(report)
    labels = []
    for pathology_name in PATHOLOGY_NAMES:
        asserted = False
        for sentence in re.split(r"[.;:\n]+", text):
            for pattern_text in PATHOLOGY_PATTERNS[pathology_name]:
                pattern = re.compile(pattern_text, flags=re.IGNORECASE)
                for match in pattern.finditer(sentence):
                    if not _is_negated(sentence, match.start(), match.end()):
                        asserted = True
                        break
                if asserted:
                    break
            if asserted:
                break
        labels.append(float(asserted))
    return labels


def load_abnormal_labels(path):
    """Load study-level labels from a JSON dictionary or list of records."""
    if not path:
        return None
    label_path = Path(path).expanduser()
    if not label_path.is_file():
        raise FileNotFoundError(f"Abnormal label file not found: {label_path}")

    with label_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if isinstance(data, dict):
        items = data.items()
    elif isinstance(data, list):
        items = []
        for record in data:
            if not isinstance(record, dict) or "id" not in record:
                raise ValueError("Every label record must contain an 'id' field")
            value = record.get("is_abnormal", record.get("abnormal"))
            items.append((record["id"], value))
    else:
        raise ValueError("Abnormal labels must be a JSON dictionary or list")

    def coerce_label(value, study_id):
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)) and value in (0, 1):
            return bool(value)
        if isinstance(value, str):
            normalised = value.strip().lower()
            if normalised in ("1", "true", "abnormal", "positive"):
                return True
            if normalised in ("0", "false", "normal", "negative"):
                return False
        raise ValueError(f"Invalid abnormal label for study {study_id}: {value!r}")

    labels = {}
    for study_id, value in items:
        if isinstance(value, dict):
            value = value.get("is_abnormal", value.get("abnormal"))
        if value is None:
            raise ValueError(f"Missing abnormal label for study {study_id}")
        labels[str(study_id)] = coerce_label(value, study_id)
    return labels


