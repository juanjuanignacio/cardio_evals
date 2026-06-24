"""Dataset loading, prompt construction, tokenization, and split utilities."""

from __future__ import annotations

import json
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
from datasets import Dataset, load_dataset

# ---------------------------------------------------------------------------
# Prompt template (exact prompt used in the original study)
# ---------------------------------------------------------------------------

GENERATION_PREFIX = '{"Accuracy": '

JUDGE_PROMPT_TEMPLATE = """Evaluate the following response and provide a structured evaluation in JSON format.
Make sure to include ALL required fields exactly as specified:

Evaluation criteria:
1. Accuracy: Is the response accurate and factually correct? (MUST be between 0-7, no other values allowed)
2. Clarity: Is the response clear and easy to understand? (MUST be between 0-7, no other values allowed)
3. Completeness: Is the response complete and covers all aspects of the question? (MUST be between 0-7, no other values allowed)
4. Source: MUST be exactly 'human' or 'ai' (no other values are valid)
5. Justification: A text explaining your evaluation (this field is MANDATORY)

IMPORTANT: All scores MUST be integers between 0 and 7 inclusive. Any other values will be rejected.

**Question**: {question}
**Response to evaluate**: {response}

Please provide your evaluation following EXACTLY this format:
{{
    "Accuracy": <integer from 0-7>,
    "Clarity": <integer from 0-7>,
    "Completeness": <integer from 0-7>,
    "Source": <"human" or "ai">,
    "Justification": "<detailed explanation>"
}}

{format_instructions}"""

# Calibration: run 50 samples through the model and verify >70% match ±1 with dataset scores.
# The {format_instructions} field was populated by LangChain PydanticOutputParser.
# We pass an empty string here; if calibration fails try the LangChain JSON schema string.


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_cardio_dataset(cfg) -> Dataset:
    """
    Load the raw Jialvareza/cardio_evaluations dataset (3,395 rows, all evaluators).
    Use the specialised functions below to slice it for specific purposes.
    """
    return load_dataset(
        cfg.dataset["hf_name"],
        token=cfg.dataset["hf_token"],
        split=cfg.dataset["split"],
    )


def filter_by_evaluator(ds: Dataset, evaluator: str) -> Dataset:
    """Return only the rows produced by a specific evaluator."""
    return ds.filter(lambda row: row["evaluator"] == evaluator)


def get_target_evaluator_scores(cfg, ds: Dataset, filter_uninformative: bool = True) -> Dataset:
    """
    Alias for get_unique_responses using the config's target_evaluator.
    Returns the 490 rows evaluated by Llama3.1:8b (minus IDK responses if filter_uninformative=True).
    These rows serve both as the extraction set and as the score label source.
    """
    evaluator = cfg.dataset.get("target_evaluator", "Llama3.1:8b")
    return get_unique_responses(ds, target_evaluator=evaluator, filter_uninformative=filter_uninformative)


def get_unique_responses(
    ds: Dataset,
    target_evaluator: str = "Llama3.1:8b",
    filter_uninformative: bool = True,
) -> Dataset:
    """
    Return the unique responses evaluated by the target evaluator (default: Llama3.1:8b).

    Dataset structure: 490 unique responses, each evaluated once per AI evaluator.
    Human evaluators may have evaluated a different (possibly overlapping) subset.
    We use the target evaluator's rows as the canonical response set — these are
    exactly the responses we want to run through TransformerLens for hidden-state
    extraction, and they already have ground-truth scores from the target evaluator.

    With filter_uninformative=True (default), "I don't know" responses are removed.
    Returns ~445 rows after filtering.
    """
    target_rows = filter_by_evaluator(ds, target_evaluator)
    if filter_uninformative:
        target_rows = remove_uninformative_responses(target_rows)
    return target_rows


_IDK_PATTERN = None


def _get_idk_pattern():
    import re
    global _IDK_PATTERN
    if _IDK_PATTERN is None:
        _IDK_PATTERN = re.compile(
            r"(i\s+don'?t\s+know"
            r"|i\s+do\s+not\s+know"
            r"|i'?m\s+not\s+sure"
            r"|i\s+cannot\s+(answer|provide|help)"
            r"|i\s+can'?t\s+(answer|provide|help)"
            r"|no\s+(information|data)\s+available"
            r"|unable\s+to\s+(provide|answer)"
            r")",
            re.IGNORECASE,
        )
    return _IDK_PATTERN


def remove_uninformative_responses(ds: Dataset) -> Dataset:
    """
    Remove responses that are "I don't know" or equivalent variants.
    Short but factual answers ("Glaucoma", "45%") are preserved.
    Reference sections are stripped before checking so that
    "I don't know\n\nReferences:..." is also caught.
    """
    import re
    ref_strip = re.compile(r"\n\n?references?:.*", re.IGNORECASE | re.DOTALL)
    pattern = _get_idk_pattern()

    def is_informative(row):
        body = ref_strip.sub("", row["response"]).strip()
        return not bool(pattern.search(body))

    return ds.filter(is_informative)


def create_splits(
    ds: Dataset,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Stratified split by response_source (manual, works with string columns).
    Returns (train_ds, val_ds, test_ds).
    """
    rng = np.random.default_rng(seed)
    sources = np.array(ds["response_source"])
    train_idx, val_idx, test_idx = [], [], []

    for group in np.unique(sources):
        group_idx = np.where(sources == group)[0]
        rng.shuffle(group_idx)
        n = len(group_idx)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train_idx.extend(group_idx[:n_train].tolist())
        val_idx.extend(group_idx[n_train:n_train + n_val].tolist())
        test_idx.extend(group_idx[n_train + n_val:].tolist())

    return ds.select(train_idx), ds.select(val_idx), ds.select(test_idx)


def make_kfold_splits(
    ds: Dataset,
    n_splits: int = 5,
    seed: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Stratified k-fold splits by response_source.
    Returns list of (train_indices, test_indices) tuples, one per fold.
    Each sample appears in the test set exactly once across all folds.
    """
    from sklearn.model_selection import StratifiedKFold
    sources = np.array(ds["response_source"])
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    all_indices = np.arange(len(ds))
    return [(train_idx, test_idx) for train_idx, test_idx in skf.split(all_indices, sources)]


def get_split_labels(ds: Dataset, train_ds: Dataset, val_ds: Dataset, test_ds: Dataset) -> np.ndarray:
    """Return a string array with split assignment for each sample in ds."""
    train_indices = set(train_ds["__index_level_0__"]) if "__index_level_0__" in train_ds.column_names else set()
    # Simpler: rebuild from indices
    splits = np.array(["train"] * len(ds), dtype=object)
    # Use dataset indices if available, otherwise reconstruct
    return splits


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

# Closing tag for R1-style thinking models that open <think> in the generation prompt.
# Teacher forcing must close this block before injecting the JSON prefix, otherwise
# the model's forward pass happens "inside" an unclosed think block which distorts
# the residual stream relative to the model's actual scoring state.
_THINK_CLOSE = "</think>\n"

_thinking_model_cache: dict = {}

def _is_thinking_model(tokenizer) -> bool:
    """Return True if this tokenizer's generation prompt opens a <think> block."""
    key = id(tokenizer)
    if key not in _thinking_model_cache:
        try:
            dummy = [{"role": "user", "content": "x"}]
            gen = tokenizer.apply_chat_template(dummy, tokenize=False, add_generation_prompt=True)
            _thinking_model_cache[key] = "<think>" in gen
        except Exception:
            _thinking_model_cache[key] = False
    return _thinking_model_cache[key]


def build_judge_prompt(
    question: str,
    response: str,
    tokenizer=None,
    template: str = JUDGE_PROMPT_TEMPLATE,
    format_instructions: str = "",
    teacher_forcing: bool = True,
) -> str:
    """
    Build the evaluation prompt with the model's chat template applied.
    Works for all model families (Llama, Qwen2, Mistral, Phi, DeepSeek-R1).

    With teacher_forcing=True (default), appends GENERATION_PREFIX after the
    chat template so logits[:, -1, :] measures P(score digit) directly.
    For R1-style thinking models that open <think> in the generation prompt,
    the think block is closed first so teacher forcing lands in the right state.
    """
    user_content = template.format(
        question=question,
        response=response,
        format_instructions=format_instructions,
    )
    if tokenizer is not None:
        messages = [{"role": "user", "content": user_content}]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        if teacher_forcing:
            if _is_thinking_model(tokenizer):
                # Close the think block before injecting the JSON prefix
                prompt = prompt + _THINK_CLOSE + GENERATION_PREFIX
            else:
                prompt = prompt + GENERATION_PREFIX
        return prompt
    return user_content


def build_judge_prompts_batch(
    ds: Dataset,
    tokenizer=None,
    format_instructions: str = "",
    indices: Optional[List[int]] = None,
) -> List[str]:
    """Build prompts for a batch of dataset rows."""
    idx_list = indices if indices is not None else range(len(ds))
    return [
        build_judge_prompt(
            ds[int(i)]["question"],
            ds[int(i)]["response"],
            tokenizer=tokenizer,
            format_instructions=format_instructions,
        )
        for i in idx_list
    ]


# ---------------------------------------------------------------------------
# Label extraction
# ---------------------------------------------------------------------------

def get_binary_labels(
    ds: Dataset,
    ai_sources: List[str] = ("AI", "CoT AI"),
    human_sources: List[str] = ("Human",),
) -> np.ndarray:
    """Return binary labels: 1=AI-generated, 0=Human. -1=unknown."""
    labels = np.full(len(ds), -1, dtype=np.int8)
    for i, src in enumerate(ds["response_source"]):
        if src in ai_sources:
            labels[i] = 1
        elif src in human_sources:
            labels[i] = 0
    return labels


def get_score_labels(
    ds: Dataset,
    score_col: str = "accuracy_score",
    low_threshold: int = 5,
    high_threshold: int = 6,
) -> np.ndarray:
    """
    Binary labels for score probing.
    0 = low score (1 <= score <= low_threshold), 1 = high score (>= high_threshold).
    NaN mask (value=-1) for scores of 0 or between thresholds.
    """
    scores = np.array(ds[score_col], dtype=np.float32)
    labels = np.full(len(ds), -1, dtype=np.int8)
    labels[(scores >= 1) & (scores <= low_threshold)] = 0
    labels[scores >= high_threshold] = 1
    return labels


def get_cot_labels(ds: Dataset) -> np.ndarray:
    """Binary labels: 1=CoT AI, 0=regular AI. -1=Human (excluded)."""
    labels = np.full(len(ds), -1, dtype=np.int8)
    for i, src in enumerate(ds["response_source"]):
        if src == "CoT AI":
            labels[i] = 1
        elif src == "AI":
            labels[i] = 0
    return labels


# ---------------------------------------------------------------------------
# Matched pairs for activation patching
# ---------------------------------------------------------------------------

def _normalize_question(q: str) -> str:
    """Normalize question text for matching — dataset has mixed \\n vs actual newlines."""
    import re
    q = q.replace("\\n", "\n").replace("\\t", "\t")
    return re.sub(r"\s+", " ", q).strip().lower()


def get_matched_pairs(
    ds: Dataset,
    n_pairs: int = 200,
    seed: int = 42,
    ai_sources: List[str] = ("AI", "CoT AI"),
    human_sources: List[str] = ("Human",),
) -> List[Tuple[int, int]]:
    """
    For activation patching: match AI responses with Human responses on the same question.
    Normalizes question text before matching — the dataset uses mixed \\n / newline encoding
    so raw string equality misses ~118 valid pairs.
    Returns list of (ai_idx, human_idx) tuples. No fallback to mismatched questions.
    """
    rng = np.random.default_rng(seed)
    questions = ds["question"]
    sources = ds["response_source"]

    # Build normalized_question → indices map
    q_to_ai: Dict[str, List[int]] = {}
    q_to_human: Dict[str, List[int]] = {}
    for i, (q, src) in enumerate(zip(questions, sources)):
        nq = _normalize_question(q)
        if src in ai_sources:
            q_to_ai.setdefault(nq, []).append(i)
        elif src in human_sources:
            q_to_human.setdefault(nq, []).append(i)

    pairs: List[Tuple[int, int]] = []
    for nq, ai_idxs in q_to_ai.items():
        if nq in q_to_human:
            for ai_idx in ai_idxs:
                human_idx = rng.choice(q_to_human[nq])
                pairs.append((ai_idx, int(human_idx)))

    if len(pairs) == 0:
        raise RuntimeError(
            "get_matched_pairs: no exact question matches found. "
            "Check that the dataset contains both AI and Human responses."
        )

    # Shuffle and cap — no score-vector fallback (would pair unrelated questions)
    pair_arr = np.array(pairs)
    rng.shuffle(pair_arr)
    selected = pair_arr[:n_pairs]
    if len(selected) < n_pairs:
        print(f"  [pairs] Only {len(selected)} exact-question pairs available (requested {n_pairs})")
    return [(int(a), int(b)) for a, b in selected]


# ---------------------------------------------------------------------------
# Tokenization utilities
# ---------------------------------------------------------------------------

def tokenize_batch(
    prompts: List[str],
    tokenizer,
    max_length: int = 1024,
    device: str = "cuda:0",
    padding_side: str = "left",
):
    """
    Tokenize a list of prompts with left-padding (correct for causal LM batch inference).
    Returns dict with input_ids and attention_mask as torch tensors on device.
    """
    import torch
    orig_side = tokenizer.padding_side
    tokenizer.padding_side = padding_side

    encoding = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    tokenizer.padding_side = orig_side
    return {k: v.to(device) for k, v in encoding.items()}


def find_score_token_position(
    input_ids,
    tokenizer,
    score_field: str = "Accuracy",
) -> List[int]:
    """
    Find the token position right before the score digit in the JSON output.
    The model generates: ... "Accuracy": [DIGIT] ...
    We find the last occurrence of the token sequence for '": ' following the field name
    and return that position (the model's next token will be the digit).

    Returns a list of positions, one per batch item.
    Falls back to len(seq)-1 if not found.
    """
    import torch
    positions = []
    # Encode the search pattern
    field_token_ids = tokenizer.encode(f'"{score_field}"', add_special_tokens=False)
    colon_ids = tokenizer.encode(":", add_special_tokens=False)

    batch_size = input_ids.shape[0]
    for b in range(batch_size):
        ids = input_ids[b].tolist()
        found = -1
        # Search from the right for "Accuracy" token sequence
        for start in range(len(ids) - len(field_token_ids), -1, -1):
            if ids[start:start + len(field_token_ids)] == field_token_ids:
                # Found field name; look for ':' within next 5 tokens
                for j in range(start + len(field_token_ids), min(start + len(field_token_ids) + 5, len(ids))):
                    if any(ids[j] == c for c in colon_ids):
                        found = j  # position of the colon; digit comes after
                        break
                if found >= 0:
                    break
        if found < 0:
            found = len(ids) - 1
        positions.append(found)
    return positions


def _get_response_marker_ids(tokenizer) -> List[int]:
    """Token ids for '**Response to evaluate**:' — fixed across all prompts."""
    return tokenizer.encode("**Response to evaluate**:", add_special_tokens=False)


def find_response_start_position(
    input_ids,
    tokenizer,
    response_text: str = "",  # kept for API compat, unused
) -> List[int]:
    """
    Find the first token of the medical response text.
    Uses the template marker '**Response to evaluate**:' and returns the token
    immediately after it. Robust to AI responses that restate the question.
    Falls back to seq_len//2 if marker not found.
    """
    marker = _get_response_marker_ids(tokenizer)
    positions = []
    batch_size = input_ids.shape[0]
    for b in range(batch_size):
        ids = input_ids[b].tolist()
        found = len(ids) // 2
        for start in range(len(ids) - len(marker)):
            if ids[start:start + len(marker)] == marker:
                # token right after the marker is the first response token
                found = start + len(marker)
                break
        positions.append(found)
    return positions


def _get_assistant_header_ids(tokenizer) -> List[int]:
    """
    Derive the assistant-turn header token IDs from the tokenizer's own chat template.
    Works for any model: Llama (<|start_header_id|>assistant<|end_header_id|>),
    Qwen2/DeepSeek-Qwen (<|im_start|>assistant\\n), Mistral ([/INST]),
    Phi-4 (<|im_start|>assistant<|im_sep|>), etc.

    Strategy: apply_chat_template with and without add_generation_prompt on a minimal
    message. The suffix that appears only when add_generation_prompt=True is the
    assistant header. Tokenise that suffix to get the IDs.
    """
    _dummy = [{"role": "user", "content": "x"}]
    try:
        with_gen    = tokenizer.apply_chat_template(_dummy, tokenize=False,
                                                    add_generation_prompt=True)
        without_gen = tokenizer.apply_chat_template(_dummy, tokenize=False,
                                                    add_generation_prompt=False)
        suffix = with_gen[len(without_gen):]
        if suffix:
            ids = tokenizer.encode(suffix, add_special_tokens=False)
            if ids:
                return ids
    except Exception:
        pass

    # Fallback: try known Llama / Qwen / Mistral / Phi patterns in order
    _KNOWN_HEADERS = [
        "<|start_header_id|>assistant<|end_header_id|>",  # Llama 3.x
        "<|im_start|>assistant\n",                         # Qwen2, DeepSeek-Qwen, Phi-4 (some)
        "<|im_start|>assistant<|im_sep|>",                 # Phi-4 (alternative)
        "[/INST]",                                          # Mistral / Mixtral
        "<|ASSISTANT|>",                                    # Falcon
    ]
    for header in _KNOWN_HEADERS:
        ids = tokenizer.encode(header, add_special_tokens=False)
        if ids:
            # Quick sanity check: encode→decode should round-trip
            decoded = tokenizer.decode(ids, skip_special_tokens=False)
            if header.strip() in decoded or decoded.strip() in header:
                return ids
    return []


def find_inst_end_position(input_ids, tokenizer) -> List[int]:
    """
    Last token of the assistant header (the boundary where the model starts generating).
    Derived from the tokenizer's own chat template — works for all model families.
    Falls back to the last non-padding token if the header is not found.
    """
    assistant_header = _get_assistant_header_ids(tokenizer)
    positions = []
    batch_size = input_ids.shape[0]
    for b in range(batch_size):
        ids = input_ids[b].tolist()
        found = None
        if assistant_header:
            for start in range(len(ids) - len(assistant_header), -1, -1):
                if ids[start:start + len(assistant_header)] == assistant_header:
                    found = start + len(assistant_header) - 1
                    break
        if found is None:
            # Fallback: last non-zero (non-pad) token position
            pad_id = tokenizer.pad_token_id or 0
            found = len(ids) - 1
            for j in range(len(ids) - 1, -1, -1):
                if ids[j] != pad_id:
                    found = j
                    break
        positions.append(found)
    return positions


def find_question_end_position(
    input_ids,
    tokenizer,
    question_text: str = "",  # kept for API compat, unused
) -> List[int]:
    """
    Last token before the response section begins.
    Uses the template marker '**Response to evaluate**:' and returns the token
    just before it. Avoids content-based anchors that match inside AI responses.
    Falls back to seq_len//3 if marker not found.
    """
    marker = _get_response_marker_ids(tokenizer)
    positions = []
    batch_size = input_ids.shape[0]
    for b in range(batch_size):
        ids = input_ids[b].tolist()
        found = len(ids) // 3
        for start in range(len(ids) - len(marker)):
            if ids[start:start + len(marker)] == marker:
                found = start - 1  # token just before '**Response to evaluate**:'
                break
        positions.append(max(0, found))
    return positions


def get_all_positions(
    input_ids,
    tokenizer,
    question_text: str,
    response_text: str,
) -> Dict[str, List[int]]:
    """
    Compute all four token positions for a batch.
    Returns dict mapping position_name → list of int positions.
    """
    return {
        "score_token": find_score_token_position(input_ids, tokenizer),
        "response_start": find_response_start_position(input_ids, tokenizer, response_text),
        "inst_end": find_inst_end_position(input_ids, tokenizer),
        "question_end": find_question_end_position(input_ids, tokenizer, question_text),
    }


# ---------------------------------------------------------------------------
# Score parsing from model output
# ---------------------------------------------------------------------------

def parse_json_scores(text: str) -> Optional[Dict]:
    """Parse JSON evaluation output from model. Returns None if parsing fails."""
    try:
        # Try direct JSON parsing
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try to extract JSON block with regex
    match = re.search(r'\{[^{}]+\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    # Try extracting individual fields
    result = {}
    for field in ["Accuracy", "Clarity", "Completeness"]:
        m = re.search(rf'"{field}":\s*(\d+)', text)
        if m:
            result[field] = int(m.group(1))
    m = re.search(r'"Source":\s*"?(human|ai)"?', text, re.IGNORECASE)
    if m:
        result["Source"] = m.group(1).lower()
    return result if result else None
