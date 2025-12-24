from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Set, Tuple

import torch


_PUNCT: Set[str] = {
    "，",
    "。",
    "！",
    "？",
    "；",
    "：",
    "、",
    "（",
    "）",
    "《",
    "》",
    "“",
    "”",
    "‘",
    "’",
    "·",
    ",",
    ".",
    "?",
    "!",
    ";",
    ":",
    "(",
    ")",
    "[",
    "]",
    "{",
    "}",
}


@dataclass(frozen=True)
class DecodeOptions:
    strategy: str = "constrained"  # argmax | constrained | beam
    topk: int = 20
    beam_size: int = 5
    no_copy: bool = False
    max_repeat: int = 2
    match_punct: bool = False
    banned_ids: Tuple[int, ...] = ()


def _token(tokenizer, token_id: int) -> str:
    return tokenizer.ix_to_token.get(int(token_id), "[UNK]")


def _is_punct(tokenizer, token_id: int) -> bool:
    return _token(tokenizer, token_id) in _PUNCT


def _allowed(
    *,
    tokenizer,
    pos: int,
    cand_id: int,
    input_ids: Sequence[int],
    chosen_ids: Sequence[int],
    options: DecodeOptions,
) -> bool:
    if int(cand_id) in options.banned_ids:
        return False
    if int(cand_id) == int(tokenizer.pad_id):
        return False
    if int(cand_id) == int(tokenizer.unk_id):
        return False

    if options.match_punct and pos < len(input_ids) and _is_punct(tokenizer, input_ids[pos]):
        # When the input is punctuation, force output to match punctuation.
        return int(cand_id) == int(input_ids[pos])

    if options.no_copy and pos < len(input_ids) and int(cand_id) == int(input_ids[pos]):
        return False

    if options.max_repeat > 0:
        # Limit occurrences of the same token in the whole output.
        if sum(1 for x in chosen_ids if int(x) == int(cand_id)) >= options.max_repeat:
            return False

    return True


def decode_argmax(logits: torch.Tensor) -> List[int]:
    # logits: (seq_len, vocab)
    return logits.argmax(dim=-1).tolist()


def decode_constrained_greedy(
    logits: torch.Tensor,
    input_ids: Sequence[int],
    tokenizer,
    options: DecodeOptions,
) -> List[int]:
    # logits: (seq_len, vocab)
    seq_len = int(logits.shape[0])
    logprobs = torch.log_softmax(logits, dim=-1)
    chosen: List[int] = []

    for pos in range(seq_len):
        if options.match_punct and pos < len(input_ids) and _is_punct(tokenizer, input_ids[pos]):
            chosen.append(int(input_ids[pos]))
            continue

        k = max(1, int(options.topk))
        vals, ids = torch.topk(logprobs[pos], k=min(k, logprobs.shape[-1]))
        picked: Optional[int] = None
        for cand_id in ids.tolist():
            if _allowed(
                tokenizer=tokenizer,
                pos=pos,
                cand_id=int(cand_id),
                input_ids=input_ids,
                chosen_ids=chosen,
                options=options,
            ):
                picked = int(cand_id)
                break
        if picked is None:
            picked = int(logprobs[pos].argmax().item())
        chosen.append(picked)

    return chosen


def decode_beam_search(
    logits: torch.Tensor,
    input_ids: Sequence[int],
    tokenizer,
    options: DecodeOptions,
) -> List[int]:
    # logits: (seq_len, vocab)
    seq_len = int(logits.shape[0])
    logprobs = torch.log_softmax(logits, dim=-1)

    beams: List[Tuple[float, List[int]]] = [(0.0, [])]
    beam_size = max(1, int(options.beam_size))
    topk = max(1, int(options.topk))

    for pos in range(seq_len):
        new_beams: List[Tuple[float, List[int]]] = []

        # If punctuation position is forced, just advance beams deterministically.
        if options.match_punct and pos < len(input_ids) and _is_punct(tokenizer, input_ids[pos]):
            forced_id = int(input_ids[pos])
            for score, seq in beams:
                new_beams.append((score, seq + [forced_id]))
            beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[:beam_size]
            continue

        vals, ids = torch.topk(logprobs[pos], k=min(topk, logprobs.shape[-1]))
        cand_ids = ids.tolist()
        cand_vals = vals.tolist()

        for score, seq in beams:
            added_any = False
            for cand_id, cand_val in zip(cand_ids, cand_vals):
                if not _allowed(
                    tokenizer=tokenizer,
                    pos=pos,
                    cand_id=int(cand_id),
                    input_ids=input_ids,
                    chosen_ids=seq,
                    options=options,
                ):
                    continue
                new_beams.append((score + float(cand_val), seq + [int(cand_id)]))
                added_any = True
            if not added_any:
                # Fallback: extend with argmax if constraints are too strict.
                fallback_id = int(logprobs[pos].argmax().item())
                new_beams.append((score + float(logprobs[pos][fallback_id].item()), seq + [fallback_id]))

        beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[:beam_size]

    return beams[0][1]


def decode_with_options(
    logits: torch.Tensor,
    input_ids: Sequence[int],
    tokenizer,
    options: Optional[DecodeOptions] = None,
) -> List[int]:
    options = options or DecodeOptions()
    strategy = (options.strategy or "argmax").lower()

    if strategy == "argmax":
        return decode_argmax(logits)
    if strategy in {"constrained", "greedy"}:
        return decode_constrained_greedy(logits, input_ids, tokenizer, options)
    if strategy in {"beam", "beam_search"}:
        return decode_beam_search(logits, input_ids, tokenizer, options)

    raise ValueError(f"Unknown decode strategy: {options.strategy}")
