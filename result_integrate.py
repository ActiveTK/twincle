#!/usr/bin/env python3
import argparse
import glob
import json
import math
import sys
from decimal import Decimal, localcontext
from pathlib import Path


DECIMAL_PREC = 80
UNIT_ROUND = Decimal(1) / Decimal(1 << 53)
TWIN_PRIME_CONSTANT_C2 = Decimal("0.6601618158468695739")
FULL_NOTE = (
    "checkpoint sums are exact for limit_requested, computed as full k<k_floor "
    "plus partial k_floor by residue filter; error bounds are IEEE-754 gamma_n "
    "upper bounds."
)


def gamma_n(n: int) -> Decimal:
    with localcontext() as ctx:
        ctx.prec = DECIMAL_PREC
        nu = Decimal(n) * UNIT_ROUND
        return nu / (Decimal(1) - nu)


def accum_error_bound(sum_abs: Decimal, n_terms: int) -> Decimal:
    with localcontext() as ctx:
        ctx.prec = DECIMAL_PREC
        n = Decimal(n_terms)
        kahan = (Decimal(2) * UNIT_ROUND + Decimal(2) * n * UNIT_ROUND * UNIT_ROUND) * sum_abs
        reduction = Decimal(64) * UNIT_ROUND * sum_abs
        return kahan + reduction


def is_prime_small(n: int) -> bool:
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    d = 5
    while d * d <= n:
        if n % d == 0 or n % (d + 2) == 0:
            return False
        d += 6
    return True


def gcd_u64(a: int, b: int) -> int:
    while b != 0:
        a, b = b, a % b
    return a


def decimal_from_f64(value: float) -> Decimal:
    return Decimal(repr(value))


def count_wheel_missed_twins(wheel_m: int, limit: int) -> tuple[int, Decimal]:
    count = 0
    total = 0.0
    max_p = min(limit - 2, wheel_m)
    for p in range(3, max_p + 1):
        if not is_prime_small(p) or not is_prime_small(p + 2):
            continue
        pc = gcd_u64(p, wheel_m) == 1
        p2c = gcd_u64(p + 2, wheel_m) == 1
        if not (pc and p2c):
            count += 1
            total += 1.0 / p + 1.0 / (p + 2)
    return count, decimal_from_f64(total)


def compute_residues(wheel_m: int) -> list[int]:
    residues = []
    for r in range(1, wheel_m):
        if r % 2 == 0:
            continue
        if gcd_u64(r, wheel_m) != 1:
            continue
        if gcd_u64((r + 2) % wheel_m, wheel_m) != 1:
            continue
        residues.append(r)
    return residues


def count_candidates_upto(limit: int, wheel_m: int, residues: list[int]) -> int:
    if limit < 1:
        return 0
    full_k, rem = divmod(limit, wheel_m)
    total = full_k * len(residues)
    total += sum(1 for r in residues if r <= rem)
    if limit >= 1 and 1 in residues:
        total -= 1
    return total


def count_candidates_interval(start_exclusive: int, end_inclusive: int, wheel_m: int, residues: list[int]) -> int:
    return count_candidates_upto(end_inclusive, wheel_m, residues) - count_candidates_upto(start_exclusive, wheel_m, residues)


def load_part(path: Path) -> dict:
    meta = None
    final = None
    checkpoints: list[dict] = []
    checkpoint_by_limit: dict[int, dict] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line, parse_float=Decimal)
            rec_type = rec.get("type")
            if rec_type == "meta":
                meta = rec
            elif rec_type == "checkpoint":
                lim = int(rec["limit_requested"])
                checkpoints.append(rec)
                checkpoint_by_limit[lim] = rec
            elif rec_type == "final":
                final = rec
    if meta is None:
        raise ValueError(f"{path}: missing meta record")
    if final is None:
        raise ValueError(f"{path}: missing final record")
    return {
        "path": str(path),
        "meta": meta,
        "final": final,
        "checkpoints": checkpoints,
        "checkpoint_by_limit": checkpoint_by_limit,
    }


def validate_parts(parts: list[dict]) -> tuple[int, int, int | None]:
    first_meta = parts[0]["meta"]
    limit = int(first_meta["limit"])
    wheel_m = int(first_meta["wheel_m"])
    exp = first_meta.get("exp")
    for part in parts[1:]:
        meta = part["meta"]
        if int(meta["limit"]) != limit:
            raise SystemExit("All part files must share the same limit.")
        if int(meta["wheel_m"]) != wheel_m:
            raise SystemExit("All part files must share the same wheel_m.")
        if meta.get("exp") != exp:
            raise SystemExit("All part files must share the same exp.")
    return limit, wheel_m, exp


def summarize_coverage(parts: list[dict], limit: int) -> dict:
    ranges = sorted(
        (int(p["final"]["range_start_exclusive"]), int(p["final"]["range_end_inclusive"]))
        for p in parts
    )
    merged: list[list[int]] = []
    has_overlap = False
    for start_exclusive, end_inclusive in ranges:
        start = start_exclusive + 1
        end = end_inclusive
        if not merged or start > merged[-1][1] + 1:
            merged.append([start, end])
            continue
        if start <= merged[-1][1]:
            has_overlap = True
        merged[-1][1] = max(merged[-1][1], end)
    gaps = []
    cursor = 1
    for start, end in merged:
        if start > cursor:
            gaps.append([cursor, start - 1])
        cursor = max(cursor, end + 1)
    if cursor <= limit:
        gaps.append([cursor, limit])
    complete_for_limit = len(merged) == 1 and merged[0] == [1, limit] and not has_overlap
    return {
        "ranges": ranges,
        "merged_ranges": merged,
        "gaps": gaps,
        "has_overlap": has_overlap,
        "complete_for_limit": complete_for_limit,
    }


def default_output_path(parts: list[dict], exp: int | None) -> str:
    if exp is not None:
        return f"result_e{exp}_summary.txt"
    return "result_integrated_summary.txt"


def build_meta(first_meta: dict, limit: int) -> dict:
    return {
        "type": "meta",
        "format": "jsonl",
        "git_sha": first_meta.get("git_sha"),
        "exp": first_meta.get("exp"),
        "limit": limit,
        "wheel_m": int(first_meta["wheel_m"]),
        "part_mode": False,
        "split_mode": False,
        "split": None,
        "split_count": None,
        "range_start_exclusive": 0,
        "range_end_inclusive": limit,
        "segment_k": int(first_meta["segment_k"]),
        "segment_mem_frac": first_meta["segment_mem_frac"],
        "auto_tune_seg": bool(first_meta["auto_tune_seg"]),
        "residues_len": int(first_meta["residues_len"]),
        "gpu_names": list(first_meta["gpu_names"]),
        "note": FULL_NOTE,
    }


def component_checkpoint_for_limit(part: dict, limit_requested: int) -> dict | None:
    start = int(part["final"]["range_start_exclusive"])
    end = int(part["final"]["range_end_inclusive"])
    if limit_requested <= start:
        return None
    if limit_requested >= end:
        rec = part["checkpoint_by_limit"].get(end)
        if rec is None:
            raise SystemExit(f"{part['path']}: missing boundary checkpoint for {end}")
        return rec
    rec = part["checkpoint_by_limit"].get(limit_requested)
    if rec is None:
        raise SystemExit(f"{part['path']}: missing checkpoint for {limit_requested}")
    return rec


def aggregate_checkpoint(parts: list[dict], limit_requested: int, wheel_m: int) -> dict:
    component_recs = [
        rec for rec in (component_checkpoint_for_limit(part, limit_requested) for part in parts)
        if rec is not None
    ]
    total_twins = sum(int(rec["twins"]) for rec in component_recs)
    with localcontext() as ctx:
        ctx.prec = DECIMAL_PREC
        total_sum = sum((rec["sum"] for rec in component_recs), Decimal(0))
        accum_err = accum_error_bound(total_sum, total_twins)
        term_eval_err = gamma_n(5) * total_sum
        total_err = accum_err + term_eval_err
    return {
        "type": "checkpoint",
        "limit_requested": limit_requested,
        "split": None,
        "range_start_exclusive": 0,
        "range_end_inclusive": int(parts[0]["meta"]["limit"]),
        "k_floor": limit_requested // wheel_m,
        "limit_covered": limit_requested,
        "twins": total_twins,
        "sum": total_sum,
        "accum_err_bound": accum_err,
        "term_eval_err_bound": term_eval_err,
        "total_err_bound": total_err,
    }


def aggregate_final(parts: list[dict], limit: int, wheel_m: int) -> dict:
    shard_twins = sum(int(part["final"]["twins"]) for part in parts)
    with localcontext() as ctx:
        ctx.prec = DECIMAL_PREC
        shard_sum = sum((part["final"]["sum"] for part in parts), Decimal(0))
    missed_count, missed_sum = count_wheel_missed_twins(wheel_m, limit)
    final_twins = shard_twins + missed_count
    with localcontext() as ctx:
        ctx.prec = DECIMAL_PREC
        final_sum = shard_sum + missed_sum
        accum_err = accum_error_bound(final_sum, final_twins)
        term_eval_err = gamma_n(5) * final_sum
        total_err = accum_err + term_eval_err
        b2_star_term = decimal_from_f64(4.0 * float(TWIN_PRIME_CONSTANT_C2) / math.log(limit))
        b2_star = final_sum + b2_star_term
    return {
        "type": "final",
        "part_mode": False,
        "limit": limit,
        "split": None,
        "range_start_exclusive": 0,
        "range_end_inclusive": limit,
        "twins": final_twins,
        "sum": final_sum,
        "b2_star": b2_star,
        "accum_err_bound": accum_err,
        "term_eval_err_bound": term_eval_err,
        "total_err_bound": total_err,
        "missed_count": missed_count,
        "missed_sum": missed_sum,
    }


def json_number(value: Decimal) -> str:
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    if text == "-0":
        text = "0"
    if "." not in text and "E" not in text and "e" not in text:
        return text
    return text


def quantize_decimal_places(value: Decimal, places: int) -> Decimal:
    with localcontext() as ctx:
        ctx.prec = max(DECIMAL_PREC, places + 20)
        quantum = Decimal(1).scaleb(-places)
        return value.quantize(quantum)


def json_dumps_decimal(value) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, Decimal):
        return json_number(value)
    if isinstance(value, list):
        return "[" + ",".join(json_dumps_decimal(v) for v in value) + "]"
    if isinstance(value, dict):
        items = []
        for key in sorted(value.keys()):
            items.append(json.dumps(str(key), ensure_ascii=False) + ":" + json_dumps_decimal(value[key]))
        return "{" + ",".join(items) + "}"
    raise TypeError(f"Unsupported type for JSON serialization: {type(value)!r}")


def format_aligned_decimals(
    values: list[tuple[str, Decimal, str | None]],
    min_label: int = 0,
    min_int: int = 1,
    min_frac: int = 0,
    min_suffix: int = 0,
) -> list[str]:
    rendered = []
    max_label = min_label
    max_int = min_int
    max_frac = min_frac
    max_suffix = min_suffix
    for label, value, suffix in values:
        text = json_number(value)
        if "." in text:
            int_part, frac_part = text.split(".", 1)
        else:
            int_part, frac_part = text, ""
        suffix_text = suffix or ""
        rendered.append((label, int_part, frac_part, suffix_text))
        max_label = max(max_label, len(label))
        max_int = max(max_int, len(int_part))
        max_frac = max(max_frac, len(frac_part))
        max_suffix = max(max_suffix, len(suffix_text))

    lines = []
    for label, int_part, frac_part, suffix_text in rendered:
        label_text = label.ljust(max_label)
        left = int_part.rjust(max_int)
        if max_frac > 0:
            right = frac_part.ljust(max_frac)
            value_text = f"{left}.{right}"
        else:
            value_text = left
        if max_suffix > 0:
            suffix_padded = suffix_text.ljust(max_suffix)
            if suffix_text:
                lines.append(f"{label_text}: {value_text} {suffix_padded}")
            else:
                lines.append(f"{label_text}: {value_text}")
        else:
            lines.append(f"{label_text}: {value_text}")
    return lines


def print_aligned_decimals(values: list[tuple[str, Decimal]]) -> None:
    for line in format_aligned_decimals(values):
        print(f"  {line}")


def format_with_commas(n: int) -> str:
    return f"{n:,}"


def format_target(limit: int, exp: int | None) -> str:
    if exp is not None:
        return f"N=10^{exp}"
    return f"N={format_with_commas(limit)}"


def format_gpu_summary(gpu_names: list[str]) -> str:
    if not gpu_names:
        return "unknown GPU"
    first = gpu_names[0]
    if all(name == first for name in gpu_names):
        return f"{len(gpu_names)}x {first}"
    counts: dict[str, int] = {}
    for name in gpu_names:
        counts[name] = counts.get(name, 0) + 1
    parts = [f"{count}x {name}" for name, count in counts.items()]
    return " + ".join(parts)


def format_elapsed(elapsed_value) -> str:
    if elapsed_value is None:
        return "n/a"
    elapsed = float(elapsed_value)
    return f"{elapsed:,.2f}s"


def format_speed(candidates: int, elapsed_value) -> str:
    if elapsed_value is None:
        return "n/a"
    elapsed = float(elapsed_value)
    if elapsed <= 0.0:
        return "n/a"
    text = f"{candidates / elapsed:.2e}"
    mantissa, exp = text.split("e")
    exp_i = int(exp)
    return f"{mantissa}e{exp_i} cand/s"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Integrate split outputs and write a plain-text summary."
    )
    parser.add_argument("files", nargs="*", help="Part files to integrate. Default: result_e*_part_*.json")
    parser.add_argument(
        "--output",
        default=None,
        help="Output text summary path. Default: result_e<exp>_summary.txt",
    )
    args = parser.parse_args()

    paths = [Path(p) for p in args.files] if args.files else [Path(p) for p in sorted(glob.glob("result_e*_part_*.json"))]
    if not paths:
        raise SystemExit("No part files found.")

    parts = [load_part(path) for path in paths]
    limit, wheel_m, exp = validate_parts(parts)
    coverage = summarize_coverage(parts, limit)
    if not coverage["complete_for_limit"]:
        raise SystemExit(f"Coverage incomplete or overlapping: {coverage}")

    first_meta = parts[0]["meta"]
    residues = compute_residues(wheel_m)
    checkpoint_limits = sorted({int(rec["limit_requested"]) for part in parts for rec in part["checkpoints"]})
    meta = build_meta(first_meta, limit)
    checkpoints = [aggregate_checkpoint(parts, lim, wheel_m) for lim in checkpoint_limits]
    final = aggregate_final(parts, limit, wheel_m)

    part_sums = []
    for part in sorted(parts, key=lambda p: int(p["final"].get("split") or 0)):
        start_exclusive = int(part["final"]["range_start_exclusive"])
        start = start_exclusive + 1
        end = int(part["final"]["range_end_inclusive"])
        twins = int(part["final"]["twins"])
        label = f"({format_with_commas(start)}..{format_with_commas(end)}, twins={format_with_commas(twins)})"
        candidates = count_candidates_interval(start_exclusive, end, wheel_m, residues)
        suffix = (
            f"({format_elapsed(part['final'].get('elapsed_secs'))} @ "
            f"{format_gpu_summary(list(part['meta'].get('gpu_names', [])))}; "
            f"{format_speed(candidates, part['final'].get('elapsed_secs'))})"
        )
        part_sums.append((label, part["final"]["sum"], suffix))
    final_values = [
        ("Final Brun partial sum  (twins=" + format_with_commas(int(final["twins"])) + ")", final["sum"], None),
        ("Final B2*", final["b2_star"], None),
        ("Accumulation error bound", quantize_decimal_places(final["accum_err_bound"], 19), None),
        ("Per-term error bound", quantize_decimal_places(final["term_eval_err_bound"], 19), None),
        ("Total error bound", quantize_decimal_places(final["total_err_bound"], 19), None),
    ]
    all_values = part_sums + final_values
    max_label = 0
    max_int = 1
    max_frac = 0
    max_suffix = 0
    for label, value, suffix in all_values:
        text = json_number(value)
        if "." in text:
            int_part, frac_part = text.split(".", 1)
        else:
            int_part, frac_part = text, ""
        max_label = max(max_label, len(label))
        max_int = max(max_int, len(int_part))
        max_frac = max(max_frac, len(frac_part))
        max_suffix = max(max_suffix, len(suffix or ""))

    output_lines = [f"Target: {format_target(limit, exp)}", "Part final sums:"]
    output_lines.extend(
        f"  {line}" for line in format_aligned_decimals(part_sums, max_label, max_int, max_frac, max_suffix)
    )
    output_lines.append("Final values:")
    output_lines.extend(
        f"  {line}" for line in format_aligned_decimals(final_values, max_label, max_int, max_frac, max_suffix)
    )

    output_path = Path(args.output or default_output_path(parts, exp))
    output_path.write_text("\n".join(output_lines) + "\n", encoding="utf-8", newline="\n")
    for line in output_lines:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
