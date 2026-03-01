#!/usr/bin/env python3
import argparse
import glob
import json
import math
from datetime import datetime, timezone
from pathlib import Path


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


def count_wheel_missed_twins(wheel_m: int, limit: int) -> tuple[int, float]:
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
    return count, total


def load_part(path: Path) -> dict:
    meta = None
    final = None
    checkpoints = 0
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rec_type = rec.get("type")
            if rec_type == "meta":
                meta = rec
            elif rec_type == "checkpoint":
                checkpoints += 1
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
        "checkpoint_count": checkpoints,
    }


def summarize_coverage(parts: list[dict], full_k_end: int) -> dict:
    ranges = sorted((p["final"]["k_start"], p["final"]["k_end"]) for p in parts)
    merged: list[list[int]] = []
    has_overlap = False
    for start, end in ranges:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
            continue
        if start < merged[-1][1]:
            has_overlap = True
        merged[-1][1] = max(merged[-1][1], end)
    gaps = []
    cursor = 0
    for start, end in merged:
        if start > cursor:
            gaps.append([cursor, start])
        cursor = max(cursor, end)
    if cursor < full_k_end:
        gaps.append([cursor, full_k_end])
    complete_for_limit = len(merged) == 1 and merged[0] == [0, full_k_end] and not has_overlap
    return {
        "ranges": ranges,
        "merged_ranges": merged,
        "gaps": gaps,
        "has_overlap": has_overlap,
        "complete_for_limit": complete_for_limit,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Integrate result_part[start-end].json shard outputs into result.json."
    )
    parser.add_argument("files", nargs="*", help="Part files to integrate. Default: result_part[*].json")
    parser.add_argument(
        "--output",
        default="result.json",
        help="Output JSON path (default: result.json)",
    )
    args = parser.parse_args()

    paths = [Path(p) for p in args.files] if args.files else [Path(p) for p in sorted(glob.glob("result_part[[]*-*[]].json"))]
    if not paths:
        raise SystemExit("No part files found.")

    parts = [load_part(path) for path in paths]
    first_meta = parts[0]["meta"]
    limit = int(first_meta["limit"])
    wheel_m = int(first_meta["wheel_m"])
    full_k_end = limit // wheel_m + 2

    for part in parts[1:]:
        meta = part["meta"]
        if int(meta["limit"]) != limit:
            raise SystemExit("All part files must share the same limit.")
        if int(meta["wheel_m"]) != wheel_m:
            raise SystemExit("All part files must share the same wheel_m.")

    coverage = summarize_coverage(parts, full_k_end)

    shard_twins = sum(int(p["final"]["twins"]) for p in parts)
    shard_sum = math.fsum(float(p["final"]["sum"]) for p in parts)
    accum_err_bound = math.fsum(float(p["final"].get("accum_err_bound", 0.0)) for p in parts)
    term_eval_err_bound = math.fsum(float(p["final"].get("term_eval_err_bound", 0.0)) for p in parts)
    total_err_bound = math.fsum(float(p["final"].get("total_err_bound", 0.0)) for p in parts)
    elapsed_secs_sum = math.fsum(float(p["final"].get("elapsed_secs", 0.0)) for p in parts)

    missed_count, missed_sum = count_wheel_missed_twins(wheel_m, limit)
    final_twins = shard_twins + missed_count
    final_sum = shard_sum + missed_sum
    b2_star = final_sum + 4.0 * 0.6601618158468695739 / math.log(limit)

    result = {
        "type": "integrated_result",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "limit": limit,
        "wheel_m": wheel_m,
        "coverage": coverage,
        "wheel_missed": {
            "count": missed_count,
            "sum": missed_sum,
        },
        "parts": [
            {
                "path": part["path"],
                "k_start": int(part["final"]["k_start"]),
                "k_end": int(part["final"]["k_end"]),
                "twins": int(part["final"]["twins"]),
                "sum": float(part["final"]["sum"]),
                "accum_err_bound": float(part["final"].get("accum_err_bound", 0.0)),
                "term_eval_err_bound": float(part["final"].get("term_eval_err_bound", 0.0)),
                "total_err_bound": float(part["final"].get("total_err_bound", 0.0)),
                "elapsed_secs": float(part["final"].get("elapsed_secs", 0.0)),
                "checkpoint_count": part["checkpoint_count"],
            }
            for part in sorted(parts, key=lambda p: (int(p["final"]["k_start"]), int(p["final"]["k_end"])))
        ],
        "totals": {
            "shard_twins": shard_twins,
            "shard_sum": shard_sum,
            "accum_err_bound": accum_err_bound,
            "term_eval_err_bound": term_eval_err_bound,
            "total_err_bound": total_err_bound,
            "elapsed_secs_sum": elapsed_secs_sum,
            "final_twins": final_twins,
            "final_sum": final_sum,
            "b2_star": b2_star,
        },
    }

    output_path = Path(args.output)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Wrote {output_path}")
    print(f"Parts: {len(parts)}")
    print(f"Coverage complete: {coverage['complete_for_limit']}")
    print(f"Coverage overlap: {coverage['has_overlap']}")
    print(f"Shard twins: {shard_twins}")
    print(f"Shard sum: {shard_sum:.17f}")
    print(f"Wheel-missed twins: {missed_count}")
    print(f"Wheel-missed sum: {missed_sum:.17f}")
    print(f"Final twins: {final_twins}")
    print(f"Final Brun partial sum: {final_sum:.17f}")
    print(f"Final B2*: {b2_star:.17f}")
    print(f"Accumulation error bound sum: {accum_err_bound:.3e}")
    print(f"Per-term error bound sum: {term_eval_err_bound:.3e}")
    print(f"Total error bound sum: {total_err_bound:.3e}")
    if coverage["gaps"]:
        print(f"Gaps: {coverage['gaps']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
