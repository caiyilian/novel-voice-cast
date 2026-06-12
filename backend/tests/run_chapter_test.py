"""
测试 LLM 章节提取 - 对 novels/other_novels 目录下所有小说运行测试

用法:
    python tests/run_chapter_test.py              # 测试所有小说
    python tests/run_chapter_test.py --start 0 --end 5   # 测试前5本
    python tests/run_chapter_test.py --file "狼与辛香料 02（台）.txt"  # 测试单本
"""
import sys, os, re, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.core.ollama_client import OllamaClient
from app.core.chapter_extractor import _run_llm_extraction
from app.core.parser import extract_chapters


def find_expected_chapters(lines):
    """Find chapter markers in the file."""
    expected_lines = []
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped and re.match(r'^第[一二三四五六七八九十百千\d]+幕', stripped):
            expected_lines.append(i)
    # deduplicate consecutive lines
    expected = []
    for ln in expected_lines:
        if not expected or ln - expected[-1] > 1:
            expected.append(ln)
    return expected


def check_match(expected, found, tolerance=2):
    """Check how many expected chapters were found."""
    match_count = 0
    for exp in expected:
        for f_ln in found:
            if abs(f_ln - exp) <= tolerance:
                match_count += 1
                break
    return match_count


def print_progress(current, total, width=40):
    """Print a progress bar."""
    pct = current / total
    filled = int(width * pct)
    bar = "█" * filled + "░" * (width - filled)
    sys.stdout.write(f"\r  [{bar}] {current}/{total}")
    sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(description="Test LLM chapter extraction")
    parser.add_argument("--start", type=int, default=0, help="Start index (0-based)")
    parser.add_argument("--end", type=int, default=None, help="End index (exclusive)")
    parser.add_argument("--file", type=str, default=None, help="Test single file")
    parser.add_argument("--novels-dir", type=str, default="E:/projects/novel-voice-cast/novels/other_novels")
    args = parser.parse_args()

    # find novels
    files = sorted([f for f in os.listdir(args.novels_dir) if f.endswith(".txt")])
    if args.file:
        files = [f for f in files if args.file in f]
    elif args.end:
        files = files[args.start:args.end]
    else:
        files = files[args.start:]

    if not files:
        print("No novels found!")
        return

    # connect to Ollama
    client = OllamaClient()
    status = client.check_connection()
    if not status.ok:
        print(f"Cannot connect to Ollama: {status.message}")
        return

    print(f"Testing {len(files)} novels...")
    print(f"Ollama: {client.config.model} @ {client.config.base_url}")
    print()

    results = []
    total_start = time.time()

    for idx, fname in enumerate(files):
        path = os.path.join(args.novels_dir, fname)

        # read file
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        lines = text.splitlines()

        # find expected chapters
        expected = find_expected_chapters(lines)

        # run LLM
        t0 = time.time()
        regex_chapters = extract_chapters(text)
        llm_chapters = _run_llm_extraction(text, client, 12, regex_chapters)
        elapsed = time.time() - t0
        found = sorted([ch["line_number"] for ch in llm_chapters])

        # check match
        match_count = check_match(expected, found)
        match = match_count == len(expected) if expected else len(found) == 0

        results.append({
            "file": fname,
            "lines": len(lines),
            "expected": len(expected),
            "matched": match_count,
            "found_count": len(found),
            "match": match,
            "time": elapsed,
        })

        # print result
        status_str = "PASS" if match else "FAIL"
        total_elapsed = time.time() - total_start
        print(f"[{status_str}] {idx+1}/{len(files)}: {fname[:40]} ({elapsed:.0f}s, {match_count}/{len(expected)})")

    # summary
    total_elapsed = time.time() - total_start
    pass_count = sum(1 for r in results if r["match"])
    total_expected = sum(r["expected"] for r in results)
    total_matched = sum(r["matched"] for r in results)

    print()
    print("=" * 60)
    print(f"Result: {pass_count}/{len(results)} novels passed")
    print(f"Chapters: {total_matched}/{total_expected} found")
    print(f"Time: {total_elapsed:.0f}s ({total_elapsed/len(files):.0f}s per novel)")
    print()

    if pass_count < len(results):
        print("Failed novels:")
        for r in results:
            if not r["match"]:
                print(f"  {r['file']}: expected={r['expected']}, matched={r['matched']}")


if __name__ == "__main__":
    main()
