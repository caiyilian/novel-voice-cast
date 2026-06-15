## Problem Description

When running the full pipeline, the labels from `labels.txt` are being assigned to the wrong dialogues. This causes characters to speak with the wrong voice.

**Example:** Dialogue #1347 should be spoken by 赫萝 (according to `labels.txt`), but the pipeline assigns it to 罗伦斯.

## Root Cause Analysis

The parser has multiple dialogue detection patterns:
1. `DIALOGUE_JP`: `「对话」——说话人` (speaker after dialogue)
2. `SPEAKER_PREFIX`: `说话人：「对话」` (speaker before dialogue)
3. `DIALOGUE_CN`: `「对话」` (standalone dialogue)
4. `DIALOGUE_INLINE`: `「对话」` (inline in middle of line)
5. `narrative`: Non-dialogue lines (added as "旁白")

The issue is that:
- Simple regex (`re.findall(r'「(.*?)」', novel)`) finds **1349 dialogues**
- Parser finds **1287 dialogues** + **1740 narrative** = **3022 total**

The 62 missing dialogues are detected by `DIALOGUE_INLINE` but NOT by the simple regex. This causes the `dialogue_index` counter to be off, misaligning the labels.

Additionally, narrative lines are added to the dialogues list as "旁白" but do NOT increment `dialogue_index`. This means:
- If a narrative line appears before a dialogue, the dialogue gets the wrong label
- The misalignment accumulates throughout the novel

## How to Reproduce

1. Run `run_full.py` with emotion labeling enabled
2. Check the audio files around index 1347
3. Compare with `labels.txt` - the speaker should be 赫萝 but is assigned to 罗伦斯

## Files to Check

- `novels/novel.txt` - Novel text
- `novels/labels.txt` - Speaker labels
- `backend/app/core/parser.py` - Parser logic
- `run_full.py` - Pipeline script

## Expected Behavior

Each dialogue should be correctly matched with its corresponding label from `labels.txt`.

## Additional Context

The parser was recently modified to add `DIALOGUE_INLINE` pattern to catch dialogues that appear in the middle of lines. This may have introduced the alignment issue.

@nightt5789 Please help investigate and fix this alignment issue.
