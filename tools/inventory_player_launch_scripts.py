"""Inventory original EU player battle-script launch commands.

Reads only data/files/ from the user's extracted disc. The JSON report holds
IDs, command positions, and hashes; it contains no model, sprite, or sound
assets. Run with --check after generation to detect source/report drift.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from struct import unpack_from


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data/files"
DEFAULT_OUTPUT = ROOT / "config/player_script_baselines.json"
PARTNER_LIST_OFFSET = 0x10
PARTNER_RECORDS_OFFSET = 0x2458
PARTNER_RECORD_SIZE = 0xC4
EFFECTS_POINTER_OFFSET = 8
PARTNER_COUNT = 55
FIRST_KIND = 9
LAST_KIND = 52
SCRIPT_INDICES = (9, 13)
SCRIPT_TABLE_ENTRIES = 19


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def u32(blob: bytes, offset: int) -> int:
    require(0 <= offset <= len(blob) - 4, f"u32 outside file at {offset:#x}")
    return unpack_from("<I", blob, offset)[0]


def s16_words(blob: bytes, start: int, end: int) -> list[int]:
    require(0 <= start < end <= len(blob) and (end - start) % 2 == 0,
            f"invalid script bounds {start:#x}..{end:#x}")
    return list(unpack_from(f"<{(end - start) // 2}h", blob, start))


def name_from_table(table: bytes, kind: int) -> str:
    start = u32(table, 4 * (kind + 1))
    require(0 <= start < len(table), f"invalid name offset for kind {kind}")
    end = table.find(b"\0", start)
    require(end >= 0, f"unterminated name for kind {kind}")
    chars = []
    for value in table[start:end]:
        if value == 1:
            chars.append(" ")
        elif 4 <= value <= 13:
            chars.append(chr(ord("0") + value - 4))
        elif 14 <= value <= 39:
            chars.append(chr(ord("A") + value - 14))
        elif 40 <= value <= 65:
            chars.append(chr(ord("a") + value - 40))
        else:
            raise ValueError(f"unknown name glyph {value} for kind {kind}")
    return "".join(chars)


def positions(words: list[int], pattern: tuple[int, ...]) -> list[int]:
    return [i for i in range(len(words) - len(pattern) + 1)
            if tuple(words[i:i + len(pattern)]) == pattern]


def one_position(words: list[int], pattern: tuple[int, ...], label: str) -> int:
    found = positions(words, pattern)
    require(len(found) == 1, f"{label}: expected one {pattern}, found {len(found)}")
    return found[0]


def nearby_wait(words: list[int], start: int, command_word: int,
                after: bool = False) -> dict | None:
    wait_word = command_word + 3 if after else command_word - 2
    if wait_word >= 0 and wait_word + 1 < len(words) and words[wait_word] == 11:
        return {"file_offset": start + 2 * wait_word,
                "frames": words[wait_word + 1]}
    return None


def inventory_script(blob: bytes, archive: int, offsets: list[int], index: int) -> dict:
    relative_start = offsets[index]
    next_offsets = [value for value in offsets[index + 1:] if value > relative_start]
    require(relative_start > 0 and bool(next_offsets),
            f"missing script {index} or next archive offset")
    start = archive + relative_start
    end = archive + next_offsets[0]
    words = s16_words(blob, start, end)
    label = f"script {index} at {start:#x}"

    require(len(words) >= 9 and words[:2] == [6, 1] and words[3:5] == [3, 1],
            f"{label}: unexpected launch preloads")
    model = words[2]
    sprite = words[5]
    require(0 < model < 1000 and 1000 <= sprite < 2000,
            f"{label}: unexpected model/sprite {model}/{sprite}")
    require(one_position(words, (6, 1, model), label) == 0,
            f"{label}: model preload moved")
    require(one_position(words, (3, 1, sprite), label) == 3,
            f"{label}: sprite preload moved")

    spawn_words = positions(words, (6, 0, model))
    require(1 <= len(spawn_words) <= 3,
            f"{label}: unexpected model spawn count {len(spawn_words)}")
    spawns = []
    for word in spawn_words:
        require(word + 15 <= len(words) and words[word + 9:word + 12] == [3, 0, sprite],
                f"{label}: model spawn at word {word} lacks adjacent sprite play")
        raw_position = words[word + 3:word + 6]
        raw_rotation = words[word + 6:word + 9]
        spawns.append({
            "model_spawn_offset": start + 2 * word,
            "sprite_play_offset": start + 2 * (word + 9),
            "raw_model_position": raw_position,
            "raw_model_rotation": raw_rotation,
            "sprite_position": words[word + 12:word + 15],
            "wait_immediately_before_first_spawn": (
                nearby_wait(words, start, word) if word == spawn_words[0] else None
            ),
        })

    sound_word = one_position(words, (10, 23, 0), label)
    placeholder_counts = {
        "impact_sprite_preload": len(positions(words, (3, 1, 9999))),
        "impact_sprite_play": len(positions(words, (3, 0, 9999))),
        "stage_preload": len(positions(words, (4, 1, 0x38))),
        "stage_change": len(positions(words, (4, 0, 0x38))),
        "hit_sound_98": len(positions(words, (10, 98, 0))),
        "hit_sound_99": len(positions(words, (10, 99, 0))),
    }
    expected = {
        "impact_sprite_preload": 1,
        "impact_sprite_play": 1 if index == 9 else 3,
        "stage_preload": 1,
        "stage_change": 1,
        "hit_sound_98": 0 if index == 9 else 2,
        "hit_sound_99": 1,
    }
    require(placeholder_counts == expected,
            f"{label}: unexpected dynamic placeholders {placeholder_counts}")

    return {
        "index": index,
        "file_start": start,
        "file_end": end,
        "script_sha256": hashlib.sha256(blob[start:end]).hexdigest(),
        "original_projectile_model": model,
        "model_preload_offset": start,
        "original_launch_sprite": sprite,
        "sprite_preload_offset": start + 6,
        "spawns": spawns,
        "original_launch_sound": {
            "id": words[sound_word + 1],
            "duration": words[sound_word + 2],
            "file_offset": start + 2 * sound_word,
            "wait_immediately_before": nearby_wait(words, start, sound_word),
            "wait_immediately_after": nearby_wait(words, start, sound_word, after=True),
        },
        "dynamic_placeholders": placeholder_counts,
    }


def inventory(data_dir: Path) -> dict:
    fighter_table = (data_dir / "0460.bin").read_bytes()
    name_table = (data_dir / "0080.bin").read_bytes()
    fighters = []
    for list_index in range(PARTNER_COUNT):
        fighter_id, record_index, kind = unpack_from(
            "<hBB", fighter_table, PARTNER_LIST_OFFSET + 4 * list_index)
        if not FIRST_KIND <= kind <= LAST_KIND:
            continue
        pointer_offset = (PARTNER_RECORDS_OFFSET
                          + PARTNER_RECORD_SIZE * record_index
                          + EFFECTS_POINTER_OFFSET)
        effects_pointer = u32(fighter_table, pointer_offset)
        file_id, archive_entry = effects_pointer >> 16, effects_pointer & 0xFFFF
        path = data_dir / f"{file_id:04d}.bin"
        blob = path.read_bytes()
        archive = u32(blob, archive_entry * 4)
        require(0 < archive < len(blob),
                f"invalid archive pointer for {path.name} entry {archive_entry}")
        offsets = [u32(blob, archive + 4 * i) for i in range(SCRIPT_TABLE_ENTRIES)]
        scripts = {str(index): inventory_script(blob, archive, offsets, index)
                   for index in SCRIPT_INDICES}
        fighters.append({
            "kind": kind,
            "name": name_from_table(name_table, kind),
            "fighter_id": fighter_id,
            "effects_file_id": file_id,
            "effects_archive_entry": archive_entry,
            "source_file_sha256": hashlib.sha256(blob).hexdigest(),
            "scripts": scripts,
        })

    fighters.sort(key=lambda fighter: fighter["kind"])
    require(len(fighters) == LAST_KIND - FIRST_KIND + 1,
            f"expected 44 playable fighters, found {len(fighters)}")
    require([fighter["kind"] for fighter in fighters] == list(range(FIRST_KIND, LAST_KIND + 1)),
            "missing or duplicated playable fighter kind")
    require(len({fighter["effects_file_id"] for fighter in fighters}) == len(fighters),
            "playable fighters unexpectedly share effects files")
    return {
        "schema_version": 1,
        "source": "original European disc extraction in data/files; never data/mods",
        "fighter_table_sha256": hashlib.sha256(fighter_table).hexdigest(),
        "name_table_sha256": hashlib.sha256(name_table).hexdigest(),
        "fighters": fighters,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true", help="compare with the saved report without writing")
    args = parser.parse_args()
    result = inventory(args.data_dir)
    rendered = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.check:
        require(args.output.read_text(encoding="utf-8") == rendered,
                f"inventory report differs from {args.output}")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8", newline="\n")

    counts = Counter(len(fighter["scripts"][str(index)]["spawns"])
                     for fighter in result["fighters"] for index in SCRIPT_INDICES)
    print(f"{len(result['fighters'])} fighters, {sum(counts.values())} scripts; "
          + ", ".join(f"{count} with {spawns} spawn(s)" for spawns, count in sorted(counts.items())))
    print(f"{'Verified' if args.check else 'Wrote'} {args.output}")


if __name__ == "__main__":
    main()
