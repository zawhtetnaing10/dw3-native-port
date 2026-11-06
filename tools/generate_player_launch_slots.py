"""Mark player launch operands and align projectile spawn with launch sound.

Each marked operand becomes the negative of its original signed 16-bit ID.
The C interpreter uses the positive ID as the per-fighter, per-script fallback.
Where a post-spawn wait precedes the fixed launch sound, move that same wait
before the entire model/sprite launch block and negate its duration as a
runtime timing marker. C uses the selected technique's timing mode to wait
there (aligned) or defer that same wait until after the launch (original).
Script length and later timing stay unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from struct import pack_into, unpack_from


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data/files"
DEFAULT_INVENTORY = ROOT / "config/player_script_baselines.json"
DEFAULT_OUTPUT = ROOT / "build/launch-slots"
SCRIPT_INDICES = (9, 13)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def within(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def mark_command(source: bytes, output: bytearray, used: set[int],
                 start: int, opcode: int, mode: int | None, original: int,
                 script_start: int, script_end: int, label: str) -> None:
    """Validate a recorded command and mark only its ID operand."""
    operand = start + (4 if mode is not None else 2)
    require(script_start <= start < operand + 2 <= script_end,
            f"{label}: command outside script at {start:#x}")
    require(operand not in used, f"{label}: duplicate operand at {operand:#x}")
    require(0 < original <= 32767, f"{label}: invalid original ID {original}")
    found_opcode = unpack_from("<h", source, start)[0]
    require(found_opcode == opcode,
            f"{label}: expected opcode {opcode} at {start:#x}, found {found_opcode}")
    if mode is not None:
        found_mode = unpack_from("<h", source, start + 2)[0]
        require(found_mode == mode,
                f"{label}: expected mode {mode} at {start:#x}, found {found_mode}")
    found_id = unpack_from("<h", source, operand)[0]
    require(found_id == original,
            f"{label}: expected ID {original} at {operand:#x}, found {found_id}")
    pack_into("<h", output, operand, -original)
    used.add(operand)


def move_launch_wait(output: bytearray, script: dict, label: str) -> tuple[int, set[int]]:
    """Move the existing post-spawn wait before all launch model/sprite pairs."""
    sound = script["original_launch_sound"]
    wait = sound["wait_immediately_before"]
    spawns = script["spawns"]
    first = spawns[0]["model_spawn_offset"]
    if wait is None:
        require(len(spawns) == 1 and sound["file_offset"] == first + 30,
                f"{label}: unexpected launch block without a post-spawn wait")
        return 0, set()

    wait_at = wait["file_offset"]
    require(script["file_start"] <= first < wait_at and
            wait_at + 4 == sound["file_offset"] <= script["file_end"] - 6,
            f"{label}: wait is not directly before the launch sound")
    require(wait_at - first == 30 * len(spawns),
            f"{label}: unexpected commands inside model/sprite launch block")
    for i, spawn in enumerate(spawns):
        require(spawn["model_spawn_offset"] == first + 30 * i and
                spawn["sprite_play_offset"] == first + 30 * i + 18,
                f"{label}: model and sprite commands are not adjacent pairs")
    require(unpack_from("<hh", output, wait_at) == (11, wait["frames"]) and
            wait["frames"] > 0,
            f"{label}: unexpected post-spawn wait command")

    launches = bytes(output[first:wait_at])
    wait_command = bytes(output[wait_at:wait_at + 4])
    output[first:wait_at + 4] = wait_command + launches
    require(output[first:first + 4] == wait_command and
            output[first + 4:wait_at + 4] == launches,
            f"{label}: launch block did not move intact")
    pack_into("<h", output, first + 2, -wait["frames"])
    return 1, set(range(first, wait_at + 4))


def verify_switchable_launch(output: bytes, script: dict, label: str) -> None:
    """Decode the generated launch block and its technique-timing marker."""
    sound = script["original_launch_sound"]
    wait = sound["wait_immediately_before"]
    shift = 4 if wait is not None else 0
    first = script["spawns"][0]["model_spawn_offset"]
    if wait is not None:
        require(unpack_from("<hh", output, first) == (11, -wait["frames"]),
                f"{label}: relocated wait is not before the first projectile")
    for spawn in script["spawns"]:
        model_at = spawn["model_spawn_offset"] + shift
        sprite_at = spawn["sprite_play_offset"] + shift
        require(unpack_from("<hhh", output, model_at) ==
                (6, 0, -script["original_projectile_model"]),
                f"{label}: marked model spawn was not preserved")
        require(unpack_from("<hhh", output, sprite_at) ==
                (3, 0, -script["original_launch_sprite"]),
                f"{label}: marked sprite play was not preserved")
    require(unpack_from("<hhh", output, sound["file_offset"]) ==
            (10, -sound["id"], sound["duration"]),
            f"{label}: launch sound moved or changed")
    require(script["spawns"][-1]["sprite_play_offset"] + shift + 12 ==
            sound["file_offset"],
            f"{label}: launch sound is not adjacent to the final sprite play")


def generate_file(source: bytes, fighter: dict) -> tuple[bytes, Counter]:
    name = fighter["name"]
    require(sha256(source) == fighter["source_file_sha256"],
            f"{name}: original archive hash differs from inventory")
    output = bytearray(source)
    used: set[int] = set()
    counts: Counter = Counter()

    for index in SCRIPT_INDICES:
        script = fighter["scripts"][str(index)]
        start, end = script["file_start"], script["file_end"]
        label = f"{name} script {index}"
        require(0 <= start < end <= len(source) and (end - start) % 2 == 0,
                f"{label}: invalid bounds {start:#x}..{end:#x}")
        require(sha256(source[start:end]) == script["script_sha256"],
                f"{label}: original script hash differs from inventory")

        model = script["original_projectile_model"]
        sprite = script["original_launch_sprite"]
        sound = script["original_launch_sound"]
        mark_command(source, output, used, script["model_preload_offset"],
                     6, 1, model, start, end, label)
        counts["model_preloads"] += 1
        mark_command(source, output, used, script["sprite_preload_offset"],
                     3, 1, sprite, start, end, label)
        counts["sprite_preloads"] += 1
        for spawn in script["spawns"]:
            mark_command(source, output, used, spawn["model_spawn_offset"],
                         6, 0, model, start, end, label)
            counts["model_spawns"] += 1
            mark_command(source, output, used, spawn["sprite_play_offset"],
                         3, 0, sprite, start, end, label)
            counts["sprite_plays"] += 1
        sound_offset = sound["file_offset"]
        mark_command(source, output, used, sound_offset, 10, None,
                     sound["id"], start, end, label)
        require(unpack_from("<h", source, sound_offset + 4)[0] == sound["duration"],
                f"{label}: launch sound duration changed")
        counts["launch_sounds"] += 1

    # Before moving any command, the only changes are the recorded ID operands.
    allowed = {offset + byte for offset in used for byte in (0, 1)}
    changed = {i for i, (before, after) in enumerate(zip(source, output))
               if before != after}
    require(len(output) == len(source) and changed <= allowed,
            f"{name}: unexpected byte changes")
    require(all(unpack_from("<h", output, offset)[0]
                == -unpack_from("<h", source, offset)[0] for offset in used),
            f"{name}: a marker was not written")

    for index in SCRIPT_INDICES:
        script = fighter["scripts"][str(index)]
        moved, touched = move_launch_wait(output, script, f"{name} script {index}")
        verify_switchable_launch(output, script, f"{name} script {index}")
        counts["relocated_waits"] += moved
        counts["already_aligned"] += 1 - moved
        allowed.update(touched)
    changed = {i for i, (before, after) in enumerate(zip(source, output))
               if before != after}
    require(len(output) == len(source) and changed <= allowed,
            f"{name}: unexpected changes after wait relocation")
    return bytes(output), counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true",
                        help="verify existing output files without writing")
    args = parser.parse_args()

    source_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    active_mods = (ROOT / "data/mods").resolve()
    require(not within(output_dir, source_dir) and not within(output_dir, active_mods),
            "output directory must be separate from original files and active mods")
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    require(inventory["schema_version"] == 1, "unsupported inventory schema")
    fighters = inventory["fighters"]
    require(len(fighters) == 44, f"expected 44 fighters, found {len(fighters)}")
    require(sorted(fighter["kind"] for fighter in fighters) == list(range(9, 53)),
            "missing or repeated playable fighter kind")
    file_ids = [fighter["effects_file_id"] for fighter in fighters]
    require(len(set(file_ids)) == 44, "fighters unexpectedly share effects files")

    generated: list[tuple[Path, bytes]] = []
    counts: Counter = Counter()
    for fighter in fighters:
        filename = f"{fighter['effects_file_id']:04d}.bin"
        source = (source_dir / filename).read_bytes()
        result, file_counts = generate_file(source, fighter)
        counts.update(file_counts)
        generated.append((output_dir / filename, result))

    if args.check:
        found = {path.name for path in output_dir.glob("*.bin")}
        expected = {path.name for path, _ in generated}
        require(found == expected,
                f"output file set differs: missing={sorted(expected - found)}, "
                f"extra={sorted(found - expected)}")
        for path, result in generated:
            require(path.read_bytes() == result, f"generated archive differs: {path}")
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        for path, result in generated:
            path.write_bytes(result)

    print(f"{'Verified' if args.check else 'Generated'} {len(generated)} inactive archives "
          f"for {len(fighters) * len(SCRIPT_INDICES)} scripts in {output_dir}")
    print(", ".join(f"{key}={counts[key]}" for key in
                    ("model_preloads", "model_spawns", "sprite_preloads",
                     "sprite_plays", "launch_sounds", "relocated_waits",
                     "already_aligned")))


if __name__ == "__main__":
    main()
