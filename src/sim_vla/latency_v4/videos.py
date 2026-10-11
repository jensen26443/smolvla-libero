"""A fixed 75-second comparison, retaining failures and cached-image provenance."""
import json

import numpy as np

from ..latency_v2.integrity import file_sha256
from ..latency_v2.storage import canonical_hash, publish, trial_key
from ..tasks import TASK_BY_KEY
from .continuation import AUTHORIZATION
from .layout import DOCS, ROOT, RUN, read
from .schema import METHODS


def build_demo(values):
    formal = len(values["formal_async"]) == 240
    continuing = (RUN/AUTHORIZATION).exists()
    stage = "formal_async" if formal else "develop_period"
    selected = [row for row in values[stage] if row["init_index"] == 0
                and (formal or row["config"]["period"] == 25)]
    if len(selected) != 12:
        return {"available": False, "reason": "preselected four-method clips are incomplete; no score-selected substitute"}
    inputs, clips = {}, {}
    for row in selected:
        identity = {key: row[key] for key in ("task", "init_index", "state_id", "seed", "latency", "profile", "config")}
        directory = RUN/stage/"trials"/trial_key(identity)
        video = directory/"episode.mp4"
        source = video if video.exists() else directory/"initial_no_control.png"
        if not source.exists():
            return {"available": False, "reason": "a preselected raw control/initial clip is missing"}
        key = row["task"], row["config"]["method"]
        clips[key] = row, directory, source
        for path in (source, directory/"result.json", directory/"events.jsonl"):
            inputs[str(path.relative_to(ROOT))] = file_sha256(path)
    inputs[str((ROOT/"src/sim_vla/latency_v4/videos.py").relative_to(ROOT))] = file_sha256(
        ROOT/"src/sim_vla/latency_v4/videos.py")
    identity = canonical_hash(inputs)
    metadata_path, destination = DOCS/"video_validation.json", DOCS/"demo.mp4"
    if metadata_path.exists() and destination.exists():
        metadata = read(metadata_path)
        if metadata.get("input_sha256") == identity and file_sha256(destination) == metadata["video_sha256"]:
            return metadata
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont

    font_root = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font, small = ImageFont.truetype(font_root, 22), ImageFont.truetype(font_root, 16)
    writer = imageio.get_writer(destination, fps=20, codec="libx264", quality=7, macro_block_size=1)
    cached = {}
    try:
        for task in TASK_BY_KEY:
            for method in METHODS:
                row, directory, source = clips[(task, method)]
                if source.suffix == ".mp4":
                    reader = imageio.get_reader(source)
                    try:
                        frames = [Image.fromarray(frame).resize((240, 240)) for frame in reader]
                    finally:
                        reader.close()
                else:
                    frames = [Image.open(source).convert("RGB").resize((240, 240))]
                events = [json.loads(line) for line in (directory/"events.jsonl").read_text().splitlines()]
                steps = [event for event in events if event["event"] == "step"]
                decisions = [event for event in events if event["event"] in ("activate", "late_discard", "expired_discard")]
                cached[method] = row, frames, steps, decisions
            for frame_index in range(500):
                canvas = Image.new("RGB", (1280, 720), (18, 23, 34))
                draw = ImageDraw.Draw(canvas)
                label = "FORMAL: fresh state rank 0" if formal else "DEVELOPMENT: exposed init0, nominal K25"
                if continuing and formal:
                    label = "AUTHORIZED CONTINUATION: fresh rank 0 / original gate failed"
                draw.text((22, 10), "PLAN-06 / "+task.replace("goal_bowl_", "bowl -> "), font=font, fill="white")
                draw.text((22, 40), label+" / all four methods retained / encoded 20 fps", font=small, fill="#c0cbdc")
                for method_index, method in enumerate(METHODS):
                    row, frames, steps, decisions = cached[method]
                    x, y = 22+(method_index % 2)*630, 68+(method_index // 2)*316
                    index = min(frame_index, len(frames)-1)
                    canvas.paste(frames[index], (x, y+28))
                    state = "SUCCESS" if row["success"] else row["termination_reason"]
                    ended = frame_index >= len(frames)
                    draw.text((x, y), f"{method} / {row['config']['sampling_steps']} steps / K{row['config']['period']}"
                              +(f" / END: {state}" if ended else ""), font=small, fill="white")
                    tick = min(frame_index, max(0, row["steps"]-1))
                    step = steps[tick] if steps else {}
                    image_tick = step.get("sampling", {}).get("image", {}).get("tick", tick+1)
                    eligible = [event for event in decisions if event["tick"] <= tick]
                    last = eligible[-1] if eligible else {}
                    accepted = sum(event.get("catchup_accepted", False) for event in eligible)
                    facts = ["Policy camera (cached)", f"Source: {step.get('source', 'initial')}",
                        f"Source index / age: {step.get('index', '-')} / {step.get('age', '-')}",
                        f"Last result age: {last.get('receive_age', last.get('age', '-'))} ticks",
                        f"Last late: {last.get('late_ticks', 0)} ticks",
                        f"Catch-up accepted: {last.get('catchup_accepted', False)}",
                        f"Accepted so far: {accepted}", f"Outcome: {state}"]
                    for line_index, fact in enumerate(facts):
                        draw.text((x+254, y+36+line_index*26), fact, font=small, fill="#c0cbdc")
                    draw.text((x, y+270), f"tick {tick} / actual {row['actual_control_hz']:.2f} Hz (episode mean)",
                              font=small, fill="#c0cbdc")
                    draw.text((x, y+290), f"Image age: {tick+1-image_tick} ticks / end marker retains failures",
                              font=small, fill="#c0cbdc")
                writer.append_data(np.asarray(canvas))
            cached.clear()
    finally:
        writer.close()
    reader = imageio.get_reader(destination)
    try:
        decoded = sum(1 for _ in reader)
        video_metadata = reader.get_meta_data()
    finally:
        reader.close()
    if decoded != 1500:
        raise ValueError("demo full decode count differs from 75 seconds at 20 fps")
    metadata = {"available": True, "stage": stage, "formal": formal, "authorized_continuation": continuing,
        "duration_s": 75, "encoded_fps": 20,
        "decoded_frames": decoded, "dimensions": list(video_metadata["size"]), "input_sha256": identity,
        "video_sha256": file_sha256(destination), "input_inventory": inputs,
        "selection": "each task rank0, all four methods; exposed init0 K25 only if no complete formal matrix",
        "actual_hz_annotation": "measured episode control Hz; encoding fps is not control frequency",
        "age_annotations": "source index/age is the active plan or pending commitment window; last result age is receipt minus its t_obs; image age is actual camera freshness"}
    publish(metadata_path, metadata)
    return metadata
