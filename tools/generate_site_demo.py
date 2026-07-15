from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

WIDTH = 960
HEIGHT = 540
FPS = 10
FRAMES_PER_STAGE = 10
STAGES = (
    "Profile traffic",
    "Build plans",
    "Measure",
    "Validate",
    "Select",
    "Cache",
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "site" / "assets" / "optimizer-demo-v3.gif"
POSTER = ROOT / "site" / "assets" / "optimizer-demo-v3-poster.webp"

COLORS = {
    "background": "#09100d",
    "panel": "#111a16",
    "panel_alt": "#15211b",
    "line": "#2c3a33",
    "white": "#f4f7f5",
    "muted": "#8fa198",
    "muted_dark": "#63756c",
    "green": "#42d680",
    "green_dark": "#173b29",
    "cyan": "#35b8c5",
    "cyan_dark": "#17343a",
    "coral": "#e77461",
    "coral_dark": "#3d211d",
    "amber": "#dfac43",
}

FONT_FALLBACKS = {
    "segoeui.ttf": (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "DejaVuSans.ttf",
    ),
    "seguisb.ttf": (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "DejaVuSans-Bold.ttf",
    ),
    "consola.ttf": (
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "DejaVuSansMono.ttf",
    ),
    "consolab.ttf": (
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
        "DejaVuSansMono-Bold.ttf",
    ),
}


def font(name: str, size: int) -> ImageFont.FreeTypeFont:
    candidates = (str(Path("C:/Windows/Fonts") / name), *FONT_FALLBACKS[name])
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    raise RuntimeError(f"Could not find a usable font for {name}")


SANS = font("segoeui.ttf", 16)
SANS_SMALL = font("segoeui.ttf", 13)
SANS_BOLD = font("seguisb.ttf", 16)
SANS_BOLD_SMALL = font("seguisb.ttf", 13)
MONO = font("consola.ttf", 13)
MONO_SMALL = font("consola.ttf", 11)
MONO_BOLD = font("consolab.ttf", 13)


def ease(value: float) -> float:
    value = min(1.0, max(0.0, value))
    return 1 - (1 - value) ** 3


def rounded(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    fill: str,
    *,
    outline: str | None = None,
    radius: int = 6,
    width: int = 1,
) -> None:
    draw.rounded_rectangle(
        box,
        radius=radius,
        fill=fill,
        outline=outline,
        width=width,
    )


def text(
    draw: ImageDraw.ImageDraw,
    position: tuple[int, int],
    value: str,
    fill: str,
    selected_font: ImageFont.FreeTypeFont = SANS,
    *,
    anchor: str | None = None,
) -> None:
    draw.text(position, value, fill=fill, font=selected_font, anchor=anchor)


def stage_progress(draw: ImageDraw.ImageDraw, stage: int, progress: float) -> None:
    y = 508
    start_x = 34
    available = WIDTH - 68
    segment = available / len(STAGES)
    for index, label in enumerate(STAGES):
        x = start_x + index * segment
        text(
            draw,
            (int(x), y),
            f"{index + 1:02d}  {label}",
            COLORS["white"] if index == stage else COLORS["muted_dark"],
            MONO_SMALL,
        )
        line_start = int(x)
        line_end = int(x + segment - 12)
        draw.line(
            (line_start, y - 10, line_end, y - 10),
            fill=COLORS["line"],
            width=3,
        )
        if index < stage:
            fill_end = line_end
        elif index == stage:
            fill_end = int(line_start + (line_end - line_start) * progress)
        else:
            fill_end = line_start
        if fill_end > line_start:
            draw.line(
                (line_start, y - 10, fill_end, y - 10),
                fill=COLORS["green"],
                width=3,
            )


def workload_panel(draw: ImageDraw.ImageDraw, stage: int, progress: float) -> None:
    rounded(draw, (26, 68, 270, 472), COLORS["panel"], outline=COLORS["line"])
    text(draw, (46, 91), "WORKLOAD PROFILE", COLORS["green"], MONO_BOLD)
    text(draw, (46, 119), "serving-mix", COLORS["white"], SANS_BOLD)
    text(draw, (46, 143), "50,000 expected calls", COLORS["muted"], SANS_SMALL)

    cases = (("batch-1", 70, COLORS["green"]), ("batch-8", 25, COLORS["cyan"]), ("batch-32", 5, COLORS["amber"]))
    for index, (label, weight, color) in enumerate(cases):
        y = 195 + index * 72
        text(draw, (46, y), label, COLORS["white"], MONO)
        text(draw, (246, y), f"{weight}%", COLORS["muted"], MONO, anchor="ra")
        rounded(draw, (46, y + 27, 246, y + 35), "#26332d", radius=3)
        reveal = ease(progress) if stage == 0 else 1.0
        width = max(2, int(200 * weight / 70 * reveal))
        rounded(draw, (46, y + 27, 46 + width, y + 35), color, radius=3)

    rounded(draw, (46, 405, 246, 446), COLORS["panel_alt"], outline=COLORS["line"])
    text(draw, (62, 419), "shape-aware", COLORS["cyan"], MONO_SMALL)
    text(draw, (62, 435), "weighted selection", COLORS["muted"], MONO_SMALL)


def candidate_status(stage: int, index: int) -> tuple[str, str, str]:
    if stage < 1:
        return "queued", COLORS["muted_dark"], COLORS["panel_alt"]
    if stage == 1:
        return "building", COLORS["cyan"], COLORS["cyan_dark"]
    if stage == 2:
        return "measuring", COLORS["amber"], "#3a301b"
    if stage >= 3 and index == 3:
        return "parity failed", COLORS["coral"], COLORS["coral_dark"]
    if stage >= 4 and index == 2:
        return "selected", COLORS["green"], COLORS["green_dark"]
    if stage >= 3:
        return "valid", COLORS["muted"], COLORS["panel_alt"]
    return "measuring", COLORS["amber"], "#3a301b"


def candidate_panel(draw: ImageDraw.ImageDraw, stage: int, progress: float) -> None:
    rounded(draw, (288, 68, 934, 400), COLORS["panel"], outline=COLORS["line"])
    text(draw, (310, 91), "CANDIDATE RACE", COLORS["green"], MONO_BOLD)
    text(draw, (912, 91), "median latency", COLORS["muted_dark"], MONO_SMALL, anchor="ra")

    candidates = (
        ("Eager FP32", 100.0, "#76877f"),
        ("Native AMP", 43.0, COLORS["cyan"]),
        ("FX + Inductor", 39.8, COLORS["green"]),
        ("External provider", 35.4, COLORS["coral"]),
    )
    for index, (name, latency, color) in enumerate(candidates):
        y = 132 + index * 62
        if index:
            draw.line((310, y - 10, 912, y - 10), fill=COLORS["line"], width=1)
        status, status_color, status_background = candidate_status(stage, index)
        if stage >= 4 and index == 2:
            rounded(draw, (300, y - 7, 922, y + 46), COLORS["green_dark"], radius=4)
        text(draw, (310, y + 2), name, COLORS["white"], SANS_BOLD_SMALL)
        rounded(draw, (310, y + 28, 667, y + 36), "#27352e", radius=3)
        if stage >= 2:
            measure_progress = ease(progress) if stage == 2 else 1.0
            bar_width = max(3, int(357 * latency / 100 * measure_progress))
            rounded(draw, (310, y + 28, 310 + bar_width, y + 36), color, radius=3)
            shown_latency = latency / max(measure_progress, 0.22)
            shown_latency = max(latency, min(180.0, shown_latency))
            value = f"{shown_latency:05.1f} ms"
        else:
            build_progress = ease(progress) if stage == 1 else 0.0
            if stage == 1:
                rounded(
                    draw,
                    (310, y + 28, 310 + int(357 * build_progress), y + 36),
                    COLORS["cyan_dark"],
                    radius=3,
                )
            value = "--.- ms"
        text(draw, (756, y + 2), value, COLORS["muted"], MONO, anchor="ra")
        status_width = max(82, int(draw.textlength(status, font=MONO_SMALL)) + 22)
        rounded(draw, (912 - status_width, y + 24, 912, y + 43), status_background, radius=4)
        text(draw, (901, y + 28), status, status_color, MONO_SMALL, anchor="ra")


def result_panel(draw: ImageDraw.ImageDraw, stage: int, progress: float) -> None:
    rounded(draw, (288, 414, 934, 472), COLORS["panel_alt"], outline=COLORS["line"])
    if stage < 3:
        messages = (
            "Reading representative signatures",
            "Building candidates in isolation",
            "Synchronizing and collecting serial samples",
        )
        icon_color = COLORS["cyan"]
        message = messages[min(stage, 2)]
        detail = "Parity gate and deployment policy remain armed"
    elif stage == 3:
        icon_color = COLORS["coral"]
        message = "External provider rejected"
        detail = "Fastest raw latency, but output parity exceeded tolerance"
    elif stage == 4:
        icon_color = COLORS["green"]
        message = "FX + Inductor selected"
        detail = "2.51x vs eager | parity passed | native fallback ready"
    else:
        icon_color = COLORS["green"]
        message = "Decision cached and ready"
        detail = "Next launch revalidates parity, constraints, and latency"

    pulse = 4 + int(2 * (0.5 + 0.5 * math.sin(progress * math.pi * 2)))
    draw.ellipse((308 - pulse, 443 - pulse, 308 + pulse, 443 + pulse), fill=icon_color)
    text(draw, (328, 424), message, COLORS["white"], SANS_BOLD_SMALL)
    text(draw, (328, 445), detail, COLORS["muted"], SANS_SMALL)
    if stage == 5:
        text(draw, (912, 436), "key  91ac...7f2", COLORS["green"], MONO_SMALL, anchor="ra")


def render_frame(frame_index: int) -> Image.Image:
    stage = min(len(STAGES) - 1, frame_index // FRAMES_PER_STAGE)
    progress = (frame_index % FRAMES_PER_STAGE + 1) / FRAMES_PER_STAGE
    image = Image.new("RGB", (WIDTH, HEIGHT), COLORS["background"])
    draw = ImageDraw.Draw(image)

    text(draw, (28, 24), "CDL", COLORS["green"], MONO_BOLD)
    text(draw, (68, 24), "/ LIVE PLAN SELECTION", COLORS["white"], MONO)
    text(draw, (932, 24), "CUDA  /  SERVING-MIX", COLORS["muted"], MONO_SMALL, anchor="ra")

    workload_panel(draw, stage, progress)
    candidate_panel(draw, stage, progress)
    result_panel(draw, stage, progress)
    stage_progress(draw, stage, progress)
    return image


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    frames = [render_frame(index) for index in range(len(STAGES) * FRAMES_PER_STAGE)]
    poster = frames[-1]
    poster.save(POSTER, "WEBP", quality=90, method=6)

    palette = frames[0].quantize(colors=128, method=Image.Quantize.MEDIANCUT)
    quantized = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]
    quantized[0].save(
        OUTPUT,
        save_all=True,
        append_images=quantized[1:],
        duration=round(1000 / FPS),
        loop=0,
        optimize=True,
        disposal=2,
    )
    print(f"Generated {OUTPUT} ({OUTPUT.stat().st_size / 1024:.1f} KiB)")
    print(f"Generated {POSTER} ({POSTER.stat().st_size / 1024:.1f} KiB)")


if __name__ == "__main__":
    main()
