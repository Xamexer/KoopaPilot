"""Generate the README architecture diagram with OpenCV.

Run from the repository root:
    uv run python ./docs/generate_architecture_diagram.py
"""

from pathlib import Path

import cv2
import numpy as np


WIDTH = 1800
HEIGHT = 1080
OUTPUT = Path(__file__).parent / "images" / "architecture.png"

BG = (250, 249, 247)
INK = (55, 63, 75)
MUTED = (125, 132, 142)
LINE = (194, 198, 204)
BLUE = (196, 116, 42)
BLUE_FILL = (248, 235, 220)
GREEN = (94, 137, 55)
GREEN_FILL = (232, 246, 224)
PURPLE = (147, 80, 112)
PURPLE_FILL = (245, 231, 239)
AMBER = (42, 143, 204)
AMBER_FILL = (224, 244, 255)
TEAL = (145, 129, 25)
TEAL_FILL = (231, 247, 245)

FONT = cv2.FONT_HERSHEY_SIMPLEX


def rounded_box(image, rect, fill, border, radius=22, thickness=3):
    x1, y1, x2, y2 = rect
    cv2.rectangle(image, (x1 + radius, y1), (x2 - radius, y2), fill, -1)
    cv2.rectangle(image, (x1, y1 + radius), (x2, y2 - radius), fill, -1)
    for center in (
        (x1 + radius, y1 + radius),
        (x2 - radius, y1 + radius),
        (x1 + radius, y2 - radius),
        (x2 - radius, y2 - radius),
    ):
        cv2.circle(image, center, radius, fill, -1)
    cv2.line(image, (x1 + radius, y1), (x2 - radius, y1), border, thickness)
    cv2.line(image, (x1 + radius, y2), (x2 - radius, y2), border, thickness)
    cv2.line(image, (x1, y1 + radius), (x1, y2 - radius), border, thickness)
    cv2.line(image, (x2, y1 + radius), (x2, y2 - radius), border, thickness)
    cv2.ellipse(image, (x1 + radius, y1 + radius), (radius, radius), 0, 180, 270, border, thickness)
    cv2.ellipse(image, (x2 - radius, y1 + radius), (radius, radius), 0, 270, 360, border, thickness)
    cv2.ellipse(image, (x1 + radius, y2 - radius), (radius, radius), 0, 90, 180, border, thickness)
    cv2.ellipse(image, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, border, thickness)


def centered_text(image, text, center, scale, color=INK, thickness=1):
    size, baseline = cv2.getTextSize(text, FONT, scale, thickness)
    x = int(center[0] - size[0] / 2)
    y = int(center[1] + (size[1] - baseline) / 2)
    cv2.putText(image, text, (x, y), FONT, scale, color, thickness, cv2.LINE_AA)


def box(image, rect, title, subtitle, fill, border):
    rounded_box(image, rect, fill, border)
    cx = (rect[0] + rect[2]) // 2
    cy = (rect[1] + rect[3]) // 2
    centered_text(image, title, (cx, cy - 15), 0.72, INK, 2)
    centered_text(image, subtitle, (cx, cy + 25), 0.48, MUTED, 1)


def arrow(image, start, end, color=LINE, thickness=3, dashed=False):
    start = np.array(start, dtype=float)
    end = np.array(end, dtype=float)
    direction = end - start
    length = float(np.linalg.norm(direction))
    if length == 0:
        return
    unit = direction / length
    arrow_end = end - unit * 15
    if dashed:
        position = 0.0
        while position < length - 20:
            segment_end = min(position + 11, length - 20)
            a = start + unit * position
            b = start + unit * segment_end
            cv2.line(image, tuple(a.astype(int)), tuple(b.astype(int)), color, thickness, cv2.LINE_AA)
            position += 20
    else:
        cv2.line(
            image,
            tuple(start.astype(int)),
            tuple(arrow_end.astype(int)),
            color,
            thickness,
            cv2.LINE_AA,
        )
    normal = np.array([-unit[1], unit[0]])
    tip = end.astype(int)
    left = (end - unit * 18 + normal * 8).astype(int)
    right = (end - unit * 18 - normal * 8).astype(int)
    cv2.fillConvexPoly(image, np.array([tip, left, right]), color, cv2.LINE_AA)


def label(image, text, center, color=MUTED):
    centered_text(image, text.upper(), center, 0.42, color, 1)


def main():
    image = np.full((HEIGHT, WIDTH, 3), BG, dtype=np.uint8)

    centered_text(image, "KoopaPilot architecture", (WIDTH // 2, 40), 1.15, INK, 2)
    centered_text(
        image,
        "One policy and reward contract, two interchangeable emulator backends",
        (WIDTH // 2, 76),
        0.50,
        MUTED,
        1,
    )

    box(image, (640, 105, 1160, 180), "CLI + config.json", "mode and runtime settings", BLUE_FILL, BLUE)
    box(image, (560, 240, 1240, 340), "Mode orchestration", "server.main", BLUE_FILL, BLUE)
    box(image, (590, 400, 1210, 505), "Stable-Baselines3 PPO", "TrainingManager + shared reward logic", BLUE_FILL, BLUE)

    label(image, "live-demo only", (250, 220), AMBER)
    box(
        image,
        (45, 245, 455, 335),
        "Model checkpoints",
        "immutable model_live generations",
        AMBER_FILL,
        AMBER,
    )
    box(image, (45, 410, 455, 505), "Deterministic viewer", "DemoManager + visible BizHawk", AMBER_FILL, AMBER)

    label(image, "observability", (1550, 220), TEAL)
    box(image, (1345, 245, 1755, 335), "JSON metrics", "training + viewer episodes", TEAL_FILL, TEAL)
    box(image, (1345, 410, 1755, 505), "Flask dashboard", "browser charts and comparisons", TEAL_FILL, TEAL)

    label(image, "backend A", (450, 580), GREEN)
    box(image, (145, 605, 755, 705), "BizHawk adapter", "SMWVecEnv + TCP socket server", GREEN_FILL, GREEN)
    box(image, (145, 800, 755, 900), "BizHawk + Lua agent", "WRAM reads, overlays, controller input", GREEN_FILL, GREEN)

    label(image, "backend B", (1350, 580), PURPLE)
    box(image, (1045, 605, 1655, 705), "RetroJet adapter", "RetroJetVecEnv + native Python API", PURPLE_FILL, PURPLE)
    box(image, (1045, 800, 1655, 900), "libretro + Snes9x", "headless native batched stepping", PURPLE_FILL, PURPLE)

    rounded_box(image, (275, 965, 1525, 1040), (244, 244, 242), LINE, radius=18, thickness=2)
    centered_text(
        image,
        "Shared contract: normalized observations  |  12 discrete actions  |  episode rewards",
        (900, 1003),
        0.56,
        INK,
        1,
    )

    arrow(image, (900, 180), (900, 240), BLUE)
    arrow(image, (900, 340), (900, 400), BLUE)
    arrow(image, (720, 505), (450, 605), GREEN)
    arrow(image, (1080, 505), (1350, 605), PURPLE)
    arrow(image, (450, 705), (450, 800), GREEN)
    arrow(image, (1350, 705), (1350, 800), PURPLE)
    arrow(image, (450, 900), (570, 965), GREEN)
    arrow(image, (1350, 900), (1230, 965), PURPLE)

    arrow(image, (590, 450), (455, 290), AMBER, dashed=True)
    arrow(image, (250, 335), (250, 410), AMBER, dashed=True)
    arrow(image, (1210, 450), (1345, 290), TEAL)
    arrow(image, (1550, 335), (1550, 410), TEAL)

    label(image, "TCP sockets", (450, 755), GREEN)
    label(image, "native calls", (1350, 755), PURPLE)
    label(image, "files", (345, 375), AMBER)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(OUTPUT), image):
        raise RuntimeError(f"Could not write {OUTPUT}")
    print(f"Wrote {OUTPUT} ({WIDTH}x{HEIGHT})")


if __name__ == "__main__":
    main()
