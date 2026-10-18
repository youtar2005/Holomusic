import math
import os
import time
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import numpy as np
import sounddevice as sd

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
FRAME_WIDTH = 1280
FRAME_HEIGHT = 780

ROOTS = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]

# (label on wheel, chord-name suffix, semitone intervals above the root)
QUALITIES = [
    ("maj",  "",     (0, 4, 7)),
    ("min",  "m",    (0, 3, 7)),
    ("7",    "7",    (0, 4, 7, 10)),
    ("maj7", "maj7", (0, 4, 7, 11)),
    ("m7",   "m7",   (0, 3, 7, 10)),
    ("sus2", "sus2", (0, 2, 7)),
    ("sus4", "sus4", (0, 5, 7)),
    ("dim",  "dim",  (0, 3, 6)),
    ("aug",  "aug",  (0, 4, 8)),
    ("add9", "add9", (0, 4, 7, 14)),
    ("6",    "6",    (0, 4, 7, 9)),
    ("m7b5", "m7b5", (0, 3, 6, 10)),
]

WHEEL_RADIUS = 190
WHEEL_INNER = 40
MARGIN = 15

# If True, the last chosen quality stays selected when that hand leaves the
# wheel (so you can play with one hand). If False it falls back to major.
STICKY_QUALITY = True

# Selection stability
CONFIRM_FRAMES = 3        # frames a new sector must be held before switching
LOST_FRAMES = 5           # frames without a hand before releasing
ANGLE_HYSTERESIS = 4.0    # degrees of extra room before leaving a sector
RADIAL_HYSTERESIS = 10    # pixels of extra room at the wheel edges
SMOOTHING = 0.55          # fingertip EMA factor (1 = no smoothing)

# Hand model
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "hand_landmarker.task")
INDEX_TIP = 8
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
    (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]

# Audio
SAMPLE_RATE = 44100
BLOCK_SIZE = 512
MASTER_GAIN = 0.7
PIANO_ATTACK = 0.004      # seconds (hammer strike)
RELEASE = 0.18            # seconds (damper falling on the strings)
SUSTAIN_FLOOR = 0.30      # how much of the tone is held while you keep pointing
                          # (0 = natural piano decay, higher = longer hold)
MAX_VOICES = 24


# --------------------------------------------------------------------------
# Piano synth (additive, runs inside the audio callback)
# --------------------------------------------------------------------------
class Voice:
    """One piano note: inharmonic partials, two slightly detuned strings,
    higher partials decaying faster, and a damper release."""

    def __init__(self, midi):
        self.midi = midi
        f0 = 440.0 * 2 ** ((midi - 69) / 12)
        B = 0.0004                                         # inharmonicity
        n_part = max(2, min(8, int(SAMPLE_RATE * 0.45 / f0)))
        k = np.arange(1, n_part + 1, dtype=np.float64)

        fk = f0 * k * np.sqrt(1 + B * k * k)
        freqs = np.concatenate([fk, fk * 1.0009])           # 2 strings, ~1.5 cents

        # Brighter in the bass, rounder in the treble.
        a = k ** -1.05 * np.exp(-k * max(0, midi - 36) / 120.0)
        self.amps = np.concatenate([a, a]) * 0.5

        tau1 = 0.6 / (1 + 0.6 * (k - 1))                    # fast initial decay
        tau2 = 6.0 / (1 + 0.35 * (k - 1))                   # slow "aftersound"
        self.tau1 = np.concatenate([tau1, tau1])
        self.tau2 = np.concatenate([tau2, tau2])
        floor = SUSTAIN_FLOOR / k
        self.floor = np.concatenate([floor, floor])

        self.inc = 2 * np.pi * freqs / SAMPLE_RATE
        self.phases = np.zeros(len(freqs))
        self.level = 0.20 + 0.15 * max(0.0, (60 - midi) / 24.0)  # bass a bit louder
        self.age = 0                                        # samples since strike
        self.gain = 1.0                                     # damper gain
        self.released = False

    def _env(self, t):
        fast = np.exp(-t / self.tau1)
        slow = np.exp(-t / self.tau2)
        return self.floor + (1 - self.floor) * (0.6 * fast + 0.4 * slow)

    def render(self, frames, n, u):
        t0 = self.age / SAMPLE_RATE
        t1 = (self.age + frames) / SAMPLE_RATE
        e0, e1 = self._env(t0), self._env(t1)
        env = e0[:, None] + (e1 - e0)[:, None] * u[None, :]  # linear within block

        ph = self.phases[:, None] + self.inc[:, None] * n[None, :]
        wave = (self.amps[:, None] * env * np.sin(ph)).sum(axis=0)

        t = (self.age + n) / SAMPLE_RATE
        atk = np.minimum(1.0, t / PIANO_ATTACK)

        g0 = self.gain
        g1 = max(0.0, g0 - frames / (SAMPLE_RATE * RELEASE)) if self.released else 1.0
        damper = np.linspace(g0, g1, frames)
        self.gain = g1

        self.age += frames
        self.phases = (self.phases + self.inc * frames) % (2 * np.pi)
        return self.level * atk * damper * wave


class PianoSynth:
    def __init__(self):
        self.voices = []
        self.requested = frozenset()        # replaced atomically by the main thread
        self._current = frozenset()
        self.stream = sd.OutputStream(
            samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE, channels=1,
            dtype="float32", latency="low", callback=self._callback)

    def start(self):
        self.stream.start()

    def stop(self):
        self.stream.stop()
        self.stream.close()

    def set_notes(self, midi_notes):
        """Hold exactly these MIDI notes (empty = release everything)."""
        self.requested = frozenset(midi_notes)

    def _apply(self, req):
        for v in self.voices:
            if not v.released and v.midi not in req:
                v.released = True                     # damper down
        held = {v.midi for v in self.voices if not v.released}
        for m in sorted(req - held):                  # shared notes keep ringing
            self.voices.append(Voice(m))
        if len(self.voices) > MAX_VOICES:
            self.voices = self.voices[-MAX_VOICES:]
        self._current = req

    def _callback(self, outdata, frames, time_info, status):
        req = self.requested
        if req != self._current:
            self._apply(req)

        n = np.arange(frames, dtype=np.float64)
        u = np.linspace(0.0, 1.0, frames)
        out = np.zeros(frames)
        alive = []
        for v in self.voices:
            out += v.render(frames, n, u)
            if not (v.released and v.gain <= 0.0):
                alive.append(v)
        self.voices = alive
        outdata[:, 0] = np.tanh(out * MASTER_GAIN).astype(np.float32)


def chord_notes(root, quality):
    """MIDI notes for a root (0-11) and quality index: bass note + voicing."""
    rr = root if root < 6 else root - 12           # keep voicing in a tidy range
    notes = {60 + rr + iv for iv in QUALITIES[quality][2]}
    notes.add(36 + rr)                             # bass root
    return frozenset(notes)


# --------------------------------------------------------------------------
# Wheels
# --------------------------------------------------------------------------
class Wheel:
    def __init__(self, center, labels, title):
        self.center = center
        self.labels = labels
        self.title = title
        self.n = len(labels)
        self.step = 360 / self.n
        self.radius = WHEEL_RADIUS
        self.inner = WHEEL_INNER

    def center_angle(self, i):
        return -90 + i * self.step

    def sector_at(self, point):
        dx, dy = point[0] - self.center[0], point[1] - self.center[1]
        r = math.hypot(dx, dy)
        if not (self.inner <= r <= self.radius):
            return None
        angle = math.degrees(math.atan2(dy, dx))
        return int(((angle + 90 + self.step / 2) % 360) / self.step) % self.n

    def polygon(self, i):
        cx, cy = self.center
        c = self.center_angle(i)
        angles = np.linspace(c - self.step / 2, c + self.step / 2, 12)
        outer = [(int(cx + self.radius * math.cos(math.radians(a))),
                  int(cy + self.radius * math.sin(math.radians(a))))
                 for a in angles]
        inner = [(int(cx + self.inner * math.cos(math.radians(a))),
                  int(cy + self.inner * math.sin(math.radians(a))))
                 for a in angles[::-1]]
        return np.array(outer + inner, dtype=np.int32)


class WheelSelector:
    """Turns a (noisy) fingertip position into a stable sector index."""

    def __init__(self, wheel):
        self.wheel = wheel
        self.active = None
        self.candidate = None
        self.count = 0
        self.missing = 0

    def _raw_sector(self, point):
        w = self.wheel
        if self.active is not None:
            dx, dy = point[0] - w.center[0], point[1] - w.center[1]
            r = math.hypot(dx, dy)
            angle = math.degrees(math.atan2(dy, dx))
            d = (angle - w.center_angle(self.active) + 180) % 360 - 180
            if (abs(d) <= w.step / 2 + ANGLE_HYSTERESIS
                    and w.inner - RADIAL_HYSTERESIS <= r
                    <= w.radius + RADIAL_HYSTERESIS):
                return self.active
        return w.sector_at(point)

    def update(self, point):
        if point is None:
            self.missing += 1
            if self.missing >= LOST_FRAMES:
                self.active = None
                self.candidate = None
                self.count = 0
            return self.active

        self.missing = 0
        raw = self._raw_sector(point)
        if raw == self.active:
            self.candidate, self.count = None, 0
        elif raw == self.candidate:
            self.count += 1
            if self.count >= CONFIRM_FRAMES:
                self.active = raw
                self.candidate, self.count = None, 0
        else:
            self.candidate, self.count = raw, 1
        return self.active


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------
COLOR_IDLE = (90, 90, 90)
COLOR_ACTIVE = (0, 200, 0)
COLOR_REMEMBERED = (170, 120, 0)     # BGR: dim blue for the sticky quality


def draw_wheels(frame, specs):
    """specs: list of (wheel, active_index, remembered_index)."""
    overlay = frame.copy()
    for wheel, active, remembered in specs:
        for i in range(wheel.n):
            if i == active:
                color = COLOR_ACTIVE
            elif i == remembered:
                color = COLOR_REMEMBERED
            else:
                color = COLOR_IDLE
            cv2.fillPoly(overlay, [wheel.polygon(i)], color)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    for wheel, _, _ in specs:
        cx, cy = wheel.center
        for i in range(wheel.n):
            cv2.polylines(frame, [wheel.polygon(i)], True, (255, 255, 255), 1,
                          cv2.LINE_AA)
            a = math.radians(wheel.center_angle(i))
            r = (wheel.inner + wheel.radius) / 2 + 8
            tx, ty = int(cx + r * math.cos(a)), int(cy + r * math.sin(a))
            text = wheel.labels[i]
            (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.putText(frame, text, (tx - w // 2, ty + h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                        cv2.LINE_AA)
        (w, _), _ = cv2.getTextSize(wheel.title, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.putText(frame, wheel.title, (cx - w // 2, cy - wheel.radius - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                    cv2.LINE_AA)


def put_text(frame, text, pos, scale, color, thick=2):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0),
                thick + 2, cv2.LINE_AA)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                thick, cv2.LINE_AA)


# --------------------------------------------------------------------------
# Hand tracking helpers
# --------------------------------------------------------------------------
def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


def assign_tips(tips, centers):
    """Match fingertips to wheels (index 0 = root, 1 = quality) by distance."""
    out = [None, None]
    if len(tips) == 1:
        d = [np.hypot(*(tips[0] - c)) for c in centers]
        out[int(np.argmin(d))] = tips[0]
    elif len(tips) >= 2:
        a, b = tips[0], tips[1]
        straight = (np.hypot(*(a - centers[0])) + np.hypot(*(b - centers[1])))
        swapped = (np.hypot(*(a - centers[1])) + np.hypot(*(b - centers[0])))
        out = [a, b] if straight <= swapped else [b, a]
    return out


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    ensure_model()

    cy = FRAME_HEIGHT // 2 + 15
    root_wheel = Wheel((FRAME_WIDTH - WHEEL_RADIUS - MARGIN, cy), ROOTS, "ROOT")
    qual_wheel = Wheel((WHEEL_RADIUS + MARGIN, cy),
                       [q[0] for q in QUALITIES], "QUALITY")
    centers = [np.array(root_wheel.center, float),
               np.array(qual_wheel.center, float)]

    root_sel = WheelSelector(root_wheel)
    qual_sel = WheelSelector(qual_wheel)

    synth = PianoSynth()
    synth.start()

    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    hands = vision.HandLandmarker.create_from_options(options)
    start = time.monotonic()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    smoothed = [None, None]
    last_quality = 0
    last_key = "unset"

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            frame = cv2.flip(frame, 1)

            rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms = int((time.monotonic() - start) * 1000)
            result = hands.detect_for_video(mp_image, ts_ms)

            # Collect index fingertips and draw skeletons.
            tips = []
            for landmarks in result.hand_landmarks or []:
                pts = [(int(l.x * FRAME_WIDTH), int(l.y * FRAME_HEIGHT))
                       for l in landmarks]
                for a, b in HAND_CONNECTIONS:
                    cv2.line(frame, pts[a], pts[b], (255, 255, 255), 1,
                             cv2.LINE_AA)
                t = landmarks[INDEX_TIP]
                tips.append(np.array([t.x * FRAME_WIDTH, t.y * FRAME_HEIGHT]))

            # Give each wheel its own (smoothed) fingertip.
            assigned = assign_tips(tips, centers)
            points = []
            for i in range(2):
                if assigned[i] is None:
                    smoothed[i] = None
                    points.append(None)
                else:
                    smoothed[i] = (assigned[i] if smoothed[i] is None else
                                   SMOOTHING * assigned[i]
                                   + (1 - SMOOTHING) * smoothed[i])
                    points.append((int(smoothed[i][0]), int(smoothed[i][1])))

            root = root_sel.update(points[0])
            qual = qual_sel.update(points[1])

            if qual is not None:
                last_quality = qual
            elif not STICKY_QUALITY:
                last_quality = 0
            quality = last_quality

            # Only touch the audio engine when the chord actually changes.
            key = None if root is None else (root, quality)
            if key != last_key:
                synth.set_notes(chord_notes(*key) if key else frozenset())
                last_key = key

            remembered = last_quality if (STICKY_QUALITY and qual is None) else None
            draw_wheels(frame, [(root_wheel, root, None),
                                (qual_wheel, qual, remembered)])

            for p in points:
                if p is not None:
                    cv2.circle(frame, p, 9, (0, 255, 255), -1)
                    cv2.circle(frame, p, 11, (0, 0, 0), 1)

            if root is not None:
                name = f"{ROOTS[root]}{QUALITIES[quality][1]}"
                (w, _), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 1.6, 4)
                put_text(frame, name, ((FRAME_WIDTH - w) // 2, 70), 1.6,
                         (0, 255, 0), 4)
            elif tips:
                put_text(frame, "Point at a root note", (FRAME_WIDTH // 2 - 130, 40),
                         0.7, (255, 255, 255))
            else:
                put_text(frame, "No hand detected", (FRAME_WIDTH // 2 - 100, 40),
                         0.7, (255, 255, 255))

            cv2.imshow("Camera Input", frame)
            if cv2.waitKey(1) & 0xFF in (ord("x"), ord("q")):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
        synth.stop()


if __name__ == "__main__":
    main()