# Chord Wheel

Play chords in the air. Point at a **root** on one wheel and a **quality** on the other, and the chord sounds as a piano for as long as your fingertip stays on it. Hand tracking comes from your webcam (MediaPipe) and the sound is generated live (NumPy + sounddevice), so there are no audio files and no playback lag.

```
        QUALITY wheel                         ROOT wheel
   (maj, min, 7, maj7, m7 ...)          (C, Db, D, Eb ... B)
        ← left hand                          right hand →
```

---

## Contents

1. [Requirements and installation](#1-requirements-and-installation)
2. [Running it and playing](#2-running-it-and-playing)
3. [How it works](#3-how-it-works)
4. [Configuration reference](#4-configuration-reference)
5. [Changing the instrument](#5-changing-the-instrument)
6. [Changing the UI](#6-changing-the-ui)
7. [Using more fingers](#7-using-more-fingers)
8. [Troubleshooting](#8-troubleshooting)

The code snippets in sections 5-7 are starting points. Adapt them to the current `chord_wheel.py`; they have not all been run against a live camera.

---

## 1. Requirements and installation

- Python 3.9-3.12 (the MediaPipe version you get may not support newer Pythons)
- A webcam
- Speakers or headphones

```bash
pip install opencv-python mediapipe numpy sounddevice
```

- **Linux:** also run `sudo apt install libportaudio2`.
- **macOS:** allow your terminal/IDE to use the camera when prompted.
- **First run:** the program downloads the MediaPipe hand model (`hand_landmarker.task`) next to the script, so it needs internet once. If the download is blocked, put the file in the same folder manually.

If you see `A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x`:

```bash
pip install "numpy<2"
```

---

## 2. Running it and playing

```bash
python chord_wheel.py
```

Press **x** or **q** in the video window to quit.

### Playing

1. Stand or sit so both hands are visible, about arm's length from the camera, with decent front lighting.
2. Point your **index finger** at a sector on the **ROOT wheel** (right). The chord starts and stays on as long as your fingertip stays on that sector.
3. Point your other index finger at a sector on the **QUALITY wheel** (left). The chord name updates, for example C → Cm → Cm7, and the sound re-voices live.
4. Move your fingertip off the wheel (or out of view) to release the chord.

Things to know:

- **Hands are assigned by distance**, not left/right. Whichever fingertip is closer to a wheel controls it, so you can cross your hands or swap roles.
- **One-hand play works.** By default the last quality you picked stays selected (shown in dim blue on the quality wheel), so you can hold a quality and play roots with a single hand.
- **The center of each wheel is a dead zone**, so the hub never triggers anything.
- **Debouncing:** a new sector must be held for a few frames before it takes over, and the current sector gets a little extra room at its edges. This stops flicker at boundaries. If it feels sluggish, lower `CONFIRM_FRAMES`.
- **Shared notes keep ringing** when the chord changes (for example the root stays when going C → Cmaj7), like a pianist keeping their fingers down.

### Chords available

| Wheel label | Chord name | Intervals (semitones above the root) |
|---|---|---|
| maj | C | 0 4 7 |
| min | Cm | 0 3 7 |
| 7 | C7 | 0 4 7 10 |
| maj7 | Cmaj7 | 0 4 7 11 |
| m7 | Cm7 | 0 3 7 10 |
| sus2 | Csus2 | 0 2 7 |
| sus4 | Csus4 | 0 5 7 |
| dim | Cdim | 0 3 6 |
| aug | Caug | 0 4 8 |
| add9 | Cadd9 | 0 4 7 14 |
| 6 | C6 | 0 4 7 9 |
| m7b5 | Cm7b5 | 0 3 6 10 |

Voicing: a bass note (MIDI 36 + root) plus the chord starting around middle C (MIDI 60 + root). Roots from F# upward are shifted down an octave so the chords stay in a comfortable range.

---

## 3. How it works

```
 webcam ──► MediaPipe HandLandmarker ──► index fingertips ──► assign to wheels
                                                                  │
                         smoothing (EMA) ◄────────────────────────┘
                                │
                     WheelSelector (hysteresis + debounce)
                                │ chord changed?
                                ▼
                    PianoSynth.set_notes(frozenset)
                                │  (atomic swap)
                                ▼
          audio callback thread: Voice objects render piano partials
```

The main parts of `chord_wheel.py`:

| Piece | Job |
|---|---|
| `QUALITIES`, `ROOTS` | The chord vocabulary shown on the wheels |
| `chord_notes(root, quality)` | Turns a root and quality into a set of MIDI notes |
| `Voice` | One piano note: partials, decay, damper |
| `PianoSynth` | Keeps a list of `Voice`s, runs in the audio callback |
| `Wheel` | Geometry: which sector is at a point, sector polygons |
| `WheelSelector` | Stabilizes the sector choice (hysteresis, confirm frames, lost-hand grace) |
| `assign_tips` | Decides which fingertip belongs to which wheel |
| `draw_wheels` | Draws both wheels with one semi-transparent overlay |
| `main` | Capture loop tying everything together |

**Why there's no lag:** audio runs in its own real-time callback (`sounddevice`), and the camera loop only sends a new set of notes when the chord changes. The audio engine never waits on the camera, and the camera never waits on audio. The main thread communicates with the audio thread by replacing one immutable object (`self.requested`), so no locks are needed.

---

## 4. Configuration reference

All settings are constants near the top of `chord_wheel.py`.

### Layout

| Constant | Default | Effect |
|---|---|---|
| `FRAME_WIDTH`, `FRAME_HEIGHT` | 960, 540 | Size of the video window. Smaller is faster. |
| `WHEEL_RADIUS` | 190 | Outer radius of both wheels, in pixels |
| `WHEEL_INNER` | 40 | Radius of the dead zone in the middle |
| `MARGIN` | 15 | Gap between a wheel and the window edge |

### Gesture stability

| Constant | Default | Effect |
|---|---|---|
| `CONFIRM_FRAMES` | 3 | Frames a new sector must be held before it takes over. Higher = steadier, slower. |
| `LOST_FRAMES` | 5 | Frames without a hand before the chord is released |
| `ANGLE_HYSTERESIS` | 4.0 | Extra degrees of room before leaving the current sector |
| `RADIAL_HYSTERESIS` | 10 | Extra pixels of room at the wheel's inner and outer edge |
| `SMOOTHING` | 0.55 | Fingertip smoothing. 1.0 = none (jittery), lower = smoother but laggier. |
| `STICKY_QUALITY` | `True` | Keep the last quality when that hand leaves. `False` falls back to major. |

### Sound

| Constant | Default | Effect |
|---|---|---|
| `SAMPLE_RATE` | 44100 | Audio sample rate |
| `BLOCK_SIZE` | 512 | Samples per audio block. Raise to 1024 if you hear crackling. |
| `MASTER_GAIN` | 0.7 | Overall volume (soft-clipped, so it won't harshly distort) |
| `PIANO_ATTACK` | 0.004 | Seconds for the hammer strike to reach full level |
| `RELEASE` | 0.18 | Seconds for the damper to fade a note when you let go |
| `SUSTAIN_FLOOR` | 0.30 | How much of the tone is held while you keep pointing. 0 = natural piano decay, higher = held longer. |
| `MAX_VOICES` | 24 | Cap on simultaneous notes (including fading ones) |

---

## 5. Changing the instrument

### 5.1 How the piano is built

Each note is a `Voice` (in `Voice.__init__`) made of:

| Ingredient | Code | Effect |
|---|---|---|
| Partials | `n_part` (max 8), `k = 1, 2, 3 ...` | Number of harmonics. More = brighter and costlier. |
| Inharmonicity | `B = 0.0004` | Stretches the overtones slightly sharp, like real piano strings. `0` = perfectly harmonic. |
| Two strings | `fk * 1.0009` | A slightly detuned second string gives gentle beating and warmth. |
| Brightness | `a = k ** -1.05 * exp(-k * ...)` | Amplitude of each partial. Steeper = darker. |
| Fast decay | `tau1` | Initial decay time per partial (seconds). Higher partials die faster. |
| Slow decay | `tau2` | The long "aftersound" |
| Sustain hold | `floor` (from `SUSTAIN_FLOOR`) | Level the tone settles to while you keep pointing |
| Loudness | `level` | Per-note level, a bit louder in the bass |
| Attack / release | `PIANO_ATTACK`, `RELEASE` | Onset and damper speeds |

The decay envelope for each partial is:

```
env(t) = floor + (1 - floor) * (0.6 * exp(-t/tau1) + 0.4 * exp(-t/tau2))
```

To change the sound, change those arrays in `Voice.__init__` (and the attack/release constants). Here are some recipes. Paste them after the lines that compute `self.amps`, `self.tau1`, `self.tau2` and `self.floor`.

**Organ (steady, no decay):**

```python
self.floor = np.ones_like(self.floor)          # envelope stays at 1
# optional drawbar-like balance: strong 1st/2nd/3rd, quiet above
self.amps = np.concatenate([a, a]) * 0.5
```
Also set `PIANO_ATTACK = 0.01` and `RELEASE = 0.08`.

**Plucked / guitar-like (short decay):**

```python
self.floor = np.zeros_like(self.floor)
self.tau1 = np.concatenate([0.25 / (1 + 0.3 * k)] * 2)
self.tau2 = np.concatenate([0.9 / (1 + 0.3 * k)] * 2)
```

**Soft pad / strings (slow swell):**

```python
a = k ** -1.0                                   # sawtooth-like spectrum
self.amps = np.concatenate([a, a]) * 0.5
self.floor = np.ones_like(self.floor)
freqs = np.concatenate([fk, fk * 1.003])        # wider detune = more chorus
```
Also set `PIANO_ATTACK = 0.25` and `RELEASE = 0.6`.

**Bell / electric-piano-like:** use inharmonic partial ratios instead of `k`:

```python
ratios = np.array([1.0, 2.76, 5.40, 8.93])     # bell-ish
fk = f0 * ratios
k = np.arange(1, len(ratios) + 1, dtype=np.float64)   # keep k for the arrays below
```
Everything after that line already uses `k` for array sizes, so this works if you also keep `n_part` consistent (set `n_part = len(ratios)` before building `k`).

### 5.2 Changing chord shapes and voicing

- **Add or edit qualities** in `QUALITIES`. Each entry is `(wheel label, name suffix, intervals)`. Intervals can go above 12 (14 = a ninth). The wheel resizes its sectors automatically, so anything from about 3 to 16 entries works. Keep labels short. Example, dominant 9th:

  ```python
  ("9", "9", (0, 4, 7, 10, 14)),
  ```
- **Change the voicing** in `chord_notes`. The bass is `36 + rr` and the chord starts at `60 + rr`. Raise or lower those numbers (each 12 = one octave), or add an extra octave of the root:

  ```python
  notes.add(48 + rr)
  ```
- **Order roots differently** (e.g. circle of fifths). The root's *position* on the wheel is currently also its pitch, so you need a separate pitch table:

  ```python
  ROOTS = ["C", "G", "D", "A", "E", "B", "Gb", "Db", "Ab", "Eb", "Bb", "F"]
  ROOT_SEMITONES = [0, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5]

  def chord_notes(root, quality):
      semitone = ROOT_SEMITONES[root]
      rr = semitone if semitone < 6 else semitone - 12
      notes = {60 + rr + iv for iv in QUALITIES[quality][2]}
      notes.add(36 + rr)
      return frozenset(notes)
  ```

### 5.3 Using real instrument sounds (SoundFont)

The synth is a placeholder for any object with `start()`, `stop()` and `set_notes(set_of_midi_numbers)`. For realistic sounds, install [FluidSynth](https://www.fluidsynth.org/), `pip install pyfluidsynth`, download a `.sf2` SoundFont, and swap in this class:

```python
import fluidsynth

class SoundfontSynth:
    def __init__(self, sf2_path, program=0):
        self.fs = fluidsynth.Synth()
        self.sfid = self.fs.sfload(sf2_path)
        self.fs.program_select(0, self.sfid, 0, program)   # 0 = acoustic piano
        self.held = set()

    def start(self):
        self.fs.start()          # on Windows try: self.fs.start(driver="dsound")

    def stop(self):
        self.fs.delete()

    def set_notes(self, notes):
        notes = set(notes)
        for m in self.held - notes:
            self.fs.noteoff(0, m)
        for m in notes - self.held:
            self.fs.noteon(0, m, 90)     # 90 = velocity
        self.held = notes
```

Then in `main()` replace `synth = PianoSynth()` with `synth = SoundfontSynth("path/to/piano.sf2")`. Some General MIDI program numbers: `0` acoustic piano, `4` electric piano, `19` church organ, `24` nylon guitar, `48` strings, `88` pad.

Note that sampled instruments decay on their own and won't hold forever like `SUSTAIN_FLOOR` does. For a held sound you'd want an organ, string or pad patch.

---

## 6. Changing the UI

### Sizes, positions and colors

- **Window size:** `FRAME_WIDTH`, `FRAME_HEIGHT`.
- **Wheel size and hub:** `WHEEL_RADIUS`, `WHEEL_INNER`. Make sure `2 * WHEEL_RADIUS` fits within the frame height.
- **Swap the wheels:** in `main()`, exchange the centers of `root_wheel` and `qual_wheel`.
- **Colors (BGR order):** `COLOR_IDLE`, `COLOR_ACTIVE`, `COLOR_REMEMBERED` near the drawing code.
- **Transparency:** in `draw_wheels`, change the `0.6` in `cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)`. Lower = more see-through, and the two weights should add to 1.
- **Text:** font scale and thickness are in `draw_wheels` (labels, titles) and `main()` (chord name at the top).
- **Hide the hand skeleton:** delete the `for a, b in HAND_CONNECTIONS:` loop in `main()`.
- **Stop mirroring:** remove `frame = cv2.flip(frame, 1)`. The picture will no longer act like a mirror, so left and right will feel swapped.
- **Different camera:** change `cv2.VideoCapture(0)` to `1`, `2`, and so on.

### Labels and characters

OpenCV's built-in fonts can't draw `♭` or `♯` (they show as `?`), which is why flats are written `Db`, `Eb`, and so on. To get proper symbols, render text with Pillow (draw onto a PIL image and convert back) using a font file that has those glyphs.

### Adding a third wheel

`Wheel`, `WheelSelector` and `draw_wheels` are generic, so a third wheel (say inversion or octave) takes four steps:

1. Create it: `inv_wheel = Wheel((cx, cy), ["root", "1st", "2nd"], "INVERSION")`
2. Create `inv_sel = WheelSelector(inv_wheel)` and call `inv_sel.update(point)` each frame.
3. Add `(inv_wheel, inv_sel_active, None)` to the list passed to `draw_wheels`.
4. Teach `chord_notes` to use the new value, and extend `assign_tips` to match three fingertips to three wheels (it currently handles two).

There's no room for a third wheel at the current window size, so shrink `WHEEL_RADIUS` or enlarge the frame.

---

## 7. Using more fingers

Right now only the **index fingertip** is used. But MediaPipe reports **21 landmarks per hand** every frame, so there's a lot more to work with.

### 7.1 Landmark reference

```
Thumb:  1 (CMC)  2 (MCP)   3 (IP)    4 (TIP)
Index:  5 (MCP)  6 (PIP)   7 (DIP)   8 (TIP)    <- currently used
Middle: 9        10        11        12 (TIP)
Ring:   13       14        15        16 (TIP)
Pinky:  17       18        19        20 (TIP)
Wrist:  0
```

In the code, `landmarks[i].x` and `.y` are normalized 0-1, so multiply by `FRAME_WIDTH` / `FRAME_HEIGHT` for pixels (as `main()` already does for landmark 8).

### 7.2 Detecting which fingers are extended

A helper that compares how far each tip is from the wrist versus how far its middle joint is. It works at any hand rotation:

```python
FINGER_JOINTS = {"index": (8, 6), "middle": (12, 10),
                 "ring": (16, 14), "pinky": (20, 18)}

def extended_fingers(landmarks):
    wrist = landmarks[0]
    d = lambda a, b: math.hypot(a.x - b.x, a.y - b.y)
    up = {name: d(landmarks[tip], wrist) > 1.1 * d(landmarks[pip], wrist)
          for name, (tip, pip) in FINGER_JOINTS.items()}
    # Thumb folds sideways, so compare against the pinky base instead.
    up["thumb"] = d(landmarks[4], landmarks[17]) > d(landmarks[3], landmarks[17])
    return up
```

The thumb check is the least reliable, so avoid depending on it for anything critical. Adjust the `1.1` threshold if curled fingers register as extended.

To use it, keep each hand's landmarks together with its tip. In `main()`, replace the list of tips with a list of dicts:

```python
hands_info = []
for landmarks in result.hand_landmarks or []:
    ...  # (skeleton drawing as before)
    t = landmarks[INDEX_TIP]
    hands_info.append({"tip": np.array([t.x * FRAME_WIDTH, t.y * FRAME_HEIGHT]),
                       "fingers": extended_fingers(landmarks)})
```

And make `assign_tips` return the hand dicts instead of bare points:

```python
def assign_hands(hands, centers):
    out = [None, None]
    if len(hands) == 1:
        d = [np.hypot(*(hands[0]["tip"] - c)) for c in centers]
        out[int(np.argmin(d))] = hands[0]
    elif len(hands) >= 2:
        a, b = hands[0], hands[1]
        straight = (np.hypot(*(a["tip"] - centers[0])) +
                    np.hypot(*(b["tip"] - centers[1])))
        swapped = (np.hypot(*(a["tip"] - centers[1])) +
                   np.hypot(*(b["tip"] - centers[0])))
        out = [a, b] if straight <= swapped else [b, a]
    return out
```

After that, `assigned[0]["tip"]` replaces `assigned[0]` in the smoothing step, and `assigned[0]["fingers"]` tells you what the root hand is doing.

### 7.3 Idea: finger gestures as modifiers

Extra extended fingers can switch modes while the index finger keeps pointing. For example, octave shifts from the middle and ring finger on the root hand:

```python
octave = 0
if assigned[0] is not None:
    f = assigned[0]["fingers"]
    octave = 12 if f["middle"] else (-12 if f["ring"] else 0)
```

Then add an `octave` argument to `chord_notes` (add it to every note) and include it in the change-detection key so audio updates when it changes:

```python
key = None if root is None else (root, quality, octave)
if key != last_key:
    synth.set_notes(chord_notes(*key) if key else frozenset())
    last_key = key
```

Other modifier ideas, all using the same finger-state dictionary: pinky up = add a bass octave, middle up = pick an inversion, thumb out = switch the instrument, closed fist = release everything.

### 7.4 Idea: several fingertips as separate pointers

Each finger can point at the wheel on its own, so one hand could play several chords at once. MediaPipe keeps finger identity stable (landmark 12 is always the middle fingertip), so one selector per finger works:

```python
TIP_IDS = [8, 12, 16, 20]            # index, middle, ring, pinky

root_sels = [WheelSelector(root_wheel) for _ in TIP_IDS]

# each frame, for the hand that is over the root wheel:
roots = set()
for sel, tip_id in zip(root_sels, TIP_IDS):
    lm = landmarks[tip_id]
    point = (int(lm.x * FRAME_WIDTH), int(lm.y * FRAME_HEIGHT))
    active = sel.update(point)       # pass None if that finger is curled
    if active is not None:
        roots.add(active)

notes = frozenset().union(*(chord_notes(r, quality) for r in roots)) if roots else frozenset()
```

Practical notes:

- **Only use extended fingers.** Pass `None` for a curled finger, otherwise folded knuckles will accidentally trigger sectors. Use `extended_fingers()` for this.
- **Smooth per finger.** The current code smooths one fingertip per wheel, so keep a smoothed position for each finger.
- **Sectors are narrow.** At the current radius, neighboring sectors are about 60 px wide at mid-radius, so fingers side by side may land on adjacent roots. That can sound dissonant. Increase `WHEEL_RADIUS` or use fewer sectors if that's a problem.
- **Cap the polyphony.** `MAX_VOICES` is 24 and each chord uses 3-5 voices. Running many chords at once will also use more CPU in the audio callback.
- **A simpler variant:** instead of several chords, let each extended finger add one *chord tone* (for example middle finger adds the 7th), which uses the same detection code.

### 7.5 Idea: pinch for dynamics

Thumb-to-index distance makes a natural volume control:

```python
pinch = math.hypot(landmarks[4].x - landmarks[8].x,
                   landmarks[4].y - landmarks[8].y)       # roughly 0.02 closed ... 0.25 open
volume = float(np.clip((pinch - 0.02) / 0.23, 0.0, 1.0))
```

To use it, add `self.volume = 1.0` to `PianoSynth.__init__`, multiply the output by it in `_callback` (`out * self.volume * MASTER_GAIN`), and set `synth.volume = volume` each frame. For a smoother result, apply the same smoothing you use for fingertip positions.

### 7.6 What to expect

- Hand tracking is most reliable when the hand is well lit and the palm roughly faces the camera.
- When fingers overlap or point straight at the camera, MediaPipe's estimates of the hidden fingers get shakier, so gesture detection that relies on curled fingers is less reliable than the index-fingertip pointing.
- Test any new gesture by printing the finger dictionary first, then choose thresholds that work in your own lighting.

---

## 8. Troubleshooting

| Problem | What to try |
|---|---|
| `module 'mediapipe' has no attribute 'solutions'` | Use the current `chord_wheel.py`, which uses the Tasks API. Don't pin an old MediaPipe. |
| `A module that was compiled using NumPy 1.x ...` | `pip install "numpy<2"`, or upgrade `mediapipe` and `opencv-python`. |
| Model download fails | Download `hand_landmarker.task` (MediaPipe hand landmarker, float16) and place it next to the script. |
| No sound | Check your system output device. In Python: `import sounddevice as sd; print(sd.query_devices())`, then set `sd.default.device = <index>`. On Linux make sure `libportaudio2` is installed. |
| Crackling or dropouts | Raise `BLOCK_SIZE` to 1024. Close other heavy programs. Lower `FRAME_WIDTH/HEIGHT` to free CPU. |
| Video is laggy | Lower the frame size. Make sure nothing else is using the camera. Only one hand needs to be visible at a time for one-hand play. |
| Camera window is black or wrong camera | Try `cv2.VideoCapture(1)` (or 2). On macOS check camera permissions. |
| Hand isn't detected | Improve lighting, avoid a busy or skin-colored background, keep the hand fully in frame. Lowering `min_hand_detection_confidence` in `main()` can help (at the cost of false detections). |
| Chord flickers between sectors | Raise `CONFIRM_FRAMES`, `ANGLE_HYSTERESIS`, or lower `SMOOTHING`. |
| Chord reacts too slowly | Lower `CONFIRM_FRAMES` or raise `SMOOTHING` toward 1. |
| Wrong hand controls the wrong wheel | Hands are matched by distance to each wheel. Keep your hands closer to their own wheel, or swap the wheel positions in `main()`. |
| Chord fades while I hold it | Raise `SUSTAIN_FLOOR` (up to 1.0 for no decay). |
| Sound is too quiet or too harsh | Adjust `MASTER_GAIN`. If it distorts with many notes, lower it. |
