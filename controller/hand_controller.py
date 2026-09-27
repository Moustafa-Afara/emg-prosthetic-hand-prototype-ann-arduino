"""
EMG -> grip controller for a 5-servo prosthetic hand.

Pipeline (every 100 ms): the last 200 ms of 8-channel forearm EMG -> time-domain features
(MAV, RMS, WL, ZC, SSC per channel) -> MLP gesture classifier -> majority vote over the last
few decisions (to stop the hand twitching between grips) -> grip preset -> one serial command
"G a1 a2 a3 a4 a5" (servo angles for thumb, index, middle, ring, little) for the Arduino sketch
in ../arduino/prosthetic_hand.

The feature set and classifier are the best cross-subject configuration of the author's study
https://github.com/Moustafa-Afara/emg-gesture-recognition-mfcc-vs-td-features (TD + MLP).

NOT TESTED ON HARDWARE. `replay` runs the whole controller offline on recordings of people the
model was not trained on and reports what the hand would have done; `serial` sends the same
commands to a real board but has not been run against one.

Usage:
    python hand_controller.py replay --data EMG_data_for_gestures-master --test-subjects 1 2 3 4 5 6
    python hand_controller.py train  --data EMG_data_for_gestures-master --model model.joblib
    python hand_controller.py serial --model model.joblib --port COM3 --demo-file <raw_data_file.txt>
"""
import argparse
import time
from collections import Counter, deque
from pathlib import Path

import numpy as np

from features import FS, GESTURES, load_uci, td_features, windows

WIN, STEP, TRIM = 200, 100, 200          # samples at 1 kHz = ms
FINGERS = ("thumb", "index", "middle", "ring", "little")

# Gesture -> grip preset, servo angle per finger (0 = open, 180 = fully closed; calibrate per hand).
# The UCI gestures include wrist movements a 5-finger hand cannot copy, so, as in commercial
# pattern-recognition prostheses, each detected muscle pattern *selects* a grip.
GRIPS = {
    "rest":             ("open",    (0, 0, 0, 0, 0)),
    "fist":             ("power",   (180, 180, 180, 180, 180)),
    "wrist flexion":    ("pinch",   (150, 150, 0, 0, 0)),
    "wrist extension":  ("point",   (180, 0, 180, 180, 180)),
    "radial deviation": ("tripod",  (150, 150, 150, 0, 0)),
    "ulnar deviation":  ("hook",    (0, 120, 120, 120, 120)),
}


def command(gesture_id: int) -> str:
    return "G " + " ".join(str(a) for a in GRIPS[GESTURES[gesture_id]][1])


def build_model(seed=0):
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(),
                         MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, early_stopping=True,
                                       random_state=seed))


def training_matrix(segments):
    X, y = [], []
    for s in segments:
        for w in windows(s["signal"], WIN, STEP, TRIM):
            X.append(td_features(w))
            y.append(s["gesture"])
    return np.array(X), np.array(y)


class GripController:
    """Turns a stream of 200 ms EMG windows into stable grip commands."""

    def __init__(self, model, vote=5):
        self.model, self.votes, self.current = model, deque(maxlen=vote), None

    def step(self, window: np.ndarray):
        """Returns (raw prediction, voted gesture, command or None if the grip did not change)."""
        raw = int(self.model.predict(td_features(window)[None])[0])
        self.votes.append(raw)
        voted = Counter(self.votes).most_common(1)[0][0]
        changed = voted != self.current
        self.current = voted
        return raw, voted, (command(voted) if changed else None)


def replay(a):
    """Offline, subject-independent simulation of the controller on full recordings."""
    import pandas as pd
    segments = load_uci(a.data)
    test = set(a.test_subjects)
    model = build_model(a.seed).fit(*training_matrix([s for s in segments if s["subject"] not in test]))

    rows = []  # per window: subject, true gesture, raw, voted
    commands = 0
    for f in sorted(Path(a.data).glob("*/*_raw_data_*.txt")):
        if int(f.parent.name) not in test:
            continue
        table = pd.read_csv(f, sep="\t").dropna()
        signal = table[[f"channel{i}" for i in range(1, 9)]].to_numpy(float)
        labels = table["class"].to_numpy(int)
        ctl = GripController(model, vote=a.vote)
        for start in range(0, len(signal) - WIN + 1, STEP):
            raw, voted, cmd = ctl.step(signal[start:start + WIN])
            commands += cmd is not None
            window_labels = labels[start:start + WIN]
            true = int(window_labels[0]) if (window_labels == window_labels[0]).all() else 0
            rows.append((int(f.parent.name), true, raw, voted, start / FS))
    rows = np.array(rows)
    labelled = np.isin(rows[:, 1], list(GESTURES))
    raw_acc = np.mean(rows[labelled, 2] == rows[labelled, 1])
    vote_acc = np.mean(rows[labelled, 3] == rows[labelled, 1])
    minutes = len(rows) * STEP / FS / 60
    latency = WIN + (a.vote // 2) * STEP
    print(f"test subjects {sorted(test)} (never seen in training), {minutes:.1f} min of recording")
    print(f"window accuracy on labelled gestures: raw {raw_acc:.3f}, after {a.vote}-vote smoothing {vote_acc:.3f}")
    print(f"grip commands sent: {commands} ({commands / minutes:.0f} per minute, incl. unlabelled transitions)")
    print(f"decision latency ≈ {latency} ms (window {WIN} ms + {a.vote // 2} voting steps of {STEP} ms)")
    if a.figure:
        plot_timeline(rows, a.figure, a.vote)


def plot_timeline(rows, path, vote):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    subject = int(rows[0, 0])
    r = rows[(rows[:, 0] == subject) & (rows[:, 4] <= 70)]  # first 70 s of the first test file
    t, true, voted = r[:, 4], r[:, 1].astype(int), r[:, 3].astype(int)
    fig, ax = plt.subplots(figsize=(10, 3.4))
    # true gestures: grey bars at their own level, only where a gesture is being held
    for k in range(1, 7):
        on = true == k
        edges = np.flatnonzero(np.diff(np.r_[0, on.astype(int), 0]))
        spans = [(t[a], t[min(b, len(t) - 1)] - t[a]) for a, b in zip(edges[::2], edges[1::2])]
        ax.broken_barh(spans, (k - 0.3, 0.6), color="#c9d1db", label="true gesture (held)" if k == 1 else None)
    ax.step(t, voted, where="post", color="#1f5f99", lw=1.5, label=f"controller output ({vote}-vote)")
    ax.set_yticks(range(1, 7), [f"{GESTURES[k]} → {GRIPS[GESTURES[k]][0]}" for k in range(1, 7)], fontsize=8)
    ax.set_ylim(0.4, 6.6)
    ax.set_xlabel("time (s)")
    ax.set_title(f"Offline replay, subject {subject} (unseen in training); grey = gesture actually held", fontsize=10)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"timeline saved to {path}")


def train(a):
    import joblib
    model = build_model(a.seed).fit(*training_matrix(load_uci(a.data)))
    joblib.dump(model, a.model)
    print(f"model trained on all {len(np.unique([s['subject'] for s in load_uci(a.data)]))} subjects -> {a.model}")


def serial_demo(a):
    """Stream a recorded file through the controller to a real board, in real time (untested on hardware)."""
    import joblib
    import pandas as pd
    import serial  # pyserial
    model = joblib.load(a.model)
    signal = pd.read_csv(a.demo_file, sep="\t").dropna()[[f"channel{i}" for i in range(1, 9)]].to_numpy(float)
    ctl = GripController(model, vote=a.vote)
    with serial.Serial(a.port, 115200, timeout=0.1) as port:
        time.sleep(2)  # the board resets when the port opens
        for start in range(0, len(signal) - WIN + 1, STEP):
            t0 = time.time()
            _, voted, cmd = ctl.step(signal[start:start + WIN])
            if cmd:
                port.write((cmd + "\n").encode())
                print(f"{start / FS:6.1f} s  {GESTURES[voted]:16s} -> {cmd}")
            time.sleep(max(0.0, STEP / FS - (time.time() - t0)))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("replay", help="offline simulation on unseen subjects")
    r.add_argument("--data", required=True, type=Path)
    r.add_argument("--test-subjects", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    r.add_argument("--vote", type=int, default=5)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--figure", default="")
    t = sub.add_parser("train", help="train on all subjects and save the model")
    t.add_argument("--data", required=True, type=Path)
    t.add_argument("--model", default="model.joblib")
    t.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("serial", help="send commands to the Arduino (not tested on hardware)")
    s.add_argument("--model", default="model.joblib")
    s.add_argument("--port", required=True)
    s.add_argument("--demo-file", required=True, help="a UCI raw_data file streamed as if live")
    s.add_argument("--vote", type=int, default=5)
    a = p.parse_args()
    {"replay": replay, "train": train, "serial": serial_demo}[a.cmd](a)


if __name__ == "__main__":
    main()
