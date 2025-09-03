import argparse
import os
import random
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from formula_all import *

def main(path, label_path):
    if not os.path.exists(label_path):
        os.makedirs(label_path)

    open(os.path.join(label_path, "train_label.ark"), "w").close()
    open(os.path.join(label_path, "test_label.ark"), "w").close()
    open(os.path.join(label_path, "cv_label.ark"), "w").close()

    train_count = test_count = cv_count = 0
    for root, _, files in os.walk(path):
        for f0 in files:
            if not f0.lower().endswith(".f0"):
                continue

            rel_path = os.path.relpath(os.path.join(root, f0), start=path).replace("\\", "/")
            pitch = [rel_path]

            with open(os.path.join(root, f0), "r", encoding="utf-8", errors="ignore") as file:
                for line in file:
                    parts = line.strip().split()
                    # parts[0] = estimated fundamental frequency (Hz)
                    # parts[1] = voicing decision (0 = unvoiced, 1 = voiced)
                    # parts[2] = local RMS energy of the signal
                    # parts[3] = normalized peak autocorrelation value
                    if len(parts) < 3:
                        continue
                    try:
                        hz = float(parts[0])
                        is_voiced = float(parts[1])
                    except:
                        continue
                    bin_val = 0 if is_voiced == 0.0 else Convert.convert_hz_to_bin(hz)
                    pitch.append(str(round(bin_val)))

            if len(pitch) <= 1:
                continue

            message = " ".join(pitch) + "\n"
            a = random.uniform(0, 1)
            if a < 0.8:
                train_count += 1
                out = "train_label.ark"
            elif a < 0.93:
                test_count += 1
                out = "test_label.ark"
            else:
                cv_count += 1
                out = "cv_label.ark"

            with open(os.path.join(label_path, out), "a+") as file:
                file.write(message)

    total = train_count + test_count + cv_count
    print(f"Processed: {total}  Train: {train_count}  Test: {test_count}  CV: {cv_count}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("--path", "-p", required=True, help="ground truth path")
    p.add_argument("--label-path", "-l", required=True, help="output ark folder")
    args = p.parse_args()
    main(args.path, args.label_path)
