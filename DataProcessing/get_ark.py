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
            if not f0.endswith("csv"):
                continue
            pitch = [f0]
            with open(os.path.join(root, f0), "r") as file:
                for line in file:
                    x = line.split(",")[1].strip()
                    hz = float(x)
                    bin = Convert.convert_hz_to_bin(hz) if hz >= 10 else 0
                    pitch.append(str(round(bin)))

            massage = " ".join(pitch) + "\n"
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
                file.write(massage)

    total = train_count + test_count + cv_count
    print(f"Processed: {total}  Train: {train_count}  Test: {test_count}  CV: {cv_count}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("--path", "-p", required=True, help="ground truth path")
    p.add_argument("--label-path", "-l", required=True, help="output ark folder")
    args = p.parse_args()
    main(args.path, args.label_path)
