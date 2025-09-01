import argparse
import os
import random
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from formula_all import *

def main(path, label_path):
    if not os.path.exists(label_path):
        os.makedirs(label_path)

    open(os.path.join(label_path, "train_lable.ark"), "w").close()
    open(os.path.join(label_path, "test_lable.ark"), "w").close()
    open(os.path.join(label_path, "cv_lable.ark"), "w").close()

    train_count = test_count = cv_count = 0
    for root, _, files in os.walk(path):
        for fn in files:
            if not fn.lower().endswith(".csv"):
                continue

            rel = os.path.relpath(os.path.join(root, fn), start=path).replace("\\", "/")
            bins = []
            with open(os.path.join(root, fn), "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) < 2:
                        continue
                    try:
                        hz = float(parts[1].strip())
                    except:
                        continue
                    binv = Convert.convert_hz_to_bin(hz) if hz >= 10 else 0
                    bins.append(str(round(binv)))

            if not bins:
                continue

            line = " ".join([rel] + bins) + "\n"
            r = random.random()
            if r < 0.8:
                out = "train_lable.ark"; train_count += 1
            elif r < 0.93:
                out = "test_lable.ark"; test_count += 1
            else:
                out = "cv_lable.ark"; cv_count += 1

            with open(os.path.join(label_path, out), "a+") as of:
                of.write(line)

    total = train_count + test_count + cv_count
    print(f"Processed: {total}  Train: {train_count}  Test: {test_count}  CV: {cv_count}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("--path", "-p", required=True, help="ground truth path")
    p.add_argument("--label-path", "-l", required=True, help="output ark folder")
    args = p.parse_args()
    main(args.path, args.label_path)
