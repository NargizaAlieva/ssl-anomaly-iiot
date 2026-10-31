import csv
import os
import urllib.request

BASE_URL = "https://engineering.case.edu/sites/default/files/{}.mat"
OUT_DIR = os.path.join("data", "raw", "cwru")

FILES = [
    ("normal", 0.0,   [97, 98, 99, 100]),
    ("IR",     0.007, [105, 106, 107, 108]),
    ("B",      0.007, [118, 119, 120, 121]),
    ("OR",     0.007, [130, 131, 132, 133]),
    ("IR",     0.014, [169, 170, 171, 172]),
    ("B",      0.014, [185, 186, 187, 188]),
    ("OR",     0.014, [197, 198, 199, 200]),
    ("IR",     0.021, [209, 210, 211, 212]),
    ("B",      0.021, [222, 223, 224, 225]),
    ("OR",     0.021, [234, 235, 236, 237]),
]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows, failed = [], []
    for fault, size, numbers in FILES:
        for load_hp, num in enumerate(numbers):
            path = os.path.join(OUT_DIR, f"{num}.mat")
            if not os.path.exists(path):
                try:
                    print(f"Downloading {num}.mat ({fault} {size} {load_hp}HP)...")
                    urllib.request.urlretrieve(BASE_URL.format(num), path)
                except Exception as e:
                    print(f"  ERROR {num}.mat: {e}")
                    failed.append(num)
                    continue
            rows.append({"file": f"{num}.mat", "type": fault,
                         "size": size, "load_hp": load_hp})

    with open(os.path.join(OUT_DIR, "files.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "type", "size", "load_hp"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nFinished: {len(rows)} files, errors: {len(failed)} {failed or ''}")


if __name__ == "__main__":
    main()