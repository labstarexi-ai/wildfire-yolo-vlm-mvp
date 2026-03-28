import argparse
from pathlib import Path
import shutil
from ultralytics import YOLO

def load_gt_labels(label_path):
    """
    YOLO txt → class list
    """
    classes = []
    if not label_path.exists():
        return classes

    with open(label_path, "r") as f:
        for line in f:
            cls = int(line.strip().split()[0])
            classes.append(cls)
    return classes


def has_fire_gt(label_path):
    classes = load_gt_labels(label_path)
    return 1 in classes  # fire = 1


def has_fire_pred(result):
    """
    prediction에서 fire 있는지 확인
    """
    if result.boxes is None:
        return False

    cls = result.boxes.cls.cpu().numpy()
    return 1 in cls


def main(args):
    model = YOLO(args.model)

    image_dir = Path(args.images)
    label_dir = Path(args.labels)
    out_dir = Path(args.out)

    miss_dir = out_dir / "fire_miss"
    miss_dir.mkdir(parents=True, exist_ok=True)

    images = list(image_dir.glob("*.*"))

    total_fire = 0
    miss_count = 0

    print(f"[INFO] total images: {len(images)}")
    print(f"[INFO] conf: {args.conf}")

    for img_path in images:
        label_path = label_dir / (img_path.stem + ".txt")

        # 1️⃣ GT에 fire 있는지
        if not has_fire_gt(label_path):
            continue

        total_fire += 1

        # 2️⃣ prediction
        results = model.predict(
            source=str(img_path),
            conf=args.conf,
            verbose=False
        )

        result = results[0]

        # 3️⃣ fire detection 여부
        if not has_fire_pred(result):
            miss_count += 1

            # 이미지 복사
            shutil.copy(img_path, miss_dir / img_path.name)

    print("\n===== RESULT =====")
    print(f"Total fire images: {total_fire}")
    print(f"Missed fire images: {miss_count}")

    if total_fire > 0:
        print(f"Miss rate: {miss_count / total_fire:.3f}")

    print(f"\nSaved to: {miss_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--images", type=str, required=True)
    parser.add_argument("--labels", type=str, required=True)
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--out", type=str, default="runs/analysis")

    args = parser.parse_args()

    main(args)
