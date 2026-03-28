from __future__ import annotations

from pathlib import Path
import argparse

import cv2
from ultralytics import YOLO

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def make_gallery(out_dir: Path, title: str) -> None:
    imgs = sorted([p for p in out_dir.glob("*") if p.suffix.lower() in IMG_EXTS])

    html = []
    html.append("<!doctype html><html><head><meta charset='utf-8'>")
    html.append(f"<title>{title}</title>")
    html.append(
        """
    <style>
      body{font-family:system-ui, sans-serif; margin:20px;}
      .grid{display:grid; grid-template-columns:repeat(auto-fill, minmax(260px, 1fr)); gap:14px;}
      .card{border:1px solid #ddd; border-radius:12px; padding:10px;}
      img{width:100%; height:auto; border-radius:10px;}
      .name{margin-top:8px; font-size:12px; color:#333; word-break:break-all;}
    </style>
    </head><body>
    """
    )
    html.append(f"<h2>{title}</h2>")
    html.append("<div class='grid'>")

    for img in imgs:
        html.append("<div class='card'>")
        html.append(f"<a href='{img.name}' target='_blank'><img src='{img.name}'></a>")
        html.append(f"<div class='name'>{img.name}</div>")
        html.append("</div>")

    html.append("</div></body></html>")
    (out_dir / "index.html").write_text("\n".join(html), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="best.pt 경로")
    ap.add_argument("--source", default="eval_images", help="이미지 폴더(또는 단일 이미지)")
    ap.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    ap.add_argument("--tag", default="default", help="결과 폴더 태그(예: sweep, baseline 등)")
    ap.add_argument("--out_root", default="vis", help="시각화 결과 루트 폴더 (default: vis)")
    ap.add_argument("--imgsz", type=int, default=320, help="predict 시 이미지 사이즈(기본 320)")

    # baseline 모델학습후 Conf 조절한결과 0.1이 최적확인후 추가
    parser.add_argument("--conf", type=float, default=0.10) 

    args = ap.parse_args()

    model_path = Path(args.model)
    source = Path(args.source)
    out_root = Path(args.out_root)

    if not model_path.exists():
        raise FileNotFoundError(f"model not found: {model_path}")
    if not source.exists():
        raise FileNotFoundError(f"source not found: {source}")

    # 단층 구조: vis/<tag>/conf0.01/
    tag = args.tag.strip() or "default"
    out_dir = out_root / tag / f"conf{args.conf:.2f}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 모델 이름(표시용)
    # weights/best.pt -> 상위 폴더명이 실험명인 경우가 많음
    model_name = model_path.parent.parent.name if model_path.as_posix().find("weights") != -1 else model_path.stem

    yolo = YOLO(str(model_path))

    # ✅ save=True 사용하지 않음: 우리가 result.plot()으로 직접 저장해서 "박스 보장"
    results = yolo.predict(
        source=str(source),
        conf=args.conf,
        imgsz=args.imgsz,
        verbose=True,
    )

    # 결과 이미지 저장: result.plot()은 박스/라벨/점수 그려진 ndarray 반환
    saved = 0
    for r in results:
        # r.path: 입력 이미지 경로
        src_path = Path(getattr(r, "path", ""))
        if src_path.name == "":
            # 혹시 path가 비어있으면 임의 이름
            src_name = f"img_{saved:04d}.jpg"
        else:
            src_name = src_path.name

        out_path = out_dir / src_name

        # plot: labels/conf 표시
        im = r.plot(labels=True, conf=True)

        # OpenCV로 저장 (plot 결과가 BGR/uint8 형태로 나오므로 그대로 저장 가능)
        ok = cv2.imwrite(str(out_path), im)
        if ok:
            saved += 1

    title = f"{model_name} | {tag} | conf={args.conf:.2f} | saved={saved}"
    make_gallery(out_dir, title)

    print(f"[OK] Saved to: {out_dir}")
    print(f"Open: {out_dir / 'index.html'}")


if __name__ == "__main__":
    main()