import os
import random
from PIL import Image, ImageDraw, ImageFilter
import numpy as np

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def make_part_image(size=512, anomaly=False, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # base background
    img = Image.new("RGB", (size, size), (235, 235, 235))
    d = ImageDraw.Draw(img)

    # draw a "metal plate" rectangle with rounded-ish corners
    margin = int(size * 0.12)
    plate = (margin, margin, size - margin, size - margin)
    d.rectangle(plate, fill=(200, 200, 205), outline=(120, 120, 125), width=6)

    # add some "holes" (circles)
    holes = []
    for _ in range(6):
        x = random.randint(margin + 60, size - margin - 60)
        y = random.randint(margin + 60, size - margin - 60)
        r = random.randint(18, 28)
        holes.append((x, y, r))
        d.ellipse((x - r, y - r, x + r, y + r), fill=(60, 60, 65))

    # add a faint texture
    noise = (np.random.randn(size, size, 3) * 6).astype(np.int16)
    arr = np.array(img).astype(np.int16)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr, "RGB").filter(ImageFilter.GaussianBlur(radius=0.6))

    # anomalies: scratch, missing hole, extra blob, burn mark
    if anomaly:
        kind = random.choice(["scratch", "missing_hole", "blob", "burn"])
        d = ImageDraw.Draw(img)

        if kind == "scratch":
            x1 = random.randint(margin, size - margin)
            y1 = random.randint(margin, size - margin)
            x2 = random.randint(margin, size - margin)
            y2 = random.randint(margin, size - margin)
            for w in [2, 3, 4]:
                d.line((x1, y1, x2, y2), fill=(120, 70, 70), width=w)

        elif kind == "missing_hole":
            x, y, r = random.choice(holes)
            # paint over with plate color
            d.ellipse((x - r - 2, y - r - 2, x + r + 2, y + r + 2), fill=(200, 200, 205))

        elif kind == "blob":
            x = random.randint(margin + 40, size - margin - 40)
            y = random.randint(margin + 40, size - margin - 40)
            r = random.randint(25, 45)
            d.ellipse((x - r, y - r, x + r, y + r), fill=(80, 80, 120))
            img = img.filter(ImageFilter.GaussianBlur(radius=1.2))

        elif kind == "burn":
            x = random.randint(margin + 80, size - margin - 80)
            y = random.randint(margin + 80, size - margin - 80)
            r = random.randint(40, 70)
            d.ellipse((x - r, y - r, x + r, y + r), fill=(110, 90, 70))
            img = img.filter(ImageFilter.GaussianBlur(radius=2.0))

    return img

def main():
    out_normal = "data/normal"
    out_test = "data/test"
    ensure_dir(out_normal)
    ensure_dir(out_test)

    # 8 normal references (5-10 arası dediğin için)
    for i in range(8):
        img = make_part_image(size=640, anomaly=False, seed=1000+i)
        img.save(os.path.join(out_normal, f"normal_{i:02d}.png"))

    # 12 test: 6 normal + 6 anomalili
    for i in range(6):
        img = make_part_image(size=640, anomaly=False, seed=2000+i)
        img.save(os.path.join(out_test, f"test_normal_{i:02d}.png"))

    for i in range(6):
        img = make_part_image(size=640, anomaly=True, seed=3000+i)
        img.save(os.path.join(out_test, f"test_anom_{i:02d}.png"))

    print("Toy factory dataset written to data/normal and data/test")

if __name__ == "__main__":
    main()
