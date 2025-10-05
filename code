# 1️⃣ Upload your dataset zip
from google.colab import files
uploaded = files.upload()  # choose Dataset.zip

# 2️⃣ Extract zip
import zipfile, os, shutil, cv2, albumentations as A
import numpy as np # Import numpy
zip_name = list(uploaded.keys())[0]  # get uploaded zip name
with zipfile.ZipFile(zip_name, 'r') as zip_ref:
    zip_ref.extractall("/content/dataset")

# 3️⃣ Define augmentation pipeline (black background applied)
augment = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Rotate(limit=90, border_mode=cv2.BORDER_CONSTANT, value=(0,0,0), p=0.7),   # rotation with black bg
    A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.2, p=0.5),
    A.ColorJitter(saturation=0.3, p=0.5),
    A.CropAndPad(percent=(-0.4, 0.0), border_mode=cv2.BORDER_CONSTANT, value=(0,0,0), p=0.5), # crop with black bg
    A.ToGray(p=0.3)   # grayscale
])

# 4️⃣ Set input/output paths
input_root = "/content/dataset"      # extracted dataset folder
output_root = "/content/augmented"   # augmented images folder
os.makedirs(output_root, exist_ok=True)

num_augmented = 7  # how many augmented images per original

# 5️⃣ Loop through all subfolders and augment images
for subdir, dirs, files_in_subdir in os.walk(input_root):
    rel_path = os.path.relpath(subdir, input_root)
    out_subdir = os.path.join(output_root, rel_path)
    os.makedirs(out_subdir, exist_ok=True)

    for filename in files_in_subdir:
        img_path = os.path.join(subdir, filename)
        img = cv2.imread(img_path)
        if img is None:  # skip non-image files
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        for i in range(num_augmented):
            augmented = augment(image=img)["image"]
            out_path = os.path.join(out_subdir, f"{os.path.splitext(filename)[0]}_aug{i}.jpg")
            cv2.imwrite(out_path, cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR))

 # =======================
        # Extra Custom Augmentations
        # =======================
        base_name = os.path.splitext(filename)[0]

        # Flipped (force apply)
        flipped = cv2.flip(img, 1)
        cv2.imwrite(os.path.join(out_subdir, f"{base_name}_flipped.jpg"),
                    cv2.cvtColor(flipped, cv2.COLOR_RGB2BGR))

        # Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        cv2.imwrite(os.path.join(out_subdir, f"{base_name}_grayscale.jpg"),
                    cv2.cvtColor(gray_rgb, cv2.COLOR_RGB2BGR))

        # Saturation ×3
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype("float32")
        hsv[...,1] = np.clip(hsv[...,1] * 2.5, 0, 255)  # scale saturation
        saturated = cv2.cvtColor(hsv.astype("uint8"), cv2.COLOR_HSV2RGB)
        cv2.imwrite(os.path.join(out_subdir, f"{base_name}_saturated.jpg"),
                    cv2.cvtColor(saturated, cv2.COLOR_RGB2BGR))

        # Brightness +0.4 (scale factor)
        bright = np.clip(img.astype("float32") * 1.4, 0, 255).astype("uint8")
        cv2.imwrite(os.path.join(out_subdir, f"{base_name}_bright.jpg"),
                    cv2.cvtColor(bright, cv2.COLOR_RGB2BGR))

        # Cropped 40%
        h, w = img.shape[:2]
        ch, cw = int(h * 0.4), int(w * 0.4)
        startx = (w - cw) // 2
        starty = (h - ch) // 2
        crop = img[starty:starty+ch, startx:startx+cw]
        crop_resized = cv2.resize(crop, (w, h))
        cv2.imwrite(os.path.join(out_subdir, f"{base_name}_crop40.jpg"),
                    cv2.cvtColor(crop_resized, cv2.COLOR_RGB2BGR))

        # Rotation 90 degree
        rot90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        cv2.imwrite(os.path.join(out_subdir, f"{base_name}_rot90.jpg"),
                    cv2.cvtColor(rot90, cv2.COLOR_RGB2BGR))

        # Random Brightness (delta=2.0)
        factor = np.random.uniform(0.5, 2.0)
        rand_bright = np.clip(img.astype("float32") * factor, 0, 255).astype("uint8")
        cv2.imwrite(os.path.join(out_subdir, f"{base_name}_randbright.jpg"),
                    cv2.cvtColor(rand_bright, cv2.COLOR_RGB2BGR))

# 6️⃣ Zip augmented folder and download
shutil.make_archive("augmented_images", 'zip', output_root)
files.download("augmented_images.zip")
