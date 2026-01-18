import json
import glob
import numpy as np
import cv2
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from tqdm import tqdm
import pandas as pd
import torch
from torch.utils.data import Dataset


class DataLoader(Dataset):
    """Multi-Camera Semantic Segmentation Dataset"""
    def __init__(
        self,
        data_root: str,
        split: str = "train",
        camera: str = "left",
        transform=None,
        img_size: Tuple[int, int] = (512, 1024),
    ):
        """
        Args:
            data_root: 데이터셋 루트 경로
            split: 'train' or 'val'
            camera: 'left' or 'right'
            transform: albumentations transform
            img_size: (height, width)
        """
        self.data_root = Path(data_root)
        self.split = split
        self.camera = camera
        self.transform = transform
        self.img_size = img_size

        # 경로 설정
        self.labels_dir = self.data_root / "labels" / split
        self.img_dir = self.data_root / f"{camera}Img" / split

        # 클래스 매핑 로드
        self.classes = self.load_classes()
        self.num_classes = len(self.classes)

        # 파일 목록 로드
        self.samples = self.load_samples()

    def load_classes(self) -> Dict[str, int]:
        """클래스 라벨 → ID 매핑 (CSV에서 로드)"""
        csv_path = self.data_root / "class_info.csv"

        if csv_path.exists():
            class_df = pd.read_csv(csv_path)
            classes = dict(zip(class_df['class_name'], class_df['class_id']))
        else:
            # CSV 없으면 생성
            print(f"class_info.csv not found. Generating...")
            class_df = generate_class_info(self.data_root, save=True)
            classes = dict(zip(class_df['class_name'], class_df['class_id']))

        return classes

    def load_samples(self) -> List[Dict]:
        """이미지-라벨 쌍 로드"""
        samples = []
        label_files = sorted(glob.glob(str(self.labels_dir / "**/*.json"), recursive=True))

        for label_path in label_files:
            label_path = Path(label_path)

            # 이미지 경로 생성
            rel_path = label_path.relative_to(self.labels_dir)
            folder = rel_path.parent
            filename = rel_path.stem.replace("_gtFine_polygons", "")

            img_suffix = "leftImg8bit" if self.camera == "left" else "rightImg8bit"
            img_path = self.img_dir / folder / f"{filename}_{img_suffix}.png"

            if img_path.exists():
                samples.append({
                    "img_path": str(img_path),
                    "label_path": str(label_path),
                    "folder": str(folder),
                })

        print(f"{self.split.capitalize()} samples: {len(samples)}")
        return samples

    def load_json(self, json_path: str) -> dict:
        """JSON 어노테이션 로드"""
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def polygon_to_mask(self, annotation: dict, img_size: Tuple[int, int]) -> np.ndarray:
        """폴리곤 어노테이션 → 세그멘테이션 마스크"""
        h, w = img_size
        mask = np.zeros((h, w), dtype=np.uint8)

        for obj in annotation.get("objects", []):
            if obj.get("deleted", 0) == 1:
                continue

            label = obj.get("label", "unlabeled")
            class_id = self.classes.get(label, 0)

            polygon = obj.get("polygon", [])
            if len(polygon) < 3:
                continue

            pts = np.array(polygon, dtype=np.int32)
            cv2.fillPoly(mask, [pts], class_id)

        return mask

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]

        # 이미지 로드 (한글 경로 지원)
        img = cv2.imdecode(np.fromfile(sample["img_path"], dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Cannot load image: {sample['img_path']}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 라벨 로드 및 마스크 생성
        annotation = self.load_json(sample["label_path"])
        orig_h, orig_w = annotation["imgHeight"], annotation["imgWidth"]
        mask = self.polygon_to_mask(annotation, (orig_h, orig_w))

        # 리사이즈
        img = cv2.resize(img, (self.img_size[1], self.img_size[0]))
        mask = cv2.resize(mask, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_NEAREST)

        # Transform 적용
        if self.transform:
            transformed = self.transform(image=img, mask=mask)
            img = transformed["image"]
            mask = transformed["mask"]
        else:
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).long()

        return img, mask

    def get_sample_info(self, idx: int) -> Dict:
        """샘플 정보 반환 (디버깅/시각화용)"""
        return self.samples[idx]


def get_dataloader(
    data_root: str,
    split: str = "train",
    batch_size: int = 4,
    transform=None,
    num_workers: int = 4,
    shuffle: Optional[bool] = None,
) -> torch.utils.data.DataLoader:
    """DataLoader 생성 헬퍼 함수"""
    if shuffle is None:
        shuffle = (split == "train")

    dataset = DataLoader(
        data_root=data_root,
        split=split,
        transform=transform,
    )

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


def generate_class_info(data_root: str, save: bool = True) -> pd.DataFrame:
    """
    JSON 파일들에서 클래스 정보 추출 후 CSV 저장
    """
    from collections import Counter

    data_root = Path(data_root)
    class_counts = Counter()

    # train, val 모두 확인
    for split in ['train', 'val']:
        labels_dir = data_root / "labels" / split
        label_files = glob.glob(str(labels_dir / "**/*.json"), recursive=True)

        for label_path in tqdm(label_files, desc=f"Scanning {split}"):
            with open(label_path, 'r', encoding='utf-8') as f:
                annotation = json.load(f)

            for obj in annotation.get("objects", []):
                if obj.get("deleted", 0) == 0:
                    label = obj.get("label", "unlabeled")
                    class_counts[label] += 1

    # DataFrame 생성 (등장 횟수 기준 정렬)
    class_df = pd.DataFrame(
        class_counts.most_common(),
        columns=['class_name', 'count']
    )
    class_df['class_id'] = range(len(class_df))
    class_df = class_df[['class_id', 'class_name', 'count']]

    # CSV 저장
    if save:
        csv_path = data_root / "class_info.csv"
        if data_root.exists():
            class_df.to_csv(str(csv_path), index=False)
            print(f"Saved: {csv_path}")
        else:
            print(f"Warning: Directory not found, CSV not saved: {data_root}")

    return class_df


def get_dataset_info(data_root: str, split: str = "train") -> pd.DataFrame:
    data_root = Path(data_root)
    labels_dir = data_root / "labels" / split
    img_dir = data_root / "leftImg" / split
    label_files = sorted(glob.glob(str(labels_dir / "**/*.json"), recursive=True))

    records = []
    for label_path in tqdm(label_files, desc=f"Loading {split}"):
        label_path = Path(label_path)

        # 이미지 경로 생성
        rel_path = label_path.relative_to(labels_dir)
        folder = rel_path.parent
        filename = rel_path.stem.replace("_gtFine_polygons", "")
        img_path = img_dir / folder / f"{filename}_leftImg8bit.png"

        # JSON 로드
        with open(label_path, 'r', encoding='utf-8') as f:
            annotation = json.load(f)

        # 클래스 추출 (deleted=0인 것만)
        classes = []
        num_objects = 0
        for obj in annotation.get("objects", []):
            if obj.get("deleted", 0) == 0:
                classes.append(obj.get("label", "unlabeled"))
                num_objects += 1

        records.append({
            "img_path": str(img_path),
            "label_path": str(label_path),
            "folder": str(folder),
            "split": split,
            "width": annotation.get("imgWidth", 0),
            "height": annotation.get("imgHeight", 0),
            "classes": classes,
            "num_objects": num_objects,
            "unique_classes": list(set(classes)),
            "num_classes": len(set(classes)),
        })

    return pd.DataFrame(records)
