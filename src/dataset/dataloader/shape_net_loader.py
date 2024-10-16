import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, Callable, Dict, Any
import numpy as np
from plyfile import PlyData, PlyElement
from src.types import TrainDataGaussianType

class PLYPointCloudDataset(Dataset):
    def __init__(self, directory: str, transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None) -> None:
        """
        初始化数据集。

        参数:
        - directory: 包含 .ply 文件的目录路径。
        - transform: 可选的转换函数，用于对数据进行预处理。
        """
        self.directory = Path(directory)
        self.transform = transform
        self.file_list = self.get_valid_file_list()

    def get_valid_file_list(self):
        temp_file_list = list(self.directory.glob('*.ply'))
        file_list = []
        for file_path in temp_file_list:
            file_dir = file_path.parent
            uuid = file_path.stem
            if (file_dir / f"{uuid}.feat.pth").exists():
                file_list.append(file_path)
        return file_list

    def __len__(self) -> int:
        # 返回数据集中样本的数量
        return len(self.file_list)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # 加载第 idx 个样本
        file_path = self.file_list[idx]

        file_dir = file_path.parent

        uuid = file_path.stem

        triplane_feature = torch.load(file_dir / f"{uuid}.feat.pth", weights_only=True)

        plydata = PlyData.read(file_path)
        max_sh_degree = 3

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])), axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names) == 3 * (max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        sample = {
            'xyz': torch.tensor(xyz, dtype=torch.float).to('cuda'),
            'f_dc': torch.tensor(features_dc, dtype=torch.float).to('cuda').transpose(1, 2).contiguous(),
            'f_rest': torch.tensor(features_extra, dtype=torch.float).to('cuda').transpose(1, 2).contiguous(),
            'opacity': torch.tensor(opacities, dtype=torch.float).to('cuda'),
            'scale': torch.tensor(scales, dtype=torch.float).to('cuda'),
            'rot': torch.tensor(rots, dtype=torch.float).to('cuda'),
            'name': file_path.stem,
            'features': triplane_feature
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


def custom_collate_fn(batch) -> TrainDataGaussianType:
    """
    自定义的 batch 数据处理函数。

    参数:
    - batch: 一个包含 N 个样本的列表。

    返回:
    - 一个字典，包含处理后的 batch 数据。
    """

    gaussian_model = {
        'xyz': [sample['xyz'] for sample in batch],
        'f_dc': [sample['f_dc'] for sample in batch],
        'f_rest': [sample['f_rest'] for sample in batch],
        'opacity': [sample['opacity'] for sample in batch],
        'scale': [sample['scale'] for sample in batch],
        'rot': [sample['rot'] for sample in batch],
        'name': [sample['name'] for sample in batch],
    }

    data = {
        'gaussian_model': gaussian_model,
        # 'labels': [None for _ in batch],
        'features': [sample['features'] for sample in batch]
    }

    return TrainDataGaussianType(**data)

def get_test_data():
    # 创建数据集
    directory_path = Path(__file__).parent.parent / 'data' / 'bag'
    dataset = PLYPointCloudDataset(directory=str(directory_path.absolute()))

    # 使用 DataLoader 加载数据
    data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)

    # 迭代数据
    for batch in data_loader:
        return batch

def get_test_data_loader():
    # 创建数据集
    directory_path = Path(__file__).parent.parent / 'data' / 'bag'
    dataset = PLYPointCloudDataset(directory=str(directory_path.absolute()))

    # 使用 DataLoader 加载数据
    data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)

    return data_loader

if __name__ == '__main__':
    # 创建数据集
    directory_path = Path(__file__).parent / 'data' / 'bag'
    dataset = PLYPointCloudDataset(directory=str(directory_path.absolute()))

    # 使用 DataLoader 加载数据
    data_loader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, collate_fn=custom_collate_fn)

    # 迭代数据
    for batch in data_loader:
        print(batch)
