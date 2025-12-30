import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple
import os

def excel8760_to_daily24_npy(excel_path, output_npy_path=None, sheet_name=None, skip_header=1):
    """
    将Excel中的8760长度数据分割为1*24的周数据，并保存为npy文件
    参数：
    excel_path: Excel文件路径（str或Path）
    output_npy_path: 输出npy文件路径，默认与Excel同目录同文件名.npy
    sheet_name: 要读取的sheet名称，None表示读取所有sheet
    skip_header: 跳过Excel开头的行数（表头），默认1（跳过第一行表头）
    """
    # 处理输出路径
    if output_npy_path is None:
        excel_path = Path(excel_path)
        output_npy_path = excel_path.parent / f"{excel_path.stem}.npy"
    # 存储所有1*24的数据
    all_weekly_data = []
    # 读取Excel文件
    if sheet_name is None:
        # 读取所有sheet
        excel_file = pd.ExcelFile(excel_path)
        sheet_names = excel_file.sheet_names
    else:
        sheet_names = [sheet_name]
    for sheet in sheet_names:
        print(f"正在处理sheet: {sheet}")
        # 读取当前sheet数据（跳过表头）
        df = pd.read_excel(excel_path, sheet_name=sheet, skiprows=skip_header)
        # 遍历每一列（每列应为8760长度的数据）
        for col_name in df.columns:
            # 获取列数据，去除NaN值，转换为numpy数组
            col_data = df[col_name].dropna().values
            # 验证数据长度是否为8760
            if len(col_data) != 8760:
                print(f"警告：列 '{col_name}' 的长度为 {len(col_data)}，不是8760，已跳过该列")
                continue
            print(f"正在处理列: {col_name} (8760个数据点)")
            # reshape为(-,24)，每个行向量是1*24的日数据
            weekly_data = col_data[:8761].reshape(-1, 24)
            # 验证分割后的数据形状
            if weekly_data.shape != (365, 24):
                print(f"警告：列 '{col_name}' 分割后形状为 {weekly_data.shape}，不是(365,24)，已跳过该列")
                continue
            # 将当前列的日数据添加到总列表
            all_weekly_data.append(weekly_data)
    # 合并所有数据（形状：(总样本数, 24)）
    if all_weekly_data:
        final_data = np.concatenate(all_weekly_data, axis=0)
        # final_data = final_data.reshape(-1, 1, 24)
        # 保存为npy文件
        np.save(output_npy_path, final_data)
        print(f"\n数据处理完成！")
        print(f"原始数据：{len(all_weekly_data)} 列 × 8760 个数据点")
        print(f"分割后：{final_data.shape[0]} 个1×24的日数据样本")
        print(f"保存路径：{output_npy_path}")
        return final_data
    else:
        print("错误：没有找到有效的8760长度数据列！")
        return None


def excel8760_to_weekly168_npy(excel_path, output_npy_path=None, sheet_name=None, skip_header=1):
    """
    将Excel中的8760长度数据分割为1*168的周数据，并保存为npy文件
    参数：
    excel_path: Excel文件路径（str或Path）
    output_npy_path: 输出npy文件路径，默认与Excel同目录同文件名.npy
    sheet_name: 要读取的sheet名称，None表示读取所有sheet
    skip_header: 跳过Excel开头的行数（表头），默认1（跳过第一行表头）
    """
    # 处理输出路径
    if output_npy_path is None:
        excel_path = Path(excel_path)
        output_npy_path = excel_path.parent / f"{excel_path.stem}.npy"
    # 存储所有1*24的数据
    all_weekly_data = []
    # 读取Excel文件
    if sheet_name is None:
        # 读取所有sheet
        excel_file = pd.ExcelFile(excel_path)
        sheet_names = excel_file.sheet_names
    else:
        sheet_names = [sheet_name]
    for sheet in sheet_names:
        print(f"正在处理sheet: {sheet}")
        # 读取当前sheet数据（跳过表头）
        df = pd.read_excel(excel_path, sheet_name=sheet, skiprows=skip_header)
        # 遍历每一列（每列应为8760长度的数据）
        for col_name in df.columns:
            # 获取列数据，去除NaN值，转换为numpy数组
            col_data = df[col_name].dropna().values
            # 验证数据长度是否为8760
            if len(col_data) != 8760:
                print(f"警告：列 '{col_name}' 的长度为 {len(col_data)}，不是8760，已跳过该列")
                continue
            print(f"正在处理列: {col_name} (8760个数据点)")
            # reshape为(-,24)，每个行向量是1*24的日数据
            weekly_data = col_data[:8737].reshape(-1, 168)
            # 验证分割后的数据形状
            if weekly_data.shape != (52, 168):
                print(f"警告：列 '{col_name}' 分割后形状为 {weekly_data.shape}，不是(365,24)，已跳过该列")
                continue
            # 将当前列的日数据添加到总列表
            all_weekly_data.append(weekly_data)
    # 合并所有数据（形状：(总样本数, 24)）
    if all_weekly_data:
        final_data = np.concatenate(all_weekly_data, axis=0)
        # final_data = final_data.reshape(-1, 1, 24)
        # 保存为npy文件
        np.save(output_npy_path, final_data)
        print(f"\n数据处理完成！")
        print(f"原始数据：{len(all_weekly_data)} 列 × 8760 个数据点")
        print(f"分割后：{final_data.shape[0]} 个1×168的日数据样本")
        print(f"保存路径：{output_npy_path}")
        return final_data
    else:
        print("错误：没有找到有效的8760长度数据列！")
        return None


class PVDataSet(Dataset):
    def __init__(self, npy_file_path: str, normalize: bool = True, max_power: float = None):
        # 1. 检查文件是否存在
        if not os.path.exists(npy_file_path):
            raise FileNotFoundError(f"找不到npy文件：{npy_file_path}")
        # 2. 加载npy数据
        self.raw_pv_data = np.load(npy_file_path, allow_pickle=False)  # shape=(N, 24)
        # 3. 数据有效性检查
        self._validate_data()
        # 4. 归一化处理（光伏出力非负，用最大值归一化）
        self.pv_data = self._normalize_data(self.raw_pv_data, normalize, max_power)
        self.pattern_features = self._compute_pattern_features()  # (N, 4)：x_avg, x_max, x_sun, x_fluc
        # 计算原始数据的特征统计量（用于后续归一化真实值输入）
        self.raw_pattern_features = self._compute_raw_pattern_features()
        self.pattern_features_max = self._compute_pattern_features_max()

    def _validate_data(self):
        """检查数据格式和有效性"""
        # 检查维度
        if self.raw_pv_data.ndim != 2:
            raise ValueError(f"npy文件数据必须是2维（样本数×24），当前维度：{self.raw_pv_data.ndim}")
        # 检查序列长度
        if self.raw_pv_data.shape[1] != 24:
            raise ValueError(f"每条数据必须是1×168维，当前序列长度：{self.raw_pv_data.shape[1]}")
        # 检查异常值（NaN/无穷大）
        if np.any(np.isnan(self.raw_pv_data)) or np.any(np.isinf(self.raw_pv_data)):
            raise ValueError("npy文件中包含NaN或无穷大值，请先清理数据")
        # 检查非负性（光伏出力不能为负）
        if np.any(self.raw_pv_data < 0):
            print("⚠️  警告：npy文件中存在负值，已自动裁剪为0")
            self.raw_pv_data = np.clip(self.raw_pv_data, 0, None)

    def _normalize_data(self, data: np.ndarray, normalize: bool, max_power: float) -> torch.Tensor:
        """归一化到 [0,1]"""
        if not normalize:
            print("⚠️  警告：未开启归一化，可能导致模型训练不稳定")
            return torch.tensor(data, dtype=torch.float32)
        # 确定归一化最大值
        if max_power is not None:
            if max_power <= 0:
                raise ValueError("max_power必须为正数")
            norm_max = max_power
        else:
            norm_max = np.max(data)
            if norm_max == 0:
                raise ValueError("所有光伏出力数据均为0，无法归一化")
        # 归一化并转为tensor
        normalized_data = data / norm_max
        # 确保归一化后在 [0,1]（避免浮点误差）
        normalized_data = np.clip(normalized_data, 0.0, 1.0)
        return torch.tensor(normalized_data, dtype=torch.float32)  # (N, 24)

    def _compute_pattern_features(self) -> torch.Tensor:
        n_data = self.pv_data.shape[0]
        features = []
        for i in range(n_data):
            seq = self.pv_data[i]  # (24,)
            # 1. 平均功率 x_avg
            x_avg = torch.mean(seq)
            # 2. 最大功率 x_max
            x_max = torch.max(seq)
            # 3. 日照时长 x_sun（出力>0的时刻数）
            x_sun = torch.sum(seq > 0).float()
            # 4. 波动功率 x_fluc（相邻差值绝对值之和）
            x_fluc = torch.sum(torch.abs(seq - torch.roll(seq, shifts=1)))
            features.append(torch.stack([x_avg, x_max, x_sun, x_fluc]))
        return torch.stack(features)

    def _compute_raw_pattern_features(self) -> np.ndarray:
        """计算原始数据的模式特征（未归一化）"""
        n_data = self.raw_pv_data.shape[0]
        features = []
        for i in range(n_data):
            seq = self.raw_pv_data[i]  # (24,)
            # 1. 平均功率 x_avg
            x_avg = np.mean(seq)
            # 2. 最大功率 x_max
            x_max = np.max(seq)
            # 3. 日照时长 x_sun（出力>0的时刻数）
            x_sun = np.sum(seq > 0)
            # 4. 波动功率 x_fluc（相邻差值绝对值之和）
            x_fluc = np.sum(np.abs(seq - np.roll(seq, shift=1)))
            features.append([x_avg, x_max, x_sun, x_fluc])
        return np.array(features)

    def _compute_pattern_features_max(self) -> np.ndarray:
        """计算模式特征的最大值，用于归一化真实值输入"""
        return np.max(self.raw_pattern_features, axis=0)

    def __len__(self) -> int:
        return self.pv_data.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回：(光伏出力序列, 对应的模式特征)"""
        return self.pv_data[idx], self.pattern_features[idx]

    def get_normalization_max(self) -> float:
        """返回归一化时使用的最大功率（用于后续反归一化真实值）"""
        return np.max(self.raw_pv_data) if self.raw_pv_data.size > 0 else 1.0

    def get_pattern_features_max(self) -> np.ndarray:
        """返回模式特征的最大值"""
        return self.pattern_features_max


class WDDataSet(Dataset):
    def __init__(self, npy_file_path: str, normalize: bool = True, max_power: float = None):
        # 1. 检查文件是否存在
        if not os.path.exists(npy_file_path):
            raise FileNotFoundError(f"找不到npy文件：{npy_file_path}")
        # 2. 加载npy数据
        self.raw_wd_data = np.load(npy_file_path, allow_pickle=False)  # shape=(N, 24)
        # 3. 数据有效性检查
        self._validate_data()
        # 4. 归一化处理（光伏出力非负，用最大值归一化）
        self.wd_data = self._normalize_data(self.raw_wd_data, normalize, max_power)
        self.pattern_features = self._compute_pattern_features()  # (N, 4)：x_avg, x_max, x_min, x_fluc
        # 计算原始数据的特征统计量（用于后续归一化真实值输入）
        self.raw_pattern_features = self._compute_raw_pattern_features()
        self.pattern_features_max = self._compute_pattern_features_max()

    def _validate_data(self):
        """检查数据格式和有效性"""
        # 检查维度
        if self.raw_wd_data.ndim != 2:
            raise ValueError(f"npy文件数据必须是2维（样本数×24），当前维度：{self.raw_wd_data.ndim}")
        # 检查序列长度
        if self.raw_wd_data.shape[1] != 24:
            raise ValueError(f"每条数据必须是1×168维，当前序列长度：{self.raw_wd_data.shape[1]}")
        # 检查异常值（NaN/无穷大）
        if np.any(np.isnan(self.raw_wd_data)) or np.any(np.isinf(self.raw_wd_data)):
            raise ValueError("npy文件中包含NaN或无穷大值，请先清理数据")
        # 检查非负性（风机出力不能为负）
        if np.any(self.raw_wd_data < 0):
            print("⚠️  警告：npy文件中存在负值，已自动裁剪为0")
            self.raw_pv_data = np.clip(self.raw_wd_data, 0, None)

    def _normalize_data(self, data: np.ndarray, normalize: bool, max_power: float) -> torch.Tensor:
        """归一化到 [0,1]"""
        if not normalize:
            print("⚠️  警告：未开启归一化，可能导致模型训练不稳定")
            return torch.tensor(data, dtype=torch.float32)
        # 确定归一化最大值
        if max_power is not None:
            if max_power <= 0:
                raise ValueError("max_power必须为正数")
            norm_max = max_power
        else:
            norm_max = np.max(data)
            if norm_max == 0:
                raise ValueError("所有光伏出力数据均为0，无法归一化")
        # 归一化并转为tensor
        normalized_data = data / norm_max
        # 确保归一化后在 [0,1]（避免浮点误差）
        normalized_data = np.clip(normalized_data, 0.0, 1.0)
        return torch.tensor(normalized_data, dtype=torch.float32)  # (N, 24)

    def _compute_pattern_features(self) -> torch.Tensor:
        n_data = self.wd_data.shape[0]
        features = []
        for i in range(n_data):
            seq = self.wd_data[i]  # (24,)
            # 1. 平均功率 x_avg
            x_avg = torch.mean(seq)
            # 2. 最大功率 x_max
            x_max = torch.max(seq)
            # 3. 日照时长 x_sun（出力>0的时刻数）
            x_sun = torch.sum(seq > 0).float()
            # 4. 波动功率 x_fluc（相邻差值绝对值之和）
            x_fluc = torch.sum(torch.abs(seq - torch.roll(seq, shifts=1)))
            features.append(torch.stack([x_avg, x_max, x_sun, x_fluc]))
        return torch.stack(features)

    def _compute_raw_pattern_features(self) -> np.ndarray:
        """计算原始数据的模式特征（未归一化）"""
        n_data = self.raw_wd_data.shape[0]
        features = []
        for i in range(n_data):
            seq = self.raw_wd_data[i]  # (24,)
            # 1. 平均功率 x_avg
            x_avg = np.mean(seq)
            # 2. 最大功率 x_max
            x_max = np.max(seq)
            # 3. 最小功率
            x_min = np.min(seq)
            # 4. 波动功率 x_fluc（相邻差值绝对值之和）
            x_fluc = np.sum(np.abs(seq - np.roll(seq, shift=1)))
            features.append([x_avg, x_max, x_min, x_fluc])
        return np.array(features)

    def _compute_pattern_features_max(self) -> np.ndarray:
        """计算模式特征的最大值，用于归一化真实值输入"""
        return np.max(self.raw_pattern_features, axis=0)

    def __len__(self) -> int:
        return self.wd_data.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回：(光伏出力序列, 对应的模式特征)"""
        return self.wd_data[idx], self.pattern_features[idx]

    def get_normalization_max(self) -> float:
        """返回归一化时使用的最大功率（用于后续反归一化真实值）"""
        return np.max(self.raw_wd_data) if self.raw_wd_data.size > 0 else 1.0

    def get_pattern_features_max(self) -> np.ndarray:
        """返回模式特征的最大值"""
        return self.pattern_features_max

