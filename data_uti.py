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
            weekly_data = col_data[:8736].reshape(-1, 168)
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
        self.raw_pv_data = np.load(npy_file_path, allow_pickle=False)  # shape=(N, 168)
        # 3. 数据有效性检查
        self._validate_data()
        # 4. 归一化处理（光伏出力非负，用最大值归一化）
        self.pv_data = self._normalize_data(self.raw_pv_data, normalize, max_power)
        self.pv_data_7x24 = self._reshape_to_7x24(self.pv_data)  # 7×24格式
        self.feature_matrix = self._compute_pattern_features_matrix()  # (N, 7)：
        # 计算原始数据的特征统计量（用于后续归一化真实值输入）
        self.raw_pattern_features = self._compute_raw_pattern_features_matrix()
        self.pattern_features_max = self._compute_pattern_features_max()

    def _validate_data(self):
        """检查数据格式和有效性"""
        # 检查维度
        if self.raw_pv_data.ndim != 2:
            raise ValueError(f"npy文件数据必须是2维（样本数×24），当前维度：{self.raw_pv_data.ndim}")
        # 检查序列长度
        if self.raw_pv_data.shape[1] != 168:
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

    def _reshape_to_7x24(self, data_168: torch.Tensor) -> torch.Tensor:
        """将1×168数据重构为7×24格式"""
        # data_168形状: (N, 168)
        # 重构为: (N, 7, 24)
        return data_168.view(-1, 7, 24)

    def _compute_entropy(self, values: torch.Tensor) -> torch.Tensor:
        """计算熵值"""
        # 确保所有值为正数
        values = torch.abs(values) + 1e-10
        # 归一化
        prob = values / torch.sum(values)
        # 计算熵
        entropy = -torch.sum(prob * torch.log(prob + 1e-10))
        return entropy

    def _compute_svd_entropy(self, matrix: torch.Tensor) -> torch.Tensor:
        """计算矩阵的奇异值熵"""
        # 计算奇异值
        u, s, v = torch.svd(matrix)
        # 计算奇异值熵
        return self._compute_entropy(s)

    def _compute_pattern_features_matrix(self) -> torch.Tensor:
        n_data = self.pv_data_7x24.shape[0]
        feature_matrixs = []
        for i in range(n_data):
            # 获取7×24格式的数据
            week_data = self.pv_data_7x24[i]  # (7, 24)
            # 总体特征
            total_features = torch.zeros(7)
            # 1. 总量
            total_features[0] = torch.sum(week_data)
            # 2. 峰值
            total_features[1] = torch.max(week_data)
            # 3. 奇异值熵
            total_features[2] = self._compute_svd_entropy(week_data)
            # 4. 行边际熵 (天与天之间的分布熵)
            row_entropy = 0.0
            for day in range(7):
                row_entropy += self._compute_entropy(week_data[day])
            total_features[3] = row_entropy / 7.0  # 平均行熵
            # 5. 列边际熵 (小时与小时之间的分布熵)
            col_entropy = 0.0
            for hour in range(24):
                col_entropy += self._compute_entropy(week_data[:, hour])
            total_features[4] = col_entropy / 24.0  # 平均列熵
            # 日间特征 (变异系数 + 峰值变化率)
            interday_features = torch.zeros(7)
            # 1. 变异系数 (放在第一列第6位)
            daily_totals = torch.sum(week_data, dim=1)  # 每天的总量
            total_features[5] = torch.std(daily_totals) / (torch.mean(daily_totals) + 1e-10)
            # 2. 1×7峰值变化率 (每天峰值与上一天相比的变化率，首尾相接)
            daily_max = torch.max(week_data, dim=1)[0]  # 每天的峰值
            for day in range(7):
                prev_day = (day - 1) % 7
                change_rate = (daily_max[day] - daily_max[prev_day]) / (daily_max[prev_day] + 1e-10)
                interday_features[day] = change_rate
            # 日内特征
            intraday_diff_mean = torch.zeros(7)
            # 日一阶差分均值
            for day in range(7):
                day_data = week_data[day]  # (24,)
                diff = day_data[1:] - day_data[:-1]  # 一阶差分
                intraday_diff_mean[day] = torch.mean(diff)
            # 日一阶差分方差
            intraday_diff_var = torch.zeros(7)
            sunlight_hours_list = []
            for day in range(7):
                day_data = week_data[day]  # (24,)
                # 一阶差分方差
                diff = day_data[1:] - day_data[:-1]  # 一阶差分
                intraday_diff_var[day] = torch.var(diff)
                # 日照时间 (计算到列表中，最后求平均)
                sunlight_hours = torch.sum(day_data > 0.01).float()  # 阈值设为0.01避免噪声
                sunlight_hours_list.append(sunlight_hours)
            # 周平均日照时间放在第一行最后一位
            avg_sunlight = torch.mean(torch.stack(sunlight_hours_list))
            total_features[6] = avg_sunlight
            # 组合成4×7矩阵
            feature_matrix = torch.stack([total_features, interday_features,
                                          intraday_diff_mean, intraday_diff_var])
            feature_matrixs.append(feature_matrix)

        return torch.stack(feature_matrixs)  # (N, 4, 7)

    def _compute_raw_pattern_features_matrix(self) -> np.ndarray:
        """计算原始数据的模式特征（未归一化）"""
        raw_data_7x24 = self.raw_pv_data.reshape(-1, 7, 24)
        n_data = raw_data_7x24.shape[0]
        feature_matrixs = []
        for i in range(n_data):
            # 获取7×24格式的数据
            week_data = self.pv_data_7x24[i]  # (7, 24)
            # 总体特征
            total_features = torch.zeros(7)
            # 1. 总量
            total_features[0] = torch.sum(week_data)
            # 2. 峰值
            total_features[1] = torch.max(week_data)
            # 3. 奇异值熵
            total_features[2] = self._compute_svd_entropy(week_data)
            # 4. 行边际熵 (天与天之间的分布熵)
            row_entropy = 0.0
            for day in range(7):
                row_entropy += self._compute_entropy(week_data[day])
            total_features[3] = row_entropy / 7.0  # 平均行熵
            # 5. 列边际熵 (小时与小时之间的分布熵)
            col_entropy = 0.0
            for hour in range(24):
                col_entropy += self._compute_entropy(week_data[:, hour])
            total_features[4] = col_entropy / 24.0  # 平均列熵
            # 日间特征 (变异系数 + 峰值变化率)
            interday_features = torch.zeros(7)
            # 1. 变异系数 (放在第一列第6位)
            daily_totals = torch.sum(week_data, dim=1)  # 每天的总量
            total_features[5] = torch.std(daily_totals) / (torch.mean(daily_totals) + 1e-10)
            # 2. 1×7峰值变化率 (每天峰值与上一天相比的变化率，首尾相接)
            daily_max = torch.max(week_data, dim=1)[0]  # 每天的峰值
            for day in range(7):
                prev_day = (day - 1) % 7
                change_rate = (daily_max[day] - daily_max[prev_day]) / (daily_max[prev_day] + 1e-10)
                interday_features[day] = change_rate
            # 日内特征
            intraday_diff_mean = torch.zeros(7)
            # 日一阶差分均值
            for day in range(7):
                day_data = week_data[day]  # (24,)
                diff = day_data[1:] - day_data[:-1]  # 一阶差分
                intraday_diff_mean[day] = torch.mean(diff)
            # 日一阶差分方差
            intraday_diff_var = torch.zeros(7)
            sunlight_hours_list = []
            for day in range(7):
                day_data = week_data[day]  # (24,)
                # 一阶差分方差
                diff = day_data[1:] - day_data[:-1]  # 一阶差分
                intraday_diff_var[day] = torch.var(diff)
                # 日照时间 (计算到列表中，最后求平均)
                sunlight_hours = torch.sum(day_data > 0.01).float()  # 阈值设为0.01避免噪声
                sunlight_hours_list.append(sunlight_hours)
            # 周平均日照时间放在第一行最后一位
            avg_sunlight = torch.mean(torch.stack(sunlight_hours_list))
            total_features[6] = avg_sunlight
            # 组合成4×7矩阵
            feature_matrix = torch.stack([total_features, interday_features,
                                          intraday_diff_mean, intraday_diff_var])
            feature_matrixs.append(feature_matrix)

        return np.array(feature_matrixs)  # (N, 4, 7)

    def _compute_pattern_features_max(self) -> np.ndarray:
        """计算模式特征的最大值，用于归一化真实值输入"""
        return np.max(self.raw_pattern_features, axis=0)

    def __len__(self) -> int:
        return self.pv_data.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回：(光伏出力序列, 对应的模式特征)"""
        return self.pv_data[idx], self.feature_matrix[idx]

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
        if self.raw_wd_data.shape[1] != 168:
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
            # ----- 周整体特征--------#
            seq = self.wd_data[i]  # (24,)
            # 1. 平均功率 x_avg
            x_avg = torch.mean(seq)
            # 2. 最大功率 x_max
            x_max = torch.max(seq)
            # 3. 日照时长 x_sun（出力>0的时刻数）
            x_sun = torch.sum(seq > 0).float()
            # 4. 波动功率 x_fluc（相邻差值绝对值之和）
            x_fluc = torch.sum(torch.abs(seq - torch.roll(seq, shifts=1)))
            # ------ 日际特征---------#

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

