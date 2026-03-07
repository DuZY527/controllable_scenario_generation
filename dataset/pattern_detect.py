import numpy as np
import pandas as pd
from pathlib import Path
from k_mediods_analysis import typical_scenarios_pattern


def excel8760_to_daily24(excel_path, output_npy_path=None, sheet_name=None, skip_header=1):
    """
    将Excel中的8760长度数据分割为1*168的周数据
    参数：
    excel_path: Excel文件路径（str或Path）
    sheet_name: 要读取的sheet名称，None表示读取所有sheet
    skip_header: 跳过Excel开头的行数（表头），默认1（跳过第一行表头）
    """
    # 处理输出路径
    if output_npy_path is None:
        excel_path = Path(excel_path)
    # 存储所有1*168的数据
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
            # reshape为(52, 168)，每个行向量是1*24的日数据
            weekly_data = col_data[:8781].reshape(-1, 24)
            # 验证分割后的数据形状
            if weekly_data.shape != (365, 24):
                print(f"警告：列 '{col_name}' 分割后形状为 {weekly_data.shape}，不是(365,24)，已跳过该列")
                continue
            # 将当前列的zhou数据添加到总列表
            all_weekly_data.append(weekly_data)
    # 合并所有数据（形状：(总样本数, 168)）
    if all_weekly_data:
        final_data = np.concatenate(all_weekly_data, axis=0)
        final_data = final_data.reshape(-1, 1, 24)
        print(f"\n数据处理完成！")
        print(f"原始数据：{len(all_weekly_data)} 列 × 8760 个数据点")
        print(f"分割后：{final_data.shape[0]} 个1×24的周数据样本")
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
        final_data = final_data.reshape(-1, 1, 168)
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

# 数据引入
excel_file_path = \
        r"D:\科研\可控场景生成\Diffusion model\diffusion model for controllable scenario generation\dataset\PVdata.xlsx"
weekly_data = excel8760_to_weekly168_npy(
            excel_path=excel_file_path,
            sheet_name=None,  # 读取所有sheet
            skip_header=0
            )
total_patterns = []
data_set = "pv"
# 全局特征
for i in range(weekly_data.shape[0]):
    daily_max = []
    daily_mean = []
    # 提取第i个1×168向量并展平
    vector = weekly_data[i, 0, :].flatten()
    # 计算各项特征
    max_val = np.max(vector)
    mean_val = np.mean(vector)
    # 最小值（wd）
    min_val = np.min(vector)
    # 非零数比例（pv）
    non_zero_ratio = np.count_nonzero(vector)
    # 相邻差值绝对值的最大值（包括首尾相连）
    for j in range(7):
        daily_max.append(np.max(vector[24 * j: 24 * (j + 1)]))
        daily_mean.append(np.mean(vector[24 * j:24 * (j + 1)]))
    # 4.最大值波动值
    max_fluc = np.sum(np.abs(daily_max - np.roll(daily_max, 1)))
    # 5.平均值波动值
    mean_fluc = np.sum(np.abs(daily_mean - np.roll(daily_mean, 1)))
    # 6. 前后两小时光伏数据差值的平均值（一阶差分均值）
    vector_rolled = np.roll(vector, 1)
    diff = vector - vector_rolled  # 前后两小时差值
    diff_mean = np.mean(np.abs(diff))  # 差值的绝对值平均值
    # 7. 前后两小时光伏数据差值的方差（一阶差分方差）
    diff_var = np.var(np.abs(diff))  # 差值的绝对值方差
    if data_set == "pv":
        pattern = [max_val, mean_val, non_zero_ratio, max_fluc, mean_fluc, diff_mean, diff_var]
    elif data_set == "wd":
        pattern = [max_val, mean_val, min_val]
    else:
        raise ValueError("不正确数据类型，请使用pv or wd")
    total_patterns.append(pattern)

max_values_global = np.max([r[0] for r in total_patterns])
min_max_values_global = np.min([r[0] for r in total_patterns])
max_mean_values_global = np.max([r[1] for r in total_patterns])
min_mean_values_global = np.min([r[1] for r in total_patterns])
max_non_zero_ratios_global = np.max([r[2] for r in total_patterns])
min_non_zero_ratios_global = np.min([r[2] for r in total_patterns])
max_maxfluc_global = np.max([r[3] for r in total_patterns])
min_maxfluc_global = np.min([r[3] for r in total_patterns])
max_meanfluc_global = np.max([r[4] for r in total_patterns])
min_meanfluc_global = np.min([r[4] for r in total_patterns])
max_diffsmean_global = np.max([r[5] for r in total_patterns])
min_diffsmean_global = np.min([r[5] for r in total_patterns])
max_diffsvar_global = np.max([r[6] for r in total_patterns])
min_diffsvar_global = np.min([r[6] for r in total_patterns])
pattern_max = [max_values_global, max_mean_values_global, max_non_zero_ratios_global, max_maxfluc_global, max_meanfluc_global, max_diffsmean_global,max_diffsvar_global]
pattern_min = [min_max_values_global, min_mean_values_global, min_non_zero_ratios_global, min_maxfluc_global, min_meanfluc_global, min_diffsmean_global,min_diffsvar_global]
print("总体统计信息:")
print(f"最大范围:{max_values_global:.4f}~{min_max_values_global:.4f}")
print(f"平均值范围：{max_mean_values_global:.4f}~{min_mean_values_global:.4f}")
print(f"光照时长/最小值范围:{max_non_zero_ratios_global:.4f}~{min_non_zero_ratios_global:.4f}")
print(f"日际波动值最大值范围：{max_maxfluc_global:.4f}~{min_maxfluc_global:.4f}")
print(f"日际波动值均值范围：{max_meanfluc_global:.4f}~{min_meanfluc_global:.4f}")
print(f"波动值均值范围：{max_diffsmean_global:.4f}~{min_diffsmean_global:.4f}")
print(f"波动值方差范围：{max_diffsvar_global:.4f}~{min_diffsvar_global:.4f}")

print("\n开始K-medoids聚类分析...")
km_result = typical_scenarios_pattern(total_patterns, pattern_max, pattern_min)
medoid_indices = km_result['medoid_indices']
cluster_sizes = km_result['cluster_sizes']
cluster_weights = km_result['cluster_weights']





