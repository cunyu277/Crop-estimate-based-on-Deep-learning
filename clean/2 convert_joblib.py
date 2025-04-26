import os
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.crs import CRS
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing

def convert_to_utm(input_path, output_path, target_crs="EPSG:32650"):
    """将单个TIFF文件转换为UTM坐标系"""
    try:
        with rasterio.open(input_path) as src:
            transform, width, height = calculate_default_transform(
                src.crs, target_crs, src.width, src.height, *src.bounds
            )
            
            kwargs = src.meta.copy()
            kwargs.update({
                'crs': target_crs,
                'transform': transform,
                'width': width,
                'height': height
            })
            
            with rasterio.open(output_path, 'w', **kwargs) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=target_crs,
                        resampling=Resampling.nearest
                    )
        return (True, input_path, "")
    except Exception as e:
        return (False, input_path, str(e))

def process_directory(dir_path, dir_name, output_root):
    """处理单个目录的所有文件"""
    tiff_files = [f for f in os.listdir(dir_path) if f.lower().endswith(('.tif', '.tiff'))]
    results = []
    
    for filename in tiff_files:
        input_file = os.path.join(dir_path, filename)
        output_filename = f"{dir_name}_{filename}"
        output_file = os.path.join(output_root, output_filename)
        results.append((input_file, output_file))
    
    return results

def process_modis_data(base_path="E:\\Crop\\NorthChina", n_jobs=-1):
    """使用多线程处理所有MODIS数据"""
    data_dirs = {
        "ENTIRE": "反射率",
        "FPAR": "FPAR",
        "LAI": "叶面积指数",
        "LULC": "土地利用",
        "Temperature": "温度"
    }
    
    # 创建统一输出目录
    output_root = os.path.join(base_path, "UTM_All")
    os.makedirs(output_root, exist_ok=True)
    
    # 准备所有任务
    all_tasks = []
    for dir_name, dir_info in data_dirs.items():
        dir_path = os.path.join(base_path, dir_name)
        if os.path.exists(dir_path):
            tasks = process_directory(dir_path, dir_name, output_root)
            all_tasks.extend(tasks)
    
    # 计算总文件数
    total_files = len(all_tasks)
    
    # 设置并行工作数（n_jobs=-1表示使用所有CPU核心）
    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()
    
    print(f"开始处理 {total_files} 个文件，使用 {n_jobs} 个线程...")
    
    # 使用Parallel并行处理
    results = Parallel(n_jobs=n_jobs)(
        delayed(convert_to_utm)(input_file, output_file)
        for input_file, output_file in tqdm(all_tasks, desc="调度任务")
    )
    
    # 统计结果
    success_count = sum(1 for r in results if r[0])
    error_count = total_files - success_count
    
    print(f"\n处理完成: {success_count} 成功, {error_count} 失败")
    
    # 打印错误信息
    if error_count > 0:
        print("\n错误详情:")
        for r in results:
            if not r[0]:
                print(f"文件 {r[1]} 错误: {r[2]}")

if __name__ == "__main__":
    process_modis_data(n_jobs=-1)  # 使用所有可用核心