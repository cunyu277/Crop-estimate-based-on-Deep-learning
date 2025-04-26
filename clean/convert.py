import os
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.crs import CRS
from tqdm import tqdm

def convert_to_utm(input_path, output_path, target_crs="EPSG:32650"):
    """将单个TIFF文件转换为UTM坐标系"""
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

def process_modis_data(base_path="E:\\Crop\\NorthChina"):
    """处理所有MODIS数据文件夹并导出到统一目录"""
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
    
    total_files = 0
    # 先统计总文件数用于进度条
    for dir_name in data_dirs.keys():
        dir_path = os.path.join(base_path, dir_name)
        total_files += len([f for f in os.listdir(dir_path) if f.lower().endswith(('.tif', '.tiff'))])
    
    # 初始化总进度条
    with tqdm(total=total_files, desc="总进度", unit="文件") as pbar_total:
        # 遍历每个数据目录
        for dir_name, dir_info in data_dirs.items():
            dir_path = os.path.join(base_path, dir_name)
            
            # 获取当前目录的文件列表
            tiff_files = [f for f in os.listdir(dir_path) if f.lower().endswith(('.tif', '.tiff'))]
            
            # 子目录进度条
            with tqdm(total=len(tiff_files), desc=f"处理 {dir_info}", leave=False, unit="文件") as pbar:
                for filename in tiff_files:
                    input_file = os.path.join(dir_path, filename)
                    # 在新文件名前加上原目录名作为前缀，避免重名
                    output_filename = f"{dir_name}_{filename}"
                    output_file = os.path.join(output_root, output_filename)
                    
                    try:
                        convert_to_utm(input_file, output_file)
                        pbar.set_postfix_str(f"{filename[:15]}...")
                    except Exception as e:
                        pbar.write(f"处理 {filename} 时出错: {str(e)}")
                    finally:
                        pbar.update(1)
                        pbar_total.update(1)  # 更新总进度条

if __name__ == "__main__":
    process_modis_data()