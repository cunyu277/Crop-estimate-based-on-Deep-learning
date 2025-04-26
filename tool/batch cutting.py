import os
import rasterio
from rasterio.mask import mask
import geopandas as gpd
from tqdm import tqdm
import numpy as np
import tempfile

# 定义输入和输出路径
vector_path = r"E:\Crop\华北平原\NCC_all.shp"  # 矢量文件路径
input_parent_dirs = [r"E:\Crop\DEM"]  # 输入栅格父目录（可多个）
output_parent_dir = r"E:\Crop\DEM_new"  # 输出父目录

# 加载矢量数据
print("加载矢量数据...")
vector_data = gpd.read_file(vector_path)
geometries = vector_data.geometry

def clip_raster_large(input_raster_path, output_raster_path, geometries, chunk_size=1000):
    """分块处理大栅格文件，避免内存溢出"""
    with rasterio.open(input_raster_path) as src:
        # 坐标系转换
        vector_data_reprojected = vector_data.to_crs(src.crs)
        geometries_reprojected = vector_data_reprojected.geometry

        # 获取原始栅格大小
        height, width = src.shape
        nodata = src.nodata if src.nodata is not None else np.nan

        # 创建临时内存映射文件（避免内存爆炸）
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        out_image_memmap = np.memmap(
            temp_file.name, 
            dtype=np.float32,  # 使用32位浮点节省内存
            shape=(src.count, height, width),
            mode='w+'
        )

        # 分块处理栅格
        for i in tqdm(range(0, height, chunk_size), desc=f"分块处理 {os.path.basename(input_raster_path)}"):
            # 计算当前块的范围
            window = rasterio.windows.Window(
                col_off=0, 
                row_off=i, 
                width=width, 
                height=min(chunk_size, height - i)
            )
            
            # 读取当前块数据
            chunk_data = src.read(window=window)
            
            # 转换为float32并处理异常值
            chunk_data = chunk_data.astype(np.float32)
            chunk_data[
                (chunk_data < 0) | 
                (chunk_data == -3.40282e+038) | 
                (chunk_data == nodata)
            ] = np.nan
            
            # 写入内存映射文件
            out_image_memmap[:, i:i+chunk_size, :] = chunk_data

        # 裁剪实际范围（避免空白区域）
        out_image, out_transform = mask(src, geometries_reprojected, crop=True, all_touched=False)
        
        # 更新元数据
        out_meta = src.meta.copy()
        out_meta.update({
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
            "dtype": "float32",
            "nodata": np.nan
        })

        # 写入最终裁剪结果
        with rasterio.open(output_raster_path, "w", **out_meta) as dest:
            dest.write(out_image.astype(np.float32))

        # 清理临时文件
        temp_file.close()
        os.unlink(temp_file.name)

# 主处理流程
for input_parent_dir in input_parent_dirs:
    parent_dir_name = os.path.basename(input_parent_dir)
    output_parent_subdir = os.path.join(output_parent_dir, parent_dir_name)
    os.makedirs(output_parent_subdir, exist_ok=True)

    print(f"\n处理父目录：{input_parent_dir}")
    tif_files = [
        os.path.join(root, file) 
        for root, _, files in os.walk(input_parent_dir) 
        for file in files if file.endswith('.tif')
    ]
    print(f"找到 {len(tif_files)} 个栅格文件。")

    for input_raster_path in tqdm(tif_files, desc=f"批量裁剪 {parent_dir_name}"):
        # 构建输出路径
        relative_path = os.path.relpath(os.path.dirname(input_raster_path), input_parent_dir)
        first_level_dir = relative_path.split(os.sep)[0] if os.sep in relative_path else relative_path
        output_raster_dir = os.path.join(output_parent_subdir, first_level_dir)
        os.makedirs(output_raster_dir, exist_ok=True)
        output_raster_path = os.path.join(output_raster_dir, os.path.basename(input_raster_path))

        # 执行裁剪（自动分块）
        clip_raster_large(input_raster_path, output_raster_path, geometries, chunk_size=1000)

print("所有栅格数据裁剪完成！")