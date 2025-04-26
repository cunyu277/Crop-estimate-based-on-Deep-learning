import os
import rasterio
from rasterio.mask import mask
from rasterio.merge import merge
import geopandas as gpd
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# 定义输入和输出路径
vector_path = r"E:\Crop\华北平原\NCC_all.shp"  # 矢量文件路径
input_parent_dirs = [
    r"E:\CB\data\「ASTGTM2高程数据-（30米分辨率，全国，85高」\山东",
    r"E:\CB\data\「ASTGTM2高程数据-（30米分辨率，全国，85高」\河北",
    r"E:\CB\data\「ASTGTM2高程数据-（30米分辨率，全国，85高」\河南"
    # r"E:\蓝碳\海水养殖\CAP_MA_China_1990_2022"
]
output_parent_dir = r"E:\Crop\DEM"  # 输出父目录
output_mosaic_path = os.path.join(output_parent_dir, "mosaic.tif")  # 镶嵌后的输出路径

# 加载矢量数据
print("加载矢量数据...")
vector_data = gpd.read_file(vector_path)

# 获取矢量数据的几何信息
geometries = vector_data.geometry

# 批量裁剪函数
def clip_raster(input_raster_path, output_raster_path, geometries):
    try:
        # 打开栅格文件
        with rasterio.open(input_raster_path) as src:
            # 将矢量数据转换为栅格数据的坐标系
            vector_data_reprojected = vector_data.to_crs(src.crs)
            geometries_reprojected = vector_data_reprojected.geometry

            # 裁剪栅格数据，严格按照矢量范围裁剪
            out_image, out_transform = mask(src, geometries_reprojected, crop=True, all_touched=False)
            
            # 将 out_image 转换为浮点数类型
            out_image = out_image.astype(float)
            
            # 获取栅格的 nodata 值
            # nodata = src.nodata if src.nodata is not None else np.nan

            # 将矢量范围内栅格空的部分设置为 0
            # out_image[(out_image == nodata) | np.isnan(out_image)] = 0

            # 更新元数据
            out_meta = src.meta.copy()
            out_meta.update({
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
                # "nodata": 0  # 设置新的 nodata 值为 0
            })

            # 保存裁剪后的栅格数据
            with rasterio.open(output_raster_path, "w", **out_meta) as dest:
                dest.write(out_image)
        return True  # 裁剪成功
    except ValueError as e:
        print(f"跳过文件 {input_raster_path}，原因：{e}")
        return False  # 裁剪失败，跳过文件

# 遍历多个父目录
clipped_raster_paths = []  # 用于存储裁剪后的栅格文件路径
for input_parent_dir in input_parent_dirs:
    # 创建对应的输出父目录
    parent_dir_name = os.path.basename(input_parent_dir)  # 获取父目录名称
    output_parent_subdir = os.path.join(output_parent_dir, parent_dir_name)
    os.makedirs(output_parent_subdir, exist_ok=True)

    print(f"\n处理父目录：{input_parent_dir}")
    print(f"输出父目录：{output_parent_subdir}")

    # 统计当前父目录下的 .tif 文件数量
    tif_files = []
    for root, dirs, files in os.walk(input_parent_dir):
        tif_files.extend([os.path.join(root, file) for file in files if file.endswith('dem.tif')])
    print(f"找到 {len(tif_files)} 个栅格文件。")

    # 使用 tqdm 显示进度条
    for input_raster_path in tqdm(tif_files, desc=f"处理 {parent_dir_name}", mininterval=2.5, dynamic_ncols=False):
        # 创建对应的输出路径
        relative_path = os.path.relpath(os.path.dirname(input_raster_path), input_parent_dir)
        # 只保留第一层子文件夹名称
        first_level_dir = relative_path.split(os.sep)[0] if os.sep in relative_path else relative_path
        output_raster_dir = os.path.join(output_parent_subdir, first_level_dir)
        os.makedirs(output_raster_dir, exist_ok=True)

        output_raster_path = os.path.join(output_raster_dir, os.path.basename(input_raster_path))

        # 执行裁剪
        clip_raster(input_raster_path, output_raster_path, geometries)
        clipped_raster_paths.append(output_raster_path)

print("所有栅格数据裁剪完成！")

# 镶嵌裁剪后的栅格数据
print("开始镶嵌裁剪后的栅格数据...")
clipped_rasters = []
for path in clipped_raster_paths:
    if os.path.exists(path):
        try:
            clipped_rasters.append(rasterio.open(path))
        except rasterio.errors.RasterioIOError as e:
            print(f"无法打开文件 {path}，跳过：{e}")
    else:
        print(f"文件不存在，跳过：{path}")

if not clipped_rasters:
    print("没有有效的栅格文件可供镶嵌！")
else:
    mosaic, out_trans = merge(clipped_rasters)

    # 获取镶嵌后的元数据
    out_meta = clipped_rasters[0].meta.copy()
    out_meta.update({
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans,
        "nodata": 0  # 设置新的 nodata 值为 0
    })

    # 保存镶嵌后的栅格数据
    with rasterio.open(output_mosaic_path, "w", **out_meta) as dest:
        dest.write(mosaic)

    print(f"镶嵌完成，结果已保存至：{output_mosaic_path}")
