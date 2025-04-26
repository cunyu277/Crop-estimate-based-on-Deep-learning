import os
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.mask import mask
import geopandas as gpd
from tqdm import tqdm
import shutil
import math

# 常量定义
TARGET_RESOLUTION = 500  # 目标分辨率(米)
BASE_PATH = "D:\\Crop\\NorthChina"
BASE = "E:\\Crop"
SHP_PATH = os.path.join(BASE, "华北平原", "NCC.shp")
LULC_TARGET_VALUE = 12  # 需要保留的LULC值

def get_resampling_method(data_type):
    """根据数据类型返回对应的重采样方法"""
    if data_type == "LULC":
        return Resampling.nearest  # 分类数据保持最近邻
    elif data_type in ["Temperature", "LAI", "FPAR"]:
        return Resampling.bilinear  # 连续型数据使用双线性
    else:
        return Resampling.bilinear  # 其他默认双线性


def parse_filename(filename):
    """解析标准化文件名（类型_省级码_区位码_年份）"""
    try:
        parts = os.path.splitext(filename)[0].split('_')
        if len(parts) != 4:
            raise ValueError
        return {
            'data_type': parts[0],  # ENTIRE/LAI/FPAR/LULC/Temperature
            'prov_code': parts[1],  # 省级码
            'loc_code': parts[2],   # 区划码
            'year': parts[3]        # 年份
        }
    except:
        raise ValueError(f"文件名格式错误，应为: 类型_省级码_区位码_年份.tif，实际得到: {filename}")

def resample_image(src_path, dst_path):
    """多波段重采样（保持波段顺序）"""
    with rasterio.open(src_path) as src:
        if abs(src.transform.a - TARGET_RESOLUTION) < 1e-6:
            shutil.copy2(src_path, dst_path)
            return False

        data_type = parse_filename(os.path.basename(src_path))['data_type']
        
        new_transform = rasterio.Affine(
            TARGET_RESOLUTION, 0, src.bounds.left,
            0, -TARGET_RESOLUTION, src.bounds.top
        )
        width = math.ceil((src.bounds.right - src.bounds.left) / TARGET_RESOLUTION)
        height = math.ceil((src.bounds.top - src.bounds.bottom) / TARGET_RESOLUTION)

        kwargs = src.meta.copy()
        kwargs.update({
            'transform': new_transform,
            'width': width,
            'height': height,
            'count': src.count,
            'dtype': 'float32' if data_type != "LULC" else src.meta['dtype'],
            'nodata': -9999 if data_type != "LULC" else src.nodata
        })

        with rasterio.open(dst_path, 'w', **kwargs) as dst:
            for band_idx in range(1, src.count + 1):
                resampled = np.empty((height, width), dtype=kwargs['dtype'])
                reproject(
                    source=rasterio.band(src, band_idx),
                    destination=resampled,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=new_transform,
                    dst_crs=src.crs,
                    resampling=get_resampling_method(data_type),
                    src_nodata=src.nodata,
                    dst_nodata=kwargs['nodata']
                )
                dst.write(resampled, band_idx)
    return True

def get_lulc_mask(lulc_path, shp_geom):
    """获取同区域同年份LULC值为12的掩膜"""
    with rasterio.open(lulc_path) as src:
        # 先裁剪到矢量范围
        lulc_data, _ = mask(src, shp_geom, crop=True, all_touched=True)
        return (lulc_data[0] == LULC_TARGET_VALUE)  # 返回布尔掩膜

def clip_with_lulc(src_path, dst_path, shp_df, lulc_dir):
    """使用LULC=12的掩膜裁剪所有数据类型"""
    file_info = parse_filename(os.path.basename(src_path))
    
    # 获取匹配的矢量几何
    geometries = shp_df[
        (shp_df['区划码'].astype(str) == file_info['loc_code']) & 
        (shp_df['省级码'].astype(str) == file_info['prov_code'])
    ].geometry
    
    if geometries.empty:
        raise ValueError(f"未找到匹配矢量: {file_info['loc_code']}-{file_info['prov_code']}")

    # 查找对应的LULC文件
    lulc_file = f"LULC_{file_info['prov_code']}_{file_info['loc_code']}_{file_info['year']}.tif"
    lulc_path = os.path.join(lulc_dir, lulc_file)
    
    if not os.path.exists(lulc_path):
        raise FileNotFoundError(f"找不到对应的LULC文件: {lulc_file}")

    # 获取LULC=12的掩膜
    lulc_mask = get_lulc_mask(lulc_path, geometries)
    
    with rasterio.open(src_path) as src:
        # 先裁剪到矢量范围
        out_image, out_transform = mask(
            src, 
            geometries, 
            crop=True, 
            all_touched=True,
            indexes=range(1, src.count + 1)
        )
        
        # 应用LULC掩膜（所有波段）
        for i in range(out_image.shape[0]):
            out_image[i][~lulc_mask] = src.nodata if src.nodata else -9999

        # 写入输出文件
        meta = src.meta.copy()
        meta.update({
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })
        
        with rasterio.open(dst_path, "w", **meta) as dst:
            dst.write(out_image)

def process_data():
    """主处理流程"""
    # 初始化目录
    input_dir = os.path.join(BASE_PATH, "UTM_All")
    resampled_dir = os.path.join(BASE_PATH, "Resampled_500m")
    clipped_dir = os.path.join(BASE_PATH, "Clipped")
    os.makedirs(resampled_dir, exist_ok=True)
    os.makedirs(clipped_dir, exist_ok=True)

    # 加载矢量数据
    shp_df = gpd.read_file(SHP_PATH)

    # 获取待处理文件列表
    tif_files = [f for f in os.listdir(input_dir) if f.endswith('.tif')]
    
    # 重采样阶段
    print(">>> 正在执行重采样 <<<")
    with tqdm(tif_files, desc="重采样进度") as pbar:
        for f in pbar:
            try:
                src = os.path.join(input_dir, f)
                dst = os.path.join(resampled_dir, f)
                resampled = resample_image(src, dst)
                pbar.set_postfix(file=f, status="跳过" if not resampled else "完成")
            except Exception as e:
                tqdm.write(f"错误 {f}: {str(e)}")

    # 裁剪阶段（使用LULC掩膜）
    print("\n>>> 正在执行LULC掩膜裁剪 <<<")
    with tqdm(tif_files, desc="裁剪进度") as pbar:
        for f in pbar:
            try:
                src = os.path.join(resampled_dir, f)
                dst = os.path.join(clipped_dir, f)
                clip_with_lulc(src, dst, shp_df, resampled_dir)
                pbar.set_postfix(file=f, status="完成")
            except Exception as e:
                tqdm.write(f"错误 {f}: {str(e)}")

if __name__ == "__main__":
    process_data()
