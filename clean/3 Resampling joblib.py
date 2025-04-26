import os
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.mask import mask
import geopandas as gpd
from joblib import Parallel, delayed
import warnings
import shutil
import math
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

# 常量定义
TARGET_RESOLUTION = 500  # 目标分辨率(米)
BASE_PATH = "D:\\Crop\\NorthChina"
BASE = "E:\\Crop"
SHP_PATH = os.path.join(BASE, "华北平原", "NCC.shp")
LULC_TARGET_VALUE = 12  # 需要保留的LULC值
N_JOBS = -1  # 使用所有CPU核心

def get_resampling_method(data_type):
    """根据数据类型返回对应的重采样方法"""
    return Resampling.nearest if data_type == "LULC" else Resampling.bilinear

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

def ensure_crs_match(shp_df, raster_crs):
    """确保矢量数据与栅格坐标系一致"""
    if str(shp_df.crs) != str(raster_crs):
        return shp_df.to_crs(raster_crs)
    return shp_df

def resample_image(src_path, dst_path):
    """多波段重采样（保持波段顺序）"""
    try:
        with rasterio.open(src_path) as src:
            if abs(src.transform.a - TARGET_RESOLUTION) < 1e-6:
                shutil.copy2(src_path, dst_path)
                return "跳过"

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
        return "完成"
    except Exception as e:
        return f"错误: {str(e)}"

def clip_with_lulc(src_path, dst_path, shp_df, lulc_dir):
    """使用LULC=12的掩膜裁剪所有数据类型（自动坐标系对齐）"""
    try:
        with rasterio.open(src_path) as src:
            # 坐标系对齐
            aligned_shp = ensure_crs_match(shp_df, src.crs)
            
            file_info = parse_filename(os.path.basename(src_path))
            
            geometries = aligned_shp[
                (aligned_shp['区划码'].astype(str) == file_info['loc_code']) & 
                (aligned_shp['省级码'].astype(str) == file_info['prov_code'])
            ].geometry
            
            if geometries.empty:
                return f"未找到匹配矢量: {file_info['loc_code']}-{file_info['prov_code']}"

            lulc_file = f"LULC_{file_info['prov_code']}_{file_info['loc_code']}_{file_info['year']}.tif"
            lulc_path = os.path.join(lulc_dir, lulc_file)
            
            if not os.path.exists(lulc_path):
                return f"找不到LULC文件: {lulc_file}"

            # 处理LULC文件（同样需要坐标系对齐）
            with rasterio.open(lulc_path) as lulc_src:
                lulc_aligned_shp = ensure_crs_match(shp_df, lulc_src.crs)
                lulc_geometries = lulc_aligned_shp[
                    (lulc_aligned_shp['区划码'].astype(str) == file_info['loc_code']) & 
                    (lulc_aligned_shp['省级码'].astype(str) == file_info['prov_code'])
                ].geometry
                
                lulc_data, _ = mask(lulc_src, lulc_geometries, crop=True, all_touched=True)
                lulc_mask = (lulc_data[0] == LULC_TARGET_VALUE)

            # 执行主裁剪
            out_image, out_transform = mask(
                src, geometries, crop=True, all_touched=True, indexes=range(1, src.count + 1)
            )
            
            # 应用LULC掩膜
            for i in range(out_image.shape[0]):
                out_image[i][~lulc_mask] = src.nodata if src.nodata else -9999

            # 写入结果
            meta = src.meta.copy()
            meta.update({
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform
            })
            
            with rasterio.open(dst_path, "w", **meta) as dst:
                dst.write(out_image)
        return "完成"
    except Exception as e:
        return f"错误: {str(e)}"

def process_data():
    """主处理流程"""
    # 初始化目录
    input_dir = os.path.join(BASE_PATH, "UTM_All")
    resampled_dir = os.path.join(BASE_PATH, "Resampled")
    clipped_dir = os.path.join(BASE_PATH, "Clipped")
    os.makedirs(resampled_dir, exist_ok=True)
    os.makedirs(clipped_dir, exist_ok=True)

    # 加载矢量数据（原始坐标系）
    shp_df = gpd.read_file(SHP_PATH)

    # 获取待处理文件列表
    tif_files = [f for f in os.listdir(input_dir) if f.endswith('.tif')]
    
    # 阶段1：并行重采样
    print(">>> 正在执行重采样 <<<")
    resample_results = Parallel(n_jobs=N_JOBS,verbose=10)(
        delayed(resample_image)(
            os.path.join(input_dir, f),
            os.path.join(resampled_dir, f)
        ) for f in tif_files
    )
    for f, status in zip(tif_files, resample_results):
        print(f"{f[:30]}... {status}")

    # 阶段2：并行裁剪（自动坐标系对齐）
    print("\n>>> 正在执行LULC掩膜裁剪 <<<")
    clip_results = Parallel(n_jobs=N_JOBS,verbose=10)(
        delayed(clip_with_lulc)(
            os.path.join(resampled_dir, f),
            os.path.join(clipped_dir, f),
            shp_df,
            resampled_dir
        ) for f in tif_files
    )
    for f, status in zip(tif_files, clip_results):
        print(f"{f[:30]}... {status}")

if __name__ == "__main__":
    process_data()
