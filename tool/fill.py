import numpy as np
import rasterio
from rasterio import features
from rasterio.windows import Window
import geopandas as gpd
import os
from scipy.spatial import cKDTree
import time
from numba import cuda
import math

def safe_window(x, y, block_size, width, height):
    """确保窗口不超出栅格范围"""
    return Window(
        col_off=x,
        row_off=y,
        width=min(block_size, width - x),
        height=min(block_size, height - y)
    )

def dynamic_block_fill(input_raster, output_raster, vector_path):
    """修复窗口越界的动态分块填补"""
    # 环境设置
    os.environ['GDAL_TMPDIR'] = r'E:\Temp'
    
    # 加载矢量数据
    print("🕒 正在加载矢量数据...")
    gdf = gpd.read_file(vector_path)
    
    with rasterio.open(input_raster) as src:
        # 获取栅格尺寸
        width, height = src.width, src.height
        
        # 确保profile是字典类型
        profile = dict(src.profile)
        profile.update({
            'dtype': 'float32',
            'nodata': np.nan,
            'compress': 'LZW'
        })
        
        # 生成全局掩膜
        print("🔄 正在生成全局掩膜...")
        start_mask = time.time()
        mask = features.geometry_mask(
            gdf.to_crs(src.crs).geometry,
            out_shape=(height, width),
            transform=src.transform,
            invert=True
        )
        print(f"✅ 掩膜生成完成 [耗时: {time.time()-start_mask:.1f}s]")
        
        # 分析缺失值密度
        print("📊 分析缺失值分布...")
        sample_window = Window(0, 0, min(1024, width), min(1024, height))
        sample_data = src.read(1, window=sample_window)
        missing_density = np.sum(np.isnan(sample_data)) / sample_data.size
        
        # 动态分块策略
        base_block_size = 512
        if missing_density > 0.3:
            block_size = base_block_size // 2
            print(f"🔍 高缺失密度区域({missing_density:.1%})，使用小分块: {block_size}x{block_size}")
        else:
            block_size = base_block_size
            print(f"🔍 低缺失密度区域({missing_density:.1%})，使用标准分块: {block_size}x{block_size}")
        
        # GPU加速准备
        has_gpu = cuda.is_available()
        if has_gpu:
            print("🎮 检测到CUDA GPU，启用加速")
        else:
            print("⚠️ 未检测到CUDA GPU，使用CPU模式")
        
        with rasterio.open(output_raster, 'w', **profile) as dst:
            total_blocks = ((height + block_size - 1) // block_size) * ((width + block_size - 1) // block_size)
            processed = 0
            start_fill = time.time()
            
            print(f"🚀 开始处理 (总区块数: {total_blocks})")
            for y in range(0, height, block_size):
                for x in range(0, width, block_size):
                    # 创建安全窗口
                    window = safe_window(x, y, block_size, width, height)
                    
                    # 读取当前块
                    data = src.read(1, window=window).astype(np.float32)
                    win_height, win_width = data.shape
                    
                    # 获取掩膜区块
                    y_slice = slice(y, y + win_height)
                    x_slice = slice(x, x + win_width)
                    mask_chunk = mask[y_slice, x_slice]
                    
                    # 识别缺失区域
                    missing = np.isnan(data) & mask_chunk
                    missing_count = np.sum(missing)
                    
                    # 进度显示
                    processed += 1
                    elapsed = time.time() - start_fill
                    progress = processed / total_blocks * 100
                    speed = processed / elapsed if elapsed > 0 else 0
                    remain = (total_blocks - processed) / speed if speed > 0 else 0
                    
                    print(
                        f"\r区块 {processed}/{total_blocks} "
                        f"[{progress:.1f}%] "
                        f"缺失: {missing_count} "
                        f"剩余: {remain:.1f}s ",
                        end="", flush=True
                    )
                    
                    if missing_count > 0:
                        # 查找有效点
                        valid_points = np.argwhere(~np.isnan(data) & mask_chunk)
                        
                        if valid_points.size > 0:
                            if has_gpu:
                                # GPU加速填补
                                data = gpu_fill(data, missing, valid_points)
                            else:
                                # CPU填补
                                tree = cKDTree(valid_points)
                                dist, idx = tree.query(np.argwhere(missing))
                                data[missing] = data[tuple(valid_points[idx].T)]
                    
                    # 写入结果
                    dst.write(data, 1, window=window)
            
            print(f"\n✨ 填补完成! 总耗时: {time.time()-start_fill:.1f}秒")

@cuda.jit
def gpu_fill_kernel(data, missing_mask, valid_points, output):
    """GPU核函数"""
    i = cuda.grid(1)
    if i < missing_mask.size:
        if missing_mask.flat[i]:
            y = i // data.shape[1]
            x = i % data.shape[1]
            
            min_dist = np.inf
            nearest_val = np.nan
            
            for j in range(valid_points.shape[0]):
                dy = y - valid_points[j, 0]
                dx = x - valid_points[j, 1]
                dist = dy*dy + dx*dx  # 平方距离避免开方
                
                if dist < min_dist:
                    min_dist = dist
                    nearest_val = data[valid_points[j,0], valid_points[j,1]]
            
            output.flat[i] = nearest_val
        else:
            output.flat[i] = data.flat[i]

def gpu_fill(data, missing_mask, valid_points):
    """GPU填补封装函数"""
    output = np.empty_like(data)
    
    # 配置GPU网格
    threads_per_block = 256
    blocks_per_grid = math.ceil(missing_mask.size / threads_per_block)
    
    # 将数据复制到设备
    d_data = cuda.to_device(data)
    d_mask = cuda.to_device(missing_mask)
    d_points = cuda.to_device(valid_points.astype(np.int32))
    d_output = cuda.to_device(output)
    
    # 启动核函数
    gpu_fill_kernel[blocks_per_grid, threads_per_block](
        d_data, d_mask, d_points, d_output
    )
    
    # 将结果复制回主机
    return d_output.copy_to_host()

if __name__ == "__main__":
    input_raster = r"E:\Crop\DEM\mosaic.tif"
    output_raster = r"E:\Crop\DEM\mosaic_filled_final.tif"
    vector_file = r"E:\Crop\华北平原\NCC_all.shp"
    
    print("=== 安全窗口动态分块填补 ===")
    start = time.time()
    dynamic_block_fill(input_raster, output_raster, vector_file)
    print(f"总运行时间: {time.time()-start:.1f}秒")