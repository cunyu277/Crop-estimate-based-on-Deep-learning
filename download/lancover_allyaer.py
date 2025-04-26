import ee
import time
import pandas as pd

# 初始化Earth Engine
ee_project = "ee-hao0801277"
try:
    ee.Initialize(project=ee_project)
    print(f"✅ 已成功初始化，当前项目: {ee.data.getAssetRoots()[0]['id']}")
except Exception as e:
    print(f"❌ 初始化失败: {str(e)}")

# 导出函数（保持不变）
def export_oneimage(img, folder, name, scale, crs):
    """导出单个图像到Google Drive"""
    task = ee.batch.Export.image.toDrive(
        image=img,
        description=name,
        folder=folder,
        fileNamePrefix=name,
        scale=scale,
        crs=crs
    )
    task.start()
    while task.status()['state'] == 'RUNNING':
        print('Running...')
        time.sleep(10)
    print('Done.', task.status())

# 1. 加载华北平原区县数据
north_china_plain = ee.FeatureCollection('projects/ee-hao0801277/assets/NCc')

# 2. 加载位置信息CSV
locations = pd.read_csv('./1 download data/locations_new.csv')

# 3. 获取MODIS土地覆盖数据
china_bbox = north_china_plain.geometry().bounds()
imgcoll = ee.ImageCollection('MODIS/061/MCD12Q1') \
    .filterBounds(china_bbox) \
    .filterDate('2013-12-31', '2023-12-31')  # 修改为您需要的日期范围

# 4. 为每个影像添加年份标记
def add_year_band(img):
    """为波段添加年份后缀（例如LC_Type1_2013）"""
    year = ee.Date(img.get('system:time_start')).format('YYYY')
    return img.select([0]).rename(ee.String('LC_Type1_').cat(year))

# 应用年份标记并合并波段
imgcoll_with_year = imgcoll.map(add_year_band)
img = imgcoll_with_year.toBands()  # 自动合并所有波段

# 5. 导出每个区县的多年度合并数据
for _, row in locations.iterrows():
    province_code = str(int(row['province']))
    district_code = str(int(row['district']))
    fname = f'{province_code}_{district_code}_multiyear'  # 文件名示例: 110000_110101_multiyear
    
    # 获取对应区县边界
    region = north_china_plain.filter(
        ee.Filter.And(
            ee.Filter.eq('省级码', province_code),
            ee.Filter.eq('区划码', district_code)
        )
    ).first()
    
    # 检查区域是否存在
    try:
        if region.getInfo() is None:
            print(f'未找到区域: {province_code}_{district_code}')
            continue
    except Exception as e:
        print(f'区域获取失败: {province_code}_{district_code}, 错误: {str(e)}')
        continue
    
    # 导出（带重试机制）
    retries = 3
    for attempt in range(retries):
        try:
            print(f'正在导出: {fname} (尝试 {attempt+1}/{retries})')
            export_oneimage(
                img.clip(region),  # 裁剪到当前区县
                'NorthChina_LandCover',
                fname,
                scale=500,
                crs='EPSG:4326'
            )
            break
        except Exception as e:
            print(f'导出失败: {str(e)}')
            if attempt == retries - 1:
                print(f'{fname} 导出最终失败')
            time.sleep(30)
