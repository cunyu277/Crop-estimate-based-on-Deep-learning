import ee
import time
import pandas as pd

# 初始化Earth Engine
ee_project = "ee-hao0801277"  # 替换为您的GEE项目ID
try:
    ee.Initialize(project=ee_project)
    print(f"✅ 已成功初始化，当前项目: {ee.data.getAssetRoots()[0]['id']}")
except Exception as e:
    print(f"❌ 初始化失败: {str(e)}")

# 导出函数
def export_oneimage(img, folder, name, scale, crs):
    """导出单个图像到Google Drive"""
    task = ee.batch.Export.image.toDrive(
        image=img,
        description=name,
        folder='LULC',
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
north_china_plain = ee.FeatureCollection('projects/ee-hao0801277/assets/NCC')

# 2. 加载位置信息CSV
locations = pd.read_csv("E:\Crop\华北平原\locations_new.csv")

# 3. 获取MODIS土地覆盖数据范围
china_bbox = north_china_plain.geometry().bounds()

# 4. 按年份处理的主循环
for year in range(2022, 2023):  # 处理2013-2022年数据
    print(f"\n=== 开始处理 {year} 年数据 ===")
    
    # 筛选当前年份数据（确保包含完整的年度）
    yearly_imgcoll = ee.ImageCollection('MODIS/061/MCD12Q1') \
        .filterBounds(china_bbox) \
        .filterDate(f'{year}-01-01', f'{year}-12-31')  # 按完整年份筛选
    
    # 检查是否有数据
    if yearly_imgcoll.size().getInfo() == 0:
        print(f"⚠️  {year} 年无数据，跳过")
        continue
    
    # 获取该年份的第一幅影像（确保日期准确）
    first_img = ee.Image(yearly_imgcoll.first())
    # img_date = first_img.date().format('YYYYMMdd').getInfo()
    img_date = first_img.date().format('YYYY').getInfo()
    
    # 逐个区县处理
    for _, row in locations.iterrows():
        province_code = str(int(row['province']))
        district_code = str(int(row['district']))
        fname = f'{province_code}_{district_code}_{img_date}'
        
        # 筛选当前区县
        region = north_china_plain.filter(
            ee.Filter.And(
                ee.Filter.eq('省级码', province_code),
                ee.Filter.eq('区划码', district_code)
            )
        ).first()
        
        # 检查区域有效性
        try:
            if region.getInfo() is None:
                print(f'  未找到区域: {province_code}_{district_code}')
                continue
        except Exception as e:
            print(f'  获取区域失败: {province_code}_{district_code} - {str(e)}')
            continue
        
        # 导出（带重试机制）
        retries = 3
        for attempt in range(retries):
            try:
                print(f'  导出: {fname} (尝试 {attempt+1}/{retries})')
                export_oneimage(
                    first_img.clip(region).select('LC_Type1'),  # 明确选择土地分类波段
                    'NorthChina_LandCover',
                    fname,
                    scale=500,
                    crs='EPSG:4326'
                )
                break
            except Exception as e:
                print(f'    出错: {str(e)}')
                if attempt == retries - 1:
                    print(f'    ❌ {fname} 导出失败，跳过')
                time.sleep(30)
    
    print(f"=== {year} 年处理完成 ===")

print("\n全部处理完毕！")
