'''
Author: cunyu277 2465899266@qq.com
Date: 2025-04-09 21:15:55
LastEditors: cunyu277 2465899266@qq.com
LastEditTime: 2025-04-14 10:11:20
FilePath: \crop_yield_prediction\cunuu\download\temperature_year.py
Description: 

Copyright (c) 2025 by yh, All Rights Reserved. 
'''
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

def export_oneimage(img, folder, name, scale, crs):
    task = ee.batch.Export.image.toDrive(
        image=img,
        description=name,
        folder=folder,
        fileNamePrefix=name,
        scale=scale,
        crs=crs,
        maxPixels=1e13
    )
    task.start()
    while task.status()['state'] == 'RUNNING':
        print('Running...')
        time.sleep(10)
    print('Done.', task.status())

# 加载数据
north_china_plain = ee.FeatureCollection('projects/ee-hao0801277/assets/NCC')
locations = pd.read_csv(r"E:\Crop\华北平原\locations_new.csv")  # 使用原始字符串避免转义
china_bbox = north_china_plain.geometry().bounds()

# 按年份处理
for year in range(2022, 2023):
    print(f"\n=== 处理 {year} 年数据 ===")
    
    # 获取当年生长季数据（3月1日-7月31日）
    yearly_coll = ee.ImageCollection('MODIS/061/MYD11A2') \
        .filterBounds(china_bbox) \
        .filterDate(f'{year}-03-01', f'{year}-07-15')
    
    if yearly_coll.size().getInfo() == 0:
        print(f"⚠️  {year} 年无数据，跳过")
        continue
    
    def process_image(img):
        date_str = img.date().format('YYYYMMdd')
        lstd = img.select('LST_Day_1km').multiply(0.02).subtract(273.15) \
            .rename(ee.String('LSTd_').cat(date_str))
        lstn = img.select('LST_Night_1km').multiply(0.02).subtract(273.15) \
            .rename(ee.String('LSTn_').cat(date_str))
        return lstd.addBands(lstn)

    yearly_img = yearly_coll.map(process_image).toBands()


    # 逐个区县处理
    for _, row in locations.iterrows():
        province_code = str(int(row['province']))
        district_code = str(int(row['district']))
        fname = f'{province_code}_{district_code}_{year}'
        
        # 获取行政区划
        region = north_china_plain.filter(
            ee.Filter.And(
                ee.Filter.eq('省级码', province_code),
                ee.Filter.eq('区划码', district_code)
            )
        ).first()
        
        if region.getInfo() is None:
            print(f'  未找到区域: {province_code}_{district_code}')
            continue
        
        # 带重试机制的导出
        retries = 3
        for attempt in range(retries):
            try:
                print(f'  导出: {fname} (尝试 {attempt+1}/{retries})')
                export_oneimage(
                    yearly_img.clip(region),
                    'Temperature_new',  # 谷歌云端文件夹名称
                    fname,
                    scale=1000,         # MODIS温度数据分辨率
                    crs='EPSG:4326'
                )
                break
            except Exception as e:
                print(f'    出错: {str(e)}')
                if attempt == retries - 1:
                    print(f'    ❌ {fname} 导出失败，跳过')
                time.sleep(30)

print("\n全部处理完毕！")