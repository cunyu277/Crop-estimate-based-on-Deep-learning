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
locations = pd.read_csv("E:\Crop\华北平原\locations_new.csv")
china_bbox = north_china_plain.geometry().bounds()

# 按年份处理
for year in range(2010, 2023):  # 2003-2016年
    print(f"\n=== 处理 {year} 年数据 ===")
    
    # 获取当年数据
    yearly_coll = ee.ImageCollection('MODIS/061/MOD09A1') \
        .filterBounds(china_bbox) \
        .filterDate(f'{year}-03-01', f'{year}-07-07')
    
    if yearly_coll.size().getInfo() == 0:
        print(f"⚠️  {year} 年无数据，跳过")
        continue
    
    # 合成当年所有影像
    def appendBand(current, previous):
        previous = ee.Image(previous)
        current = current.select([0,1,2,3,4,5,6])
        accum = ee.Algorithms.If(ee.Algorithms.IsEqual(previous, None), 
                               current, 
                               previous.addBands(ee.Image(current)))
        return accum
    
    yearly_img = yearly_coll.iterate(appendBand)
    yearly_img = ee.Image(yearly_img)
    yearly_img = yearly_img.min(16000).max(-100)  # 设置数值范围
    
    # 获取该年份的第一幅影像（确保日期准确）
    first_img = ee.Image(yearly_coll.first())
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
                    'ENTIRE',  # 修改为您的文件夹名
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

print("\n全部处理完毕！")
