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
    exit()

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
locations = pd.read_csv(r"E:\Crop\华北平原\locations_new.csv")
china_bbox = north_china_plain.geometry().bounds()

# 按年份处理
for year in range(2010, 2023):
    print(f"\n=== 处理 {year} 年数据 ===")
    
    # 获取当年生长季数据（3月1日-7月31日）
    yearly_coll = ee.ImageCollection('MODIS/061/MOD15A2H') \
        .filterBounds(china_bbox) \
        .filterDate(f'{year}-03-01', f'{year}-07-07') \
        .select('Fpar_500m')  # 明确使用Fpar_500m波段
    
    if yearly_coll.size().getInfo() == 0:
        print(f"⚠️  {year} 年无数据，跳过")
        continue
    
    # 处理每景影像（添加日期后缀）
    def process_image(img):
        date_str = img.date().format('YYYYMMdd')
        fpar = img.select('Fpar_500m').rename(ee.String('FPAR_').cat(date_str))
        return fpar

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

        geometry = region.geometry()
        if geometry.area().getInfo() == 0:  # 判断几何面积是否为0
            print(f'  区域几何无效: {province_code}_{district_code}')
            continue

        # 带重试机制的导出
        retries = 3
        for attempt in range(retries):
            try:
                print(f'  导出: {fname} (尝试 {attempt+1}/{retries})')
                export_oneimage(
                    yearly_img.reproject('EPSG:4326').clip(region.geometry()),
                    'FPAR',  # 谷歌云端文件夹名称
                    fname,
                    scale=500,    # 保持与Fpar_500m分辨率一致
                    crs='EPSG:4326'
                )
                break
            except Exception as e:
                print(f'    出错: {str(e)}')
                if attempt == retries - 1:
                    print(f'    ❌ {fname} 导出失败，跳过')
                time.sleep(30)

print("\n全部处理完毕！")
