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
north_china_plain = ee.FeatureCollection('projects/ee-hao0801277/assets/NCC')

# 2. 加载位置信息CSV
locations = pd.read_csv('locations_final.csv', header=None)

# 3. 获取MODIS数据范围
china_bbox = north_china_plain.geometry().bounds()

# 4. 处理MODIS数据
imgcoll = ee.ImageCollection('MODIS/MOD09A1') \
    .filterBounds(china_bbox) \
    .filterDate('2002-12-31', '2016-8-4')

# 转换图像集合为单图像
def appendBand(current, previous):
    previous = ee.Image(previous)
    current = current.select([0,1,2,3,4,5,6])
    accum = ee.Algorithms.If(ee.Algorithms.IsEqual(previous, None), 
                           current, 
                           previous.addBands(ee.Image(current)))
    return accum

img = imgcoll.iterate(appendBand)
img = ee.Image(img)

# 设置数值范围
img_0 = ee.Image(ee.Number(-100))
img_16000 = ee.Image(ee.Number(16000))
img = img.min(img_16000).max(img_0)

# 逐个区县处理
for loc1, loc2, lat, lon in locations.values:
    fname = f'{int(loc1)}_{int(loc2)}'
    
    # 筛选当前区县
    region = north_china_plain.filter(
        ee.Filter.And(
            ee.Filter.eq('省级码', int(loc1)),
            ee.Filter.eq('区划码', int(loc2))
        )
    ).first()
    
    # 检查区域有效性
    try:
        if region.getInfo() is None:
            print(f'未找到区域: {loc1}_{loc2}')
            continue
    except Exception as e:
        print(f'获取区域失败: {loc1}_{loc2} - {str(e)}')
        continue
    
    # 导出（带重试机制）
    retries = 3
    for attempt in range(retries):
        try:
            print(f'导出: {fname} (尝试 {attempt+1}/{retries})')
            export_oneimage(
                img.clip(region),
                'test',  # 修改为您需要的Google Drive文件夹名
                fname,
                scale=500,
                crs='EPSG:4326'
            )
            break
        except Exception as e:
            print(f'出错: {str(e)}')
            if attempt == retries - 1:
                print(f'❌ {fname} 导出失败，跳过')
            time.sleep(30)

print("\n全部处理完毕！")
