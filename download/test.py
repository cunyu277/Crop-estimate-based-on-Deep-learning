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