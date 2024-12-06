import json
import pandas as pd
from geopy.distance import geodesic
from tqdm import tqdm  # 导入 tqdm

# 加载充电站 CSV 文件
charging_stations = pd.read_csv('charging_stations.csv')

# 加载图数据（san_francisco_graph.json）
with open('san_francisco_graph.json', 'r') as f:
    graph_data = json.load(f)

# 计算两个坐标点之间的地理距离（单位：米）
def get_distance(lat1, lon1, lat2, lon2):
    return geodesic((lat1, lon1), (lat2, lon2)).meters

# 遍历充电站数据并与图中的节点进行比较
for index, row in tqdm(charging_stations.iterrows(), total=charging_stations.shape[0], desc="Processing Charging Stations"):
    station_lat = row['AddressInfo.Latitude']
    station_lon = row['AddressInfo.Longitude']

    # 遍历图中的节点
    for node in graph_data['nodes']:
        # 获取节点的经纬度，位于 'geometry.coordinates' 中
        node_lon, node_lat = node['geometry']['coordinates']
        
        # 计算充电站和节点之间的距离
        distance = get_distance(station_lat, station_lon, node_lat, node_lon)
        
        # 如果距离小于100米（可以调整该阈值），为节点添加 'has_charging_station' 特征
        if distance < 100:
            node['properties']['has_charging_station'] = 1  # 有充电站
        elif 'has_charging_station' not in node['properties']:
            node['properties']['has_charging_station'] = 0  # 没有充电站

# 保存更新后的图数据
with open('updated_san_francisco_graph.json', 'w') as f:
    json.dump(graph_data, f, indent=4)

print("Updated graph has been saved to 'updated_san_francisco_graph.json'")
