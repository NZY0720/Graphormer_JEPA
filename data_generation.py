import osmnx as ox
import pandas as pd
import torch
from torch_geometric.data import Data
import requests
import os
import math

def get_roads(place):
    """
    下载道路网络数据从OSM。

    Args:
        place (str): 地区名称，例如 'San Francisco, California, USA'

    Returns:
        roads_gdf (GeoDataFrame): 道路作为边
        nodes_gdf (GeoDataFrame): 节点及其位置
        G (networkx.Graph): 道路网络图
    """
    print(f"Downloading road network data for {place}...")
    G = ox.graph_from_place(place, network_type='drive')
    roads_gdf, nodes_gdf = ox.graph_to_gdfs(G)
    print(f"Road network downloaded: {len(nodes_gdf)} nodes, {len(roads_gdf)} edges.")
    return roads_gdf, nodes_gdf, G

def get_charging_stations(api_key, place):
    """
    从OpenChargeMap下载充电站数据。

    Args:
        api_key (str): OpenChargeMap API密钥
        place (dict): 包含 'countrycode', 'latitude', 'longitude', 'distance' 的字典

    Returns:
        charging_df (DataFrame): 充电站数据
    """
    print("Fetching charging stations data from OpenChargeMap...")
    url = 'https://api.openchargemap.io/v3/poi/'
    params = {
        'output': 'json',
        'countrycode': place['countrycode'],
        'latitude': place['latitude'],
        'longitude': place['longitude'],
        'distance': place['distance'],  # 单位为公里
        'distanceunit': 'KM',
        'maxresults': 1000,
        'compact': True,
        'verbose': False,
        'key': api_key
    }
    response = requests.get(url, params=params)
    if response.status_code != 200:
        raise Exception(f"Error fetching charging stations: {response.status_code}")
    data = response.json()
    charging_df = pd.json_normalize(data)
    print(f"Number of charging stations fetched: {len(charging_df)}")
    return charging_df

def find_nearest_node(x, y, G):
    """
    找到图G中离给定坐标最近的节点。

    Args:
        x (float): 经度
        y (float): 纬度
        G (networkx.Graph): 道路网络图

    Returns:
        nearest_node (int): 最近节点的ID
    """
    return ox.distance.nearest_nodes(G, X=x, Y=y)

def build_graph(roads_gdf, nodes_gdf, G, charging_df):
    """
    构建包含充电站的图数据集。

    Args:
        roads_gdf (GeoDataFrame): 道路作为边
        nodes_gdf (GeoDataFrame): 节点及其位置
        G (networkx.Graph): 道路网络图
        charging_df (DataFrame): 充电站数据

    Returns:
        data (PyG Data object): 包含节点和边的图数据
    """
    print("Building graph data with charging stations...")
    # 获取现有节点ID
    existing_node_ids = list(G.nodes())
    max_node_id = max(existing_node_ids)
    
    # 从max_node_id +1开始为充电站分配新的节点ID
    new_node_id = max_node_id + 1
    
    # 初始化节点特征列表和边索引列表
    node_features = []
    node_id_mapping = {}  # 将原始节点ID映射到node_features中的索引
    # 现有节点：使用其位置作为特征，例如 [x, y]（经度和纬度）
    for node_id, data in nodes_gdf.iterrows():
        lon = data['x']
        lat = data['y']
        node_features.append([lon, lat])
        node_id_mapping[node_id] = len(node_features) -1  # 映射节点ID到索引
    
    edge_index_list = []
    
    # 添加道路边
    for idx, row in roads_gdf.iterrows():
        u = row['u']
        v = row['v']
        # 获取PyG节点索引
        u_pyg = node_id_mapping.get(u)
        v_pyg = node_id_mapping.get(v)
        if u_pyg is None or v_pyg is None:
            continue  # 跳过未找到的节点
        # 无向图，添加双向边
        edge_index_list.append([u_pyg, v_pyg])
        edge_index_list.append([v_pyg, u_pyg])
    
    # 添加充电站节点及其连接
    for idx, row in charging_df.iterrows():
        lat = row.get('AddressInfo.Latitude')
        lon = row.get('AddressInfo.Longitude')
        if pd.isna(lat) or pd.isna(lon):
            continue  # 如果没有位置信息，跳过
        # 分配新的节点ID
        charging_node_id = new_node_id
        new_node_id +=1
        # 添加充电站节点特征
        node_features.append([lon, lat])
        node_id_mapping[charging_node_id] = len(node_features) -1
        # 找到最近的道路节点
        try:
            nearest_node = find_nearest_node(lon, lat, G)
        except Exception as e:
            print(f"Error finding nearest node for charging station at ({lat}, {lon}): {e}")
            continue
        # 获取PyG节点索引
        road_pyg_idx = node_id_mapping.get(nearest_node)
        charging_pyg_idx = node_id_mapping.get(charging_node_id)
        if road_pyg_idx is None:
            print(f"Road node {nearest_node} not found in node_id_mapping.")
            continue
        # 计算两个节点之间的欧氏距离作为边的长度
        road_node_feat = nodes_gdf.loc[nearest_node]
        road_lon = road_node_feat['x']
        road_lat = road_node_feat['y']
        distance = math.sqrt((lon - road_lon)**2 + (lat - road_lat)**2)
        # 添加边（双向）
        edge_index_list.append([road_pyg_idx, charging_pyg_idx])
        edge_index_list.append([charging_pyg_idx, road_pyg_idx])
    
    # 将节点特征转换为Tensor
    x = torch.tensor(node_features, dtype=torch.float)
    
    # 将边索引转换为Tensor
    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    
    # 创建PyG Data对象
    data = Data(x=x, edge_index=edge_index)
    print(f"Graph data built: {data.num_nodes} nodes, {data.num_edges//2} edges.")
    return data

def save_graph_data(data, save_path):
    """
    保存PyG Data对象到磁盘。

    Args:
        data (Data): PyG Data对象
        save_path (str): 保存路径，例如 'saved_models/graph_data.pt'
    """
    torch.save(data, save_path)
    print(f"Graph data saved to {save_path}")

def main():
    # 设定区域
    place = {
        'place': 'San Francisco, California, USA',
        'countrycode': 'US',
        'latitude': 37.7749,
        'longitude': -122.4194,
        'distance': 10  # 10 KM半径
    }
    
    # 获取API Key
    api_key = '77f93e38-f988-4da2-a184-c41ca4986445'  # 请替换为你的实际API密钥
    if api_key == 'YOUR_OPENCHARGEMAP_API_KEY':
        raise ValueError("Please replace 'YOUR_OPENCHARGEMAP_API_KEY' with your actual OpenChargeMap API key.")
    
    # 创建保存模型的目录
    model_dir = "saved_models"
    os.makedirs(model_dir, exist_ok=True)
    
    # 获取道路网络
    roads_gdf, nodes_gdf, G = get_roads(place['place'])
    
    # 获取充电站数据
    charging_df = get_charging_stations(api_key, place)
    
    # 构建图数据集
    graph_data = build_graph(roads_gdf, nodes_gdf, G, charging_df)
    
    # 保存图数据
    save_path = os.path.join(model_dir, "graph_data.pt")
    save_graph_data(graph_data, save_path)

    # 可选：可视化道路网络和充电站
    # visualize_graph(roads_gdf, charging_df)

def visualize_graph(roads_gdf, charging_df):
    """
    可视化道路网络和充电站分布。

    Args:
        roads_gdf (GeoDataFrame): 道路数据
        charging_df (DataFrame): 充电站数据
    """
    import matplotlib.pyplot as plt
    import geopandas as gpd
    from shapely.geometry import Point

    print("Visualizing road network and charging stations...")
    
    # 创建GeoDataFrame for charging stations
    charging_stations_gdf = gpd.GeoDataFrame(
        charging_df,
        geometry=gpd.points_from_xy(charging_df['AddressInfo.Longitude'], charging_df['AddressInfo.Latitude']),
        crs=roads_gdf.crs
    )
    
    fig, ax = plt.subplots(figsize=(12, 12))
    roads_gdf.plot(ax=ax, linewidth=0.5, edgecolor='grey')
    charging_stations_gdf.plot(ax=ax, marker='o', color='red', markersize=50, label='Charging Stations')
    plt.legend()
    plt.title('Road Network and Charging Stations in San Francisco')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()

if __name__ == "__main__":
    main()
