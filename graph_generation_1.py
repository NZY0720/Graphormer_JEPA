import json
import geopy.distance
from tqdm import tqdm
import itertools

# 计算两个点之间的距离
def calculate_distance(coord1, coord2):
    # 确保坐标顺序是 (纬度, 经度)
    return geopy.distance.distance((coord1[1], coord1[0]), (coord2[1], coord2[0])).km  # 返回距离（千米）

# 使用固定的最大距离（10公里）进行归一化
def calculate_normalized_distance(coord1, coord2, max_distance=10.0):
    dist = calculate_distance(coord1, coord2)
    return dist / max_distance if dist <= max_distance else 1.0  # 归一化处理

# 读取GeoJSON文件
def load_nodes_from_file(file_path):
    print("Loading nodes from file...")
    with open(file_path, 'r') as f:
        data = json.load(f)
    print(f"Loaded {len(data['features'])} nodes.")
    return {node['properties']['osmid']: node for node in tqdm(data['features'], desc="Processing nodes", unit="node")}

def load_edges_from_file(file_path):
    print("Loading edges from file...")
    with open(file_path, 'r') as f:
        data = json.load(f)
    print(f"Loaded {len(data['features'])} edges.")
    return data['features']

# 创建图结构
def build_graph(node_file, edge_file):
    nodes = load_nodes_from_file(node_file)  # 加载节点数据
    edges = load_edges_from_file(edge_file)  # 加载边数据

    graph_edges = []

    # 遍历边数据
    for feature in tqdm(edges, desc="Processing edges", unit="edge"):
        u = feature['properties']['u']
        v = feature['properties']['v']
        coordinates = feature['geometry']['coordinates']

        # 获取节点坐标
        if u in nodes and v in nodes:
            node_u_coord = (nodes[u]['properties']['x'], nodes[u]['properties']['y'])
            node_v_coord = (nodes[v]['properties']['x'], nodes[v]['properties']['y'])

            # 计算归一化距离并添加到边列表中
            normalized_dist = calculate_normalized_distance(node_u_coord, node_v_coord)
            graph_edges.append({
                'source': u,
                'target': v,
                'weight': normalized_dist
            })

    # 构建图结构
    graph = {
        'nodes': list(nodes.values()),
        'edges': graph_edges
    }

    return graph

# 将图数据保存为JSON文件
def save_graph_to_file(graph, output_file):
    with open(output_file, 'w') as f:
        json.dump(graph, f, indent=4)
    print(f"Graph saved to {output_file}")

# 主函数
def main():
    node_file = 'san_francisco_nodes.geojson'  # 节点文件路径
    edge_file = 'san_francisco_edges.geojson'  # 边文件路径
    output_file = 'san_francisco_graph.json'  # 输出文件路径

    print("Starting graph generation...")
    # 构建图
    graph = build_graph(node_file, edge_file)

    # 保存图到文件
    save_graph_to_file(graph, output_file)

if __name__ == "__main__":
    main()
