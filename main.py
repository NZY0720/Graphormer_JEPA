import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset, random_split
from tqdm import tqdm
import networkx as nx
from networkx.algorithms import community
import community as community_louvain  # 来自 python-louvain

from graph_model import GraphormerJEPA

# 输出总参数量的函数
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 使用GPU加速
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 超参数
input_dim = 4
hidden_dim = 8    # 从16减少到8
output_dim = 8    # 同样减少到8
lr = 1e-3
epochs = 20       # 根据需要调整
batch_size = 1    # 保持为1
max_degree = 128  # 从256减少到128
max_nodes = 50000 # 从100000减少到50000
num_heads = 1     # 从2减少到1
num_layers = 1    # 从2减少到1
num_communities = 10  # 将图拆分为10个社区

def load_data(json_path):
    print("Loading and processing data...")
    with open(json_path, 'r') as f:
        graph_json = json.load(f)
    nodes = graph_json['nodes']
    edges = graph_json['edges']

    G = nx.Graph()
    node_osmids = [n['properties']['osmid'] for n in nodes]
    osmid_to_index = {osmid: i for i, osmid in enumerate(node_osmids)}

    for i, n in enumerate(nodes):
        G.add_node(i)

    # 显示边的处理进度
    for e in tqdm(edges, desc="Processing Edges"):
        s = osmid_to_index[e['source']]
        t = osmid_to_index[e['target']]
        G.add_edge(s, t)

    x_list = []
    degrees = []
    # 节点处理进度条
    for i in tqdm(range(len(nodes)), desc="Processing Nodes"):
        n = nodes[i]
        props = n['properties']
        x_coord = props['x']
        y_coord = props['y']
        station_flag = props['has_charging_station']
        x_list.append([x_coord, y_coord, station_flag, 1.0])
        degrees.append(G.degree(i))

    x = torch.tensor(x_list, dtype=torch.float)
    degree = torch.tensor(degrees, dtype=torch.long)

    data = Data(x=x, degree=degree)
    data.edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()

    return G, data

class GraphPairDataset(Dataset):
    def __init__(self, subgraphs, num_samples=1000):
        self.subgraphs = subgraphs
        self.num_samples = num_samples

    def __len__(self):
        return min(self.num_samples, len(self.subgraphs))

    def __getitem__(self, idx):
        # 直接返回子图作为context和target
        subgraph = self.subgraphs[idx]
        return subgraph, subgraph.clone()

def convert_nx_to_pyg(subgraph, x_full, node_id_map):
    """
    将 NetworkX 子图转换为 torch_geometric 的 Data 对象。
    """
    # 获取子图的节点和边
    nodes = list(subgraph.nodes())
    edges = list(subgraph.edges())

    # 创建新的节点特征
    x = x_full[nodes]

    # 创建新的边索引
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    # 重新映射节点索引
    mapping = {node: i for i, node in enumerate(nodes)}
    edge_index = torch.tensor([[mapping[e[0]], mapping[e[1]]] for e in edges], dtype=torch.long).t().contiguous()

    # 重新计算度数
    degrees = torch.tensor([subgraph.degree(n) for n in nodes], dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, degree=degrees)
    return data

def split_graph_into_subgraphs_louvain(G, data, num_communities):
    """
    使用 Louvain 方法将大图拆分为多个子图，并添加进度展示。

    参数:
        G (networkx.Graph): 要拆分的图。
        data (torch_geometric.data.Data): 原始图的torch_geometric数据对象。
        num_communities (int): 目标社区数量。

    返回:
        list of torch_geometric.data.Data: 拆分后的子图列表。
    """
    print("正在使用 Louvain 方法拆分大图为多个子图...")
    
    # 使用 Louvain 方法进行社区检测
    partition = community_louvain.best_partition(G)
    
    # 统计每个社区的节点
    communities = {}
    for node, comm_id in partition.items():
        communities.setdefault(comm_id, set()).add(node)
    
    # 如果社区数量超过目标数量，进一步合并小社区
    if len(communities) > num_communities:
        # 按社区大小排序，合并最小的社区
        sorted_communities = sorted(communities.values(), key=lambda x: len(x))
        while len(sorted_communities) > num_communities:
            # 合并最小的两个社区
            smallest = sorted_communities.pop(0)
            second_smallest = sorted_communities.pop(0)
            merged = smallest.union(second_smallest)
            sorted_communities.append(merged)
            sorted_communities = sorted(sorted_communities, key=lambda x: len(x))
        communities = {i: comm for i, comm in enumerate(sorted_communities)}
    elif len(communities) < num_communities:
        print(f"警告: 检测到的社区数量 ({len(communities)}) 少于目标数量 ({num_communities})。")
    
    subgraphs = []
    
    # 使用 tqdm 显示进度
    with tqdm(total=len(communities), desc="拆分进度", unit="子图") as pbar:
        for comm_id, nodes in communities.items():
            subG = G.subgraph(nodes).copy()
            sub_data = convert_nx_to_pyg(subG, data.x, {})
            subgraphs.append(sub_data)
            pbar.update(1)
    
    print(f"拆分后的子图数量: {len(subgraphs)}")
    return subgraphs

def evaluate_model(model, dataloader, loss_fn):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for context_batch, target_batch in dataloader:
            # 将数据移动到GPU
            context_batch = context_batch.to(device, non_blocking=True)
            target_batch = target_batch.to(device, non_blocking=True)
            predicted_embeddings, target_embeddings = model(context_batch, target_batch)
            loss = loss_fn(predicted_embeddings, target_embeddings)
            total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    return avg_loss

# 自定义 collate_fn
def custom_collate(batch):
    """
    自定义 collate_fn，将 batch 中的 (context, target) 对分别批处理。
    
    参数:
        batch (list of tuples): 每个元素是 (context, target)
    
    返回:
        tuple: (context_batch, target_batch) 两个 Batch 对象
    """
    context_list, target_list = zip(*batch)  # 分离 context 和 target
    context_batch = Batch.from_data_list(context_list)  # 批处理 context
    target_batch = Batch.from_data_list(target_list)    # 批处理 target
    return context_batch, target_batch

if __name__ == "__main__":
    # 设置spawn启动方式以避免CUDA在子进程中初始化的问题
    import torch.multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # 方法已经设置

    # 加载数据
    G, data = load_data("updated_san_francisco_graph.json")

    # 检查最大节点度数
    print(f"图中节点的最大度数: {data.degree.max().item()}")

    # 将大图拆分为多个子图
    subgraphs_nx = split_graph_into_subgraphs_louvain(G, data, num_communities=num_communities)

    # 将 NetworkX 子图转换为 torch_geometric 的 Data 对象
    # 此步骤已在拆分函数中完成，故无需重复转换
    subgraphs = subgraphs_nx

    print(f"转换后的子图数量: {len(subgraphs)}")

    # 构建数据集
    dataset = GraphPairDataset(subgraphs, num_samples=len(subgraphs))  # 使用所有子图

    # 数据集划分：80%训练，10%验证，10%测试
    total_len = len(dataset)
    train_len = int(0.8 * total_len)
    val_len = int(0.1 * total_len)
    test_len = total_len - train_len - val_len
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_len, val_len, test_len])

    # 数据加载器
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True,
        collate_fn=custom_collate  # 使用自定义 collate_fn
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True,
        collate_fn=custom_collate  # 使用自定义 collate_fn
    )
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True,
        collate_fn=custom_collate  # 使用自定义 collate_fn
    )

    # 初始化模型、优化器和损失函数
    model = GraphormerJEPA(input_dim, hidden_dim, output_dim, max_degree=max_degree, max_nodes=max_nodes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # 输出模型的总参数量
    total_params = count_parameters(model)
    print(f"模型的总参数量: {total_params}")

    best_val_loss = float('inf')
    best_model_path = "model_checkpoint.pt"

    # 初始化GradScaler用于混合精度训练
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        with tqdm(train_dataloader, desc=f"Epoch {epoch}/{epochs}") as pbar:
            for context_batch, target_batch in pbar:
                # 将数据移动到GPU
                context_batch = context_batch.to(device, non_blocking=True)
                target_batch = target_batch.to(device, non_blocking=True)

                optimizer.zero_grad()

                with torch.cuda.amp.autocast():
                    predicted_embeddings, target_embeddings = model(context_batch, target_batch)
                    loss = loss_fn(predicted_embeddings, target_embeddings)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item()
                pbar.set_postfix({"train_loss": loss.item()})

        train_avg_loss = total_loss / len(train_dataloader)
        val_avg_loss = evaluate_model(model, val_dataloader, loss_fn)
        print(f"Epoch {epoch}: Train Loss={train_avg_loss:.4f}, Val Loss={val_avg_loss:.4f}")

        # 如果验证集表现更好，则保存模型
        if val_avg_loss < best_val_loss:
            best_val_loss = val_avg_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"模型在第{epoch}轮保存，验证损失={val_avg_loss:.4f}")

    # 使用最优模型在测试集上评估
    model.load_state_dict(torch.load(best_model_path))
    test_avg_loss = evaluate_model(model, test_dataloader, loss_fn)
    print(f"测试集损失: {test_avg_loss:.4f}")
    print("训练和评估完成。")
