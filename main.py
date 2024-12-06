import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, Batch
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import networkx as nx

from graph_model import GraphormerJEPA

# 输出总参数量的函数
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 使用GPU加速
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 超参数
input_dim = 4
hidden_dim = 32  # 从64减少到32
output_dim = 32  # 同样减少到32
lr = 1e-3
epochs = 20  # 根据需要调整
batch_size = 4  # 从16减少到4
max_degree = 256  # 从512减少到256
max_nodes = 100000  # 根据数据情况调整

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
    data.batch = torch.zeros(len(nodes), dtype=torch.long)
    data.ptr = torch.tensor([0, len(nodes)], dtype=torch.long)

    # 不将数据移动到GPU，避免多工作进程访问GPU
    return data

class GraphPairDataset(Dataset):
    def __init__(self, data, num_samples=1000):
        self.data = data
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 直接返回相同的图作为context和target
        return self.data.clone(), self.data.clone()

def custom_collate_fn(batch):
    context_list = [b[0] for b in batch]
    target_list = [b[1] for b in batch]

    # Batch.from_data_list保持在CPU上
    context_batch = Batch.from_data_list(context_list)
    target_batch = Batch.from_data_list(target_list)

    # 不在collate_fn中移动到GPU
    return context_batch, target_batch

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

if __name__ == "__main__":
    # 设置spawn启动方式以避免CUDA在子进程中初始化的问题
    import torch.multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # 方法已经设置

    # 加载数据
    data = load_data("updated_san_francisco_graph.json")

    # 检查最大节点度数
    print(f"Maximum node degree in the graph: {data.degree.max().item()}")

    # 构建数据集
    dataset = GraphPairDataset(data, num_samples=1000)  # 根据需要调整样本数量

    # 数据集划分：8:1:1
    total_len = len(dataset)
    train_len = int(0.8 * total_len)
    val_len = int(0.1 * total_len)
    test_len = total_len - train_len - val_len
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_len, val_len, test_len])

    # 数据加载器
    # 为了更快的数据加载，可以根据CPU核数调整num_workers
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn, num_workers=4)

    # 初始化模型、优化器和损失函数
    model = GraphormerJEPA(input_dim, hidden_dim, output_dim, max_degree=max_degree, max_nodes=max_nodes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # 输出模型的总参数量
    total_params = count_parameters(model)
    print(f"Total model parameters: {total_params}")

    best_val_loss = float('inf')
    best_model_path = "model_checkpoint.pt"

    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        with tqdm(train_dataloader, desc=f"Epoch {epoch}/{epochs}") as pbar:
            for context_batch, target_batch in pbar:
                # 将数据移动到GPU
                context_batch = context_batch.to(device, non_blocking=True)
                target_batch = target_batch.to(device, non_blocking=True)

                optimizer.zero_grad()
                predicted_embeddings, target_embeddings = model(context_batch, target_batch)
                loss = loss_fn(predicted_embeddings, target_embeddings)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                pbar.set_postfix({"train_loss": loss.item()})

        train_avg_loss = total_loss / len(train_dataloader)
        val_avg_loss = evaluate_model(model, val_dataloader, loss_fn)
        print(f"Epoch {epoch}: Train Loss={train_avg_loss:.4f}, Val Loss={val_avg_loss:.4f}")

        # 如果验证集表现更好，则保存模型
        if val_avg_loss < best_val_loss:
            best_val_loss = val_avg_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Model saved at epoch {epoch} with Val Loss={val_avg_loss:.4f}")

    # 使用最优模型在测试集上评估
    model.load_state_dict(torch.load(best_model_path))
    test_avg_loss = evaluate_model(model, test_dataloader, loss_fn)
    print(f"Test Set Loss: {test_avg_loss:.4f}")
    print("Training and evaluation completed.")
