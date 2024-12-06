import json

def extract_json_structure(obj, seen=None, level=0):
    if seen is None:
        seen = set()  # 用于追踪已经处理过的结构
    
    if isinstance(obj, dict):
        # 用于表示字典结构的起始
        dict_key = "dict"  # 通过字段类型来标识
        if dict_key not in seen:
            print("  " * level + "{")
            seen.add(dict_key)
            for key, value in obj.items():
                print("  " * (level + 1) + f"{key}:")
                extract_json_structure(value, seen, level + 2)
            print("  " * level + "}")
    elif isinstance(obj, list):
        # 用于表示列表结构的起始
        list_key = "list"  # 通过字段类型来标识
        if list_key not in seen:
            print("  " * level + "[")
            seen.add(list_key)
            for item in obj:
                extract_json_structure(item, seen, level + 1)
            print("  " * level + "]")
    else:
        # 打印出其他类型
        print("  " * level + str(type(obj)))

# 加载 JSON 文件
with open('san_francisco_graph.json', 'r') as f:
    data = json.load(f)

# 提取并显示架构
extract_json_structure(data)
