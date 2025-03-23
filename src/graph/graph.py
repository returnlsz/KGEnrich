from collections import defaultdict

class Graph:
    def __init__(self):
        # 使用字典表示邻接表,键为实体,值为一个列表,其中每个元素是一个(关系, 目标实体)元组
        self.adjacency_list = defaultdict(list)

    def add_triplet(self, triplet):
        """添加三元组"""
        if len(triplet) != 3:
            raise ValueError("三元组的长度必须为3")
        subject, relation, obj = triplet
        # 在邻接表中添加三元组信息
        self.adjacency_list[subject].append((relation, obj))

    def get_one_hop_paths(self, topics):
        """给定一个或多个topic entity,识别所有1跳路径"""
        one_hop_paths = []
        # 检查是否有任何一个topic出现在邻接表中
        has_topic_in_graph = any(topic in self.adjacency_list for topic in topics)

        if has_topic_in_graph:
            # 如果topics存在于邻接表中，遍历指定topics的1跳路径
            for topic in topics:
                if topic in self.adjacency_list:
                    # 遍历与该实体相连的所有关系和目标实体
                    for relation, obj in self.adjacency_list[topic]:
                        one_hop_paths.append([topic, relation, obj])
        else:
            # 如果没有任何一个topic在邻接表中出现，遍历所有实体的1跳路径
            for subject, relations in self.adjacency_list.items():
                for relation, obj in relations:
                    one_hop_paths.append([subject, relation, obj])

        return one_hop_paths

    def get_two_hop_paths(self, topics):
        """给定一个或多个topic entity,识别所有2跳路径"""
        two_hop_paths = []
        # 检查是否有任何一个topic出现在邻接表中
        has_topic_in_graph = any(topic in self.adjacency_list for topic in topics)

        if has_topic_in_graph:
            # 如果topics存在于邻接表中，遍历指定topics的2跳路径
            for topic in topics:
                if topic in self.adjacency_list:
                    # 遍历与该实体相连的所有关系和目标实体
                    for relation1, intermediate in self.adjacency_list[topic]:
                        if intermediate in self.adjacency_list:
                            # 遍历中间实体的所有关系和目标实体
                            for relation2, obj in self.adjacency_list[intermediate]:
                                two_hop_paths.append([[topic, relation1, intermediate], 
                                                      [intermediate, relation2, obj]])
        else:
            # 如果没有任何一个topic在邻接表中出现，遍历所有实体的2跳路径
            for subject, relations in self.adjacency_list.items():
                for relation1, intermediate in relations:
                    if intermediate in self.adjacency_list:
                        for relation2, obj in self.adjacency_list[intermediate]:
                            two_hop_paths.append([[subject, relation1, intermediate], 
                                                  [intermediate, relation2, obj]])

        return two_hop_paths

# 使用示例
if __name__ == "__main__":
    # 创建图
    graph = Graph()

    # 添加三元组
    graph.add_triplet(["e1", "r1", "e2"])
    graph.add_triplet(["e2", "r2", "e3"])
    graph.add_triplet(["e1", "r3", "e4"])
    graph.add_triplet(["e4", "r4", "e5"])

    # 查询1跳路径
    one_hop = graph.get_one_hop_paths(["e1", "e4"])
    print("1跳路径:", one_hop)

    # 查询2跳路径
    two_hop = graph.get_two_hop_paths(["e1"])
    print("2跳路径:", two_hop)
