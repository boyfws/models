from DecisionTreeBase import DecisionTreeBase
from Node import Node
import pydot
from io import BytesIO
import matplotlib.pyplot as plt
from matplotlib.image import imread

def plot_tree(
        model: "DecisionTreeBase",
        feature_names: list[str] = None,
        format: str = "png",
) -> None:
    graph = pydot.Dot(graph_type="digraph", format=format)
    graph.set_node_defaults(shape="box", style="rounded")

    def add_node(node: Node, parent_id: str = None, branch_label: str = None) -> str:
        if node is None:
            return ""

        node_id = str(id(node))

        if node.left_node is None and node.right_node is None:
            label = f"value = {node.value:.2f}"
        else:
            feature_num = node.condition.feature_num
            feature_name = feature_names[feature_num] if feature_names else f"X[{feature_num}]"
            threshold = node.condition.t
            label = f"{feature_name} >= {threshold:.2f}"

        graph_node = pydot.Node(
            node_id,
            label=label,
        )
        graph.add_node(graph_node)

        if parent_id is not None:
            edge = pydot.Edge(parent_id, node_id, label=branch_label)
            graph.add_edge(edge)

        if node.left_node is not None:
            add_node(node.left_node, node_id, "False")

        if node.right_node is not None:
            add_node(node.right_node, node_id, "True")

        return node_id

    add_node(model.start_node_)

    png_data = graph.create_png(prog="dot")

    img = imread(BytesIO(png_data))
    plt.figure(figsize=(20, 20))
    plt.imshow(img)
    plt.axis("off")
    plt.show()
