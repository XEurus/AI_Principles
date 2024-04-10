import networkx as nx
import matplotlib.pyplot as plt

# 生成平衡树状图
tree = nx.balanced_tree(3, 2)

# 设置布局
pos = nx.graphviz_layout(tree, prog='dot')

# 绘制图形
nx.draw(tree, pos, with_labels=True, node_size=500, node_color='lightblue', font_size=10, font_weight='bold')
plt.axis('off')
plt.show()
