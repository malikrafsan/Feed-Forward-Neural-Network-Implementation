# pip install networkx
import networkx as nx
import matplotlib.pyplot as plt
# from networkx.drawing.nx_agraph import graphviz_layout

def drawANNVisualization(annModelInstance: ANNModel):
  G = nx.DiGraph()

  for i, layer in enumerate(annModelInstance.layers):
    if(layer.id != 0 and layer.id != annModelInstance.countLayer() - 1): # Not input layer or hidden layer
      G.add_node(f'Bias: {i}', pos=(i, -1))

    for j, node in enumerate(layer.nodes):
      counter = 0
      G.add_node(node.id, pos=(i, j))
      if(layer.layerType == LayerType.Input):
        pass
      else:
        prevLayer = annModelInstance.getPreviousLayer(layer)
        for prevNode in prevLayer.nodes:
          currWeight = node.weight[counter]
          G.add_edge(prevNode.id, node.id, weight=currWeight)
          counter +=1

        if(prevLayer.id != 0):
            G.add_edge(f'Bias: {i - 1}', node.id, weight = node.bias)

  pos=nx.get_node_attributes(G,'pos')
  edgeLabel = nx.get_edge_attributes(G, 'weight')
  plt.figure(figsize=(10,5))
  ax = plt.gca()
  ax.set_title(f'ANN Model Visualization for {modelFileName}')
  nx.draw(G,with_labels=True,pos=pos, font_weight='bold', ax=ax)
  nx.draw_networkx_edge_labels(G, pos, edge_labels=edgeLabel)

# * Test
drawANNVisualization(annModelInstance)