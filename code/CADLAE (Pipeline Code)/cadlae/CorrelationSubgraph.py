import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms.components import connected_components

class CorrelationSubgraph:
    def __init__(self, df, threshold):
        self.df = df
        self.threshold = threshold

    def generate_edges(self):
        '''
        package: https://networkx.org
        This code is generating edges for a graph by using the Spearman correlation coefficient to measure the
        correlation between variables in a dataframe.
        The code first calculates the Spearman correlation coefficient of the dataframe using the "spearman" method.
        It filters the resulting series by keeping only values with an absolute value greater than the user defined
        threshold.
        Finally, the code creates a graph using the NetworkX library's from_edgelist() function,
        which takes in a list of edges represented as tuples (i.e. (node1, node2)) and returns a graph object.
        The edges for the graph are taken from the filtered and reset dataframe.
        The function returns the generated graph.
        :return: graph
        '''
        corr = self.df.corr(method = "spearman") # spearman corr

        mask_keep = np.triu(np.ones(corr.shape), k=1).astype('bool').reshape(corr.size) # upper triangle
        # melt (unpivot) the dataframe and apply mask
        sr = corr.stack()[mask_keep]
        # filter and get names
        edges = sr[abs(sr) > 0.6].reset_index().values[:, :2]
        g = nx.from_edgelist(edges) # create graph
        return g

    def correlation_heatmap(self, w = 50, h = 50):
        '''
        Plot correlation heatmap
        :param w: width
        :param h: height
        '''
        # Increase the size of the heatmap.
        plt.figure(figsize=(w, h))
        # Store heatmap object in a variable to easily access it when you want to include more features (such as title).
        # Set the range of values to be displayed on the colormap from -1 to 1, and set the annotation to True to display the correlation values on the heatmap.
        heatmap = sns.heatmap(self.df.corr(), vmin=-1, vmax=1, annot=True)
        # Give a title to the heatmap. Pad defines the distance of the title from the top of the heatmap.
        heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12)

    def subgraphs(self):
        '''
        This code is generating a list of subgraphs from a graph.
        It uses the connected_components() function from the NetworkX library to find all connected components
        in the graph, which are returned as a list of sets of nodes.
        The code then iterates over the connected components, and for each component it creates a subgraph using
        the subgraph() method of the graph object.
        The code returns the "subgraph_lists", which is a list of lists, each list representing the nodes of a subgraph.
        :return: list of lists
        '''
        # generate graph
        g = self.generate_edges()
        # generate subgraphs
        subgraphs = [g.subgraph(c) for c in connected_components(g)]
        # number of subgraphs
        n_subgraphs = len(subgraphs)

        # list of lists
        subgraph_lists = []
        # iterate over subgraphs
        for i in range(n_subgraphs):
            # append nodes to list of lists
            subgraph_lists.append(list(subgraphs[i].nodes()))
        # return list of lists
        return subgraph_lists

    def generate_subgraph_dict(self):
        '''
        This code is generating a dictionary of subgraphs from a graph.This code is generating a
        dictionary of subgraphs, where the key is the subgraph name and the value is a list of
        nodes in that subgraph.
        start incrementing subgraph name from A -> chr(65)
        :return:
        '''
        subgraph_list = self.subgraphs()
        # start from A
        increment = 65
        # dictionary
        subgraph_dict = {}
        # iterate over subgraph list
        for i in subgraph_list:
            subgraph_dict["Subgraph {}".format(chr(increment))] = i
            increment +=1
        return subgraph_dict

    def plot_subgraphs(self,w=12,h = 12):
        '''
        Plot subgraphs
        :param w: width
        :param h: height
        '''
        # set figure size
        plt.figure(figsize=(w, h))
        # generate graph
        g = self.generate_edges()
        # set layout
        pos = nx.spring_layout(g, k=0.25, iterations=25)
        # draw graph
        nx.draw(g,pos,with_labels = True,node_size = 750,font_size =13,font_weight = "bold",node_color = "lightskyblue")
        # adjust layout
        plt.tight_layout()
        # remove axis
        plt.axis("off")
        # show plot
        plt.show()
