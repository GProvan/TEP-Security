
import pandas as pd
import numpy as np
class LocaliseSubgraph:

    def __init__(self, rank_list, subgraph_dict):
        '''
        :param rank_list: list of tuples (component, (number of threshold violations, percent of total detected anomalies))
        :param subgraph_dict: dictionary of subgraphs
        '''
        self.rank_list = rank_list
        self.subgraph_dict = subgraph_dict

    def _generate_empty_rank_dict(self):
        '''
        Generate empty dictionary of subgraphs
        :return: dict
        '''
        new_dict = {}
        for key in self.subgraph_dict:
            new_dict[key] = []
        return new_dict

    def _generate_empty_subgraph_count_dict(self):
        '''
        Generate empty dictionary of subgraphs
        :return: dict
        '''
        subgraph_count = {}
        for key in self.subgraph_dict:
            subgraph_count[key] = [0, 0]
        return subgraph_count

    def generate_rank_dict(self):
        '''
        This code is generating a dictionary of ranked items, where the key is the subgraph name
        and the value is a list of ranked items in that subgraph.
        :return: dict
        '''
        rank_dict = self._generate_empty_rank_dict()
        for i in self.rank_list:
            for key, value in self.subgraph_dict.items():
                if i[0] in value:
                    rank_dict[key].append(i)

        return rank_dict

    def subgraph_counts(self):
        '''
        Generates a dictionary of subgraph counts, where the key is the subgraph name and the value
        is a list of two integers. The first integer is the sum of the first element of the second
        element of the tuple in the rank_list, and the second integer is the count of the number of
        ranked items in that subgraph.
        :return: dict
        '''
        rank_dict = self.generate_rank_dict()
        subgraph_count = self._generate_empty_subgraph_count_dict()
        for key, value in rank_dict.items():
            for val in value:
                subgraph_count[key][1] += 1
                subgraph_count[key][0] += val[1][0]
        return subgraph_count

    def find_max_subgraph(self):
        '''
        The function finds the maximum subgraph of a graph by calculating the average of the subgraphs and
        returning the subgraph with the highest average. It iterates through each subgraph, keeping track
        of the current highest average and the corresponding key. If a component not in the subgraph has a
        higher average than the current highest average, it returns the component instead.
        :return:
        '''
        subgraph_count = self.subgraph_counts()
        max_avg = 0
        max_key = ""
        for key, value in subgraph_count.items():
            if value[0] / value[1] > max_avg:
                max_key = key
                max_avg = value[0] / value[1]
        # check if a component not in subgraph is greater than subgraph
        return max_key

    def subgraph_components(self, graph_name):
        return self.subgraph_dict[graph_name]

    def rank_subgraph(self, graph_name):
        component_list = self.subgraph_dict[graph_name]
        counter = 1
        report ={}
        report["Subgraph"] = graph_name
        report["Components"] = component_list
        print("Ranking subgraph: {}".format(graph_name))
        for i in self.rank_list:
            for comp in component_list:
                if comp == i[0]:

                    comp_report = {}
                    comp_report["Rank"] = counter
                    comp_report["Number of Threshold Violations"] = i[1][0]
                    comp_report["Percent of Total Detected Anomalies"] = i[1][1]
                    report[comp] = comp_report
                    print("Rank {}: {}".format(counter, comp))
                    print("Number of Threshold Violations: {}".format(i[1][0]))
                    print("Percent of total detected anomalies: {}%".format(i[1][1]))
                    print("\n")
                    counter += 1
        return report

'''
Example of use

local = LocaliseSubgraph(rank, subgraph_dict)
likely_subgraph = local.find_max_subgraph()
print(likely_subgraph)
print(local.subgraph_components(likely_subgraph))
local.rank_subgraph(likely_subgraph)
'''