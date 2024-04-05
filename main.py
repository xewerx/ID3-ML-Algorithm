import pprint
import numpy as np
import pandas as pd

from entropy import calculate_entropy

csv_file_path = 'data/titanic-homework.csv'
attributes = ['Pclass','Sex','Age','SibSp','Parch']
decision_column = 'Survived'

# csv_file_path = 'data/example.csv'
# decision_column = 'decyzja'
# attributes = ['matematyka', 'biologia', 'polski']

original_data = pd.read_csv(csv_file_path)

decision_entropy = calculate_entropy(original_data[decision_column])

def get_attr_data(attr_name, data):
    attr_data = {}

    for _, row in data.iterrows():
        attr_value = row[attr_name]
        decision = row[decision_column]

        if attr_value in attr_data:
            attr_data[attr_value].append(decision)
        else:
            attr_data[attr_value] = [decision]

    return attr_data

def get_attr_cond_entropies(attr_name, data):
    attr_data = get_attr_data(attr_name, data)

    cond_entropy = 0
    for _, decisions in attr_data.items():
        cond_entropy += (len(decisions) / len(data)) * calculate_entropy(decisions)

    return cond_entropy


def get_info_gain(attr_name, data):
    return decision_entropy - get_attr_cond_entropies(attr_name, data)

def get_best_attribute(attributes, data):
    best_attr_name = ''
    best_attr_info_gain = 0.0

    for attr in attributes:
        attr_info_gain = get_info_gain(attr, data)
        if best_attr_info_gain < attr_info_gain:
            best_attr_name = attr
            best_attr_info_gain = attr_info_gain
    
    return best_attr_name


def build_tree(data, attributes, parent_node = None):
    # warunki stopu rekurencji
    if len(np.unique(data[decision_column])) == 1: # jesli pozostala jedna decyzja
        return np.unique(data[decision_column])[0]
    
    if len(attributes) == 0: # jesli uzylismy wszystkie atrybuty to zwroc poddrzewo
        return parent_node
    
    else:
        parent_node = np.unique(data[decision_column])[np.argmax(np.unique(data[decision_column], return_counts=True)[1])]
        
        best_attr = get_best_attribute(attributes, data)
        
        tree = {best_attr:{}}
        
        # usun najlepszy atrybut z listy
        attributes = [i for i in attributes if i != best_attr]
        
        # podział zbioru
        for value in np.unique(data[best_attr]): # mozliwe wartosci danej kolumny
            sub_data = data.where(data[best_attr] == value).dropna() # wez wiersze z danym value dla danej kolumny
            subtree = build_tree(sub_data, attributes, parent_node)
            tree[best_attr][value] = subtree
        
        return tree

tree = build_tree(original_data, attributes)

pprint.pp(tree)