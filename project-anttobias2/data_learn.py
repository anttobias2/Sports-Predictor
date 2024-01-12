"""Machine learning algorithm implementations.

NAME: Anthony Tobias
DATE: Fall 2023
CLASS: CPSC 322

"""

from data_table import *
from data_util import *
from decision_tree import *

from random import randint
import math


#----------------------------------------------------------------------
# HW-8
#----------------------------------------------------------------------


def random_subset(F, columns):
    """Returns F unique column names from the given list of columns. The
    column names are selected randomly from the given names.

    Args: 
        F: The number of columns to return.
        columns: The columns to select F column names from.

    Notes: If F is greater or equal to the number of names in columns,
       then the columns list is just returned.

    """
    # base case
    if F >= len(columns):
        return columns
    
    new_cols = []
    indicies = []
    
    # get random indicies
    for i in range(F):
        index = randint(0, len(columns) - 1)
        # get unique indicies
        while index in indicies:
            index = randint(0, len(columns) - 1)
        indicies.append(index)
    
    # add random cols
    for i in indicies:
        new_cols.append(columns[i])
        
    return new_cols



def tdidt_F(table, label_col, F, columns): 
    """Returns an initial decision tree for the table using information
    gain, selecting a random subset of size F of the columns for
    attribute selection. If fewer than F columns remain, all columns
    are used in attribute selection.

    Args:
        table: The table to build the decision tree from. 
        label_col: The column containing class labels. 
        F: The number of columns to randomly subselect
        columns: The categorical columns. 

    Notes: The resulting tree can contain multiple leaf nodes for a
        particular attribute value, may have attributes without all
        possible attribute values, and in general is not pruned.

    """
    cols = random_subset(F, columns)
    
    root = tdidt(table, label_col, cols)

    return root


def closest_centroid(centroids, row, columns):
    """Given k centroids and a row, finds the centroid that the row is
    closest to.

    Args:
        centroids: The list of rows serving as cluster centroids.
        row: The row to find closest centroid to.
        columns: The numerical columns to calculate distance from. 
    
    Returns: The index of the centroid the row is closest to. 

    Notes: Uses Euclidean distance (without the sqrt) and assumes
        there is at least one centroid.

    """
    distances = []
    
    for instance in centroids:
        distance = 0
        for col in columns:
            distance += (row[col] - instance[col]) ** 2
                
        distances.append(distance)
        
    best_distance = 1000000000
    best_index = 0
    
    for i in range(len(distances)):
        if distances[i] < best_distance:
            best_distance = distances[i]
            best_index = i
            
    return best_index
                
                



def select_k_random_centroids(table, k):
    """Returns a list of k random rows from the table to serve as initial
    centroids.

    Args: 
        table: The table to select rows from.
        k: The number of rows to select values from.
    
    Returns: k unique rows. 

    Notes: k must be less than or equal to the number of rows in the table. 

    """
    # base case return empty list
    if table.row_count() == 0:
        return []
    
    
    rows = []
    
    # return entire table if k >= table length
    if k >= table.row_count():
        for row in table:
            rows.append(row)
        
        return rows
    
    
    indicies = []
    # get k unique indicies and add to table
    for _ in range(k):
        index = randint(0, table.row_count() - 1)
        while index in indicies:
            index = randint(0, table.row_count() - 1)
        
        indicies.append(index) 
        rows.append(table[index])
    
    
    return rows
    



def k_means(table, centroids, columns): 
    """Returns k clusters from the table using the initial centroids for
    the given numerical columns.

    Args:
        table: The data table to build the clusters from.
        centroids: Initial centroids to use, where k is length of centroids.
        columns: The numerical columns for calculating distances.

    Returns: A list of k clusters, where each cluster is represented
        as a data table.

    Notes: Assumes length of given centroids is number of clusters k to find.

    """
    clusters = {i: [] for i in range(len(centroids))}
    
    
    # Assign each row to the closest centroid
    for row in table:
        closest_index = closest_centroid(centroids, row, columns)
        clusters[closest_index].append(row.values())
    
    # Convert clusters into data tables
    result_clusters = []
    for cluster_rows in clusters.values():
        cluster_table = DataTable(table.columns())
        for row in cluster_rows:
            cluster_table.append(row)
        result_clusters.append(cluster_table)
    
    return result_clusters



def tss(clusters, columns):
    """Return the total sum of squares (tss) for each cluster using the
    given numerical columns.

    Args:
        clusters: A list of data tables serving as the clusters
        columns: The list of numerical columns for determining distances.
    
    Returns: A list of tss scores for each cluster. 

    """
    if clusters == [] or columns == []:
        return []
    
    
    def calculate_centroid(cluster, columns):
        """Calculate the centroid of a cluster."""
        centroid = [0] * len(columns)
        for col_index, col_name in enumerate(columns):
            col_values = [point[col_name] for point in cluster]
            centroid[col_index] = sum(col_values) / len(col_values)
        return centroid

    def calculate_distance(point, centroid):
        """Calculate the squared Euclidean distance between a point and a centroid."""
        squared_distance = 0
        for i in range(len(point)):
            squared_distance += (point[i] - centroid[i]) ** 2
        return squared_distance
    
    tss_scores = []
    
    for cluster in clusters:
        centroid = calculate_centroid(cluster, columns)
        cluster_tss = 0
        
        for point in cluster:
            point_values = [point[col] for col in columns]
            distance = calculate_distance(point_values, centroid)
            cluster_tss += distance
        
        tss_scores.append(cluster_tss)
    
    return tss_scores




#----------------------------------------------------------------------
# HW-7
#----------------------------------------------------------------------

def same_class(table, label_col):
    """Returns true if all of the instances in the table have the same
    labels and false otherwise.

    Args: 
        table: The table with instances to check. 
        label_col: The column with class labels.

    """
    # set label to first one in table
    label = table[0][label_col]
    
    # if any label is different return false
    for row in table:
        if row[label_col] != label:
            return False
        
    return True


def build_leaves(table, label_col):
    """Builds a list of leaves out of the current table instances.
    
    Args: 
        table: The table to build the leaves out of.
        label_col: The column to use as class labels

    Returns: A list of LeafNode objects, one per label value in the
        table.

    """
    leaves = []
    
    # go through each row in the table
    for row in table:
        in_list = False
        
        # check if leaf exists in the list already
        for leaf in leaves:
            if leaf.label == row[label_col]:
                in_list = True
                leaf.count += 1
        
        # create new leaf node if it is not in list     
        if not in_list:
            leaves.append(LeafNode(row[label_col], 1, table.row_count()))
        
    return leaves
                


def calc_e_new(table, label_col, columns):
    """Returns entropy values for the given table, label column, and
    feature columns (assumed to be categorical). 

    Args:
        table: The table to compute entropy from
        label_col: The label column.
        columns: The categorical columns to use to compute entropy from.

    Returns: A dictionary, e.g., {e1:['a1', 'a2', ...], ...}, where
        each key is an entropy value and each corresponding key-value
        is a list of attributes having the corresponding entropy value. 

    Notes: This function assumes all columns are categorical.

    """
    
    vals = {}
    
    # base case with empty table
    if table.row_count() == 0:
        return {0.0: columns}

    # partition on each row in columns
    for col in columns:
        partitions = partition(table, [col])
        entropy = 0.0
        
        # partiton on each value in label_col
        for part in partitions:
            tmp = partition(part, [label_col])
            e_tmp = 0
            
            # calculate entropy for each col value 
            for t in tmp:
                prob = t.row_count() / part.row_count()
                e_tmp -= (prob * math.log(prob, 2))
            
            # weight the entropy
            entropy += e_tmp * (part.row_count() / table.row_count())
            
        # add to dictionary
        if entropy not in vals:
            vals[entropy] = [col]
        else:
            if col not in vals[entropy]:
                vals[entropy].append(col)

        
    return vals



def tdidt(table, label_col, columns): 
    """Returns an initial decision tree for the table using information
    gain.

    Args:
        table: The table to build the decision tree from. 
        label_col: The column containing class labels. 
        columns: The categorical columns. 

    Notes: The resulting tree can contain multiple leaf nodes for a
        particular attribute value, may have attributes without all
        possible attribute values, and in general is not pruned.

    """
    # base case with empty table
    if table.row_count() == 0:
        return None
    
    # base case where all labels are the same in table
    if same_class(table, label_col):
        return [LeafNode(table[0][label_col], table.row_count(), table.row_count())]
    
    # base case with empty columns
    if len(columns) == 0:
        return build_leaves(table, label_col)
    
    # get dictionary with entropy values
    entropy = calc_e_new(table, label_col, columns)
    best_e = 10000
    best_col = None
    
    # find the best column to use
    for key, value in entropy.items():
        if key < best_e:
            best_e = key
            best_col = value[0]
    
    # partition on best column and remove from list of columns      
    partitions = partition(table, [best_col])
    res = {}
    
    for part in partitions:
        name = part[0][best_col]
        part.drop([best_col])
        cols = columns.copy()
        cols.remove(best_col)
        node = tdidt(part, label_col, cols)
        res[name] = node    

    
    return AttributeNode(best_col, res)    
    


def summarize_instances(dt_root):
    """Return a summary by class label of the leaf nodes within the given
    decision tree.

    Args: 
        dt_root: The subtree root whose (descendant) leaf nodes are summarized. 

    Returns: A dictionary {label1: count, label2: count, ...} of class
        labels and their corresponding number of instances under the
        given subtree root.

    """
    # if dt root is already a leaf node, return its info
    if isinstance(dt_root, LeafNode):
        return {dt_root.label: dt_root.count}
    
    
    summary = {}
    
    # add each Leaf Node to summary
    for _, value in dt_root.values.items():
        
        # if it is a list, then add the leaf nodes
        if isinstance(value, list):
            for val in value:
                if val.label not in summary:
                    summary[val.label] = val.count
                else:
                    summary[val.label] += val.count
                   
        # if it is an attribute node, make recursive call 
        elif isinstance(value, AttributeNode):
            tmp = summarize_instances(value)
            
            # add other items to the dictionary
            for k, v in tmp.items():
                if k not in summary:
                    summary[k] = v
                else:
                    summary[k] += v
                
    return summary
                


def resolve_leaf_nodes(dt_root):
    """Modifies the given decision tree by combining attribute values with
    multiple leaf nodes into attribute values with a single leaf node
    (selecting the label with the highest overall count).

    Args:
        dt_root: The root of the decision tree to modify.

    Notes: If an attribute value contains two or more leaf nodes with
        the same count, the first leaf node is used.

    """
    # base cases
    if isinstance(dt_root, LeafNode):
        return LeafNode(dt_root.label, dt_root.count, dt_root.total)
    
    elif isinstance(dt_root, list):
        return [LeafNode(l.label, l.count, l.total) for l in dt_root]
    
    # create new root
    new_dt_root = AttributeNode(dt_root.name, {})
    
    # recursive step
    for val, child in dt_root.values.items():
        new_dt_root.values[val] = resolve_leaf_nodes(child)
        
    # backtracking phase 
    for val, child in new_dt_root.values.items():
        
        # make each list of leaf nodes down to length 1
        if isinstance(child, list):
            if len(child) > 1:
                best = 0
                best_label = None
                total = 0
                for leaf in child:
                    total = leaf.total
                    if leaf.count > best:
                        best = leaf.count
                        best_label = leaf.label
                new_dt_root.values[val] = [LeafNode(best_label, best, total)]
    
    return new_dt_root
        


def resolve_attribute_values(dt_root, table):
    """Return a modified decision tree by replacing attribute nodes
    having missing attribute values with the corresponding summarized
    descendent leaf nodes.
    
    Args:
        dt_root: The root of the decision tree to modify.
        table: The data table the tree was built from. 

    Notes: The table is only used to obtain the set of unique values
        for attributes represented in the decision tree.

    """
    # base cases
    if isinstance(dt_root, LeafNode):
        return LeafNode(dt_root.label, dt_root.count, dt_root.total)
    
    elif isinstance(dt_root, list):
        return [LeafNode(l.label, l.count, l.total) for l in dt_root]
    
    # create new attribute node
    tmp_root = AttributeNode(dt_root.name, dict(dt_root.values))
    
    # check all vals
    for val in distinct_values(table, tmp_root.name):
        
        # check if val is in root already
        if val not in tmp_root.values:
            leaves = summarize_instances(tmp_root)
            total = 0
            
            for val in leaves:
                total += leaves[val]
            
            final = []
            
            # fix each leaf
            for label in leaves:
                final.append(LeafNode(label, leaves[label], total))
                
            return final

        # if already in root, recursive step
        else:
            for key in tmp_root.values:
                tmp_root.values[key] = resolve_attribute_values(tmp_root.values[key], table)
    return tmp_root

def tdidt_predict(dt_root, instance): 
    """Returns the class for the given instance given the decision tree. 

    Args:
        dt_root: The root node of the decision tree. 
        instance: The instance to classify. 

    Returns: A pair consisting of the predicted label and the
       corresponding percent value of the leaf node.

    """
    if isinstance(dt_root, LeafNode):
        return (dt_root.label, dt_root.count/dt_root.total * 100)
    # # reslove attribute nodes
    # new_dt_root = resolve_attribute_values(dt_root, )
    
    # resolve leaf nodes
    new_dt_root = resolve_leaf_nodes(dt_root)
    
    curr = new_dt_root
    
    while isinstance(curr, AttributeNode):
        # print(curr.name)
        # print(instance)
        actual = instance[curr.name]
        curr = curr.values[actual]
        
    if isinstance(curr, LeafNode):
        return (curr.label, curr.count/curr.total * 100)
        
    new_label = (curr[0].label, curr[0].count/curr[0].total * 100)
        
    return new_label
    
    
    


#----------------------------------------------------------------------
# HW-6
#----------------------------------------------------------------------

def naive_bayes(table, instance, label_col, continuous_cols, categorical_cols=[]):
    """Returns the labels with the highest probabibility for the instance
    given table of instances using the naive bayes algorithm.

    Args:
       table: A data table of instances to use for estimating most probably labels.
       instance: The instance to classify.
       continuous_cols: The continuous columns to use in the estimation.
       categorical_cols: The categorical columns to use in the estimation. 

    Returns: A pair (labels, prob) consisting of a list of the labels
        with the highest probability and the corresponding highest
        probability.

    """       
    
    probs = {}
    totals = {}
    tables = partition(table, [label_col])
    
    # get odds for each label
    for t in tables:
        label = t[0][label_col]
        probs[label] = 1
        totals[label] = t.row_count()
        
        # calculate odds for categorical column values
        for col in categorical_cols:
            tmp = 0
            for row in t:
                if row[col] == instance[col]:
                    tmp += 1
            
            probs[label] *= (tmp/table.row_count())
            
        
        vals = []
        
        # calculate odds for continuous columns
        for col in continuous_cols:
            for row in t:
                vals.append(row[col])
            avg = sum(vals)/len(vals)
            dev = std_dev(t, col)
            probs[label] *= gaussian_density(instance[col], avg, dev) * (t.row_count() / table.row_count())
    
    # set max to -1 so it will be changed
    max = -1
    best_label = []
    
    # get the classes with the best probability
    for key, value in probs.items():
        if value > max:
            best_label = [key]
            max = value
            
        elif value == max:
            best_label.append(key)
            
    return (best_label, max)
            
            

def gaussian_density(x, mean, sdev):
    """Return the probability of an x value given the mean and standard
    deviation assuming a normal distribution.

    Args:
        x: The value to estimate the probability of.
        mean: The mean of the distribution.
        sdev: The standard deviation of the distribution.

    """
    # Calculate the probability density using the formula
    exponent = -((x - mean) ** 2) / (2 * sdev ** 2)
    coefficient = 1 / (math.sqrt(2 * math.pi) * sdev)
    probability_density = coefficient * math.exp(exponent)
    
    return probability_density


#----------------------------------------------------------------------
# HW-5
#----------------------------------------------------------------------

def knn(table, instance, k, numerical_columns, nominal_columns=[]):
    """Returns the k closest distance values and corresponding table
    instances forming the nearest neighbors of the given instance. 

    Args:
        table: The data table whose instances form the nearest neighbors.
        instance: The instance to find neighbors for in the table.
        k: The number of closest distances to return.
        numerical_columns: The numerical columns to use for comparison.
        nominal_columns: The nominal columns to use for comparison (if any).

    Returns: A dictionary with k key-value pairs, where the keys are
        the distances and the values are the corresponding rows.

    Notes: 
        The numerical and nominal columns must be disjoint. 
        The numerical and nominal columns must be valid for the table.
        The resulting score is a combined distance without the final
        square root applied.

    """
    # dictionary to store each distance
    distances = {}

    # Add euclidian distance of each row
    for row in table:
        distance = 0
        for col in numerical_columns:
            distance += (row[col] - instance[col]) ** 2
        for col in nominal_columns:
            if instance[col] != row[col]:
                distance += 1
        
        # if the distance already exists, add row to list of rows with same distance
        if distance in distances:
            distances[distance].append(row)
        else:
            distances[distance] = [row]
        
    # order the distances
    ordered_distances = []        
    for key in distances:
        ordered_distances.append(key)
    ordered_distances.sort()
    
    k_neighbors = {}
    
    # add smallest distances to new dictionary
    for i in range(k):
        if i < len(distances):
            k_neighbors[ordered_distances[i]] = distances[ordered_distances[i]]
        
    return k_neighbors
        



def majority_vote(instances, scores, labeled_column):
    """Returns the labels in the given instances that occur the most.

    Args:
        instances: A list of instance rows.
        labeled_column: The column holding the class labels.

    Returns: A list of the labels that occur the most in the given
    instances.

    """
    
    counts = {}
    
    # put each row in the dictionary with its value as the key
    for instance in instances:
        val = instance[labeled_column]
        if val in counts:
            counts[val].append(instance)
        else:
            counts[val] = [instance]
    
    # find out which key has the highest value and return that
    max_count = 0
    majority = []
    for key in counts:
        # append the list if there are multiple vals with same amount
        if len(counts[key]) == max_count:
            majority.append(key)
        if len(counts[key]) > max_count:
            majority = [key]
            max_count = len(counts[key])
            
    return majority
        



def weighted_vote(instances, scores, labeled_column):
    """Returns the labels in the given instances with the largest total
    sum of corresponding scores.

    Args:
        instances: The list of instance rows.
        scores: The corresponding scores for each instance.
        labeled_column: The column with class labels.

    """
    
    weighted = {}
    i = 0
    
    # add all values to weighted dictionary
    for instance in instances:
        val = instance[labeled_column]
        if val in weighted:
            # add score if value already exists
            weighted[val].append(scores[i])
        else:
            weighted[val] = [scores[i]]
        i += 1
        
    max_sum = 0
    vote = []
    
    # get sum of weights and return value with the highest sum
    for key in weighted:
        if sum(weighted[key]) == max_sum:
            vote.append(key)
        if sum(weighted[key]) > max_sum:
            vote = [key]
            max_sum = sum(weighted[key])
            
    return vote





