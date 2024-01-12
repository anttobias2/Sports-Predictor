"""Machine learning algorithm evaluation functions. 

NAME: Anthony Tobias
DATE: Fall 2023
CLASS: CPSC 322

"""

from data_table import *
from data_util import *
from data_learn import *
from random import randint



#----------------------------------------------------------------------
# HW-8
#----------------------------------------------------------------------

def bootstrap(table): 
    """Creates a training and testing set using the bootstrap method.

    Args: 
        table: The table to create the train and test sets from.

    Returns: The pair (training_set, testing_set)

    """
    test = DataTable(table.columns())
    train = DataTable(table.columns())
    
    if table.row_count() == 0:
        return (train, test)
    
    indicies = []
    
    for _ in table:
        indicies.append(randint(0, table.row_count() - 1))
        
    for i in indicies:
        train.append(table[i].values())
        
    for i in range(table.row_count()):
        if i not in indicies:
            test.append(table[i].values())
    
    return (train, test)
        
        

def stratified_holdout(table, label_col, test_set_size):
    """Partitions the table into a training and test set using the holdout
    method such that the test set has a similar distribution as the
    table.

    Args:
        table: The table to partition.
        label_col: The column with the class labels. 
        test_set_size: The number of rows to include in the test set.

    Returns: The pair (training_set, test_set)

    """
    if test_set_size == 0:
        return (table, DataTable(table.columns()))
    
    partitioned_tables = partition(table, [label_col])
    strat_vals = [0] * len(partitioned_tables)
    folds = []
    index = 0
    i = 0
    
    while i < test_set_size:
        p_index = index % len(partitioned_tables)
        if strat_vals[p_index] < partitioned_tables[p_index].row_count():
            strat_vals[p_index] += 1
            i = i + 1
        index += 1
    
    test_table = DataTable(table.columns())
    test_indicies = [[]] * len(partitioned_tables)

    for i in range(len(partitioned_tables)):
        for _ in range(strat_vals[i]):
            random_index = randint(0, partitioned_tables[i].row_count() - 1)
            test_table.append(partitioned_tables[i][random_index].values())
            test_indicies[i].append(random_index)
           
    train_table = DataTable(table.columns()) 
    
    if test_set_size == table.row_count():
        return (table, test_table)
    
    for i in range(table.row_count() - test_set_size):
        index = randint(0, table.row_count() - 1)
        train_table.append(table[i].values())   
            
    return (train_table, test_table)
    


def tdidt_eval_with_tree(dt_root, test, label_col, columns):
    """Evaluates the given test set using tdidt over the training
    set, returning a corresponding confusion matrix.

    Args:
       td_root: The decision tree to use.
       test: The testing data set.
       label_col: The column being predicted.
       columns: The categorical columns

    Returns: A data table with n rows (one per label), n+1 columns (an
        'actual' column plus n label columns), and corresponding
        prediction vs actual label counts.

    Notes: If the naive bayes returns multiple labels for a given test
        instance, the first such label is selected.

    """
    # create confusion matrix with necessary labels
    
    cols = ['actual']
    col_vals = []
    
    # get the labeled column values from both test and train
    for row in test:
        if row[label_col] not in col_vals:
            col_vals.append(row[label_col])
    col_vals.sort()
    
    for col in col_vals:
        cols.append(col)
    
    # create confusion matrix
    confusion = DataTable(cols)
    num_cols = len(cols)
    
    # populate confusion matrix with empty rows
    for col in cols:
        new_cols = [0] * num_cols
        if col != 'actual':
            new_cols[0] = col
            confusion.append(new_cols)
    
    # populate confusion matrix for each prediciton
    for row in test:
        decision = tdidt_predict(dt_root, row)
        for r in confusion:
            if r['actual'] == row[label_col]:
                r[decision[0]] += 1
                
    return confusion



def random_forest(table, remainder, F, M, N, label_col, columns):
    """Returns a random forest build from the given table. 
    
    Args:
        table: The original table for cleaning up the decision tree.
        remainder: The table to use to build the random forest.
        F: The subset of columns to use for each classifier.
        M: The number of unique accuracy values to return.
        N: The total number of decision trees to build initially.
        label_col: The column with the class labels.
        columns: The categorical columns used for building the forest.

    Returns: A list of (at most) M pairs (tree, accuracy) consisting
        of the "best" decision trees and their corresponding accuracy
        values. The exact number of trees (pairs) returned depends on
        the other parameters (F, M, N, and so on).

    """
    forest = []
    
    training_set, test_set = stratified_holdout(remainder, label_col, remainder.row_count()//3)
    
    
    for _ in range(N): 
        # build tree
        root = tdidt_F(training_set, label_col, F, columns)
        
        # clean up tree
        root = resolve_attribute_values(root, table)
        root = resolve_leaf_nodes(root)
        
        confusion = tdidt_eval_with_tree(root, test_set, label_col, columns)
        acc = 0

        # calculate accuracy for each tree
        for j in range(confusion.row_count()):
            acc += accuracy(confusion, confusion.columns()[j+1])

        acc /= confusion.row_count()
        
        # add root and accuracy to the list
        forest.append((root, acc))
    
    # return the whole forest if M >= N
    if M >= N:
        return forest
    
    # otherwise get the M best tuples
    best_M = sorted(forest, key=lambda x: x[1], reverse=True)[:M]
    
    return best_M



def random_forest_eval(table, train, test, F, M, N, label_col, columns):
    """Builds a random forest and evaluate's it given a training and
    testing set.

    Args: 
        table: The initial table.
        train: The training set from the initial table.
        test: The testing set from the initial table.
        F: Number of features (columns) to select.
        M: Number of trees to include in random forest.
        N: Number of trees to initially generate.
        label_col: The column with class labels. 
        columns: The categorical columns to use for classification.

    Returns: A confusion matrix containing the results. 

    Notes: Assumes weighted voting (based on each tree's accuracy) is
        used to select predicted label for each test row.

    """
    # create confusion matrix with necessary labels
    
    # 
    cols = ['actual']
    col_vals = []
    
    # get the labeled column values from both test and train
    for row in test:
        if row[label_col] not in col_vals:
            col_vals.append(row[label_col])
    for row in train:
        if row[label_col] not in col_vals:
            col_vals.append(row[label_col])
    col_vals.sort()
    
    for col in col_vals:
        cols.append(col)
    
    # create confusion matrix
    confusion = DataTable(cols)
    num_cols = len(cols)
    
    # populate confusion matrix with empty rows
    for col in cols:
        new_cols = [0] * num_cols
        if col != 'actual':
            new_cols[0] = col
            confusion.append(new_cols)
            
            
    # generate forest
    forest = random_forest(table, train, F, M, N, label_col, columns)
    
    
    # make predictions for each row
    for row in test:
        predictions = {}
        
        # get predictions and weights from each tree
        for t in forest:
            tree = t[0]
            p = tdidt_predict(tree, row)
            
            if p[0] not in predictions:
                predictions[p[0]] = t[1]
            else:
                predictions[p[0]] += t[1]
                
        best_confidence = 0
        best_label = None
        
        for key, value in predictions.items():
            if value > best_confidence:
                best_confidence = value
                best_label = key
                
        for r in confusion:
            if r['actual'] == row[label_col]:
                r[best_label] += 1
                
    return confusion
            
        



#----------------------------------------------------------------------
# HW-7
#----------------------------------------------------------------------


def tdidt_eval(train, test, label_col, columns):
    """Evaluates the given test set using tdidt over the training
    set, returning a corresponding confusion matrix.

    Args:
       train: The training data set.
       test: The testing data set.
       label_col: The column being predicted.
       columns: The categorical columns

    Returns: A data table with n rows (one per label), n+1 columns (an
        'actual' column plus n label columns), and corresponding
        prediction vs actual label counts.

    Notes: If the naive bayes returns multiple labels for a given test
        instance, the first such label is selected.

    """
    # create confusion matrix with necessary labels
    
    # 
    cols = ['actual']
    col_vals = []
    
    # get the labeled column values from both test and train
    for row in test:
        if row[label_col] not in col_vals:
            col_vals.append(row[label_col])
    for row in train:
        if row[label_col] not in col_vals:
            col_vals.append(row[label_col])
    col_vals.sort()
    
    for col in col_vals:
        cols.append(col)
    
    # create confusion matrix
    confusion = DataTable(cols)
    num_cols = len(cols)
    
    # populate confusion matrix with empty rows
    for col in cols:
        new_cols = [0] * num_cols
        if col != 'actual':
            new_cols[0] = col
            confusion.append(new_cols)
    
    # create tree
    tree = tdidt(train, label_col, columns)
    tree = resolve_attribute_values(tree, union_all([train, test]))
    tree = resolve_leaf_nodes(tree)
    
    # populate confusion matrix for each prediciton
    for row in test:
        decision = tdidt_predict(tree, row)
        for r in confusion:
            if r['actual'] == row[label_col]:
                r[decision[0]] += 1
                
    return confusion
    

def tdidt_stratified(table, k_folds, label_col, columns):
    """Evaluates tdidt prediction approach over the table using stratified
    k-fold cross validation, returning a single confusion matrix of
    the results.

    Args:
        table: The data table.
        k_folds: The number of stratified folds to use.
        label_col: The column with labels to predict. 
        columns: The categorical columns for tdidt. 

    Notes: Each fold created is used as the test set whose results are
        added to a combined confusion matrix from evaluating each
        fold.

    """
    # get folds from table
    strat_folds = stratify(table, label_col, k_folds)
    confusion_matricies = []
    
    # run naive bayes for each fold
    for i in range(len(strat_folds)):
        to_union = []
        
        # union all tables that are not the test fold
        for j in range(len(strat_folds)):
            if i != j:
                to_union.append(strat_folds[j])
                
    
        # add each confusion matrix to a list
        train = union_all(to_union)
        confusion_matricies.append(tdidt_eval(train, strat_folds[i], label_col, columns))
    
    # create a combined confusion matrix that is equal to first one in the list
    combined = DataTable(confusion_matricies[0].columns())
    for row in confusion_matricies[0]:
        combined.append(row.values())
        
    # combine each confusion matrix into one
    for i in range(1, len(confusion_matricies)):
        for j in range(0, confusion_matricies[0].row_count()):
            for col in confusion_matricies[i].columns():
                if col != 'actual':
                    combined[j][col] += confusion_matricies[i][j][col]
    
    return combined



#----------------------------------------------------------------------
# HW-6
#----------------------------------------------------------------------

def stratify(table, label_column, k):
    """Returns a list of k stratified folds as data tables from the given
    data table based on the label column.

    Args:
        table: The data table to partition.
        label_column: The column to use for the label. 
        k: The number of folds to return. 

    Note: Does not randomly select instances for the folds, and
        instead produces folds in order of the instances given in the
        table.

    """
    partitioned_tables = partition(table, [label_column])
    strat_vals = []
    folds = []
    
    # get the number of values to add from each table for each fold
    for t in partitioned_tables:
        r = t.row_count() % k
        fold_len = t.row_count() // k
        tmp = [fold_len] * k
        for i in range(r):
            tmp[i] += 1
        strat_vals.append(tmp)
    
    # the current index for each partitioned table
    curr_index = [0] * len(partitioned_tables)
    
    # add all necessary values to each fold
    for i in range(k):
        tmp = DataTable(table.columns())
        for j in range(len(partitioned_tables)):
            for n in range(strat_vals[j][i]):
                tmp.append(partitioned_tables[j][curr_index[j]].values())
                curr_index[j] += 1
                
        # add each fold to an array
        folds.append(tmp)
        
    return folds



def union_all(tables):
    """Returns a table containing all instances in the given list of data
    tables.

    Args:
        tables: A list of data tables. 

    Notes: Returns a new data table that contains all the instances of
       the first table in tables, followed by all the instances in the
       second table in tables, and so on. The tables must have the
       exact same columns, and the list of tables must be non-empty.

    """
    # raise error for empty list
    if tables == []:
        raise ValueError('No tables to union')
    
    # create table
    unioned_table = DataTable(tables[0].columns())
    
    # union each table
    for table in tables:
        
        # check column values for each table in list
        if table.columns() != unioned_table.columns():
            raise ValueError('Columns do not match')
        
        for row in table:
            unioned_table.append(row.values())
    
    
            
    return unioned_table

def naive_bayes_eval(train, test, label_col, continuous_cols, categorical_cols=[]):
    """Evaluates the given test set using naive bayes over the training
    set, returning a corresponding confusion matrix.

    Args:
       train: The training data set.
       test: The testing data set.
       label_col: The column being predicted.
       continuous_cols: The continuous columns (estimated via PDF)
       categorical_cols: The categorical columns

    Returns: A data table with n rows (one per label), n+1 columns (an
        'actual' column plus n label columns), and corresponding
        prediction vs actual label counts.

    Notes: If the naive bayes returns multiple labels for a given test
        instance, the first such label is selected.

    """
    # create confusion matrix with necessary labels
    
    # 
    columns = ['actual']
    col_vals = []
    
    # get the labeled column values from both test and train
    for row in test:
        if row[label_col] not in col_vals:
            col_vals.append(row[label_col])
    for row in train:
        if row[label_col] not in col_vals:
            col_vals.append(row[label_col])
    col_vals.sort()
    
    for col in col_vals:
        columns.append(col)
    
    # create confusion matrix
    confusion = DataTable(columns)
    num_cols = len(columns)
    
    # populate confusion matrix with empty rows
    for col in columns:
        new_cols = [0] * num_cols
        if col != 'actual':
            new_cols[0] = col
            confusion.append(new_cols)
            
    # populate confusion matrix for each prediciton
    for row in test:
        nb = naive_bayes(train, row, label_col, continuous_cols, categorical_cols)
        for r in confusion:
            if r['actual'] == row[label_col]:
                r[nb[0][0]] += 1
                
    return confusion
            

def naive_bayes_stratified(table, k_folds, label_col, cont_cols, cat_cols=[]):
    """Evaluates naive bayes over the table using stratified k-fold cross
    validation, returning a single confusion matrix of the results.

    Args:
        table: The data table.
        k_folds: The number of stratified folds to use.
        label_col: The column with labels to predict. 
        cont_cols: The continuous columns for naive bayes. 
        cat_cols: The categorical columns for naive bayes. 

    Notes: Each fold created is used as the test set whose results are
        added to a combined confusion matrix from evaluating each
        fold.

    """
    # get folds from table
    strat_folds = stratify(table, label_col, k_folds)
    confusion_matricies = []
    
    # run naive bayes for each fold
    for i in range(len(strat_folds)):
        to_union = []
        
        # union all tables that are not the test fold
        for j in range(len(strat_folds)):
            if i != j:
                to_union.append(strat_folds[j])
                
    
        # add each confusion matrix to a list
        train = union_all(to_union)
        confusion_matricies.append(naive_bayes_eval(train, strat_folds[i], label_col, cont_cols, cat_cols))
    
    # create a combined confusion matrix that is equal to first one in the list
    combined = DataTable(confusion_matricies[0].columns())
    for row in confusion_matricies[0]:
        combined.append(row.values())
    
    
    # combine each confusion matrix into one
    for i in range(1, len(confusion_matricies)):
        for j in range(0, confusion_matricies[0].row_count()):
            for col in confusion_matricies[i].columns():
                if col != 'actual':
                    combined[j][col] += confusion_matricies[i][j][col]
    
    return combined


def knn_stratified(table, k_folds, label_col, vote_fun, k, num_cols, nom_cols=[]):
    """Evaluates knn over the table using stratified k-fold cross
    validation, returning a single confusion matrix of the results.

    Args:
        table: The data table.
        k_folds: The number of stratified folds to use.
        label_col: The column with labels to predict. 
        vote_fun: The voting function to use with knn.
        num_cols: The numeric columns for knn.
        nom_cols: The nominal columns for knn.

    Notes: Each fold created is used as the test set whose results are
        added to a combined confusion matrix from evaluating each
        fold.

    """
    # create folds
    strat_folds = stratify(table, label_col, k_folds)
    confusion_matricies = []
    
    # run knn on each fold
    for i in range(len(strat_folds)):
        to_union = []
        # union all tables that are not the test table
        for j in range(len(strat_folds)):
            if i != j:
                to_union.append(strat_folds[j])
                
        train = union_all(to_union)
        # add each confusion matrix to a list
        confusion_matricies.append(knn_eval(train, strat_folds[i], vote_fun, k, label_col, num_cols, nom_cols))
    
    # create new confusion matrix that is equal to first matrix in list
    combined = DataTable(confusion_matricies[0].columns())
    for row in confusion_matricies[0]:
        combined.append(row.values())
    
    # combine the rest of the matricies into one
    for i in range(1, len(confusion_matricies)):
        for j in range(0, confusion_matricies[0].row_count()):
            for col in confusion_matricies[i].columns():
                if col != 'actual':
                    combined[j][col] += confusion_matricies[i][j][col]
    
    return combined


#----------------------------------------------------------------------
# HW-5
#----------------------------------------------------------------------

def holdout(table, test_set_size):
    """Partitions the table into a training and test set using the holdout method. 

    Args:
        table: The table to partition.
        test_set_size: The number of rows to include in the test set.

    Returns: The pair (training_set, test_set)

    """
    # create data tables for each
    test_table = DataTable(table.columns())
    train_table = DataTable(table.columns())
    used_rows = []
    train_indexes = []
    
    # write loops to add the first 'test_set_size' instances to test table
    for i in range(test_set_size):
        index = randint(0, table.row_count() - 1)
        
        while index in used_rows:
            index = randint(0, table.row_count() - 1)
        
        used_rows.append(index)
        test_table.append(table[index].values())
    
    # add rest of instances to train table
    for i in range(table.row_count()):
        if i not in used_rows:
            train_indexes.append(i)
            train_table.append(table[i].values())
            
    return train_table, test_table
        


def knn_eval(train, test, vote_fun, k, label_col, numeric_cols, nominal_cols=[]):
    """Returns a confusion matrix resulting from running knn over the
    given test set. 

    Args:
        train: The training set.
        test: The test set.
        vote_fun: The function to use for selecting labels.
        k: The k nearest neighbors for knn.
        label_col: The column to use for predictions. 
        numeric_cols: The columns compared using Euclidean distance.
        nominal_cols: The nominal columns for knn (match or no match).

    Returns: A data table with n rows (one per label), n+1 columns (an
        'actual' column plus n label columns), and corresponding
        prediction vs actual label counts.

    Notes: If the given voting function returns multiple labels, the
        first such label is selected.

    """
    # create confusion matrix with necessary labels
    
    # 
    columns = ['actual']
    col_vals = []
    
    for row in test:
        if row[label_col] not in col_vals:
            col_vals.append(row[label_col])
    for row in train:
        if row[label_col] not in col_vals:
            col_vals.append(row[label_col])
    col_vals.sort()
    
    for col in col_vals:
        columns.append(col)
    
    confusion = DataTable(columns)
    num_cols = len(columns)
    
    
    for col in columns:
        new_cols = [0] * num_cols
        if col != 'actual':
            new_cols[0] = col
            confusion.append(new_cols)
            
    # run knn on numeric/label cols
    for row in test:
        instances = []
        scores = []
        neighbors = knn(train, row, k, numeric_cols, nominal_cols)
        for key in neighbors:
            for neighbor in neighbors[key]:
                instances.append(neighbor)
                # just have all scores be 1 for now
                scores.append(1)
        
        vote = vote_fun(instances, scores, label_col)
        for r in confusion:
            if r['actual'] == row[label_col]:
                r[vote[0]] += 1
                
    
    return confusion




def accuracy(confusion_matrix, label):
    """Returns the accuracy for the given label from the confusion matrix.
    
    Args:
        confusion_matrix: The confusion matrix.
        label: The class label to measure the accuracy of.

    """
    row_num = 0
    cols = confusion_matrix.columns()
    
    # find out which row number the 'true positives' are in
    for row in confusion_matrix:
        if row['actual'] == label:
            break
        row_num += 1
        
    
    true_pos = confusion_matrix[row_num][label]
    true_neg = 0
    total = 0
    i = 0
    
    # add the number of true negatives, and total number of instances
    for row in confusion_matrix:
        for col in cols:
            if col != label and i != row_num and col != 'actual':
                true_neg += row[col]
            if col != 'actual':
                total += row[col]
        i += 1
    if total != 0:
        return ((true_pos + true_neg) / total)
    return 1
                

def precision(confusion_matrix, label):
    """Returns the precision for the given label from the confusion
    matrix.

    Args:
        confusion_matrix: The confusion matrix.
        label: The class label to measure the precision of.

    """
    correct = 0
    total = 0
    
    # add number of correct true positive predictions there are vs total
    for row in confusion_matrix:
        if row['actual'] == label:
            correct = row[label]
        total += row[label]
    if total != 0:
        return correct/total
    return 1


def recall(confusion_matrix, label): 
    """Returns the recall for the given label from the confusion matrix.

    Args:
        confusion_matrix: The confusion matrix.
        label: The class label to measure the recall of.

    """
    cols = confusion_matrix.columns()
    total = 0
    
    # add number of actual instances vs number of correct
    for row in confusion_matrix:
        if row['actual'] == label:
            correct = row[label]
            
            for col in cols:
                if col != 'actual':
                    total += row[col]
            break

    if total != 0:
        return correct/total
    return 1
                

