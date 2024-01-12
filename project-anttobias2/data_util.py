"""Data utility functions.

NAME: Anthony Tobias
DATE: Fall 2023
CLASS: CPSC 322

"""

from math import sqrt

from data_table import DataTable, DataRow
import matplotlib.pyplot as plt


#----------------------------------------------------------------------
# HW5
#----------------------------------------------------------------------

def normalize(table, column):
    """Normalize the values in the given column of the table. This
    function modifies the table.

    Args:
        table: The table to normalize.
        column: The column in the table to normalize.

    """
    # get min and max of the column
    max_val = max(column_values(table, column), key=lambda x: x)
    min_val = min(column_values(table, column), key=lambda x: x)

    # normalize each row in column
    for i in range(table.row_count()):
        table[i][column] = (table[i][column] - min_val)/(max_val - min_val)
        
    return table

def discretize(table, column, cut_points):
    """Discretize column values according to the given list of n-1
    cut_points to form n ordinal values from 1 to n. This function
    modifies the table.

    Args:
        table: The table to discretize.
        column: The column in the table to discretize.

    """
    # figure out which bin each value should go in
    for i in range(table.row_count()):
        bin_num = 1
        bin = 2
        for cut in cut_points:
            if table[i][column] >= cut:
                bin_num = bin
            
            bin += 1
            
        table[i][column] = bin_num
        
    return table
        

#----------------------------------------------------------------------
# HW4
#----------------------------------------------------------------------


def column_values(table, column):
    """Returns a list of the values (in order) in the given column.

    Args:
        table: The data table that values are drawn from
        column: The column whose values are returned
    
    """
    vals = []
    for row in table:
        vals.append(row[column])
    vals.sort()
    
    return vals    
    



def mean(table, column):
    """Returns the arithmetic mean of the values in the given table
    column.

    Args:
        table: The data table that values are drawn from
        column: The column to compute the mean from

    Notes: 
        Assumes there are no missing values in the column.

    """
    vals = []
    for row in table:
        vals.append(row[column])
    
    return sum(vals)/len(vals)


def variance(table, column):
    """Returns the variance of the values in the given table column.

    Args:
        table: The data table that values are drawn from
        column: The column to compute the variance from

    Notes:
        Assumes there are no missing values in the column.

    """
    avg = mean(table, column)
    
    vals = []
    for row in table:
        vals.append((row[column] - avg)**2)
        
    return sum(vals)/len(vals)
    
    


def std_dev(table, column):
    """Returns the standard deviation of the values in the given table
    column.

    Args:
        table: The data table that values are drawn from
        column: The colume to compute the standard deviation from

    Notes:
        Assumes there are no missing values in the column.

    """
    return sqrt(variance(table, column))



def covariance(table, x_column, y_column):
    """Returns the covariance of the values in the given table columns.
    
    Args:
        table: The data table that values are drawn from
        x_column: The column with the "x-values"
        y_column: The column with the "y-values"

    Notes:
        Assumes there are no missing values in the columns.        

    """
    avg_x = mean(table, x_column)
    avg_y = mean(table, y_column)
    
    sum = 0
    for row in table:
        sum += (row[x_column] - avg_x) * (row[y_column] - avg_y)
    
    return sum/table.row_count()


def linear_regression(table, x_column, y_column):
    """Returns a pair (slope, intercept) resulting from the ordinary least
    squares linear regression of the values in the given table columns.

    Args:
        table: The data table that values are drawn from
        x_column: The column with the "x values"
        y_column: The column with the "y values"

    """
    avg_x = mean(table, x_column)
    avg_y = mean(table, y_column)
    
    sum_num = 0
    sum_den = 0
    
    for i in range(table.row_count()):
        sum_num += (table[i][x_column] - avg_x) * (table[i][y_column] - avg_y)
        sum_den += (table[i][x_column] - avg_x) ** 2
        
    m = sum_num/sum_den
    
    b = avg_y - m * avg_x
    
    return (m, b)


def correlation_coefficient(table, x_column, y_column):
    """Return the correlation coefficient of the table's given x and y
    columns.

    Args:
        table: The data table that value are drawn from
        x_column: The column with the "x values"
        y_column: The column with the "y values"

    Notes:
        Assumes there are no missing values in the columns.        

    """
    return covariance(table, x_column, y_column)/(std_dev(table, x_column) * std_dev(table, y_column))


def frequency_of_range(table, column, start, end):
    """Return the number of instances of column values such that each
    instance counted has a column value greater or equal to start and
    less than end. 
    
    Args:
        table: The data table used to get column values from
        column: The column to bin
        start: The starting value of the range
        end: The ending value of the range

    Notes:
        start must be less than end

    """
    if start >= end:
        return 0
    
    count = 0
    
    for row in table:
        if row[column] >= start and row[column] < end:
            count += 1
            
    return count
            


def histogram(table, column, nbins, xlabel, ylabel, title, filename=None):
    """Create an equal-width histogram of the given table column and number of bins.
    
    Args:
        table: The data table to use
        column: The column to obtain the value distribution
        nbins: The number of equal-width bins to use
        xlabel: The label of the x-axis
        ylabel: The label of the y-axis
        title: The title of the chart
        filename: Filename to save chart to (in SVG format)

    Notes:
        If filename is given, chart is saved and not displayed.

    """
    # Extract the values from the specified column
    values = column_values(table, column)

    # Create a histogram
    plt.figure(figsize=(10, 6))
    plt.hist(values, bins=nbins, edgecolor='black', alpha=0.7)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    # Display or save the chart
    if filename:
        plt.savefig(filename, format='svg')
    else:
        plt.show()
    

def scatter_plot_with_best_fit(table, xcolumn, ycolumn, xlabel, ylabel, title, filename=None):
    """Create a scatter plot from given values that includes the "best fit" line.
    
    Args:
        table: The data table to use
        xcolumn: The column for x-values
        ycolumn: The column for y-values
        xlabel: The label of the x-axis
        ylabel: The label of the y-axis
        title: The title of the chart
        filename: Filename to save chart to (in SVG format)

    Notes:
        If filename is given, chart is saved and not displayed.

    """
    xvalues = []
    yvalues = []
    for i in range(table.row_count()):
        xvalues.append(table[i][xcolumn])
        yvalues.append(table[i][ycolumn])
    
    
    reg = linear_regression(table, xcolumn, ycolumn)
    reg_yvals = []
    for val in xvalues:
        reg_yvals.append(val * reg[0] + reg[1])
        
    plt.scatter(xvalues, yvalues)
    plt.plot(xvalues, reg_yvals)
    
    plt.plot()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    # Display or save the chart
    if filename:
        plt.savefig(filename, format='svg')
    else:
        plt.show()

    
#----------------------------------------------------------------------
# HW3
#----------------------------------------------------------------------

def distinct_values(table, column):
    """Return the unique values in the given column of the table.
    
    Args:
        table: The data table used to get column values from.
        column: The column of the table to get values from.

    Notes:
        Returns a list of unique values
    """
    vals = []
    for row in table:
        vals.append(row[column])
        
    return list(set(vals))


def remove_missing(table, columns):
    """Return a new data table with rows from the given one without
    missing values.

    Args:
        table: The data table used to create the new table.
        columns: The column names to check for missing values.

    Returns:
        A new data table.

    Notes: 
        Columns must be valid.

    """
    # TODO
    new_table = DataTable(table.columns())
    new_columns = []
    
    for row in table:
        empty = False
        for column in columns:
            if row[column] == '':
                empty = True
                
        if not empty:
            new_table.append(row.values())
        
    return new_table
                


def duplicate_instances(table):
    """Returns a table containing duplicate instances from original table.
    
    Args:
        table: The original data table to check for duplicate instances.

    """
    rows = {}
    new_table = DataTable(table.columns())
    
    # add values to dictionary
    # create list with duplicate rows
    for row in table:
        if str(row.values()) not in rows:
            rows[str(row.values())] = [row]
        else:
            rows[str(row.values())].append(row)
    
    # add the first row if there is more than one value in dictionary key
    for key in rows:
        if len(rows[key]) > 1:
            new_table.append(rows[key][0].values())

    return new_table

                    
def remove_duplicates(table):
    """Remove duplicate instances from the given table.
    
    Args:
        table: The data table to remove duplicate rows from

    """
    rows = {}
    new_table = DataTable(table.columns())
    
    # add values to dictionary
    # create list with duplicate rows
    for row in table:
        if str(row.values()) not in rows:
            rows[str(row.values())] = [row]
        else:
            rows[str(row.values())].append(row)
    
    # add the first row of every key
    for key in rows:
        new_table.append(rows[key][0].values())
        

    return new_table


def partition(table, columns):
    """Partition the given table into a list containing one table per
    corresponding values in the grouping columns.
    
    Args:
        table: the table to partition
        columns: the columns to partition the table on
    """
    partitions = []
    matches = {}
    
    for row in table:
        key = str(row.select(columns).values())
        if key not in matches:
            matches[key] = [row]
        else:
            matches[key].append(row)
            
    for key in matches:
        new_table = DataTable(table.columns())
        for row in matches[key]:
            new_table.append(row.values())
        partitions.append(new_table)

    return partitions

def summary_stat(table, column, function):
    """Return the result of applying the given function to a list of
    non-empty column values in the given table.

    Args:
        table: the table to compute summary stats over
        column: the column in the table to compute the statistic
        function: the function to compute the summary statistic

    Notes: 
        The given function must take a list of values, return a single
        value, and ignore empty values (denoted by the empty string)

    """
    # create empty list to add values to
    vals = []
    
    # add all non empty values to the list
    for row in table:
        val = row[column]
        if val != '':
            vals.append(val)
    
    # return the result of calling the provided function
    return function(vals)


def replace_missing(table, column, partition_columns, function): 
    """Replace missing values in a given table's column using the provided
     function over similar instances, where similar instances are
     those with the same values for the given partition columns.

    Args:
        table: the table to replace missing values for
        column: the coumn whose missing values are to be replaced
        partition_columns: for finding similar values
        function: function that selects value to use for missing value

    Notes: 
        Assumes there is at least one instance with a non-empty value
        in each partition

    """
    # create list of partitions wanted
    partitions = partition(table, partition_columns)
    new_table = table.copy()

    # for each row in original table, check if value is empty
    for row in new_table:
        val = row.select([column]).values()[0]
        
        # if value is empty check to see which table in the partitions it belongs in
        if val == '':
            for t in partitions:
                if t[0].select(partition_columns) == row.select(partition_columns):
                    val = summary_stat(t, column, function)
    
            row[column] = val
        
    return new_table


def summary_stat_by_column(table, partition_column, stat_column, function):
    """Returns for each partition column value the result of the statistic
    function over the given statistics column.

    Args:
        table: the table for computing the result
        partition_column: the column used to create groups from
        stat_column: the column to compute the statistic over
        function: the statistic function to apply to the stat column

    Notes:
        Returns a list of the groups and a list of the corresponding
        statistic for that group.

    """
    # Partition table
    partitions = partition(table, [partition_column])
    stats = []
    vals = []
    
    # get the summary stat and parition column value from each table
    for table in partitions:
        stat = summary_stat(table, stat_column, function)
        stats.append(stat)
        vals.append(table[0][partition_column])
    
    return vals, stats


def frequencies(table, partition_column):
    """Returns for each partition column value the number of instances
    having that value.

    Args:
        table: the table for computing the result
        partition_column: the column used to create groups

    Notes:

        Returns a list of the groups and a list of the corresponding
        instance count for that group.

    """
    # partition the table
    partitions = partition(table, [partition_column])
    lengths = []
    vals = []
    
    # for each table, get value of partitioned column and length of each tables
    for table in partitions:
        lengths.append(table.row_count())
        vals.append(table[0][partition_column])
        
    return vals, lengths


def dot_chart(xvalues, xlabel, title, filename=None):
    """Create a dot chart from given values.
    
    Args:
        xvalues: The values to display
        xlabel: The label of the x axis
        title: The title of the chart
        filename: Filename to save chart to (in SVG format)

    Notes:
        If filename is given, chart is saved and not displayed.

    """
    # reset figure
    plt.figure()
    # dummy y values
    yvalues = [1] * len(xvalues)
    # create an x-axis grid
    plt.grid(axis='x', color='0.85', zorder=0)
    # create the dot chart (with pcts)
    plt.plot(xvalues, yvalues, 'b.', alpha=0.2, markersize=16, zorder=3)
    # get rid of the y axis
    plt.gca().get_yaxis().set_visible(False)
    # assign the axis labels and title
    plt.xlabel(xlabel)
    plt.title(title)
    # save as file or just show
    if filename:
        plt.savefig(filename, format='svg')
    else:
        plt.show()
    # close the plot
    plt.close()

    
def pie_chart(values, labels, title, filename=None):
    """Create a pie chart from given values.
    
    Args:
        values: The values to display
        labels: The label to use for each value
        title: The title of the chart
        filename: Filename to save chart to (in SVG format)

    Notes:
        If filename is given, chart is saved and not displayed.

    """
    # Create a pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title(title)

    # Display or save the chart
    if filename:
        plt.savefig(filename, format='svg')
    else:
        plt.show()

def bar_chart(bar_values, bar_names, xlabel, ylabel, title, filename=None):
    """Create a bar chart from given values.
    
    Args:
        bar_values: The values used for each bar
        bar_labels: The label for each bar value
        xlabel: The label of the x-axis
        ylabel: The label of the y-axis
        title: The title of the chart
        filename: Filename to save chart to (in SVG format)

    Notes:
        If filename is given, chart is saved and not displayed.

    """
    # Create a bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(bar_names, bar_values)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    # Display or save the chart
    if filename:
        plt.savefig(filename, format='svg')
    else:
        plt.show()

    
def scatter_plot(xvalues, yvalues, xlabel, ylabel, title, filename=None):
    """Create a scatter plot from given values.
    
    Args:
        xvalues: The x values to plot
        yvalues: The y values to plot
        xlabel: The label of the x-axis
        ylabel: The label of the y-axis
        title: The title of the chart
        filename: Filename to save chart to (in SVG format)

    Notes:
        If filename is given, chart is saved and not displayed.

    """
    # Create a scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(xvalues, yvalues)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    # Display or save the chart
    if filename:
        plt.savefig(filename, format='svg')
    else:
        plt.show()


def box_plot(distributions, labels, xlabel, ylabel, title, filename=None):
    """Create a box and whisker plot from given values.
    
    Args:
        distributions: The distribution for each box
        labels: The label of each corresponding box
        xlabel: The label of the x-axis
        ylabel: The label of the y-axis
        title: The title of the chart
        filename: Filename to save chart to (in SVG format)

    Notes:
        If filename is given, chart is saved and not displayed.

    """
    # Create a box and whisker plot
    plt.figure(figsize=(10, 6))
    plt.boxplot(distributions, labels=labels)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    # Display or save the chart
    if filename:
        plt.savefig(filename, format='svg')
    else:
        plt.show()


    
