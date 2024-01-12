"""
HW-2 Data Table implementation.

NAME: Anthony Tobias
DATE: Fall 2023
CLASS: CPSC 322

"""

import csv
import tabulate


class DataRow:
    """A basic representation of a relational table row. The row maintains
    its corresponding column information.

    """
    
    def __init__(self, columns=[], values=[]):
        """Create a row from a list of column names and data values.
           
        Args:
            columns: A list of column names for the row
            values: A list of the corresponding column values.

        Notes: 
            The column names cannot contain duplicates.
            There must be one value for each column.

        """
        if len(columns) != len(set(columns)):
            raise ValueError('duplicate column names')
        if len(columns) != len(values):
            raise ValueError('mismatched number of columns and values')
        self.__columns = columns.copy()
        self.__values = values.copy()

        
    def __repr__(self):
        """Returns a string representation of the data row (formatted as a
        table with one row).

        Notes: 
            Uses the tabulate library to pretty-print the row.

        """
        return tabulate.tabulate([self.values()], headers=self.columns())

        
    def __getitem__(self, column):
        """Returns the value of the given column name.
        
        Args:
            column: The name of the column.

        """
        if column not in self.columns():
            print(f'This is a bad column name {column}')
            raise IndexError('bad column name')
        return self.values()[self.columns().index(column)]


    def __setitem__(self, column, value):
        """Modify the value for a given row column.
        
        Args: 
            column: The column name.
            value: The new value.

        """
        if column not in self.columns():
            raise IndexError('bad column name')
        self.__values[self.columns().index(column)] = value


    def __delitem__(self, column):
        """Removes the given column and corresponding value from the row.

        Args:
            column: The column name.

        """
        # TODO
        if column not in self.columns():
            raise IndexError('bad column name')
        index = self.columns().index(column)
        del self.__values[index]
        del self.__columns[index]

    
    def __eq__(self, other):
        """Returns true if this data row and other data row are equal.

        Args:
            other: The other row to compare this row to.

        Notes:
            Checks that the rows have the same columns and values.

        """
        
        # must be DataRow object
        if not isinstance(other, DataRow):
            raise ValueError('expecting DataRow object')
        # the values of both objects must be of same lenght
        if len(other.values()) != len(self.values()):
            raise ValueError('These lists are not the same length')
        # check if each value is the same
        for i in range(len(other.columns())):
            if other.values()[i] != self.values()[i]:
                return False
            
        return True

    
    def __add__(self, other):
        """Combines the current row with another row into a new row.
        
        Args:
            other: The other row being combined with this one.

        Notes:
            The current and other row cannot share column names.

        """
        if not isinstance(other, DataRow):
            raise ValueError('expecting DataRow object')
        if len(set(self.columns()).intersection(other.columns())) != 0:
            raise ValueError('overlapping column names')
        return DataRow(self.columns() + other.columns(),
                       self.values() + other.values())


    def columns(self):
        """Returns a list of the columns of the row."""
        return self.__columns.copy()


    def values(self, columns=None):
        """Returns a list of the values for the selected columns in the order
        of the column names given.
           
        Args:
            columns: The column values of the row to return. 

        Notes:
            If no columns given, all column values returned.

        """
        if columns is None:
            return self.__values.copy()
        if not set(columns) <= set(self.columns()):
            raise ValueError('duplicate column names')
        return [self[column] for column in columns]


    def select(self, columns=None):
        """Returns a new data row for the selected columns in the order of the
        column names given.

        Args:
            columns: The column values of the row to include.
        
        Notes:
            If no columns given, all column values included.

        """
        if columns == None:
            return DataRow(self.columns(), self.values())
        
        values = self.values(columns)
        return DataRow(columns, values)
        

    
    def copy(self):
        """Returns a copy of the data row."""
        return self.select()

    

class DataTable:
    """A relational table consisting of rows and columns of data.

    Note that data values loaded from a CSV file are automatically
    converted to numeric values.

    """
    
    def __init__(self, columns=[]):
        """Create a new data table with the given column names

        Args:
            columns: A list of column names. 

        Notes:
            Requires unique set of column names. 

        """
        if len(columns) != len(set(columns)):
            raise ValueError('duplicate column names')
        self.__columns = columns.copy()
        self.__row_data = []


    def __repr__(self):
        """Return a string representation of the table.
        
        Notes:
            Uses tabulate to pretty print the table.

        """
        rows = self.__row_data
        row_values = []
        for row in rows:
            row_values.append(row.values())
            
        return tabulate.tabulate(row_values, headers=self.columns())
        # return tabulate.tabulate(row_values)

    
    def __getitem__(self, row_index):
        """Returns the row at row_index of the data table.
        
        Notes:
            Makes data tables iterable over their rows.

        """
        return self.__row_data[row_index]

    
    def __delitem__(self, row_index):
        """Deletes the row at row_index of the data table.

        """
        del self.__row_data[row_index]

        
    def load(self, filename, delimiter=','):
        """Add rows from given filename with the given column delimiter.

        Args:
            filename: The name of the file to load data from
            delimeter: The column delimiter to use

        Notes:
            Assumes that the header is not part of the given csv file.
            Converts string values to numeric data as appropriate.
            All file rows must have all columns.
        """
        with open(filename, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=delimiter)
            num_cols = len(self.columns())
            for row in reader:
                row_cols = len(row)                
                if num_cols != row_cols:
                    raise ValueError(f'expecting {num_cols}, found {row_cols}')
                converted_row = []
                for value in row:
                    converted_row.append(DataTable.convert_numeric(value.strip()))
                self.__row_data.append(DataRow(self.columns(), converted_row))

                    
    def save(self, filename, delimiter=','):
        """Saves the current table to the given file.
        
        Args:
            filename: The name of the file to write to.
            delimiter: The column delimiter to use. 

        Notes:
            File is overwritten if already exists. 
            Table header not included in file output.
        """
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=delimiter, quotechar='"',
                                quoting=csv.QUOTE_NONNUMERIC)
            for row in self.__row_data:
                writer.writerow(row.values())


    def column_count(self):
        """Returns the number of columns in the data table."""
        return len(self.__columns)


    def row_count(self):
        """Returns the number of rows in the data table."""
        return len(self.__row_data)


    def columns(self):
        """Returns a list of the column names of the data table."""
        return self.__columns.copy()


    def append(self, row_values):
        """Adds a new row to the end of the current table. 

        Args:
            row_data: The row to add as a list of values.
        
        Notes:
            The row must have one value per column. 
        """
        self.__row_data.append(DataRow(self.columns(), row_values))

    
    def rows(self, row_indexes):
        """Returns a new data table with the given list of row indexes. 

        Args:
            row_indexes: A list of row indexes to copy into new table.
        
        Notes: 
            New data table has the same column names as current table.

        """
        # create new table
        new_table = DataTable(self.columns())
        
        # append new table to have the same rows as self
        for index in row_indexes:
            new_row = self[index]
            new_table.append(new_row.values())
            
        return new_table
            

    
    def copy(self):
        """Returns a copy of the current table."""
        table = DataTable(self.columns())
        for row in self:
            table.append(row.values())
        return table
    

    def update(self, row_index, column, new_value):
        """Changes a column value in a specific row of the current table.

        Args:
            row_index: The index of the row to update.
            column: The name of the column whose value is being updated.
            new_value: The row's new value of the column.

        Notes:
            The row index and column name must be valid. 

        """
        # only need to make a change to one value
        self[row_index][column] = new_value

    
    @staticmethod
    def combine(table1, table2, columns=[], non_matches=False):
        """Returns a new data table holding the result of combining table 1 and 2.

        Args:
            table1: First data table to be combined.
            table2: Second data table to be combined.
            columns: List of column names to combine on.
            nonmatches: Include non matches in answer.

        Notes:
            If columns to combine on are empty, performs all combinations.
            Column names to combine are must be in both tables.
            Duplicate column names removed from table2 portion of result.

        """
        # Make sure the columns are unique and in both tables
        if len(columns) != len(set(columns)):
            raise IndexError('duplicate column names')
        
        for column in columns:
            if (column not in table1.columns()) or (column not in table2.columns()):
                raise IndexError('bad column name')
        
        matches = {}
        unused = {}
        original_columns = table1.columns()
        temp_columns = table2.columns()
        new_columns = original_columns
        needed_columns = []
        
        # create list of columns for new table
        # and list for needed columns from table2
        for column in temp_columns:
            if column not in new_columns:
                new_columns.append(column)
                needed_columns.append(column)
            
        new_table = DataTable(new_columns)
        
        # add values to dictionary to check for matches in table 2
        for row in table1:
            key = str(row.select(columns).values())
            if key not in matches:
                matches[key] = [row]
            else:
                matches[key].append(row)
        
        # create copy of matches to find which values were used
        unused = dict(matches)
        
        # for each row in table check for match
        for row in table2:
            key = str(row.select(columns).values())
            if key in matches:
                temp_row = row.select(needed_columns)
                
                # delete keys that have been used
                if key in unused:
                    del unused[key]
                    
                # create new row and append
                for match in matches[key]:
                    new_row = match + temp_row
                    new_table.append(new_row.values())

            # if no match, create new row
            elif key not in matches and non_matches == True:
                temp_vals = []
                for i in range(len(new_columns)):
                    temp_vals.append('')
                new_row = DataRow(new_columns, temp_vals)
                
                # add values needed and leave others blank
                for column in row.columns():
                    new_row[column] = row[column]
                new_table.append(new_row.values())
            
        # if non matches == true, add values from table1
        if non_matches == True:
            temp_vals = []
            
            for i in range(len(new_columns)):
                temp_vals.append('')
            
            # for each key that is unused, add row for each value with matching key
            for key in unused:
                for i in range(len(unused[key])):
                    new_row = DataRow(new_columns, temp_vals)
                    temp_row = unused[key][i]
                    for column in temp_row.columns():
                        new_row[column] = temp_row[column]
                    new_table.append(new_row.values())
            
        
        return new_table

    
    @staticmethod
    def convert_numeric(value):
        """Returns a version of value as its corresponding numeric (int or
        float) type as appropriate.

        Args:
            value: The string value to convert

        Notes:
            If value is not a string, the value is returned.
            If value cannot be converted to int or float, it is returned.

         """
    
        # nested try except blocks to try int and float
        try:
            new_val = int(value)
        except:
            try:
                new_val = float(value)
            except:
                return value
        
        return new_val
    
    def drop(self, columns):
        """Removes the given columns from the current table.
        Args:
        column: the name of the columns to drop
        """
        original_columns = self.columns()
        new_columns = []
        
        # get list of columns for new table
        for column in original_columns:
            if column not in columns:
                new_columns.append(column)
        
        # create copy of self to choose from 
        temp_table = self.copy()
          
        # clear out table  
        self.__row_data = []
        self.__columns = new_columns
        
        # get rows from original data with only wanted columns
        for row in temp_table:
            temp_row = row.select(new_columns)
            self.append(temp_row.values())
            
        