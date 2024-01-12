from data_eval import *
from random import randint
import csv

def get_averages(table, columns):
    '''Return a table with the average for each value in row
    
        args:
            table: data table to get averages with
            columns: columns to calculate averages for
            
        note: since the tables are in chronological order already we can do it like this
    '''
    length = table.row_count()
    
    for column in columns:
        for i in range(length-1):
            average = 0
            count = 0
            for j in range(i+1, length):
                average += table[j][column]
                count += 1
            
            average /= count
        
            table[i][column] = average
            
    return table


def calc_combined_game_stats(home_table, away_table, columns):
    ''' Returns a table that has averaged values
        between the home and away teams stats
        
        args:
            home_table: Table with home team data
            away_table: Table with away team data
            columns: The columns of which stats wanted to combine
    '''
    for i in range(home_table.row_count()):
        home_row = home_table[i]
        away_row = 0
        
        for j in range(away_table.row_count()):
            if away_table[j]['Game ID'] == home_row['Game ID']:
                away_row = away_table[j]
                break
                
        
        for col in columns:
            home_table[i][col] = (home_row[col] + away_row[col]) / 2
            
    return home_table

def calc_results(results, message):
    '''Calculates and prints the accuracy, precision and recall from a given confusion matrix
    
        args:
            results: confusion matrix to use for calculation
            message: the information to include about the classifier used 
    '''
    acc = (accuracy(results, 0) + accuracy(results, 1)) / 2
    prec = (precision(results, 0) + precision(results, 1)) / 2
    rec = (recall(results, 0) + recall(results, 1)) / 2

    print(f'The accuracy of the {message} is {acc * 100:.2f}%')
    print(f'The precision of the {message} is {prec:.2f}')
    print(f'The recall of the {message} is {rec:.2f}')
    
def get_tables(data, test_set_size):
    '''Returns a test and train table based on the data given
    
        args:
            data: the original table to use for splitting into test and train
            test_set_size: the number of rows to include in test table
            
        Notes:
            The train table includes the original data except for the instances
            used in the test table
    '''
    train = DataTable(data[0].columns())
    test = DataTable(data[0].columns())
    test_indicies = []
    
    for _ in range(test_set_size):
        index = randint(0, data.row_count()-1)
        
        while index in test_indicies:
            index = randint(0, data.row_count()-1)
            
        test.append(data[index].values())
        
    for i in range(data.row_count()):
        if i not in test_indicies:
            train.append(data[i].values())
            
    return test, train

def my_dot(a, b): 
    """
   Compute the dot product of two vectors
 
    Args:
      a (ndarray (n,)):  input vector 
      b (ndarray (n,)):  input vector with same dimension as a
    
    Returns:
      x (scalar): 
    """
    x=0
    # Write a for loop to implement dot product
    for i in range(len(a)):
      x += a[i] * b[i]
    
    
    return x

def gradient_descent_eval(w, b, test, label_col, cont_cols):
    X_test = []
    y_test = []
    
    for val in test:
        tmp = []
        for col in cont_cols:
            tmp.append(val[col])
        X_test.append(tmp)
        y_test.append(val['Home Wins'])
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
            
    zeros = []
    for i in range(len(X_test)):
        prediction = my_dot(X_test[i], w) + b
        if prediction >= 0.5:
            prediction = 1
        else:
            prediction = 0
            
        for r in confusion:
            if r['actual'] == y_test[i]:
                r[prediction] += 1
    
    return confusion
    
            
        
    
    
def get_2023_data():
    '''Returns a table created from data taken from the NBA website
    '''
    rows = []
    nba2023 = DataTable(['Home Team Abbreviation', 'Home MATCH UP', 'Date', 'Home Wins', 'MIN', 'PTS Home', 'Home FGM', 
                        'Home FGA', 'FG PCT Home', 'Home 3PM', 'Home 3PA', 'FG3 PCT Home', 'Home FTM', 'Home FTA', 'FT PCT Home', 'Home OREB', 
                        'Home DREB', 'REB Home', 'AST Home', 'Home STL', 'Home BLK', 'Home TOV', 'Home PF', 'Home +/-', 
                        'Away Team Abbreviation', 'Away MATCH UP', 'GAME DATE Away', 'Away Wins', 'Away MIN', 'PTS Away', 'Away FGM', 
                        'Away FGA', 'FG PCT Away', 'Away 3PM', 'Away 3PA', 'FG3 PCT Away', 'Away FTM', 'Away FTA', 'FT PCT Away', 'Away OREB', 
                        'Away DREB', 'REB Away', 'AST Away', 'Away STL', 'Away BLK', 'Away TOV', 'Away PF', 'Away +/-', 
                        'Home Games Played', 'Home Win PCT', 'Visitor Games Played', 'Visitor Win PCT'])

    with open('nba2023.csv') as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            rows.append(row)
        
    for i in range(len(rows)):
        for j in range(len(rows[0])):
            try:
                rows[i][j] = float(rows[i][j])
            except:
                pass

    for i in range(len(rows)):
        if rows[i][3] == 'W':
            rows[i][3] = 1
        elif rows[i][3] == 'L':
            rows[i][3] = 0
            
    # combine the team box score for same game
    for i in range(len(rows)):
        teams1 = rows[i][1].split(' ')
        for j in range(len(rows)):
            if j != i:
                teams2 = rows[j][1].split(' ')
                
                if teams1[0] == teams2[2] and rows[i][2] == rows[j][2]:
                    for val in rows[j]:
                        rows[i].append(val)

    # get rid of all away game data
    to_delete = []                
    for i in range(len(rows)):
        matchup = rows[i][1].split(' ')
        if matchup[1] == '@':
            to_delete.append(i)

    to_delete = sorted(to_delete, reverse=True)

    for index in to_delete:
        del rows[index]
    
    # bug in code this fixes it for now
    for i in range(len(rows)):
        if len(rows[i]) != 48:
            del rows[i][48:]
         
    # calculate win pct   
    wins = {}
    for i in range(len(rows) - 1, -1, -1):
        
        # check if home and away teams are in dictionary
        if rows[i][0] not in wins:
            wins[rows[i][0]] = [0, 1]
        else:
            wins[rows[i][0]][1] += 1
            
        if rows[i][24] not in wins:
            wins[rows[i][24]] = [0, 1]
        else:
            wins[rows[i][24]][1] += 1
            
        # if home wins, add to home total
        if rows[i][3] == 1:
            wins[rows[i][0]][0] += 1
        # else add to away
        else:
            wins[rows[i][24]][0] += 1

       
        # add Home Total Games
        rows[i].append(wins[rows[i][0]][1])
        # add Home Win PCT
        rows[i].append(wins[rows[i][0]][0] / wins[rows[i][0]][1])
        # add Number Away Games
        rows[i].append(wins[rows[i][24]][1])
        # add Away Win PCT
        rows[i].append(wins[rows[i][24]][0] / wins[rows[i][24]][1])
    
    
    
    for row in rows:
        nba2023.append(row)
        
        
    return nba2023

def calc_combined_game_stats_2023(home_table, away_table, columns):
    ''' Returns a table that has averaged values
        between the home and away teams stats
        
        args:
            home_table: Table with home team data
            away_table: Table with away team data
            columns: The columns of which stats wanted to combine
    '''
    for i in range(home_table.row_count()):
        home_row = home_table[i]
        away_row = 0
        
        for j in range(away_table.row_count()):
            # print(away_table[j]['Home MATCH UP'])
            if away_table[j]['Home MATCH UP'] == home_row['Home MATCH UP'] and away_table[j]['Date'] == home_row['Date']:
                away_row = away_table[j]
                break
                
        
        for col in columns:
            home_table[i][col] = (home_row[col] + away_row[col]) / 2
            
    return home_table

def get_todays_matchups(home_table, away_table, columns):
    '''Returns a table that consists of matchups of teams based on the date
    
        args:
            home_table: table that consists of the home data
            away_table: table that consists of the away data
            columns: columns to include in new table
    '''
    date = '12/12/2023'
    
    for i in range(home_table.row_count()):
        home_table[i]['Home MATCH UP'] = home_table[i]['Home Team Abbreviation'] + ' vs. ' + away_table[i]['Away Team Abbreviation'] 
        home_table[i]['Date'] = date
        for col in columns:
            home_table[i][col] = (home_table[i][col] + away_table[i][col]) / 2
            
            
    return home_table