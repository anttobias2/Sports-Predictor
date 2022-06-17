'''
Author: Anthony Tobias
6/16/2022
v 1.0.0
This program contains the main file for a sports predictor that imports data and predicts the outcome and score of an NBA game
'''

# Create a list of tites for the columns and return them
def read_column_titles(filename):
    file1 = open(filename, 'r')
    column_titles = file1.readline()

    return column_titles.split(',')

def main():
    teams_column_titles = read_column_titles('teams.csv')
    print(teams_column_titles)



main()