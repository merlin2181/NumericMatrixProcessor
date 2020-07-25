"""
    Numeric Matix Processor:
    This program only adds two same sized matrices
"""


def matrix(rows, columns):
    """Takes numerical input for the amount of rows and
    columns of a matrix and creates it with user input converted to ints"""
    mtx = []
    for i in range(rows):
        mtx.append([])
    for i in range(rows):
        for j in range(columns):
            mtx[i].append(chr(ord('a') + j))
    for i in range(rows):
        mtx[i] = [int(num) for num in input().split()]
    return mtx


def add_matrices(m1, m2):
    """Compares the rows of both matrices to make sure they are the same and then compares
    the columns of each row to make sure they are the same. If it passes both tests, the
    function adds the two together.  If the rows or columns don't match it returns an error"""
    if len(m1) == len(m2):
        new_matrix = []
        for i in range(len(m1)):
            if len(m1[i]) == len(m2[i]):
                new_row = []
                for j in range(len(m1[i])):
                    new_row.append(m1[i][j] + m2[i][j])
                new_matrix.append(new_row)
            else:
                return 'ERROR'
        return new_matrix
    else:
        return 'ERROR'


def input_matrix():
    """Function that takes user input for the amount of rows and columns and passes them to
    the matrix function which returns a new user created matrix"""
    r, c = (int(num) for num in input().split())
    mtx = matrix(r, c)
    return mtx


def print_new_matrix(mtx):
    """A function that prints out the return from the add_matrices function"""
    if type(mtx) == list:
        for line in mtx:
            print(*line)
    else:
        print(mtx)


matrix1 = input_matrix()
matrix2 = input_matrix()
added_mtx = add_matrices(matrix1, matrix2)
print('\r')
print_new_matrix(added_mtx)
