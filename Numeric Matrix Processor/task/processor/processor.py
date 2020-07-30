"""
    Numeric Matrix Processor:
    This program will add two matrices, multiply two matrices
    and multiply one matrix by a scalar
"""
import math


def make_matrix(rows, columns, number=None):
    """
    Function that returns a propagated matrix that is type float
        rows: the number of rows in the matrix
        columns: the number of columns in a matrix
        number: 'zero' returns a zero matrix, any other input changes the interface
                string when asking for input for the matrix
    """
    mtx = []
    while len(mtx) < rows:
        mtx.append([])
        while len(mtx[-1]) < columns:
            mtx[-1].append(0.0)
    if number == 'zero':  # returns a zero matrix
        return mtx
    if number:
        print(f'Enter {number} matrix:\r')

        # propagates the matrix with the user's input converted to type float
        for i in range(rows):
            mtx[i] = [float(num) for num in input().split()]
        return convert_to_int(mtx)

    # User input screen if there is only 1 matrix to be input
    print('Enter matrix:\r')
    for i in range(rows):
        mtx[i] = [float(num) for num in input().split()]
    return convert_to_int(mtx)


def add_em():
    """
    Function that adds together two matrices
    """
    m1, m2 = need_matrices()
    rows_m1 = len(m1)
    cols_m1 = len(m1[0])
    rows_m2 = len(m2)
    cols_m2 = len(m2[0])

    # Checks that there are the same amount or rows and columns before adding otherwise it
    # prints an error message
    if rows_m1 == rows_m2 and cols_m1 == cols_m2:
        tmp_mtx = make_matrix(rows_m1, cols_m2, 'zero')
        for i in range(len(m1)):
            for j in range(len(m1[i])):
                tmp_mtx[i][j] = m1[i][j] + m2[i][j]
        print_new_matrix(tmp_mtx)
    else:
        print('ERROR: The number of rows and/or columns of the matrices are not the same.')


def multiply_scalar():
    """
    Function that multiplies a matrix by a scalar
    """
    r, c = (int(num) for num in input('Enter size of matrix: ').split())
    mtx = make_matrix(r, c)
    scalar = float(input('Enter constant: '))
    for i in range(len(mtx)):
        for j in range(len(mtx[i])):
            mtx[i][j] *= scalar
    print_new_matrix(mtx)


def multiply_em():
    """
    Function to multiple two matrices together
    """
    mtx_1, mtx_2 = need_matrices()
    rows_mtx1 = len(mtx_1)
    cols_mtx1 = len(mtx_1[0])
    rows_mtx2 = len(mtx_2)
    cols_mtx2 = len(mtx_2[0])
    tmp_mtx = make_matrix(rows_mtx1, cols_mtx2, 'zero')

    # Check to make sure we can multiply the two matrices
    if cols_mtx1 == rows_mtx2:

        # we can multiply the two matrices, so we transpose the 2nd matix and the both columns
        # are the same amount
        m2 = transpose(mtx_2)
        rows_m2 = len(m2)
        for i in range(rows_mtx1):  # keeps track of the rows in the first matrix
            for j in range(rows_m2):  # keeps track of the rows in the second matrix
                total = 0

                # Multiplies all the rows of the second matrix to the current row of the first
                # matrix, adds them together and stores them in a new matrix
                for _ct in range(cols_mtx1):
                    total += mtx_1[i][_ct] * m2[j][_ct]

                # Fills all the columns of the current row with the total before moving
                # on to the next row
                tmp_mtx[i][j] = total
        print_new_matrix(tmp_mtx)

    # if we can't multiply the two matrices, print an error message
    else:
        print('ERROR: Can not multiply because rows of matrix_A and columns of matrix_B are not equal.')


def multiply_matrices(mtx_1, mtx_2):
    rows_mtx1 = len(mtx_1)
    cols_mtx1 = len(mtx_1[0])
    cols_mtx2 = len(mtx_2[0])
    tmp_mtx = make_matrix(rows_mtx1, cols_mtx2, 'zero')

    # we can multiply the two matrices, so we transpose the 2nd matix and then both columns
    # are the same amount
    m2 = transpose(mtx_2)
    rows_m2 = len(m2)
    for i in range(rows_mtx1):  # keeps track of the rows in the first matrix
        for j in range(rows_m2):  # keeps track of the rows in the second matrix
            total = 0

            # Multiplies all the rows of the second matrix to the current row of the first
            # matrix, adds them together and stores them in a new matrix
            for _ct in range(cols_mtx1):
                total += mtx_1[i][_ct] * m2[j][_ct]

            # Fills all the columns of the current row with the total before moving
            # on to the next row
            tmp_mtx[i][j] = round(total)
    return tmp_mtx


def print_new_matrix(mtx):
    """
    Function that properly prints out a matrix from another function
    """
    print('The result is:')
    if type(mtx) == list:
        for line in mtx:
            print(*line)
    else:
        print(mtx)
    print('\r')


def transpose(m1):
    """
    Function to transpose a given matrix along the main diagonal
    """
    rows_m1 = len(m1)
    cols_m1 = len(m1[0])

    # Use the number of columns for new amount of rows and number of rows for new amount of
    # columns
    tmp_mtx = make_matrix(cols_m1, rows_m1, 'zero')
    for i in range(rows_m1):
        for j in range(cols_m1):
            tmp_mtx[j][i] = m1[i][j]
    return tmp_mtx


def transpose_side(m1):
    """
    Function that transposes a matrix using the side diagonal
    """
    rows_m1 = len(m1)
    cols_m1 = len(m1[0])
    tmp_mtx = make_matrix(cols_m1, rows_m1, 'zero')
    for i in range(-1, (-1 - rows_m1), -1):
        for j in range(-1, (-1 - cols_m1), -1):
            tmp_mtx[-1 - j][-1 - i] = m1[i][j]
    return tmp_mtx


def transpose_vert(m1):
    """
    Function that transposes a matrix using a vertical line
    """
    rows_m1 = len(m1)
    cols_m1 = len(m1[0])
    if cols_m1 > 1:
        tmp_mtx = make_matrix(rows_m1, cols_m1, 'zero')
        for i in range(rows_m1):
            for j in range(-1, (-1 - cols_m1), -1):
                tmp_mtx[i][-1 - j] = m1[i][j]
        return tmp_mtx
    return m1


def transpose_horz(m1):
    """
    Function that transposes a matrix using a horizontal line
    """
    rows_m1 = len(m1)
    cols_m1 = len(m1[0])
    if rows_m1 > 1:
        tmp_mtx = make_matrix(rows_m1, cols_m1, 'zero')
        for i in range(-1, (-1 - rows_m1), -1):
            for j in range(cols_m1):
                tmp_mtx[-1 - i][j] = m1[i][j]
        return tmp_mtx
    return m1


def determinant(m1):
    """
    Function to calculate the determinant of a matrix
    :param m1: the matrix in which you want to find the determinant
    :return: the determinant of the matrix
    """
    if len(m1) == 1:  # base case if matrix has only 1 number
        return m1
    elif test_square_matrix(m1):
        if len(m1) == 2:  # base case
            return m1[0][0] * m1[1][1] - m1[0][1] * m1[1][0]
        cols_m = len(m1[0])
        new_m = m1[0]
        new_mtx = [find_new_matrix(m1, i) for i in range(cols_m)]
        total = 0
        for i in range(len(new_m)):
            total += new_m[i] * ((-1) ** (i + 2)) * (determinant(new_mtx[i]))
        return total
    else:
        return 'The matrix is not square'


def find_new_matrix(mtx, col, row=0):
    """
    Function to remove columns from a matrix
    :param mtx: the matrix to perform the action
    :param col: the column to omit from the new matrix
    :param row: the row to omit from the new matrix, default is row at index 0
    :return: the minor of the given matrix
    """
    rows_mtx = len(mtx) - 1
    new = []
    while len(new) < rows_mtx:
        new.append([])
    num = 0
    while num < rows_mtx:
        for i in range(len(mtx)):
            if i == row:
                continue
            for j in range(len(mtx[0])):
                if j == col:
                    continue
                new[num].append(mtx[i][j])
            num += 1
    return new


def test_square_matrix(mtx):
    """
    Function that checks if a matrix is square before performing a calculation
    :param mtx: the matrix to test
    :return: True or False
    """
    rows_m = len(mtx)
    for i in range(rows_m):  # test to see if the matrix is square
        if len(mtx[i]) == rows_m:
            continue
        else:
            print("This is not a square matrix")
            return False
    return True


def inverse(m1):
    """
    Function to find the inverse matrix of a given matrix
    :param m1: the matrix we need to find the inverse of
    :return: the inverse matrix
    """
    det_m1 = determinant(m1)
    if det_m1 == 0:
        return "This matrix doesn't have an inverse."
    rows_m1 = len(m1)
    cols_m1 = len(m1[0])
    if rows_m1 == 2:  # find the inverse of a 2x2 matrix
        new_mtx = []
        while len(new_mtx) < rows_m1:
            new_mtx.append([])
        new_mtx[0].append(m1[1][1])
        new_mtx[0].append(-(m1[1][0]))
        new_mtx[1].append(-(m1[0][1]))
        new_mtx[1].append(m1[0][0])
        new_mtx = transpose(new_mtx)
        for i in range(len(new_mtx)):
            for j in range(len(new_mtx[i])):
                new_mtx[i][j] = truncate(((1 / det_m1) * new_mtx[i][j]), 2)
        is_m1_m2 = multiply_matrices(m1, new_mtx)
        is_m2_m1 = multiply_matrices(new_mtx, m1)
        if is_m1_m2 == is_m2_m1:
            return new_mtx
    else:
        # find the inverse of a matrix bigger than 2x2
        inv_mtx = []
        det_mtx = [find_new_matrix(m1, j, i) for i in range(rows_m1) for j in range(cols_m1)]  # find all the minors
        for i in range(len(det_mtx)):
            inv_mtx.append(determinant(det_mtx[i]))  # find the determinant of each minor
        rows_det = len(inv_mtx)
        num = rows_det // rows_m1

        # create a matrix of minors that is the same size of the input matrix
        new_mtx = [inv_mtx[i * num:(i + 1) * num] for i in range((rows_det + num - 1) // num)]

        # multiply matrix of minors elements by its co-factor
        for i in range(len(new_mtx)):
            for j in range(len(new_mtx[0])):
                new_mtx[i][j] *= (-1) ** (i + j + 2)

        # transpose the matrix of minors
        new_mtx = transpose(new_mtx)
        for i in range(len(new_mtx)):
            for j in range(len(new_mtx[i])):
                new_mtx[i][j] *= (1 / det_m1)
                if type(new_mtx[i][j]) == float:
                    new_mtx[i][j] = truncate(new_mtx[i][j], 2)  # if element is a float, store to 2 decimal places
                if new_mtx[i][j].is_integer():
                    new_mtx[i][j] = int(new_mtx[i][j])  # if element is an float integer, make it an integer

        #  test to see if M x M^-1 is the same as M^-1 x M
        is_m1_m2 = multiply_matrices(m1, new_mtx)
        is_m2_m1 = multiply_matrices(new_mtx, m1)
        if is_m1_m2 == is_m2_m1:
            return new_mtx


def truncate(f, n):
    """
    Function to store a n decimal places of a float number without rounding
    :param f: the number to truncate
    :param n: the number of decimal places to keep
    :return: the truncated float number
    """
    if f < 0:
        return math.ceil(f * 10 ** n) / 10 ** n
    return math.floor(f * 10 ** n) / 10 ** n


def convert_to_int(m1):
    """
    Function that checks user input to see if it really is type int or not because the
    program defaults the user input to a float
    """
    for row in m1:
        for col in range(len(row)):
            if row[col].is_integer():
                row[col] = int(row[col])
            else:
                continue
    return m1


def need_matrices():
    """
    Function that creates two matrices to be used in another function
    """
    r, c = (int(num) for num in input('Enter size of first matrix: ').split())
    mtx1 = make_matrix(r, c, 'first')
    r, c = (int(num) for num in input('Enter size of second matrix: ').split())
    mtx2 = make_matrix(r, c, 'second')
    return mtx1, mtx2


def run_operation(func):
    """
    Function that takes in 1 of the 4 transpose functions and prints the transposed matrix
    :param func: transpose, transpose_side, transpose_vert or transpose_horz
    """
    r, c = (int(num) for num in input('Enter size of matrix: ').split())
    mtx = make_matrix(r, c)
    mtx = func(mtx)
    print_new_matrix(mtx)


def menu():
    """
    Function that delivers a quick menu to the user and returns their choice
    """
    print('1. Add matrices\n2. Multiply matrix by a constant\n3. Multiply matrices\n4. Transpose matrix\n'
          '5. Calculate a determinant\n6. Inverse Matrix\n0. Exit')
    return input('Your choice: ')


def transpose_menu():
    """
    Function that delivers the transpose user menu
    """
    print('\n1. Main diagonal\n2. Side diagonal\n3. Vertical line\n4. Horizontal line')
    choice = input('Enter choice: ')
    if choice == '1':
        run_operation(transpose)
    elif choice == '2':
        run_operation(transpose_side)
    elif choice == '3':
        run_operation(transpose_vert)
    elif choice == '4':
        run_operation(transpose_horz)


def menu_choice():
    """
    Function that takes the return value of the menu() function and applies the choice.
    """
    while True:
        number = menu()
        if number == '1':
            add_em()
        elif number == '2':
            multiply_scalar()
        elif number == '3':
            multiply_em()
        elif number == '4':
            transpose_menu()
        elif number == '5':
            run_operation(determinant)
        elif number == '6':
            run_operation(inverse)
        elif number == '0':
            exit()
        else:
            continue


menu_choice()
