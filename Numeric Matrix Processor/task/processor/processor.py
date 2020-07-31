"""
    Numeric Matrix Processor:
    This program will add two matrices, multiply two matrices,
    multiply one matrix by a scalar, find the determinant of a
    matrix and find the inverse of a matrix
"""
import math


class Matrix:

    def __init__(self):
        self.matrix_1 = None
        self.matrix_2 = None
        self.scalar = None
        self.matrix_answer = None
        self.transpose_matrix = None
        self.trans_side_matrix = None
        self.trans_vert_matrix = None
        self.trans_horz_matrix = None
        self.matrix_minor = None
        self.print = print_new_matrix

    def menu_choice(self):
        """
        Method that outputs the main menu of the program and calls a function depending on the user's choice.
        :return: None
        """
        while True:
            print('1. Add matrices\n2. Multiply matrix by a constant\n3. Multiply matrices\n4. Transpose matrix\n'
                  '5. Calculate a determinant\n6. Inverse Matrix\n0. Exit')
            number = input('Your choice: ')
            if number == '1':
                self.matrix_1, self.matrix_2 = need_two()
                self.print(self.add(self.matrix_1, self.matrix_2))
            elif number == '2':
                self.matrix_1 = need_one()
                self.scalar = float(input('Enter constant: '))
                self.print(self.multiply_scalar(self.matrix_1, self.scalar))
            elif number == '3':
                self.matrix_1, self.matrix_2 = need_two()
                self.print(self.multiply_matrices(self.matrix_1, self.matrix_2))
            elif number == '4':
                self.transpose_menu()
            elif number == '5':
                self.matrix_1 = need_one()
                self.print(self.determinant(self.matrix_1))
            elif number == '6':
                self.matrix_1 = need_one()
                self.print(self.inverse(self.matrix_1))
            elif number == '0':
                exit()
            else:
                continue

    def add(self, matrix1, matrix2):
        """
        Method that adds together two matrices, checks to make sure the two matrices are equal sizes
        :param matrix1: the first matrix
        :param matrix2: the second matrix
        :return: a new matrix with the elements of the two matrices added together OR an Error message
        """
        rows_m1 = len(matrix1)
        cols_m1 = len(matrix1[0])
        rows_m2 = len(matrix2)
        cols_m2 = len(matrix2[0])

        # Checks that there are the same amount or rows and columns before adding otherwise it
        # prints an error message
        if rows_m1 == rows_m2 and cols_m1 == cols_m2:
            self.matrix_answer = make_matrix(rows_m1, cols_m2, 'zero')
            for i in range(rows_m1):
                for j in range(cols_m1):
                    self.matrix_answer[i][j] = matrix1[i][j] + matrix2[i][j]
            return self.matrix_answer
        else:
            return 'ERROR: The number of rows and/or columns of the matrices are not the same.'

    def multiply_scalar(self, matrix, scalar):
        """
        Method that multiplies all the elements of a matrix by a scalar
        :param matrix: The matrix to be multiplied
        :param scalar: The constant to multiply the matrix's elements
        :return: a new matrix containing the product to the input matrix and scalar
        """
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                matrix[i][j] *= scalar
        return matrix

    def multiply_matrices(self, matrix1, matrix2):
        """
        Method that multiplies two matrices together.  Checks to see if the rows in Matrix2 equal the columns
        in Matrix1.  Note: matrix1 * matrix2 != matrix2 * matrix1
        :param matrix1: the first matrix to multiply
        :param matrix2: the second matrix to multiply
        :return: a new matrix that is the product of Matrix1 and Matrix2 OR an Error message.
        """
        rows_mtx1 = len(matrix1)
        cols_mtx1 = len(matrix1[0])
        rows_mtx2 = len(matrix2)
        cols_mtx2 = len(matrix2[0])
        matrix_answer = make_matrix(rows_mtx1, cols_mtx2, 'zero')

        # Check to make sure we can multiply the two matrices
        if cols_mtx1 == rows_mtx2:

            # we can multiply the two matrices, so we transpose the 2nd matix and the both columns
            # are the same amount
            matrix2 = self.transpose(matrix2)
            rows_m2 = len(matrix2)
            for i in range(rows_mtx1):  # keeps track of the rows in the first matrix
                for j in range(rows_m2):  # keeps track of the rows in the second matrix
                    total = 0

                    # Multiplies all the rows of the second matrix to the current row of the first
                    # matrix, adds them together and stores them in a new matrix
                    for _ct in range(cols_mtx1):
                        total += matrix1[i][_ct] * matrix2[j][_ct]

                    # Fills all the columns of the current row with the total before moving
                    # on to the next row
                    matrix_answer[i][j] = total
            return matrix_answer

        # if we can't multiply the two matrices, print an error message
        else:
            return 'ERROR: Can not multiply because rows of matrix_A and columns of matrix_B are not equal.'

    def transpose_menu(self):
        """
        Method that outputs a menu to transpose a matrix using certain criteria.  It prints the return value
        of the chosen transpose option
        :return: None
        """
        print('\n1. Main diagonal\n2. Side diagonal\n3. Vertical line\n4. Horizontal line')
        choice = input('Enter choice: ')
        self.matrix_1 = need_one()
        if choice == '1':
            self.print(self.transpose(self.matrix_1))
        elif choice == '2':
            self.print(self.transpose_side(self.matrix_1))
        elif choice == '3':
            self.print(self.transpose_vert(self.matrix_1))
        elif choice == '4':
            self.print(self.transpose_horz(self.matrix_1))

    def transpose(self, matrix):
        """
        Method to transpose a matrix using its main diagonal (top left to bottom right)
        :param matrix: the matrix to transpose
        :return: the transposed matrix
        """
        rows_m1 = len(matrix)
        cols_m1 = len(matrix[0])

        # Use the number of columns for new amount of rows and number of rows for new amount of
        # columns
        self.transpose_matrix = make_matrix(cols_m1, rows_m1, 'zero')
        for i in range(rows_m1):
            for j in range(cols_m1):
                self.transpose_matrix[j][i] = matrix[i][j]
        return self.transpose_matrix

    def transpose_side(self, matrix):
        """
        Method to transpose a matrix using its side diagonal (top right to bottom left)
        :param matrix: the matrix to transpose
        :return: the transposed matrix
        """
        rows_matrix = len(matrix)
        cols_matrix = len(matrix[0])
        self.trans_side_matrix = make_matrix(cols_matrix, rows_matrix, 'zero')
        for i in range(-1, (-1 - rows_matrix), -1):
            for j in range(-1, (-1 - cols_matrix), -1):
                self.trans_side_matrix[-1 - j][-1 - i] = matrix[i][j]
        return self.trans_side_matrix

    def transpose_vert(self, matrix):
        """
        Method to transpose a matrix vertically using a midline, checks if the matrix has
        more than one column
        :param matrix: the matrix to transpose vertically
        :return: the transposed matrix OR an error message
        """
        rows_m1 = len(matrix)
        cols_m1 = len(matrix[0])
        if cols_m1 > 1:
            self.trans_vert_matrix = make_matrix(rows_m1, cols_m1, 'zero')
            for i in range(rows_m1):
                for j in range(-1, (-1 - cols_m1), -1):
                    self.trans_vert_matrix[i][-1 - j] = matrix[i][j]
            return self.trans_vert_matrix
        return 'Can not transpose the matrix vertically'

    def transpose_horz(self, matrix):
        """
        Method to transpose a matrix horizontally using a midline, checks if the matrix has
        more than one row
        :param matrix: the matrix to transpose horizontally
        :return: the transposed matrix OR an error message
        """
        rows_m1 = len(matrix)
        cols_m1 = len(matrix[0])
        if rows_m1 > 1:
            self.trans_horz_matrix = make_matrix(rows_m1, cols_m1, 'zero')
            for i in range(-1, (-1 - rows_m1), -1):
                for j in range(cols_m1):
                    self.trans_horz_matrix[-1 - i][j] = matrix[i][j]
            return self.trans_horz_matrix
        return 'Can not transpose the matrix horizontally'

    def determinant(self, matrix):
        """
        Method to calculate the determinant of a matrix
        :param matrix: the matrix in which you want to find the determinant
        :return: the determinant of the matrix
        """
        if test_square_matrix(matrix):
            if len(matrix) == 1:  # base case if matrix has only 1 number
                return matrix
            elif test_square_matrix(matrix):
                if len(matrix) == 2:  # base case if matrix is 2x2
                    return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
                cols_m = len(matrix[0])
                new_m = matrix[0]
                minor_matrix = [self.find_minor(matrix, i) for i in range(cols_m)]
                total = 0
                for i in range(len(new_m)):
                    total += new_m[i] * ((-1) ** (i + 2)) * (self.determinant(minor_matrix[i]))
                return total
        else:
            return 'The matrix is not square'

    def find_minor(self, matrix, col, row=0):
        """
        Method to remove columns from a matrix
        :param matrix: the matrix to perform the action
        :param col: the column to omit from the new matrix
        :param row: the row to omit from the new matrix, default is row at index 0
        :return: the minor of the given matrix
        """
        rows_mtx = len(matrix) - 1
        self.matrix_minor = []
        while len(self.matrix_minor) < rows_mtx:
            self.matrix_minor.append([])
        num = 0
        while num < rows_mtx:
            for i in range(len(matrix)):
                if i == row:
                    continue
                for j in range(len(matrix[0])):
                    if j == col:
                        continue
                    self.matrix_minor[num].append(matrix[i][j])
                num += 1
        return self.matrix_minor

    def inverse_2x2_matrix(self, matrix, det):
        """
        Method to determine the inverse of a 2x2 matrix
        :param matrix: matrix to find the inverse of
        :param det: the determinant of the matrix
        :return: the inverse of the input matrix
        """
        minor_matrix = []
        while len(minor_matrix) < 2:
            minor_matrix.append([])
        minor_matrix[0].append(matrix[1][1])
        minor_matrix[0].append(-(matrix[1][0]))
        minor_matrix[1].append(-(matrix[0][1]))
        minor_matrix[1].append(matrix[0][0])
        transposed_minor = self.transpose(minor_matrix)
        matrix_answer = self.multiply_scalar(transposed_minor, (1 / det))
        for i in range(len(matrix_answer)):
            for j in range(len(matrix_answer[i])):
                matrix_answer[i][j] = truncate(matrix_answer[i][j], 2)
        self.matrix_answer = convert_to_int(matrix_answer, 'inverse')
        test = self.inverse_test(matrix, self.matrix_answer)
        if test:
            return self.matrix_answer

    def inverse(self, matrix):
        """
        Method to find the inverse matrix of a given matrix
        :param matrix: the matrix we need to find the inverse of
        :return: the inverse matrix
        """
        det_m1 = self.determinant(matrix)
        if det_m1 == 0:
            return "This matrix doesn't have an inverse."
        rows_m1 = len(matrix)
        cols_m1 = len(matrix[0])
        if rows_m1 == 2:  # find the inverse of a 2x2 matrix
            return self.inverse_2x2_matrix(matrix, det_m1)
        else:
            # find the inverse of a matrix bigger than 2x2
            inv_mtx = []

            # find all the minors
            det_mtx = [self.find_minor(matrix, j, i) for i in range(rows_m1) for j in range(cols_m1)]
            for i in range(len(det_mtx)):
                inv_mtx.append(self.determinant(det_mtx[i]))  # find the determinant of each minor
            rows_det = len(inv_mtx)
            num = rows_det // rows_m1

            # create a matrix of minors that is the same size of the input matrix
            temp_matrix = [inv_mtx[i * num:(i + 1) * num] for i in range((rows_det + num - 1) // num)]

            # multiply matrix of minors elements by its co-factor
            for i in range(len(temp_matrix)):
                for j in range(len(temp_matrix[i])):
                    temp_matrix[i][j] *= (-1) ** (i + j + 2)
            # transpose the matrix of minors
            matrix_answer = self.transpose(temp_matrix)

            # multiply the transposed matrix of minors by determinant reciprocal
            matrix_answer = self.multiply_scalar(matrix_answer, (1 / det_m1))
            self.matrix_answer = convert_to_int(matrix_answer, 'inverse')
            test = self.inverse_test(matrix, self.matrix_answer)
            if test:
                return self.matrix_answer

    def inverse_test(self, matrix1, matrix2):
        """
        Method test to see if M x M^-1 and M^-1 x M are both identity matrices
        :param matrix1: user inputted matrix
        :param matrix2: inverse of the inputted matrix
        :return: True or False
        """
        is_m1_m2 = self.multiply_matrices(matrix1, matrix2)
        for i, row in enumerate(is_m1_m2):
            for j, num in enumerate(row):
                is_m1_m2[i][j] = round(num)
        is_m2_m1 = self.multiply_matrices(matrix2, matrix1)
        for i, row in enumerate(is_m2_m1):
            for j, num in enumerate(row):
                is_m2_m1[i][j] = round(num)
        if is_m1_m2 == is_m2_m1:
            return True
        return False


"""
########################################################
##########  Utility Functions for the program ##########
########################################################
"""


def need_one():
    """
    Function that creates a matrix from user input
    :return: the function make_matrix with row and column inputs
    """
    row, column = (int(num) for num in input('Enter matrix size: ').split())
    return make_matrix(row, column)


def need_two():
    """
    Function that creates two matrices both from user input
    :return: matrix1 and matrix2
    """
    row, column = (int(num) for num in input('Enter size of first matrix: ').split())
    matrix1 = make_matrix(row, column, 'first')
    row, column = (int(num) for num in input('Enter size of second matrix: ').split())
    matrix2 = make_matrix(row, column, 'second')
    return matrix1, matrix2


def make_matrix(rows, columns, number=None):
    """
    Function that returns a propagated matrix that is type float. If number='zero', the function returns a zero
    matrix. If number='string', it uses that string for UX, i.e. number='first' -> "Enter first matrix",
    number='second' -> "Enter second matrix", etc.
    If number=None, the function just asks to "Enter matrix"
    :param rows: the number of rows to create in the matrix
    :param columns: the number of columns to create in a matrix
    :param number: None (default), 'zero' or 'first', 'second', etc.
    :return: the propagated matrix
    """
    matrix = []
    if number == 'zero':  # returns a zero matrix
        while len(matrix) < rows:
            matrix.append([])
            while len(matrix[-1]) < columns:
                matrix[-1].append(0.0)
        return matrix
    while len(matrix) < rows:
        matrix.append([])
    if number:
        print(f'Enter {number} matrix:\r')

        # propagates the matrix with the user's input converted to type float
        for i in range(rows):
            matrix[i] = [float(num) for num in input().split()]
        return matrix

    # User input screen if there is only 1 matrix to be input
    print('Enter matrix:\r')
    for i in range(rows):
        matrix[i] = [float(num) for num in input().split()]
    return matrix


def test_square_matrix(matrix):
    """
    Function that checks if a matrix is square before performing a calculation
    :param matrix: the matrix to test
    :return: True or False
    """
    rows_m = len(matrix)
    for i in range(rows_m):  # test to see if the matrix is square
        if len(matrix[i]) == rows_m:
            continue
        else:
            print("This is not a square matrix")
            return False
    return True


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


def convert_to_int(matrix, func=None):
    """
    Function converts floats to ints if possible. If 'inverse' option is called then the function
    truncates a float number to two decimal places (no rounding) first and then converts floats to ints if possible. 
    :param matrix: the matrix whose elements to convert
    :param func: None (default) or 'inverse'
    :return: returns the updated matrix
    """""
    for i, row in enumerate(matrix):
        for j, num in enumerate(row):
            if func == 'inverse':
                # if element is a float, store to 2 decimal places
                if type(num) == float:
                    matrix[i][j] = truncate(num, 2)

            # if element is an float and integer, make it an integer
            if num.is_integer():
                matrix[i][j] = int(num)
    return matrix


def print_new_matrix(matrix):
    """
    Function that properly outputs a given matrix
    :param matrix: the matrix to ouput
    :return: None
    """
    print('The result is:')
    if type(matrix) == list:
        for line in matrix:
            print(*line)
    else:
        print(matrix)
    print('\r')


if __name__ == '__main__':
    matrix = Matrix()
    matrix.menu_choice()