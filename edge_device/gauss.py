
class GaussianObj():
    
    def __init__(self, field):
        self.field = field
        self.results = []

    def find_constant_for_row_addition(self, row1, row2, index):
        additive_inverse = row2[index]
        constant = row1[index]
        if additive_inverse == 0:
            return -1
        for i in range(0, self.field.max_num):
            if self.field.mult(i, constant) ^ additive_inverse == 0:
                return i
    
    def triangular_form(self, matrix):

        for row in range(0,len(matrix)-1):

            for row_inner in range(row,len(matrix)):
                if row < row_inner:
                    #print(row)
                    tmp =  matrix[row]
                    tmp2 = matrix[row_inner]
                    multiplier = self.find_constant_for_row_addition(tmp,tmp2,row)
                    if multiplier > 0:
                        for i in range(row,len(tmp)):
                            tmp[i] = self.field.mult(multiplier, tmp[i])
                            tmp2[i] ^= tmp[i]
        return matrix
    
    def substitute_values(self, matrix):
        values = []
        for i in range(0, len(matrix)):
            values.append(-1)

        for row in range(len(matrix)-1, -1, -1):
            if row == len(matrix)-1:
                S_i = matrix[row][-1]
                col = len(matrix[row])-2
                C = matrix[row][col]
                values[-1] = self.field.div(S_i,C)
            else:
                holder = 0
                S_i = matrix[row][-1]
                for i in range(len(matrix[row])-2,-1,-1):
                    if values[i] != -1:
                        S_i ^= self.field.mult(values[i], matrix[row][i])
                    else:
                        values[i] = self.field.div(S_i, matrix[row][i])
                        break
        return values
        
    def solve_system(self, matrix):
        return self.substitute_values(self.triangular_form(matrix))
        

    def print_matrix(self,matrix):
        for i in matrix:
            if type(i) == type([]):
                for j in i:
                    print(f"{j}\t",end='')
                print()
            else:
                print(i)