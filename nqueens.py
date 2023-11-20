N = int(input("Enter the number of queens : "))
board = [[0]*N for _ in range(N)]
def is_attack(i, j):
    for k in range(0, N):
        if board[i][k] == 1 or board[k][j] == 1:
            return True
    for k in range(0, N):
        for l in range(0, N):
            if (k+l == i+j) or (k-l == i-j):
                if board[k][l] == 1:
                    return True
    return False

def N_queen(n):
    if n == 0:
        return True
    for i in range(0, N):
        for j in range(0, N):
            if (not(is_attack(i, j))) and (board[i][j] != 1):
                board[i][j] = 1
                if N_queen(n-1) == True:
                    return True
                board[i][j] = 0
    return False

N_queen(N)
for i in board:
    print(i)
=========================================================================
def is_safe(board, row, col, n):
    # Check if there is a queen in the same column
    for i in range(row):
        if board[i][col] == 1:
            return False

    # Check upper-left diagonal
    for i, j in zip(range(row, -1, -1), range(col, -1, -1)):
        if board[i][j] == 1:
            return False

    # Check upper-right diagonal
    for i, j in zip(range(row, -1, -1), range(col, n)):
        if board[i][j] == 1:
            return False

    return True

def solve_n_queens_util(board, row, n, solutions):
    if row == n:
        # Found a solution
        solutions.append(["".join(["Q" if col == 1 else "." for col in row]) for row in board])
        return

    for col in range(n):
        if is_safe(board, row, col, n):
            board[row][col] = 1
            solve_n_queens_util(board, row + 1, n, solutions)
            board[row][col] = 0  # Backtrack

def solve_n_queens(n):
    board = [[0] * n for _ in range(n)]
    solutions = []
    solve_n_queens_util(board, 0, n, solutions)
    return solutions

# Example usage for N = 4
n = 4
solutions = solve_n_queens(n)
for solution in solutions:
    for row in solution:
        print(row)
    print()
