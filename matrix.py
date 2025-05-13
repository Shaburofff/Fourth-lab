import numpy as np
import matplotlib.pyplot as plt

def load_matrix():
    return np.loadtxt("matrix_data.txt", dtype=int)

def split_blocks(A):
    h = A.shape[0] // 2
    return A[:h, :h], A[:h, h:], A[h:, :h], A[h:, h:]

def calculate_perimeter_product(matrix):
    top = np.prod(matrix[0, :])
    bottom = np.prod(matrix[-1, :])
    left = np.prod(matrix[:, 0])
    right = np.prod(matrix[:, -1])
    return top * bottom * left * right

def count_zeros_even_indices(matrix):
    count = 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if (i + j) % 2 == 0 and matrix[i, j] == 0:
                count += 1
    return count

def build_F(A):
    F = A.copy()
    E, B, D, C = split_blocks(A)
    zeros_in_C = count_zeros_even_indices(C)
    perimeter_product = calculate_perimeter_product(A)
    print(f"Количество нулей в C на чётных индексах: {zeros_in_C}")
    print(f"Произведение чисел по периметру матрицы A: {perimeter_product}")
    if zeros_in_C > perimeter_product:
        print("Условие выполнено: нулей больше, чем произведение периметра. Меняем B и D симметрично.")
        F[:B.shape[0], B.shape[1]:], F[B.shape[0]:, :B.shape[1]] = D.T, B.T
    else:
        print("Условие не выполнено. Меняем C и E несимметрично.")
        F[:B.shape[0], :B.shape[1]], F[B.shape[0]:, B.shape[1]:] = C, E
    return F

def compute_result(A, F, K):
    det_A = np.linalg.det(A)
    trace_F = np.trace(F)
    G = np.tril(A)
    if det_A > trace_F:
        print("Определитель A больше следа F. Используем формулу с обратной матрицей.")
        return np.linalg.inv(A) @ A.T - K * np.linalg.inv(F)
    else:
        print("Определитель A меньше или равен следу F. Используем формулу с нижней треугольной матрицей.")
        return (A + G - F) * K

def plot_graphs(F):
    plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(F, cmap='coolwarm')
    plt.title("Тепловая карта F")
    plt.colorbar()
    plt.subplot(1, 3, 2)
    plt.plot(np.mean(F, axis=1), marker='o')
    plt.title("Среднее значение по строкам F")
    plt.grid(True)
    plt.subplot(1, 3, 3)
    x, y = np.meshgrid(range(F.shape[0]), range(F.shape[1]))
    plt.scatter(x.flatten(), y.flatten(), c=F.flatten(), cmap='viridis')
    plt.title("Диаграмма рассеяния значений F")
    plt.colorbar()
    plt.tight_layout()
    plt.show()

K = int(input("Введите значение K: "))
A = load_matrix()
print("\nМатрица A:\n", A)
F = build_F(A)
print("\nМатрица F:\n", F)
result = compute_result(A, F, K)
print("\nРезультат вычислений:\n", result)
plot_graphs(F)
