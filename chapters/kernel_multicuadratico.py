import numpy as np
from sklearn.svm import SVC

# 1. Datos
X = np.array([
    [0.2, 0.1, 0.9],
    [0.0, -0.6, 0.6],
    [-0.7, 0.3, 0.6],
    [0.4, 0.5, -0.5],
    [2.0, 0.2, 0.8],
    [-1.8, -1.0, -0.8],
    [0.5, 2.1, -1.2],
    [-2.2, 0.4, 0.7]
])

y = np.array([1, 1, 1, 1, -1, -1, -1, -1])

# 2. Kernel multicuadrático inverso
def multiquad_inv_kernel(X, Y, c=1.0):
    K = np.zeros((X.shape[0], Y.shape[0]))
    for i in range(X.shape[0]):
        for j in range(Y.shape[0]):
            K[i, j] = 1 / np.sqrt(np.sum((X[i] - Y[j])**2) + c**2)
    return K

# 3. Función para SVC
def kernel_func(X, Y):
    return multiquad_inv_kernel(X, Y, c=1.0)

# 4. Entrenar SVM
svc = SVC(kernel=kernel_func, C=1e5)  # C grande → margen duro
svc.fit(X, y)

# 5. Coeficientes duales (alpha_i)
alphas = np.zeros(X.shape[0])
alphas[svc.support_] = svc.dual_coef_[0]

# 6. Vectores soporte y sesgo b
support_vectors = svc.support_vectors_
b = svc.intercept_[0]

print("Coeficientes duales (alpha_i):", alphas)
print("Índices de vectores soporte:", svc.support_)
print("Vectores soporte:\n", support_vectors)
print("Sesgo b:", b)

# 7. Clasificación de un nuevo punto
x_test = np.array([[0.1, 0.0, 0.7]])
pred = svc.predict(x_test)
print("\nPredicción para x_test:", pred)

# 8. Clasificación de todos los puntos originales
pred_all = svc.predict(X)
print("\nClasificación de puntos originales:", pred_all)
