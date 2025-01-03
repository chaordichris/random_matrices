import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd

st.title("Introduction to Random Matrix Theory")
st.sidebar.title("Options")

# overview of what random matrix theory is
st.header("What is Random Matrix Theory?")
st.markdown("""
Random Matrix Theory (RMT) studies the properties of matrices whose entries are random variables.
It has applications in physics, finance, statistics, and more. Key concepts include:
- The eigenvalue distribution of large random matrices.
- The Wigner Semicircle Law.
- Universality and applications in statistical mechanics.
""")
# create a slider for the matrix size
matrix_size = st.sidebar.slider("Matrix size (NxN)", 10, 500, 100, step=10)
matrix_type = st.sidebar.selectbox("Matrix type", ["Gaussian", "Uniform"])

# add a noise level slider
noise_level = st.sidebar.slider("Noise Level", 0.0, 2.0, 1.0, step=0.1)
# random_matrix = (random_matrix + random_matrix.T) / 2  # Ensure symmetry
# add a checkbox to enforce symmetry    

# generate a random matrix  
def generate_random_matrix(size, dist):
    if dist == "Gaussian":
        return np.random.normal(0, noise_level, (size, size))
    elif dist == "Uniform":
        return np.random.uniform(-1, 1, (size, size))

matrix = generate_random_matrix(matrix_size, matrix_type)
# add a toggle to enforce matrix symmetry
symmetric = st.sidebar.checkbox("Enforce Symmetry", value=True)
if symmetric:
    matrix = (matrix + matrix.T) / 2
# display the matrix
st.subheader("Random Matrix")
st.write(matrix)

# display the eigenvalue distribution
eigenvalues = np.linalg.eigvals(matrix)
fig, ax = plt.subplots()
ax.hist(eigenvalues.real, bins=50, density=True, alpha=0.7, color='blue', label="Eigenvalue Distribution")
ax.set_title("Eigenvalue Distribution")
ax.set_xlabel("Real Part of Eigenvalues")
ax.set_ylabel("Density")
ax.legend()

st.subheader("Eigenvalue Distribution")
st.pyplot(fig)

# explain the wigner semicircle law 
st.markdown("""
### Wigner Semicircle Law
The **Wigner Semicircle Law** states that the eigenvalue density \( \rho(\lambda) \) of certain random symmetric matrices 
follows a semicircular distribution:
\[
\rho(\lambda) = \frac{1}{2\pi} \sqrt{4 - \lambda^2}, \quad |\lambda| \leq 2.
\]
The histogram shows the eigenvalues of a random symmetric matrix, and the red curve is the theoretical prediction.
""")
# display the wigner semicircle law 
st.subheader("Wigner Semicircle Law")
x = np.linspace(-2, 2, 500)
wigner_pdf = (1 / (2 * np.pi)) * np.sqrt(4 - x**2)
fig2, ax2 = plt.subplots()
ax2.plot(x, wigner_pdf, label="Wigner Semicircle Law", color="red")
ax2.hist(eigenvalues.real / np.sqrt(matrix_size), bins=50, density=True, alpha=0.7, color='blue', label="Scaled Eigenvalues")
ax2.set_title("Comparison with Wigner Semicircle")
ax2.set_xlabel("Scaled Eigenvalues")
ax2.set_ylabel("Density")
ax2.legend()

st.pyplot(fig2)

# allow user to upload a matrix file    
uploaded_file = st.sidebar.file_uploader("Upload a matrix file (CSV format)", type=["csv"])
if uploaded_file:
    user_matrix = pd.read_csv(uploaded_file, header=None).values
    user_eigenvalues = np.linalg.eigvals(user_matrix)
    st.write("Uploaded Matrix:")
    st.write(user_matrix)

    fig3, ax3 = plt.subplots()
    ax3.hist(user_eigenvalues.real, bins=50, density=True, alpha=0.7, color='green', label="Eigenvalue Distribution")
    ax3.set_title("Eigenvalue Distribution of Uploaded Matrix")
    ax3.set_xlabel("Real Part of Eigenvalues")
    ax3.set_ylabel("Density")
    ax3.legend()
    st.pyplot(fig3)

# add a section for the marchenko pastur law   
st.subheader("Marchenko-Pastur Law")

# User inputs for covariance matrix dimensions
p = st.sidebar.slider("Rows (p) in Matrix X", 10, 500, 100, step=10)
n = st.sidebar.slider("Columns (n) in Matrix X", 10, 500, 200, step=10)
q = p / n  # Aspect ratio
lambda_plus = (1 + np.sqrt(q))**2
lambda_minus = (1 - np.sqrt(q))**2

# Generate random matrix and compute covariance matrix
X = np.random.normal(0, 1, (p, n))
M = np.dot(X.T, X) / n
eigenvalues = np.linalg.eigvalsh(M)

# Marchenko-Pastur theoretical distribution
x = np.linspace(lambda_minus, lambda_plus, 500)
mp_pdf = (1 / (2 * np.pi * q * x)) * np.sqrt((lambda_plus - x) * (x - lambda_minus))
mp_pdf[x < lambda_minus] = 0
mp_pdf[x > lambda_plus] = 0

# Visualization
fig, ax = plt.subplots()
ax.hist(eigenvalues, bins=50, density=True, alpha=0.7, label="Eigenvalue Histogram")
ax.plot(x, mp_pdf, color="red", lw=2, label="Marchenko-Pastur PDF")
ax.set_title("Marchenko-Pastur Law")
ax.set_xlabel("Eigenvalue")
ax.set_ylabel("Density")
ax.legend()

st.pyplot(fig)
st.markdown("""
The **Marchenko-Pastur Law** describes the eigenvalue distribution of covariance matrices for large random matrices. 
The red curve shows the theoretical distribution, and the histogram represents the empirical eigenvalues.
""")





