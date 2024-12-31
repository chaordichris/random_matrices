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

# generate a random matrix  
def generate_random_matrix(size, dist):
    if dist == "Gaussian":
        return np.random.normal(0, 1, (size, size))
    elif dist == "Uniform":
        return np.random.uniform(-1, 1, (size, size))

matrix = generate_random_matrix(matrix_size, matrix_type)
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




