import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd
import yfinance as yf

st.title("Introduction to Random Matrix Theory")
st.sidebar.title("Options")

# Sidebar options
matrix_size = st.sidebar.slider("Matrix size (NxN)", 10, 500, 100, step=10)
matrix_type = st.sidebar.selectbox("Matrix type", ["Gaussian", "Uniform"])
noise_level = st.sidebar.slider("Noise Level", 0.0, 2.0, 1.0, step=0.1)
symmetric = st.sidebar.checkbox("Enforce Symmetry", value=True)

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Overview & Basics",
    "Wigner Semicircle Law",
    "Marchenko-Pastur Law",
    "Applications",
    "Advanced Topics", 
    "Example from Finance"
])

# Tab 1: Overview & Basics
with tab1:
    st.header("What is Random Matrix Theory?")
    st.markdown("""
    Random Matrix Theory (RMT) studies the properties of matrices whose entries are random variables.
    It has applications in physics, finance, statistics, and more. Key concepts include:
    - The eigenvalue distribution of large random matrices.
    - The Wigner Semicircle Law.
    - Universality and applications in statistical mechanics.
    """)

    # Generate random matrix
    def generate_random_matrix(size, dist, noise):
        if dist == "Gaussian":
            return np.random.normal(0, noise, (size, size))
        elif dist == "Uniform":
            return np.random.uniform(-1, 1, (size, size))

    matrix = generate_random_matrix(matrix_size, matrix_type, noise_level)
    if symmetric:
        matrix = (matrix + matrix.T) / 2

    st.subheader("Random Matrix")
    st.markdown(f"""Generate a random matrix by selected a size and type: /
                * Guassian with mean 0 and standard deviation defined by the Noise Level parameter)
                * Uniform with range -1, 1""")
    st.write(matrix)

    # Eigenvalue distribution
    eigenvalues = np.linalg.eigvals(matrix)
    fig, ax = plt.subplots()
    ax.hist(eigenvalues.real, bins=50, density=True, alpha=0.7, color='blue', label="Eigenvalue Distribution")
    ax.set_title("Eigenvalue Distribution")
    ax.set_xlabel("Real Part of Eigenvalues")
    ax.set_ylabel("Density")
    ax.legend()

    st.subheader("Eigenvalue Distribution")
    st.pyplot(fig)

# Tab 2: Wigner Semicircle Law
with tab2:
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
    st.markdown("""
    The **Wigner Semicircle Law** describes the eigenvalue density of random symmetric matrices.
    The red curve shows the theoretical distribution, while the histogram shows scaled eigenvalues.
    """)

# Tab 3: Marchenko-Pastur Law
with tab3:
    st.subheader("Marchenko-Pastur Law")

    # User inputs for covariance matrix dimensions
    p = st.sidebar.slider("Rows (p) in Matrix X", 10, 500, 100, step=10)
    n = st.sidebar.slider("Columns (n) in Matrix X", 10, 500, 200, step=10)
    q = p / n
    lambda_plus = (1 + np.sqrt(q))**2
    lambda_minus = (1 - np.sqrt(q))**2

    # Generate covariance matrix
    X = np.random.normal(0, 1, (p, n))
    M = np.dot(X.T, X) / n
    mp_eigenvalues = np.linalg.eigvalsh(M)

    # Marchenko-Pastur theoretical distribution
    x = np.linspace(lambda_minus, lambda_plus, 500)
    mp_pdf = (1 / (2 * np.pi * q * x)) * np.sqrt((lambda_plus - x) * (x - lambda_minus))
    mp_pdf[(x < lambda_minus) | (x > lambda_plus)] = 0

    fig3, ax3 = plt.subplots()
    ax3.hist(mp_eigenvalues, bins=50, density=True, alpha=0.7, label="Eigenvalue Histogram")
    ax3.plot(x, mp_pdf, color="red", lw=2, label="Marchenko-Pastur PDF")
    ax3.set_title("Marchenko-Pastur Law")
    ax3.set_xlabel("Eigenvalue")
    ax3.set_ylabel("Density")
    ax3.legend()

    st.pyplot(fig3)
    st.markdown("""
    The **Marchenko-Pastur Law** describes the eigenvalue distribution of covariance matrices.
    The histogram represents empirical eigenvalues, while the red curve shows the theoretical prediction.
    """)

# Tab 4: Applications
with tab4:
    st.header("Applications of Random Matrix Theory")
    st.markdown("""
    ### Physics
    - Describes energy levels in quantum systems (e.g., atomic nuclei).
    ### Finance
    - Models correlations in financial time series and portfolio optimization.
    ### Machine Learning
    - Analyzes the behavior of large neural networks (e.g., spectrum of weight matrices).
    """)

# Tab 5: Advanced Topics
with tab5:
    st.subheader("Covariance Matrix Spectrum")
    cov_matrix = np.cov(X.T)
    cov_eigenvalues = np.linalg.eigvalsh(cov_matrix)

    fig4, ax4 = plt.subplots()
    ax4.hist(cov_eigenvalues, bins=50, density=True, alpha=0.7, label="Eigenvalue Histogram")
    ax4.set_title("Covariance Matrix Eigenvalue Spectrum")
    ax4.set_xlabel("Eigenvalue")
    ax4.set_ylabel("Density")
    st.pyplot(fig4)

    st.subheader("Eigenvector Component Distribution")
    eigenvectors = np.linalg.eig(matrix)[1]
    fig5, ax5 = plt.subplots()
    ax5.hist(eigenvectors.flatten(), bins=50, density=True, alpha=0.7, label="Eigenvector Components")
    ax5.set_title("Distribution of Eigenvector Components")
    ax5.set_xlabel("Component Value")
    ax5.set_ylabel("Density")
    st.pyplot(fig5)
    st.markdown("""
    This plot shows the distribution of eigenvector components, which often follow a Gaussian distribution for large random matrices.
    """)

with tab6:
    st.header("Finance Example: Correlation Matrix of Real World Stock Returns")
    st.markdown("""
    In finance, Random Matrix Theory is often applied to analyze correlation matrices of stock returns. 
    By studying the eigenvalue spectrum, we can identify whether correlations arise from noise or meaningful patterns.
    """)
    
    # User inputs
    st.sidebar.subheader("Finance Example Options")
    stock_list = st.sidebar.text_input("Enter stock symbols separated by commas (e.g., AAPL, MSFT, GOOGL)", value="AAPL, MSFT, GOOGL, AMZN, FB, TSLA, JPM, JNJ, V, PG")
    num_days = st.sidebar.slider("Number of Days of Historical Data", 30, 1000, 252, step=10)
    
    # Parse stock symbols
    symbols = [symbol.strip().upper() for symbol in stock_list.split(',')]
    
    # Fetch historical stock data
    try:
        data = yf.download(symbols, period=f"{num_days}d")['Adj Close']
        returns = data.pct_change().dropna()
        
        # Compute correlation matrix
        correlation_matrix = returns.corr().values
        
        st.subheader("Correlation Matrix")
        st.write(pd.DataFrame(correlation_matrix, index=symbols, columns=symbols))
        
        # Compute eigenvalues
        eigenvalues = np.linalg.eigvalsh(correlation_matrix)
        
        # Visualize eigenvalue spectrum
        fig, ax = plt.subplots()
        ax.hist(eigenvalues, bins=50, density=True, alpha=0.7, color="blue", label="Eigenvalue Histogram")
        ax.set_title("Eigenvalue Spectrum of Correlation Matrix")
        ax.set_xlabel("Eigenvalue")
        ax.set_ylabel("Density")
        ax.legend()
        st.pyplot(fig)
        
        st.markdown("""
        The histogram shows the eigenvalues of the correlation matrix constructed from real stock returns. 
        Eigenvalues significantly different from the bulk may indicate market factors or correlated movements.
        """)
        
        # Display largest eigenvalues and associated stocks
        num_largest = st.sidebar.slider("Number of Largest Eigenvalues to Display", 1, len(symbols), 3)
        idx_largest = np.argsort(eigenvalues)[-num_largest:]
        st.subheader("Largest Eigenvalues")
        for idx in reversed(idx_largest):
            st.write(f"Eigenvalue: {eigenvalues[idx]:.4f}")
        
        st.markdown("""
        Large eigenvalues may correspond to common factors affecting multiple stocks, such as overall market movements.
        """)
        
    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.write("Please check the stock symbols and try again.")