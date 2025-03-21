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
tab1, tab2, tab3= st.tabs([
    "Overview & Basics",
    "Marchenko-Pastur Law",
    "Example from Finance", 
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
    st.markdown(f"""
                Generate a random matrix by selected a size and type:
                - Guassian with mean 0 and standard deviation defined by the Noise Level parameter)
                - Uniform with range -1, 1
                """)
    st.write(matrix)

    # Eigenvalue distribution
    eigenvalues = np.linalg.eigvals(matrix)
    st.subheader("Eigenvalue Distribution - Wigner Semicircle Law")
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
with tab2:
    st.subheader("Marchenko-Pastur Law")

    # User inputs for covariance matrix dimensions
    p = st.slider("Rows (p) in Matrix X", 10, 500, 100, step=10)
    n = st.slider("Columns (n) in Matrix X", 10, 500, 200, step=10)
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
    # contect on the page
    st.markdown("""
    The **Marchenko-Pastur Law** describes the eigenvalue distribution of covariance matrices.
    The histogram represents empirical eigenvalues, while the red curve shows the theoretical prediction.
    """)
    fig3, ax3 = plt.subplots()
    ax3.hist(mp_eigenvalues, bins=50, density=True, alpha=0.7, label="Eigenvalue Histogram")
    ax3.plot(x, mp_pdf, color="red", lw=2, label="Marchenko-Pastur PDF")
    ax3.set_title("Marchenko-Pastur Law")
    ax3.set_xlabel("Eigenvalue")
    ax3.set_ylabel("Density")
    ax3.legend()

    st.pyplot(fig3)
    # plot the eigenvalue component distribution
    st.markdown("""
    - The red curve shows the theoretical **normal distribution** \( \\mathcal{{N}}(0, 1/N) \), which eigenvector components are expected to follow in large random matrices (here N = {n}).
    """)
    st.subheader("Eigenvector Component Distribution")
    eigenvectors = np.linalg.eig(M)[1]
    components = eigenvectors.flatten()
    n = M.shape[0]

    # Create histogram
    fig5, ax5 = plt.subplots()
    ax5.hist(components, bins=50, density=True, alpha=0.7, label="Eigenvector Components")

    # Theoretical normal distribution: N(0, 1/N)
    x_vals = np.linspace(-0.2, 0.2, 500)
    pdf = norm.pdf(x_vals, loc=0, scale=1/np.sqrt(n))
    ax5.plot(x_vals, pdf, color='red', lw=2, label='Theoretical Normal (0, 1/N)')

    # Labels and legend
    ax5.set_title("Distribution of Eigenvector Components")
    ax5.set_xlabel("Component Value")
    ax5.set_ylabel("Density")
    ax5.legend()

    # Show plot
    st.pyplot(fig5)

with tab3:
    st.header("Finance Example: Correlation Matrix of Real World Stock Returns")
    st.markdown("""
    In finance, Random Matrix Theory is often applied to analyze correlation matrices of stock returns. 
    By studying the eigenvalue spectrum, we can identify whether correlations arise from noise or meaningful patterns.
    """)
    
    # User inputs
    sp500_top50 = [
    "MMM", "AOS", "ABT", "ABBV", "ACN", "ATVI", "AYI", "ADBE", "AAP", "AMD",
    "AES", "AFL", "A", "APD", "AKAM", "ALK", "ALB", "ARE", "ALGN", "ALLE",
    "LNT", "ALL", "GOOGL", "GOOG", "MO", "AMZN", "AMCR", "AEE", "AAL", "AEP",
    "AXP", "AIG", "AMT", "AWK", "AMP", "ABC", "AME", "AMGN", "APH", "ADI",
    "ANSS", "AON", "APA", "AAPL", "AMAT", "APTV", "ANET", "AJG", "AIZ", "T",
    "ATO", "ADSK", "AZO", "AVB", "AVY", "BKR", "BALL", "BAC", "BBWI", "BAX",
    "BDX", "WRB", "BRK.B", "BBY", "BIO", "TECH", "BIIB", "BLK", "BK", "BA",
    "BKNG", "BWA", "BXP", "BSX", "BMY", "AVGO", "BR", "BRO", "BF.B", "CHRW",
    "CDNS", "CZR", "CPT", "CPB", "COF", "CAH", "KMX", "CCL", "CARR", "CTLT",
    "CAT", "CBOE", "CBRE", "CDW", "CE", "CNC", "CNP", "CDAY", "CERN", "CF",
    "CRL", "SCHW", "CHTR", "CVX", "CMG", "CB", "CHD", "CI", "CINF", "CTAS",
    "CSCO", "C", "CFG", "CTXS", "CLX", "CME", "CMS", "KO", "CTSH", "CL",
    "CMCSA", "CMA", "CAG", "COP", "ED", "STZ", "CEG", "GLW", "CTVA", "COST",
    "CTRA", "CCI", "CSX", "CMI", "CVS", "DHI", "DHR", "DRI", "DVA", "DE",
    "DAL", "XRAY", "DVN", "DXCM", "FANG", "DLR", "DFS", "DISCA", "DISCK",
    "DISH", "DIS", "DG", "DLTR", "D", "DPZ", "DOV", "DOW", "DTE", "DUK",
    "DRE", "DD", "DXC", "EMN", "ETN", "EBAY", "ECL", "EIX", "EW", "EA",
    "EMR", "ENPH", "ETR", "EOG", "EPAM", "EFX", "EQIX", "EQR", "ESS", "EL",
    "ETSY", "RE", "EVRG", "ES", "EXC", "EXPE", "EXPD", "EXR", "XOM", "FFIV",
    "FDS", "FAST", "FRT", "FDX", "FITB", "FRC", "FE", "FIS", "FISV", "FLT",
    "FMC", "F", "FTNT", "FTV", "FBHS", "FOXA", "FOX", "BEN", "FCX", "GRMN",
]
    # Ticker selection from sp500_top50
    # Add a "Select All" checkbox
    select_all = st.checkbox("Select All Tickers", value=False)

    # Set default selection based on "Select All" checkbox
    if select_all:
        symbols = st.multiselect("Select Tickers", sp500_top50, default=sp500_top50)
    else:
        symbols = st.multiselect("Select Tickers", ["MMM", "AOS", "ABT", "ABBV", "ACN", "ATVI", "AYI", "ADBE", "AAP", "AMD",
    "AES", "AFL", "A", "APD", "AKAM", "ALK", "ALB", "ARE", "ALGN", "ALLE",
    "LNT", "ALL", "GOOGL", "GOOG", "MO", "AMZN", "AMCR", "AEE", "AAL", "AEP"], default=sp500_top50)
    # set default date range for returns
    start_date = st.date_input("Start Date",
                                    value=pd.to_datetime('2020-01-01'))
    end_date = st.date_input("End Date",
                                    value=pd.to_datetime('2024-12-31'))
    
    # Fetch historical stock data
    try:
        data = yf.download(symbols, start=start_date, end=end_date)['Close']
        returns = data.pct_change().dropna()
        
        # Compute correlation matrix
        correlation_matrix = returns.corr().values.round(2)
        # Compute eigenvalues
        epsilon = 1e-6  # Regularization for numerical stability
        correlation_matrix += epsilon * np.eye(correlation_matrix.shape[0])
        eigenvalues = np.linalg.eigvalsh(correlation_matrix)
        # create the 
        col1, col2 = st.columns([1,1])
        with col1:
            st.subheader("Correlation Matrix")
            # st.write(returns)
            st.write(pd.DataFrame(correlation_matrix, index=symbols, columns=symbols))
        with col2:
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
        num_largest = st.slider("Number of Largest Eigenvalues to Display", 1, len(symbols), 3)
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