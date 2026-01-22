import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import matplotlib.pyplot as plt
import networkx as nx
import time
from sklearn.metrics import adjusted_rand_score, accuracy_score

# ==============================================================================
# SECTION 1: PARAMETER SETUP (ID: 06 -> X=6, Y=6)
# ==============================================================================
print("="*60)
print("PHASE 1: INITIALIZING PARAMETERS & CONSTRAINTS")
print("="*60)

# 1. ID Parsing
X, Y = 6, 6
N = 10**6  # 1 Million Nodes

# 2. Constraints Calculation
# Average Degree constraint: XY (concatenation) = 66
k_avg = 66 

# Cluster Sizes (Ratio X/Y = 1.0)
n1 = int(N * (X / (X + Y)))
n2 = N - n1

# Probability Constraints
# Equation 1: p2 / p1 = 1 / (10 + XY) => p1 = p2 * (10 + XY)
# Equation 2: Total Edges M = N * k_avg / 2
ratio_const = 10 + X * Y  # 46
coeff_p2 = (0.5 * n1**2 * ratio_const) + (0.5 * n2**2 * ratio_const) + (n1 * n2)
target_edges = (N * k_avg) / 2

p2 = target_edges / coeff_p2
p1 = p2 * ratio_const

print(f"[-] Configuration:")
print(f"    Nodes (N): {N:,}")
print(f"    Target Avg Degree: {k_avg}")
print(f"    Cluster Split: N1={n1:,} | N2={n2:,}")
print(f"[-] Calculated Probabilities:")
print(f"    p1 (Intra-cluster): {p1:.8f}")
print(f"    p2 (Inter-cluster): {p2:.8f} (Ratio 1:{ratio_const})")

# Ground Truth Labels (for final comparison)
true_labels = np.concatenate([np.zeros(n1, dtype=int), np.ones(n2, dtype=int)])

# ==============================================================================
# SECTION 2: EFFICIENT GRAPH GENERATION (SPARSE)
# ==============================================================================
print("\n" + "="*60)
print("PHASE 2: GENERATING 1,000,000 NODE GRAPH (Sparse Stream)")
print("="*60)

def generate_sparse_block(rows, cols, p, r_offset, c_offset, seed=None):
    """Generates edges for a block without creating dense arrays."""
    if seed: np.random.seed(seed)
    n_edges = int(rows * cols * p)
    
    # Generate edge coordinates directly
    r_idx = np.random.randint(0, rows, size=n_edges, dtype=np.int32)
    c_idx = np.random.randint(0, cols, size=n_edges, dtype=np.int32)
    
    return r_idx + r_offset, c_idx + c_offset

t0 = time.time()

# Generate Blocks
r1, c1 = generate_sparse_block(n1, n1, p1, 0, 0, seed=42)
r2, c2 = generate_sparse_block(n2, n2, p1, n1, n1, seed=43)
r3, c3 = generate_sparse_block(n1, n2, p2, 0, n1, seed=44)

# Combine all edges
rows = np.concatenate([r1, r2, r3, c3])
cols = np.concatenate([c1, c2, c3, r3])

# Create CSR Matrix
data = np.ones(len(rows), dtype=np.float32)
A = sp.csr_matrix((data, (rows, cols)), shape=(N, N))

# Clean up
A.setdiag(0)
A.sum_duplicates()

print(f"[-] Graph Generated in {time.time()-t0:.2f}s")
print(f"[-] Actual Edges: {A.nnz // 2:,}")

# ==============================================================================
# SECTION 3: COARSENING via MAXIMAL INDEPENDENT SETS (FIXED)
# ==============================================================================
print("\n" + "="*60)
print("PHASE 3: COARSENING (Maximal Independent Set)")
print("="*60)

def luby_mis_coarsening(Adjacency, max_iter=10):
    n = Adjacency.shape[0]
    mis_mask = np.zeros(n, dtype=bool)
    active_mask = np.ones(n, dtype=bool)
    
    for i in range(max_iter):
        if not np.any(active_mask): break
        active_indices = np.where(active_mask)[0]
        
        perm = np.random.permutation(active_indices)
        candidates = []
        covered = set()
        
        for node in perm:
            if node not in covered:
                candidates.append(node)
                covered.add(node)
                nbs = Adjacency[node].indices
                covered.update(nbs)
        
        mis_mask[candidates] = True
        break 

    return mis_mask

t_c = time.time()
mis_mask = luby_mis_coarsening(A)
mis_indices = np.where(mis_mask)[0]
n_coarse = len(mis_indices)

print(f"[-] Coarsening Complete in {time.time()-t_c:.2f}s")
print(f"[-] Original Nodes: {N}")
print(f"[-] Super-Nodes (MIS size): {n_coarse}")

# --- Build Prolongation Matrix P (FIXED LOGIC) ---
print("[-] Building Interpolation Matrix P...")

map_global_to_coarse = np.full(N, -1, dtype=np.int32)
map_global_to_coarse[mis_indices] = np.arange(n_coarse)

mis_diag = sp.diags(mis_mask.astype(int))
A_mis = A @ mis_diag 

assignments = np.full(N, -1, dtype=np.int32)
valid_rows = np.diff(A_mis.indptr) > 0
row_starts = A_mis.indptr[:-1]
first_neighbors = A_mis.indices[row_starts[valid_rows]]

assignments[valid_rows] = map_global_to_coarse[first_neighbors]
assignments[mis_mask] = map_global_to_coarse[mis_indices]

if np.any(assignments == -1):
    assignments[assignments == -1] = 0

row_idx = np.arange(N)
P = sp.csr_matrix((np.ones(N), (row_idx, assignments)), shape=(N, n_coarse))

col_sums = np.array(P.sum(axis=0)).flatten()
col_sums[col_sums==0] = 1
P_norm = P @ sp.diags(1.0/np.sqrt(col_sums))

A_c = P_norm.T @ A @ P_norm
print("[-] Coarse Matrix Constructed.")

# ==============================================================================
# SECTION 4: INITIAL PARTITION & RQI
# ==============================================================================
print("\n" + "="*60)
print("PHASE 4: PARTITIONING & RAYLEIGH QUOTIENT ITERATION")
print("="*60)

print("[-] Solving Coarse Eigenproblem...")
D_c = sp.diags(np.array(A_c.sum(axis=1)).flatten())
L_c = D_c - A_c

vals, vecs = sla.eigsh(L_c, k=2, which='SM', tol=1e-3)
fiedler_coarse = vecs[:, 1]
labels_coarse = (fiedler_coarse > 0).astype(int)

print("[-] Expanding Partition...")
x_fine = P_norm @ fiedler_coarse
x_fine = x_fine / np.linalg.norm(x_fine)

print("[-] Running Rayleigh Quotient Iteration (Refinement)...")
D_vec = np.array(A.sum(axis=1)).flatten()
mu = x_fine.T @ (D_vec * x_fine - A @ x_fine)

for i in range(5): 
    def matvec(v):
        return (D_vec - mu) * v - A @ v
    
    Op = sla.LinearOperator((N, N), matvec=matvec, dtype=float)
    
    try:
        y, info = sla.minres(Op, x_fine, tol=1e-4, maxiter=50)
    except:
        break
        
    norm_y = np.linalg.norm(y)
    if norm_y < 1e-10: break
    
    x_new = y / norm_y
    mu_new = x_new.T @ (D_vec * x_new - A @ x_new)
    
    diff = abs(mu_new - mu)
    print(f"    Iter {i+1}: d(mu) = {diff:.6e}")
    
    x_fine = x_new
    mu = mu_new
    if diff < 1e-6: break

labels_fine = (x_fine > 0).astype(int)

# ==============================================================================
# SECTION 5: EVALUATION & VISUALIZATION
# ==============================================================================
print("\n" + "="*60)
print("PHASE 5: QUANTITATIVE COMPARISON & VISUALIZATION")
print("="*60)

# Ground Truth for Coarse
weighted_votes = P.T @ true_labels 
cluster_sizes = np.array(P.sum(axis=0)).flatten()
cluster_sizes[cluster_sizes==0] = 1
ratios = weighted_votes / cluster_sizes
coarse_truth = (ratios > 0.5).astype(int)

def get_metrics(truth, pred):
    acc = accuracy_score(truth, pred)
    acc = max(acc, 1 - acc) 
    ari = adjusted_rand_score(truth, pred)
    return acc, ari

acc_c, ari_c = get_metrics(coarse_truth, labels_coarse)
acc_f, ari_f = get_metrics(true_labels, labels_fine)

print(f"{'Graph Level':<15} | {'Accuracy':<10} | {'Adj. Rand Index':<15}")
print("-" * 45)
print(f"{'Coarse (MIS)':<15} | {acc_c*100:.2f}%{'':<5} | {ari_c:.4f}")
print(f"{'Fine (Final)':<15} | {acc_f*100:.2f}%{'':<5} | {ari_f:.4f}")

# Plotting
print("\n[-] Generating Plots for Coarsen Graphs...")
center_node = n_coarse // 2
try:
    subgraph_nodes = list(nx.bfs_tree(nx.from_scipy_sparse_array(A_c), center_node, depth_limit=2))
    if len(subgraph_nodes) > 200: subgraph_nodes = subgraph_nodes[:200]
    
    sub_adj = A_c[subgraph_nodes, :][:, subgraph_nodes]
    G_viz = nx.from_scipy_sparse_array(sub_adj)
    pos = nx.spring_layout(G_viz, seed=42)
    
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    
    true_colors = ['#1f77b4' if coarse_truth[n] == 0 else '#ff7f0e' for n in subgraph_nodes]
    nx.draw(G_viz, pos, node_size=50, node_color=true_colors, edge_color='gray', alpha=0.6, ax=ax[0])
    ax[0].set_title(f"Coarsened Graph\n(Colored by Ground Truth)", fontsize=14)
    
    algo_colors = ['#1f77b4' if labels_coarse[n] == 0 else '#ff7f0e' for n in subgraph_nodes]
    nx.draw(G_viz, pos, node_size=50, node_color=algo_colors, edge_color='gray', alpha=0.6, ax=ax[1])
    ax[1].set_title(f"Coarsened Graph\n(Colored by Spectral Algorithm)", fontsize=14)
    
    plt.tight_layout()
    plt.savefig("Clustering_Result.png")
    plt.show()
except Exception as e:
    print(f"Visualization skipped: {e}")