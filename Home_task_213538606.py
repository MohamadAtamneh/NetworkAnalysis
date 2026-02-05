import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import networkx as nx
import time


# ============================================================
# Phase 1: Parameters (SET YOUR X,Y HERE)
# ============================================================
print("Phase 1: Setup Parameters")

# >>> CHANGE THESE to your ID last digits (ignore zeros) <<<
X, Y = 6, 6

N = 10**6
K_AVG = int(f"{X}{Y}")  # XY concatenation (e.g., 6 and 6 -> 66)

# Cluster sizes: N1/N2 = X/Y
N1 = (N * X) // (X + Y)
N2 = N - N1

# Constraint: p2/p1 = 1/(10 + X*Y) where X*Y is multiplication
prob_ratio = 1.0 / (10 + X * Y)

# Target expected number of undirected edges: M = N * K_AVG / 2
target_edges = (N * K_AVG) / 2.0

# Expected edges in undirected SBM:
# E[M] = C(N1,2)*p1 + C(N2,2)*p1 + (N1*N2)*p2
c11 = N1 * (N1 - 1) / 2.0
c22 = N2 * (N2 - 1) / 2.0
c12 = N1 * N2

p1 = target_edges / (c11 + c22 + c12 * prob_ratio)
p2 = p1 * prob_ratio

# Clip for safety
p1 = float(np.clip(p1, 0.0, 1.0))
p2 = float(np.clip(p2, 0.0, 1.0))

print(f"N={N}, N1={N1}, N2={N2}, K_AVG(XY concat)={K_AVG}")
print(f"Constraint ratio p2/p1 = {prob_ratio:.6f}")
print(f"Computed p1={p1:.6e}, p2={p2:.6e}, achieved ratio={p2/p1:.6f}")

# Ground truth labels
true_labels = np.concatenate([
    np.zeros(N1, dtype=np.int8),
    np.ones(N2, dtype=np.int8)
])


# ============================================================
# Phase 2: Generate SBM graph (undirected sparse, unweighted)
# ============================================================
print("\nPhase 2: Generating SBM Graph (undirected, unweighted)")
t0 = time.time()


def sample_within_block_undirected(n, p, offset=0, rng=None):
    """
    Approximate SBM within-block edge sampling:
    - m ~ Binomial(n choose 2, p)
    - sample m random endpoint pairs (i,j), i!=j
    - map to undirected by (min,max) then symmetrize into (u,v) and (v,u)
    Duplicates may occur; csr.sum_duplicates() fixes it.
    """
    if rng is None:
        rng = np.random.default_rng()

    total_pairs = n * (n - 1) // 2
    m = rng.binomial(total_pairs, p)
    if m == 0:
        return np.empty(0, np.int32), np.empty(0, np.int32)

    i = rng.integers(0, n, size=m, dtype=np.int32)
    j = rng.integers(0, n, size=m, dtype=np.int32)

    mask = (i != j)
    i, j = i[mask], j[mask]
    if i.size == 0:
        return np.empty(0, np.int32), np.empty(0, np.int32)

    lo = np.minimum(i, j).astype(np.int32) + offset
    hi = np.maximum(i, j).astype(np.int32) + offset

    # symmetrize
    rows = np.concatenate([lo, hi]).astype(np.int32)
    cols = np.concatenate([hi, lo]).astype(np.int32)
    return rows, cols


def sample_between_blocks_undirected(n1, n2, p, off1=0, off2=0, rng=None):
    """
    Between-block undirected sampling:
    - m ~ Binomial(n1*n2, p)
    - sample endpoints (a in block1, b in block2), then symmetrize
    """
    if rng is None:
        rng = np.random.default_rng()

    total_pairs = n1 * n2
    m = rng.binomial(total_pairs, p)
    if m == 0:
        return np.empty(0, np.int32), np.empty(0, np.int32)

    a = rng.integers(0, n1, size=m, dtype=np.int32) + off1
    b = rng.integers(0, n2, size=m, dtype=np.int32) + off2

    rows = np.concatenate([a, b]).astype(np.int32)
    cols = np.concatenate([b, a]).astype(np.int32)
    return rows, cols


rng = np.random.default_rng(12345)

r1, c1_ = sample_within_block_undirected(N1, p1, offset=0, rng=rng)
r2, c2_ = sample_within_block_undirected(N2, p1, offset=N1, rng=rng)
r3, c3_ = sample_between_blocks_undirected(N1, N2, p2, off1=0, off2=N1, rng=rng)

all_rows = np.concatenate([r1, r2, r3])
all_cols = np.concatenate([c1_, c2_, c3_])

A = sp.csr_matrix(
    (np.ones(all_rows.shape[0], dtype=np.float32), (all_rows, all_cols)),
    shape=(N, N)
)

# Remove self-loops & clean duplicates
A.setdiag(0)
A.sum_duplicates()
A.eliminate_zeros()

build_time = time.time() - t0
nnz = A.nnz
m_undirected = nnz // 2  # because we stored both directions
emp_avg_deg = nnz / N

print(f"Graph built in {build_time:.2f}s")
print(f"A.nnz={nnz:,} (directed entries), undirected edges≈{m_undirected:,}")
print(f"Empirical average degree ≈ {emp_avg_deg:.3f} (target {K_AVG})")


# ============================================================
# Phase 3: Coarsening (Greedy MIS)
# ============================================================
print("\nPhase 3: Coarsening using Greedy MIS")
t1 = time.time()

degrees = np.diff(A.indptr).astype(np.int32)

# degree-based ordering (ascending). Sorting 1e6 takes time but matches the method.
nodes_sorted = np.argsort(degrees, kind="mergesort")

mis_mask = np.zeros(N, dtype=bool)
covered = np.zeros(N, dtype=bool)

for u in nodes_sorted:
    if not covered[u]:
        mis_mask[u] = True
        covered[u] = True
        neigh = A.indices[A.indptr[u]:A.indptr[u + 1]]
        covered[neigh] = True

mis_indices = np.where(mis_mask)[0]
n_coarse = mis_indices.size

print(f"MIS size = {n_coarse:,}")
print(f"MIS computed in {time.time() - t1:.2f}s")

print("Constructing assignment map...")

global_to_coarse = np.full(N, -1, dtype=np.int32)
global_to_coarse[mis_indices] = np.arange(n_coarse, dtype=np.int32)

assignments = np.full(N, -1, dtype=np.int32)
assignments[mis_mask] = global_to_coarse[mis_indices]

non_mis_nodes = np.where(~mis_mask)[0]
for u in non_mis_nodes:
    neigh = A.indices[A.indptr[u]:A.indptr[u + 1]]
    mis_neigh = neigh[mis_mask[neigh]]
    if mis_neigh.size > 0:
        assignments[u] = global_to_coarse[mis_neigh[0]]
    else:
        assignments[u] = 0

assignments = np.clip(assignments, 0, n_coarse - 1)

# Prolongation / aggregation matrix P (N x n_coarse), one-hot rows
P = sp.csr_matrix(
    (np.ones(N, dtype=np.float32), (np.arange(N, dtype=np.int32), assignments)),
    shape=(N, n_coarse)
)

# Normalize columns for Galerkin coarsening
col_counts = np.array(P.sum(axis=0)).ravel()
col_counts[col_counts == 0] = 1.0
P_norm = P @ sp.diags(1.0 / np.sqrt(col_counts))

# Coarse adjacency
A_coarse = (P_norm.T @ A @ P_norm).tocsr()
A_coarse.setdiag(0)
A_coarse.sum_duplicates()
A_coarse.eliminate_zeros()

print(f"Coarsened to {n_coarse:,} nodes, A_coarse.nnz={A_coarse.nnz:,}")


# ============================================================
# Phase 4: Spectral clustering on coarse + prolongation + RQI refinement
# ============================================================
print("\nPhase 4: Spectral clustering (coarse) + Prolongation + RQI refinement")

# Coarse Laplacian Lc = Dc - Ac
d_c = np.array(A_coarse.sum(axis=1)).ravel()
L_c = sp.diags(d_c) - A_coarse

# Smallest two eigenpairs: second is Fiedler vector
t2 = time.time()
vals, vecs = sla.eigsh(L_c, k=2, which="SM")
print(f"eigsh on coarse done in {time.time()-t2:.2f}s; eigenvalues={vals}")

fiedler_coarse = vecs[:, 1]
pred_coarse = (fiedler_coarse > 0).astype(np.int8)

# Prolongate to fine
x_fine = (P_norm @ fiedler_coarse).astype(np.float64)
x_fine /= (np.linalg.norm(x_fine) + 1e-12)

# RQI refinement on fine Laplacian L = D - A
d_fine = np.array(A.sum(axis=1)).ravel().astype(np.float64)

def apply_L(v):
    return d_fine * v - (A @ v)

mu = float(x_fine @ apply_L(x_fine))

t3 = time.time()
for it in range(3):
    def lhs(v):
        return apply_L(v) - mu * v

    Op = sla.LinearOperator((N, N), matvec=lhs, dtype=np.float64)
    y, info = sla.minres(Op, x_fine, maxiter=50)

    if info != 0 or np.linalg.norm(y) < 1e-10:
        print(f"RQI stopped early at iter {it}, info={info}")
        break

    x_fine = y / (np.linalg.norm(y) + 1e-12)
    mu = float(x_fine @ apply_L(x_fine))

print(f"RQI refinement time: {time.time()-t3:.2f}s")

pred_fine = (x_fine > 0).astype(np.int8)


# ============================================================
# Phase 5: Quantitative evaluation (primary + coarse) + conclusion prints
# ============================================================
print("\nPhase 5: Quantitative Comparison (primary + coarse)")

# Accuracy on primary (fine)
acc_fine = accuracy_score(true_labels, pred_fine)
acc_fine = max(acc_fine, 1.0 - acc_fine)

# Accuracy on coarse requires coarse ground truth
P_counts = np.array(P.sum(axis=0)).ravel()
P_counts[P_counts == 0] = 1.0

# Majority label in each aggregate
coarse_gt = (((P.T @ true_labels) / P_counts) > 0.5).astype(np.int8)

acc_coarse = accuracy_score(coarse_gt, pred_coarse)
acc_coarse = max(acc_coarse, 1.0 - acc_coarse)

print(f"Accuracy (COARSE vs coarse ground-truth): {acc_coarse:.4f}")
print(f"Accuracy (FINE after RQI vs fine ground-truth): {acc_fine:.4f}")

print("\nConclusion (short, for report):")
if acc_fine >= acc_coarse:
    print(f"- Coarsening + spectral gives {acc_coarse:.4f} on coarse.")
    print(f"- Prolongation + RQI improves/maintains on fine to {acc_fine:.4f}.")
else:
    print(f"- Coarsening + spectral gives {acc_coarse:.4f} on coarse.")
    print(f"- After prolongation + RQI, fine accuracy is {acc_fine:.4f} (not higher);")
    print("  this can happen if refinement iterations are insufficient or graph is very noisy.")


# ============================================================
# Phase 6: Visualization (coarse graphs) + optional primary sample
# ============================================================
print("\nPhase 6: Visualization")

# ---- Coarse graph visualization (required) ----
m = min(150, n_coarse)  # visualize first 150 coarse nodes
A_vis = A_coarse[:m, :m].tocsr()

G_c = nx.from_scipy_sparse_array(A_vis)
pos = nx.kamada_kawai_layout(G_c)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
nx.draw(G_c, pos, node_size=30,
        node_color=coarse_gt[:m], cmap="coolwarm")
plt.title(
    "Coarsened Graph – Ground Truth Clustering (Majority Vote)",
    fontsize=11,
    fontweight="bold",
    color="black"
)

plt.subplot(1, 2, 2)
nx.draw(G_c, pos, node_size=30,
        node_color=pred_coarse[:m], cmap="coolwarm")
plt.title(
    "Coarsened Graph – Spectral Clustering Result (Fiedler Vector)",
    fontsize=11,
    fontweight="bold",
    color="black"
)


plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.subplots_adjust(top=0.88)

plt.show()

# ---- Optional: visualize a tiny primary induced subgraph (not required by #4, but nice) ----
# Warning: primary graph is huge; we only show a tiny induced subgraph of s nodes.
s = 250
print(f"Optional: Visualizing PRIMARY induced subgraph (first {s} nodes) ...")
A_small = A[:s, :s].tocsr()
G_small = nx.from_scipy_sparse_array(A_small)
pos2 = nx.spring_layout(G_small, seed=7)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
nx.draw(G_small, pos2, node_size=40, node_color=true_labels[:s], cmap="coolwarm")
plt.title(
   "Primary (sample) Ground Truth",
    fontsize=11,
    fontweight="bold",
    color="black"
)


plt.subplot(1, 2, 2)
nx.draw(G_small, pos2, node_size=40, node_color=pred_fine[:s], cmap="coolwarm")
plt.title(
    "Primary (sample) Predicted (after RQI)",
    fontsize=11,
    fontweight="bold",
    color="black"
)


plt.tight_layout()
plt.show()

print("\nDONE.")
