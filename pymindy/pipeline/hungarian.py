import numpy as np

# BEGIN JAKE CODE

###### Hungarian() ######
# Hungarian() matches the labels of fit states to the labels of true states using
# the Hungarian (Kuhn-Munkres) algorithm, then relabels the fit state history so
# that each fit state carries the label of the true state it was matched to.
# HMM state labels are arbitrary (fit state 0 may correspond to true state 1), so
# this must be applied before scoring a fit state history against the truth.
#
# The previous version stopped after steps 1-2 (row/column reduction) and looped
# forever whenever those steps did not leave a unique zero in every row (e.g. a
# fit that collapsed onto a single state). It also relabeled in the wrong
# direction (fit state == true index -> fit index), which only gave the right
# answer for 2 states. This version runs the full algorithm.
#
# Parameters:
#   cost_mat : np.array (n_true x n_fit)
#       Score for matching each true state (row) with each fit state (column);
#       higher is better. Pass timepoint counts to maximize raw accuracy, or the
#       row-normalized confusion matrix to maximize balanced accuracy. Need not be
#       square. NaN entries (e.g. a true state that never occurs) are treated as 0.
#   mod_ind_hist : np.array
#       Fit state history (one fit state label per timepoint)
#
# Returns:
#   labels : list of (true_state, fit_state) tuples
#       The matched pairs, sorted by true state
#   mod_ind_hist_new : np.array (int)
#       mod_ind_hist relabeled into true state labels. Fit states left unmatched
#       (only possible when n_fit > n_true) get new labels n_true, n_true+1, ...
#       so they never count as correct.
def Hungarian(cost_mat, mod_ind_hist):
    score = np.nan_to_num(np.asarray(cost_mat, dtype=float))
    n_true, n_fit = score.shape

    # Pad to a square matrix with zero-score dummy rows/columns, then convert the
    # maximization into the standard minimization problem
    n = max(n_true, n_fit)
    padded = np.zeros((n, n))
    padded[:n_true, :n_fit] = score
    cost = padded.max() - padded

    fit_for_true = _min_cost_assignment(cost)
    labels = [(j, int(fit_for_true[j])) for j in range(n_true) if fit_for_true[j] < n_fit]

    # Re-label mod_ind_hist: every timepoint assigned fit state k becomes true state j
    mod_ind_hist = np.asarray(mod_ind_hist).astype(int)
    relabel = np.arange(n_fit) + n_true # unmatched fit states -> labels no true state uses
    for true_state, fit_state in labels:
        relabel[fit_state] = true_state
    mod_ind_hist_new = relabel[mod_ind_hist]

    return labels, mod_ind_hist_new

###### align_to_truth() ######
# align_to_truth() builds the true-vs-fit timepoint count matrix and runs
# Hungarian() on it, so the returned labeling maximizes the number of correctly
# labeled timepoints. With 2 states, the aligned accuracy is therefore >= 0.5.
#
# Parameters:
#   mod_ind_true : np.array
#       True state history
#   mod_ind_fit : np.array
#       Fit (or baseline) state history to align
#   n_true : int
#       Number of true states (default: inferred from mod_ind_true)
#   n_fit : int
#       Number of fit states (default: inferred from mod_ind_fit)
#
# Returns:
#   labels : list of (true_state, fit_state) tuples
#   mod_ind_fit_new : np.array (int)
#       mod_ind_fit relabeled into true state labels
def align_to_truth(mod_ind_true, mod_ind_fit, n_true=None, n_fit=None):
    mod_ind_true = np.asarray(mod_ind_true).astype(int)
    mod_ind_fit = np.asarray(mod_ind_fit).astype(int)
    if n_true is None:
        n_true = mod_ind_true.max() + 1
    if n_fit is None:
        n_fit = mod_ind_fit.max() + 1

    count_mat = np.zeros((n_true, n_fit))
    np.add.at(count_mat, (mod_ind_true, mod_ind_fit), 1)

    return Hungarian(count_mat, mod_ind_fit)

###### _min_cost_assignment() ######
# _min_cost_assignment() solves the square assignment problem (pick one column per
# row, each column used once, minimizing the total cost) with the O(n^3)
# shortest-augmenting-path form of the Hungarian algorithm. Row potentials u and
# column potentials v keep every reduced cost (cost - u - v) non-negative; each
# row is added by growing a tree of tight (zero reduced cost) edges, shifting the
# potentials by the smallest slack until a free column is reached, then flipping
# the assignments along that path.
#
# Parameters:
#   cost : np.array (n x n)
#       Cost of assigning each row to each column
#
# Returns:
#   col_for_row : np.array (int, length n)
#       The column assigned to each row
def _min_cost_assignment(cost):
    n = cost.shape[0]
    # Index 0 is a sentinel; rows/columns are 1-indexed below
    u = np.zeros(n + 1)
    v = np.zeros(n + 1)
    row_for_col = np.zeros(n + 1, dtype=int) # 0 means the column is free
    prev_col = np.zeros(n + 1, dtype=int) # path back to the sentinel column

    for row in range(1, n + 1):
        row_for_col[0] = row
        col = 0
        min_slack = np.full(n + 1, np.inf)
        in_tree = np.zeros(n + 1, dtype=bool)

        # Grow the tree until it reaches a free column
        while True:
            in_tree[col] = True
            tree_row = row_for_col[col]
            delta = np.inf
            next_col = 0
            for c in range(1, n + 1):
                if not in_tree[c]:
                    slack = cost[tree_row - 1, c - 1] - u[tree_row] - v[c]
                    if slack < min_slack[c]:
                        min_slack[c] = slack
                        prev_col[c] = col
                    if min_slack[c] < delta:
                        delta = min_slack[c]
                        next_col = c
            # Shift the potentials so the cheapest edge out of the tree becomes tight
            for c in range(n + 1):
                if in_tree[c]:
                    u[row_for_col[c]] += delta
                    v[c] -= delta
                else:
                    min_slack[c] -= delta
            col = next_col
            if row_for_col[col] == 0:
                break

        # Flip assignments along the augmenting path
        while col != 0:
            back = prev_col[col]
            row_for_col[col] = row_for_col[back]
            col = back

    col_for_row = np.zeros(n, dtype=int)
    for c in range(1, n + 1):
        col_for_row[row_for_col[c] - 1] = c - 1
    return col_for_row

# END JAKE CODE
