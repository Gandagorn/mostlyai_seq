import gc
from datetime import datetime
import logging
import numba
from joblib import Parallel, delayed
from typing import Optional, Dict
from itertools import combinations
from typing import List, Tuple
from scipy import sparse
from sklearn.metrics import mutual_info_score
import pandas as pd
import numpy as np
from tqdm import tqdm
from .utils import coherence_report_columns, calculate_accuracy

logger = logging.getLogger(__name__)
SEQUENCE_LEN_COL = "sequence_length"

@numba.njit(cache=True)
def _rows_sum_csr_int32(data, indices, indptr, rows, K):
    """
    Sum CSR rows into a dense 1-D vector without creating
    intermediate scipy / numpy objects.
    """
    out = np.zeros(K, dtype=np.int32)
    for r in rows:
        start = indptr[r]
        end = indptr[r + 1]
        for p in range(start, end):
            out[indices[p]] += data[p]
    return out


def fast_rows_sum(mat: sparse.csr_matrix, rows: np.ndarray) -> np.ndarray:
    return _rows_sum_csr_int32(mat.data, mat.indices, mat.indptr,
                               rows.astype(np.int32), mat.shape[1])


def _make_spec(df: pd.DataFrame, bins: int = 10) -> Dict[str, Tuple]:
    spec = {}
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            edges = np.unique(np.quantile(s.dropna().astype("float64"), np.linspace(0, 1, bins + 1)))
            if len(edges) < 2:
                edges = np.linspace(s.min(), s.max(), 2)
            spec[col] = ("num", edges)
        else:
            top_categories = s.value_counts(dropna=False).index[:bins - 1]
            mapping = {val: i for i, val in enumerate(top_categories)}
            spec[col] = ("cat", mapping, bins - 1)
    return spec


def _bin_df(df: pd.DataFrame, spec: Dict[str, Tuple]) -> pd.DataFrame:
    binned_df = pd.DataFrame(index=df.index)
    for col, info in spec.items():
        kind = info[0]
        if kind == "num":
            edges = info[1]
            binned_df[col] = np.searchsorted(edges[1:-1], df[col].values, side="right")
        else:
            mapping, other_code = info[1], info[2]
            binned_df[col] = df[col].map(mapping).fillna(other_code)
    return binned_df.astype("int64")


def _top_triples(df: pd.DataFrame, cols: List[str], k: int = 50, bins: int = 10) -> List[Tuple[str, str, str]]:
    if k == 0 or len(cols) < 3:
        return []
    mi_pairs = []
    for c1, c2 in combinations(cols, 2):
        mi = mutual_info_score(df[c1], df[c2])
        mi_pairs.append(((c1, c2), mi))
    mi_pairs.sort(key=lambda t: -t[1])
    top_cols = set([c for pair, _ in mi_pairs[:min(len(mi_pairs), 4 * k)] for c in pair])
    if len(top_cols) < 3:
        return []
    triples = []
    for c1, c2, c3 in combinations(top_cols, 3):
        joint_numerical = df[c1].values * bins + df[c2].values
        mi = mutual_info_score(joint_numerical, df[c3].values)
        triples.append(((c1, c2, c3), mi))
    triples.sort(key=lambda t: -t[1])
    return [t[0] for t in triples[:k]]


def _per_group_counts(code_vec: np.ndarray,
                      g_idx: np.ndarray,
                      G: int,
                      K: int) -> sparse.csr_matrix:
    """
    Return a sparse (G × K) matrix
    """
    flat = g_idx.astype(np.int32) * K + code_vec.astype(np.int32)

    # 1-row sparse histogram, then reshape to (G, K)
    M = sparse.coo_matrix(
        (np.ones_like(flat, dtype=np.int32),
         (np.zeros_like(flat), flat)),
        shape=(1, G * K),
        dtype=np.int32
    ).tocsr()
    return sparse.csr_matrix(M.reshape(G, K))


@numba.njit(nopython=True, fastmath=True)
def _calculate_l1_gain_numba_sparse(resid,
                                    sparse_data,
                                    sparse_indices,
                                    sparse_indptr,
                                    is_removal):
    """
    Calculates L1 gain for each row in a sparse matrix slice.
    resid: residual vector (target - current)
    sparse_data, sparse_indices, sparse_indptr: CSR representation of candidate contributions
    is_removal: whether this is a removal (True) or addition (False)
    """
    num_candidates = len(sparse_indptr) - 1
    gains = np.empty(num_candidates, dtype=np.float64)

    for i in range(num_candidates):
        l1_change = 0.0
        start = sparse_indptr[i]
        end = sparse_indptr[i + 1]

        for ptr in range(start, end):
            col_idx = sparse_indices[ptr]
            value = sparse_data[ptr]
            original_val = resid[col_idx]

            if is_removal:
                new_val = original_val + value
            else:  # is_addition
                new_val = original_val - value

            l1_change += np.abs(new_val) - np.abs(original_val)

        gains[i] = -l1_change

    return gains


def refine_subset_by_coherence(
        train_df: pd.DataFrame,
        pool_df: pd.DataFrame,
        target_size: int,
        *,
        iterations: int = 20_000,
        max_time: Optional[int] = None,
        initial_candidates_per_swap: int = 10,
        min_candidates_per_swap: int = 2,
        group_col: str = "group_id",
        n_bins: int = 10,
        top_n: int = 9,
        col_sample: float = 1,
        rng: np.random.Generator | None = None
) -> np.ndarray:
    """Refines a subset of groups to improve sequential coherence metrics.

    This function uses an iterative swapping algorithm to select a subset of
    groups from a pool that best matches the coherence characteristics of a
    training set. Coherence is measured by 'categories per sequence' and
    'sequences per category' distributions. It greedily swaps groups to
    minimize the L1 error between the subset's and the training set's
    coherence distributions.

    Args:
        train_df: The original training data.
        pool_df: The pool of synthetic groups to select from.
        target_size: The desired number of groups in the final subset.
        iterations: The maximum number of swap iterations.
        max_time: Maximum execution time in minutes for this step.
        initial_candidates_per_swap: The initial number of candidates to evaluate
                                     for each swap.
        min_candidates_per_swap: The minimum number of candidates to evaluate.
        group_col: The name of the column identifying sequences/groups.
        n_bins: Number of bins for 'categories per sequence' metric.
        top_n: Number of top categories for 'sequences per category' metric.
        col_sample: Fraction of columns to use for calculating metrics.
        rng: A numpy random number generator instance.

    Returns:
        An array of group IDs for the selected subset.
    """
    rng = rng or np.random.default_rng(42)
    feat_cols = [c for c in train_df.columns if c != group_col]
    if not feat_cols:
        logger.warning("No feature columns found to process for coherence refinement.")
        return np.array([], dtype=int)

    cols_used = rng.choice(feat_cols, size=max(1, int(len(feat_cols) * col_sample)), replace=False)

    logger.info("Coherence Refinement Step 1: Pre-calculating target distributions from train_df...")
    targets = {}
    for col in cols_used:
        uniq_tr = train_df.groupby(group_col)[col].nunique()
        if uniq_tr.empty: continue
        lo, hi = uniq_tr.min(), uniq_tr.max()
        edges = np.unique(np.linspace(lo, hi, n_bins + 1))
        if len(edges) < 2: edges = np.array([lo, hi if hi > lo else lo + 1])
        tgt_cps, _ = np.histogram(uniq_tr, bins=edges)
        targets[f"cps_{col}"] = (tgt_cps, edges)  # Not normalizing here

        trn_counts = train_df.groupby(col, observed=False)[group_col].nunique()
        if trn_counts.empty: continue
        cats = trn_counts.nlargest(top_n).index.tolist()
        tgt_spc = trn_counts[cats].to_numpy()
        targets[f"spc_{col}"] = (tgt_spc, cats)

    logger.info("Coherence Refinement Step 2: Pre-calculating contributions of each group in pool_df...")
    pool_ids = pool_df[group_col].unique()

    contributions = {}
    for group_id, group_df in tqdm(pool_df.groupby(group_col), desc="Analyzing pool groups"):
        group_contrib = {}
        for col in cols_used:
            cps_key = f"cps_{col}"
            spc_key = f"spc_{col}"
            if cps_key not in targets or spc_key not in targets: continue

            _, edges = targets[cps_key]
            n_unique = group_df[col].nunique()
            group_contrib[cps_key] = np.searchsorted(edges[1:-1], n_unique, side="right")

            _, cats = targets[spc_key]
            unique_cats_in_group = set(group_df[col].unique())
            group_contrib[spc_key] = np.array([1 if cat in unique_cats_in_group else 0 for cat in cats])
        contributions[group_id] = group_contrib

    logger.info("Coherence Refinement Step 3: Initializing with a random subset...")
    current_selection = set(rng.choice(pool_ids, size=target_size, replace=False))

    current_hists = {}
    for key in targets:
        target_hist, _ = targets[key]
        current_hists[key] = np.zeros_like(target_hist, dtype=float)

    for group_id in current_selection:
        for key, value in contributions[group_id].items():
            if key.startswith("cps_"):
                current_hists[key][value] += 1
            else:
                current_hists[key] += value

    def calculate_error(hists):
        total_error = 0
        for key, current_hist in hists.items():
            target_hist, _ = targets[key]
            # Normalize both histograms to sum to 1 to compare distributions
            current_sum = current_hist.sum()
            target_sum = target_hist.sum()
            current_norm = current_hist / current_sum if current_sum > 0 else current_hist
            target_norm = target_hist / target_sum if target_sum > 0 else target_hist
            total_error += np.sum(np.abs(current_norm - target_norm))
        return total_error

    current_error = calculate_error(current_hists)
    logger.info(f"Initial Coherence Error of Random Subset: {current_error:.4f}")

    logger.info("Coherence Refinement Step 4: Iteratively refining the subset by swapping groups...")
    unused_pool = list(set(pool_ids) - current_selection)
    selection_list = list(current_selection)

    # --- Initialize dynamic candidate size ---
    current_candidates_per_swap = initial_candidates_per_swap
    coherence_start_time = datetime.now()

    for i in tqdm(range(iterations), desc="Refining Subset"):
        if max_time is not None and ((datetime.now() - coherence_start_time).total_seconds() / 60)  > max_time:
            logger.info(f"Coherence refinement time limit of {max_time} minutes reached. Stopping early.")
            break
        if not unused_pool: break  # No more groups to swap in

        # Pick a random group to remove from the current selection
        idx_to_remove_pos = rng.integers(0, len(selection_list))
        group_to_remove = selection_list[idx_to_remove_pos]

        # Pick candidate groups to add from the unused pool
        num_to_sample = min(current_candidates_per_swap, len(unused_pool))
        if num_to_sample < 1: continue  # Not enough candidates to sample

        candidate_indices_pos = rng.choice(len(unused_pool), size=num_to_sample, replace=False)
        candidates = [unused_pool[pos] for pos in candidate_indices_pos]

        best_candidate = None
        best_candidate_error = current_error

        for candidate in candidates:
            # Quickly calculate the potential new error without rebuilding everything
            error_after_swap = 0
            for key in targets:
                target_hist, _ = targets[key]
                if key not in contributions[group_to_remove] or key not in contributions[candidate]: continue

                if key.startswith("cps_"):
                    h = current_hists[key].copy()
                    h[contributions[group_to_remove][key]] -= 1
                    h[contributions[candidate][key]] += 1
                else:
                    h = current_hists[key] - contributions[group_to_remove][key] + contributions[candidate][key]

                h_sum = h.sum()
                target_sum = target_hist.sum()
                h_norm = h / h_sum if h_sum > 0 else h
                target_norm = target_hist / target_sum if target_sum > 0 else target_hist
                error_after_swap += np.sum(np.abs(h_norm - target_norm))

            if error_after_swap < best_candidate_error:
                best_candidate_error = error_after_swap
                best_candidate = candidate

        swapped = False
        if best_candidate is not None:
            swapped = True
            current_error = best_candidate_error

            # Update histograms
            for key in targets:
                if key not in contributions[group_to_remove] or key not in contributions[best_candidate]: continue
                if key.startswith("cps_"):
                    current_hists[key][contributions[group_to_remove][key]] -= 1
                    current_hists[key][contributions[best_candidate][key]] += 1
                else:
                    current_hists[key] = current_hists[key] - contributions[group_to_remove][key] + \
                                         contributions[best_candidate][key]

            # Find position of best_candidate in the candidates list to get its original index
            best_candidate_original_pos = candidate_indices_pos[candidates.index(best_candidate)]

            # Swap the string IDs in the lists
            selection_list[idx_to_remove_pos] = best_candidate
            unused_pool[best_candidate_original_pos] = group_to_remove

        # --- Update dynamic candidate size ---
        if swapped:
            current_candidates_per_swap += 1
        else:
            current_candidates_per_swap = max(min_candidates_per_swap, current_candidates_per_swap - 3)

        if (i + 1) % 1_000 == 0 or i == (iterations - 1):
            temp_subset_df = pool_df[pool_df[group_col].isin(selection_list)]
            coherence = coherence_report_columns(train_df, temp_subset_df, group_key=group_col)
            logger.info(f"Iter {i + 1}/{iterations}: L1 Error: {current_error:.4f}, Coherence: {coherence:.4f}")

    logger.info(f"Final Coherence Error after Refinement: {current_error:.4f}")
    return np.array(selection_list)


def choose_groups_by_refinement(
        train_df: pd.DataFrame,
        pool_df: pd.DataFrame,
        *,
        group_col: str = "group_id",
        bins: int = 10,
        max_groups: int = 20_000,
        swap_iterations: int = 1000,
        swap_size: int = 100,
        top_k_pairs: int = 45,
        top_k_triples: int = 150,
        swap_size_multiplier: int = 5,
        coherence_iterations: int = 5_000,
        coherence_max_time: Optional[int] = None,
        refinement_max_time: Optional[int] = None,
        n_jobs: int = 2,
        rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """Selects a subset of groups to match statistical distributions.

    This function is a two-stage refinement process. First, it calls
    `refine_subset_by_coherence` to get an initial set of groups optimized for
    sequential coherence. Second, it performs a more intensive refinement on
    that subset, swapping groups to minimize the L1 distance of univariate,
    bivariate, and trivariate distributions.
    This process is computationally intensive and uses sparse matrices and
    parallel processing for efficiency and memory usage.

    Args:
        train_df: The original training data.
        pool_df: The pool of synthetic groups to select from.
        group_col: The column name for group identifiers.
        bins: The number of bins for discretizing numeric features.
        max_groups: The target number of groups in the final subset.
        swap_iterations: Max iterations for the statistical refinement phase.
        swap_size: The initial number of groups to swap in each iteration.
        top_k_pairs: The number of top bivariate features to match.
        top_k_triples: The number of top trivariate features to match.
        swap_size_multiplier: Factor for the candidate pool size.
        coherence_iterations: Max iterations for the initial coherence refinement.
        coherence_max_time: Time limit in minutes for coherence refinement.
        refinement_max_time: Time limit in minutes for statistical refinement.
        n_jobs: The number of parallel jobs to use.
        rng: A numpy random number generator instance.

    Returns:
        An array of the final selected group IDs.
    """
    logger.info("Starting initial group selection using coherence optimization.")
    initial_string_ids = refine_subset_by_coherence(
        train_df.drop(columns=SEQUENCE_LEN_COL),
        pool_df.drop(columns=SEQUENCE_LEN_COL),
        target_size=max_groups,
        iterations=coherence_iterations,
        max_time=coherence_max_time,
        initial_candidates_per_swap=50
    )
    gc.collect()

    start_time = datetime.now()
    rng = rng or np.random.default_rng(0)
    n_jobs = min(n_jobs, 2)
    feat_cols = [c for c in train_df.columns if c != group_col]

    spec = _make_spec(train_df[feat_cols], bins)
    tr_bin = _bin_df(train_df[feat_cols], spec)
    pl_bin = _bin_df(pool_df[feat_cols], spec)

    all_biv_feats = list(combinations(feat_cols, 2))
    biv_mi_pairs = sorted([(c, mutual_info_score(tr_bin[c[0]], tr_bin[c[1]])) for c in all_biv_feats],
                          key=lambda t: -t[1])
    bivariate_features = [p[0] for p in biv_mi_pairs[:top_k_pairs]]
    trivariate_features = _top_triples(tr_bin, feat_cols, k=top_k_triples, bins=bins)

    cols = {
        'uni': feat_cols,
        'bi': [f"{c1}×{c2}" for c1, c2 in bivariate_features],
        'tri': [f"{c1}×{c2}×{c3}" for c1, c2, c3 in trivariate_features]
    }

    tr_bin_np, pl_bin_np = {c: tr_bin[c].values for c in feat_cols}, {c: pl_bin[c].values for c in feat_cols}
    for c1, c2 in bivariate_features:
        name = f"{c1}×{c2}";
        tr_bin_np[name] = (tr_bin_np[c1] * bins + tr_bin_np[c2]);
        pl_bin_np[name] = (pl_bin_np[c1] * bins + pl_bin_np[c2])
    for c1, c2, c3 in trivariate_features:
        name = f"{c1}×{c2}×{c3}";
        tr_bin_np[name] = (tr_bin_np[c1] * bins * bins + tr_bin_np[c2] * bins + tr_bin_np[c3]);
        pl_bin_np[name] = (pl_bin_np[c1] * bins * bins + pl_bin_np[c2] * bins + pl_bin_np[c3])

    def get_bincount(c_name):
        if "×" not in c_name: return np.bincount(tr_bin_np[c_name], minlength=bins)
        return np.bincount(tr_bin_np[c_name], minlength=bins ** 2 if c_name.count("×") == 1 else bins ** 3)

    all_cols_flat = cols['uni'] + cols['bi'] + cols['tri']
    targets_flat = [get_bincount(c) for c in all_cols_flat]
    gid, g_idx = np.unique(pool_df[group_col].to_numpy(), return_inverse=True)
    G = len(gid)

    logger.info("Calculating group contributions in parallel...")
    contrib_flat = Parallel(n_jobs=n_jobs)(
        delayed(_per_group_counts)(pl_bin_np[c], g_idx, G, t.size)
        for c, t in zip(all_cols_flat, targets_flat)
    )

    split1 = len(cols['uni']);
    split2 = split1 + len(cols['bi'])
    contrib = {'uni': contrib_flat[:split1], 'bi': contrib_flat[split1:split2], 'tri': contrib_flat[split2:]}
    targets = {'uni': targets_flat[:split1], 'bi': targets_flat[split1:split2], 'tri': targets_flat[split2:]}

    logger.info("Starting statistical refinement with the coherence-optimized set...")
    chosen_mask = np.zeros(G, dtype=bool)

    gid_to_int_idx = {gid_str: i for i, gid_str in enumerate(gid)}
    initial_indices = [gid_to_int_idx[id_str] for id_str in initial_string_ids]
    chosen_mask[initial_indices] = True

    initial_chosen_indices = np.where(chosen_mask)[0]
    current_hists = {phase: [fast_rows_sum(c, initial_chosen_indices) for c in C] for phase, C in contrib.items()}

    def calculate_normalized_l1(hists_dict, targets_dict):
        total_norm_error = 0.0
        num_phases_with_features = 0
        for phase in ['uni', 'bi', 'tri']:
            if not targets_dict.get(phase): continue
            num_phases_with_features += 1
            phase_error = sum(
                np.abs(hists_dict[phase][j] - targets_dict[phase][j]).sum() for j in range(len(targets_dict[phase])))
            max_possible_phase_error = 2 * len(train_df) * len(targets_dict[phase])
            if max_possible_phase_error > 0:
                total_norm_error += phase_error / max_possible_phase_error
        return total_norm_error / num_phases_with_features if num_phases_with_features > 0 else 0

    current_error = calculate_normalized_l1(current_hists, targets)
    logger.info(f"Initial solution error (normalized): {current_error:.6f}")

    current_swap_size = swap_size
    min_swap_size = 1
    initial_temp = 0.00001

    for i in range(swap_iterations):
        if refinement_max_time is not None and ((datetime.now() - start_time).total_seconds() / 60) > refinement_max_time:
            logger.info(f"Refinement loop time limit of {refinement_max_time} minutes reached. Stopping early.")
            break
        temperature = initial_temp * (1 - (i / swap_iterations)) ** 2

        idx_chosen, idx_pool = np.where(chosen_mask)[0], np.where(~chosen_mask)[0]
        if len(idx_pool) < current_swap_size: break

        removal_gains = np.zeros(len(idx_chosen), dtype=np.float64)
        for phase in ['uni', 'bi', 'tri']:
            if not targets[phase]: continue
            for j in range(len(targets[phase])):
                max_l1_dist = targets[phase][j].sum() * 2
                if max_l1_dist == 0: continue

                resid = targets[phase][j] - current_hists[phase][j]
                sparse_slice = contrib[phase][j][idx_chosen].copy()
                raw_removal_gain = _calculate_l1_gain_numba_sparse(
                    resid,
                    sparse_slice.data,
                    sparse_slice.indices,
                    sparse_slice.indptr,
                    is_removal=True
                )
                removal_gains += raw_removal_gain / max_l1_dist

        worst_indices = idx_chosen[np.argsort(removal_gains)[-current_swap_size:]]
        hists_of_worst = {
            p: [fast_rows_sum(c, worst_indices) for c in C]
            for p, C in contrib.items()
        }
        cand_indices = rng.choice(idx_pool, size=min(len(idx_pool), current_swap_size * swap_size_multiplier),
                                  replace=False)

        addition_gains = np.zeros(len(cand_indices), dtype=np.float64)
        for phase in ['uni', 'bi', 'tri']:
            if not targets[phase]: continue
            for j in range(len(targets[phase])):
                max_l1_dist = 2 * targets[phase][j].sum()
                if max_l1_dist == 0: continue

                resid_after_removal = targets[phase][j] - (current_hists[phase][j] - hists_of_worst[phase][j])

                sparse_slice = contrib[phase][j][cand_indices].copy()
                raw_addition_gain = _calculate_l1_gain_numba_sparse(
                    resid_after_removal,
                    sparse_slice.data,
                    sparse_slice.indices,
                    sparse_slice.indptr,
                    is_removal=False
                )

                addition_gains += raw_addition_gain / max_l1_dist

        best_replacements = cand_indices[np.argsort(addition_gains)[-current_swap_size:]]

        hists_of_cand = {
            p: [fast_rows_sum(contrib[p][j], best_replacements)  # contrib[p][j] is a csr_matrix
                for j in range(len(C))]
            for p, C in contrib.items()
        }
        new_hists = {p: [current_hists[p][j] - hists_of_worst[p][j] + hists_of_cand[p][j] for j in range(len(C))] for
                     p, C in contrib.items()}
        new_error = calculate_normalized_l1(new_hists, targets)

        accepted = False
        if new_error < current_error:
            accepted = True
            # logger.info(f"Iter {i + 1:3d}/{iterations}: ACCEPTED (IMPROVED) Swapped {current_swap_size}. Err: {current_error:.6f} -> {new_error:.6f}.")
        elif temperature > 1e-9 and np.exp((current_error - new_error) / temperature) > rng.random():
            accepted = True
            # logger.info(f"Iter {i + 1:3d}/{iterations}: ACCEPTED (ANNEALING) Swapped {current_swap_size}. Err: {current_error:.6f} -> {new_error:.6f}.")

        if accepted:
            chosen_mask[worst_indices], chosen_mask[best_replacements] = False, True
            current_hists = new_hists
            current_error = new_error
            current_swap_size = min(swap_size * 2, current_swap_size + 1)
        else:
            current_swap_size = max(min_swap_size, current_swap_size - 5)

        if (i + 1) % 500 == 0 or i == (swap_iterations - 1):
            temp_subset_df = pool_df.loc[pool_df["group_id"].isin(gid[chosen_mask])]
            acc = calculate_accuracy(train_df.drop(columns="group_id"), temp_subset_df.drop(columns="group_id"))
            coherence = coherence_report_columns(train_df, temp_subset_df)
            combined_score = acc.get('overall_accuracy', 0) * 0.75 + coherence * 0.25
            logger.info(f"Iter {i + 1:4d}/{swap_iterations}: Swap Size: {current_swap_size:3d}, Norm. L1 Err: {current_error:.6f}, Accuracy: {acc.get('overall_accuracy', 0):.6f}, Coherence: {coherence:.4f}, Combined: {combined_score:.4f}")

    logger.info(f"Finished refinement in {(datetime.now() - start_time).total_seconds():.2f} seconds.")
    return gid[chosen_mask]

def run_refinement(
    synthetic_pool: pd.DataFrame,
    train_df: pd.DataFrame,
    params: dict,
):
    """Orchestrates the full post-processing pipeline for sequential data.

    This function manages the end-to-end refinement process. It calls the main two-stage
    `choose_groups_by_refinement` function to select the best group IDs based
    on both coherence and statistical accuracy. Finally, it filters the
    synthetic pool to the chosen groups and ensures data types match the
    original training data.

    Returns:
        A refined DataFrame containing the final synthetic data.
    """
    train_df = train_df.copy()
    # Refine accuracy including sequence_length
    synthetic_pool[SEQUENCE_LEN_COL] = synthetic_pool["group_id"].map(synthetic_pool["group_id"].value_counts())
    train_df[SEQUENCE_LEN_COL] = train_df["group_id"].map(train_df["group_id"].value_counts())

    chosen_group_ids = choose_groups_by_refinement(
        train_df,
        synthetic_pool,
        group_col="group_id",
        **params
    )
    synthetic_pool = synthetic_pool.drop(columns=SEQUENCE_LEN_COL)
    train_df = train_df.drop(columns=SEQUENCE_LEN_COL)

    subset_df = (synthetic_pool
                 .loc[synthetic_pool["group_id"].isin(chosen_group_ids)]
                 .reset_index(drop=True))

    # match dtypes of training
    for c in train_df.columns:
        subset_df[c] = subset_df[c].astype(train_df[c].dtype)

    return subset_df