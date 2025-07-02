import pandas as pd
from mostlyai.qa._accuracy import bin_data, calculate_univariates, calculate_bivariates, calculate_trivariates
import numpy as np


def calculate_accuracy(original_data, synthetic_data, variate_level=3):
    ori_bin, bins = bin_data(df=original_data, bins=10)
    syn_bin, _ = bin_data(df=synthetic_data, bins=bins)

    # mimick mostly columns
    ori_bin.columns = ["tgt::" + c for c in ori_bin.columns]
    syn_bin.columns = ["tgt::" + c for c in syn_bin.columns]

    res = {}

    if variate_level >= 1:
        acc_uni = calculate_univariates(ori_bin, syn_bin)
        res["univariate_accuracy"] = acc_uni["accuracy"].mean()
    if variate_level >= 2:
        acc_biv = calculate_bivariates(ori_bin, syn_bin)
        res["bivariate_accuracy"] = acc_biv["accuracy"].mean()
    if variate_level >= 3:
        acc_triv = calculate_trivariates(ori_bin, syn_bin)
        res["trivariate_accuracy"] = acc_triv["accuracy"].mean()

    res["overall_accuracy"] = np.mean(list(res.values()))
    return res


def _accuracy_from_counts(c_a: pd.Series, c_b: pd.Series) -> float:
    p = c_a / c_a.sum()
    q = c_b / c_b.sum()
    idx = p.index.union(q.index)
    return 1.0 - 0.5 * np.abs(p.reindex(idx, fill_value=0) -
                              q.reindex(idx, fill_value=0)).sum()


def _cats_per_seq_accuracy(train_df: pd.DataFrame,
                           syn_df: pd.DataFrame,
                           group_key: str,
                           n_bins: int = 10) -> pd.Series:

    def _cats_per_seq(df: pd.DataFrame, group_key: str) -> pd.DataFrame:
        return df.groupby(group_key).nunique().reset_index(drop=True)

    trn = _cats_per_seq(train_df, group_key)
    syn = _cats_per_seq(syn_df, group_key)

    def _bin_with_edges(s: pd.Series, edges: np.ndarray) -> pd.Series:
        return pd.cut(s, edges, include_lowest=True).astype(str)

    scores = {}
    for col in trn.columns:
        lo, hi = trn[col].min(), trn[col].max()
        if lo == hi:  # constant → accuracy 1 by definition
            scores[col] = 1.0
            continue
        edges = np.linspace(lo, hi + 1e-9, n_bins + 1)
        trn_binned = _bin_with_edges(trn[col], edges)
        # NB: **reuse the same edges for synthetic**
        syn_binned = _bin_with_edges(syn[col].clip(lo, hi), edges)

        scores[col] = _accuracy_from_counts(
            trn_binned.value_counts(),
            syn_binned.value_counts()
        )
    return pd.Series(scores, name="cats_per_seq_acc")


def _seqs_per_cat(df: pd.DataFrame, group_key: str) -> dict[str, pd.Series]:
    return {
        col: df.groupby(col, observed=False)[group_key].nunique()
        for col in df.columns if col != group_key
    }


def _seqs_per_cat_accuracy(
        train_df: pd.DataFrame,
        syn_df: pd.DataFrame,
        group_key: str,
        top_n: int = 9
) -> pd.Series:
    trn_counts = _seqs_per_cat(train_df, group_key)
    syn_counts = _seqs_per_cat(syn_df, group_key)

    scores = {}
    for col in trn_counts:
        trn_top = trn_counts[col].nlargest(top_n)
        trn_rest = trn_counts[col].sum() - trn_top.sum()
        syn_top = syn_counts[col].reindex(trn_top.index, fill_value=0)
        syn_rest = syn_counts[col].sum() - syn_top.sum()

        scores[col] = _accuracy_from_counts(
            pd.concat([trn_top, pd.Series({'(other)': trn_rest})]),
            pd.concat([syn_top, pd.Series({'(other)': syn_rest})]),
        )
    return pd.Series(scores, name="seqs_per_cat_acc")


def coherence_column_scores(
        train_df: pd.DataFrame,
        syn_df: pd.DataFrame,
        group_key: str = "group_id",
        n_bins: int = 10,
        top_n: int = 9
) -> pd.DataFrame:
    cats_per_seq = _cats_per_seq_accuracy(train_df, syn_df, group_key, n_bins)
    seqs_per_cat = _seqs_per_cat_accuracy(train_df, syn_df, group_key, top_n)

    df1 = (cats_per_seq
           .reset_index().rename(columns={'index': 'column',
                                          'cats_per_seq_acc': 'accuracy'})
           .assign(metric='cats_per_seq'))
    df2 = (seqs_per_cat
           .reset_index().rename(columns={'index': 'column',
                                          'seqs_per_cat_acc': 'accuracy'})
           .assign(metric='seqs_per_cat'))

    return pd.concat([df1, df2], ignore_index=True)


def coherence_report_columns(
        train_df: pd.DataFrame,
        syn_df: pd.DataFrame,
        group_key: str = "group_id",
        n_bins: int = 10,
        top_n: int = 9
):
    """
    Faster (near) replication of the per-column coherence section of MOSTLY AI’s HTML report.
    Returns one row per column with:
        cats_per_seq , seqs_per_cat , coherence
    """
    cats = _cats_per_seq_accuracy(train_df, syn_df, group_key, n_bins)
    seqs = _seqs_per_cat_accuracy(train_df, syn_df, group_key, top_n)

    cols = sorted(set(cats.index) | set(seqs.index))

    out = pd.DataFrame(index=cols, columns=["cats_per_seq", "seqs_per_cat", "coherence"])
    out["cats_per_seq"] = cats
    out["seqs_per_cat"] = seqs

    out["coherence"] = out[["cats_per_seq", "seqs_per_cat"]].mean(axis=1)

    out = out.round(6)
    return out["coherence"].mean()