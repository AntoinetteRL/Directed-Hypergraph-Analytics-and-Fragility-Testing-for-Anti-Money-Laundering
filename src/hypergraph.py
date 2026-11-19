import numpy as np
import pandas as pd
# Optional reconstruction of the transaction-hypergraph features from
# AddrTx_edgelist.csv and TxAddr_edgelist.csv.
#
# This cell is not required when the canonical enriched CSV is used.

def construct_hypergraph_features(addr_tx_path, tx_addr_path):
    addr_tx = pd.read_csv(
        addr_tx_path, usecols=["input_address", "txId"],
        dtype={"input_address": "string"}
    ).drop_duplicates()
    tx_addr = pd.read_csv(
        tx_addr_path, usecols=["txId", "output_address"],
        dtype={"output_address": "string"}
    ).drop_duplicates()

    in_card = addr_tx.groupby("txId")["input_address"].nunique().rename("n_input")
    out_card = tx_addr.groupby("txId")["output_address"].nunique().rename("n_output")
    cards = pd.concat([in_card, out_card], axis=1).fillna(0)
    cards["hyperedge_size"] = cards["n_input"] + cards["n_output"]
    cards["expansion_ratio"] = cards["n_output"] / cards["n_input"].clip(lower=1)

    input_participation = addr_tx.merge(cards, on="txId", how="left")
    output_participation = tx_addr.merge(cards, on="txId", how="left")

    input_stats = input_participation.groupby("input_address").agg(
        input_incident_count=("txId", "nunique"),
        input_sum_n_input=("n_input", "sum"),
        input_sum_n_output=("n_output", "sum"),
        input_max_hyperedge=("hyperedge_size", "max"),
        input_sum_expansion=("expansion_ratio", "sum"),
        addr_co_input_degree=("n_input", lambda s: (s - 1).clip(lower=0).sum()),
    )

    output_stats = output_participation.groupby("output_address").agg(
        output_incident_count=("txId", "nunique"),
        output_sum_n_input=("n_input", "sum"),
        output_sum_n_output=("n_output", "sum"),
        output_max_hyperedge=("hyperedge_size", "max"),
        output_sum_expansion=("expansion_ratio", "sum"),
        addr_co_output_degree=("n_output", lambda s: (s - 1).clip(lower=0).sum()),
    )

    stats = input_stats.join(output_stats, how="outer").fillna(0)
    stats["incident_count"] = (
        stats["input_incident_count"] + stats["output_incident_count"]
    )
    denom = stats["incident_count"].replace(0, np.nan)

    stats["addr_avg_input_set_size"] = (
        stats["input_sum_n_input"] + stats["output_sum_n_input"]
    ) / denom
    stats["addr_avg_output_set_size"] = (
        stats["input_sum_n_output"] + stats["output_sum_n_output"]
    ) / denom
    stats["addr_max_hyperedge_size"] = stats[
        ["input_max_hyperedge", "output_max_hyperedge"]
    ].max(axis=1)
    stats["addr_avg_expansion_ratio"] = (
        stats["input_sum_expansion"] + stats["output_sum_expansion"]
    ) / denom

    result_cols = [
        "addr_avg_input_set_size",
        "addr_avg_output_set_size",
        "addr_max_hyperedge_size",
        "addr_avg_expansion_ratio",
        "addr_co_input_degree",
        "addr_co_output_degree",
    ]
    return stats[result_cols].fillna(0).reset_index(names="address")