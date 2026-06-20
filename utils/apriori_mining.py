"""Apriori association rule mining utilities."""

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder


def _transactions_from_csv(df):
    """Convert CSV rows to list of transactions (sets of items)."""
    transactions = []
    for _, row in df.iterrows():
        items = []
        for val in row.values:
            if pd.isna(val):
                continue
            text = str(val).strip()
            if text:
                items.append(text)
        if items:
            transactions.append(items)
    return transactions


def run_apriori(df, min_support=0.2, min_confidence=0.5, min_lift=1.0):
    """Run Apriori algorithm and return frequent itemsets and association rules."""
    transactions = _transactions_from_csv(df)
    if not transactions:
        raise ValueError("No valid transactions found in the uploaded dataset.")

    te = TransactionEncoder()
    te_array = te.fit(transactions).transform(transactions)
    encoded_df = pd.DataFrame(te_array, columns=te.columns_)

    frequent_itemsets = apriori(encoded_df, min_support=min_support, use_colnames=True)
    if frequent_itemsets.empty:
        return {
            "transaction_count": len(transactions),
            "item_count": len(te.columns_),
            "frequent_itemsets": [],
            "rules": [],
            "message": "No frequent itemsets found. Try lowering min_support.",
        }

    frequent_itemsets["itemsets"] = frequent_itemsets["itemsets"].apply(
        lambda x: sorted(list(x))
    )
    itemsets_records = frequent_itemsets.to_dict(orient="records")

    rules_df = association_rules(
        frequent_itemsets, metric="confidence", min_threshold=min_confidence
    )
    if not rules_df.empty and min_lift > 0:
        rules_df = rules_df[rules_df["lift"] >= min_lift]

    rules_records = []
    if not rules_df.empty:
        for _, row in rules_df.iterrows():
            rules_records.append(
                {
                    "antecedents": sorted(list(row["antecedents"])),
                    "consequents": sorted(list(row["consequents"])),
                    "support": round(float(row["support"]), 4),
                    "confidence": round(float(row["confidence"]), 4),
                    "lift": round(float(row["lift"]), 4),
                    "conviction": round(float(row["conviction"]), 4)
                    if row["conviction"] != float("inf")
                    else None,
                }
            )

    return {
        "transaction_count": len(transactions),
        "item_count": len(te.columns_),
        "frequent_itemsets": [
            {
                "itemsets": r["itemsets"],
                "support": round(float(r["support"]), 4),
            }
            for r in itemsets_records
        ],
        "rules": sorted(rules_records, key=lambda x: x["lift"], reverse=True),
        "parameters": {
            "min_support": min_support,
            "min_confidence": min_confidence,
            "min_lift": min_lift,
        },
    }
