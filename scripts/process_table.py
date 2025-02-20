import pandas as pd


def main():
    df = pd.read_parquet("all_proofs.parquet")

    df["proof_length"] = df["proof"].apply(lambda x: len(x))
    df["statement_length"] = df["statement"].apply(lambda x: len(x))
    df["previous_section_length"] = df["previous_section"].apply(lambda x: len(x))

    df.to_parquet("all_proofs_processed.parquet")


if __name__ == "__main__":
    main()
