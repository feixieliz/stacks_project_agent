import pandas as pd
import re

def main():
    with open("all.tex", "r") as f:
        all_text = f.read()

    all_proofs = {"statement": [], "proof": [], "type": [], "previous_section": []}
    for key in ["theorem", "lemma", "proposition"]:
        pattern = r"\\section([\d\D]*?)\n*\\begin{" + key + r"}\n([\d\D]*?)\\end{" + key + r"}\n+\\begin{proof}\n([\d\D]*?)\\end{proof}"
        matches = re.findall(pattern, all_text)
        for match in matches:
            all_proofs["statement"].append(match[1])
            all_proofs["proof"].append(match[2])
            all_proofs["type"].append(key)
            all_proofs["previous_section"].append(match[0])

    df = pd.DataFrame(all_proofs)
    df.to_parquet("all_proofs.parquet")


if __name__ == '__main__':
    main()
