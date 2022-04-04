import pandas as pd


def get_compound_descriptors():
    url = "https://raw.githubusercontent.com/ersilia-os/griddify/main/data/test_molecules.csv"
    data = pd.read_csv(url)
    return data[list(data.columns)[2:]]
