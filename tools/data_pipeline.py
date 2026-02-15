import polars as pl


class DataPipeline:
    def __init__(self, caminho=None):
        self.df = None
        self.caminho = caminho

    def carregar_csv(self):
        self.df = pl.read_csv(self.caminho)
        return self.df

    def carregar_parquet(self):
        self.df = pl.read_parquet(self.caminho)
        return self.df

    def selecionar_target(self, target):
        X = self.df.drop(target)
        y = self.df.select(target)
        return X, y

    def train_test_split(self, target, test_size=0.2, seed=42):
        df = self.df.sample(fraction=1.0, shuffle=True, seed=seed)

        split_index = int(len(df) * (1 - test_size))
        train = df[:split_index]
        test = df[split_index:]

        return train, test
