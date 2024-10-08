from cbr_fox import cbr_fox


class cbr_fox_factory:
    @staticmethod
    def create_single(technique: str):
        return cbr_fox(technique)

    @staticmethod
    def create_multiple(techniques: list):
        return [cbr_fox(technique) for technique in techniques]