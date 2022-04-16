from os import PathLike


class EdgeLabeler:
    def __init__(self, model_path: PathLike) -> None:
        self.model = self.__load_model(model_path)

    def predict(
        self, from_node, edge, to_node
    ):  # -> Tuple[SBN Role, SBN_EDGE_TYPE]
        pass

    def _encode(self, from_node, edge, to_node):
        pass

    @staticmethod
    def __load_model(model_path: PathLike):
        raise NotImplementedError("TODO")
