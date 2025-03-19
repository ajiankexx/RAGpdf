from unstructured.partition.auto import partition

def simple_parser(file_path: str):
    elements = partition(file_path)
    return ("\n\n".join([str(el) for el in elements]))