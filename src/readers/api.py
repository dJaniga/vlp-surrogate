from pathlib import Path


from readers.base import ReaderInterface
from readers.eclipse import EclipseReader

MAPPING = {".unsmry": EclipseReader}


def initialize_reader_from_path(file_path: Path) -> ReaderInterface:
    suffix = file_path.suffix.lower()
    return MAPPING[suffix].from_file(file_path)
