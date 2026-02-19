from pathlib import Path
from typing import Type


from readers.base import ReaderInterface
from readers.eclipse import EclipseReader

MAPPING = {".unsmry": EclipseReader}


def get_reader_by_file_suffix(file_path: Path) -> Type[ReaderInterface]:
    suffix = file_path.suffix.lower()
    return MAPPING[suffix]
