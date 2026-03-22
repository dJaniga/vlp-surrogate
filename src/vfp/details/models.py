from __future__ import annotations

from typing import Dict, ItemsView, Iterator, KeysView, ValuesView

from pydantic import BaseModel, ConfigDict, Field, RootModel, model_validator


class VFPTableDetails(BaseModel):
    table_number: int = Field(alias="table number")
    bhp_depth: float = Field(alias="bhp depth")

    model_config = ConfigDict(populate_by_name=True, extra="forbid")


class StreamVFP(BaseModel):
    VFPPROD: VFPTableDetails
    VFPINJ: VFPTableDetails

    model_config = ConfigDict(extra="forbid")


class VFPDetails(RootModel[Dict[str, StreamVFP]]):
    """
    Root model where the JSON itself is the dict:
      { "<well>": {"VFPPROD": {...}, "VFPINJ": {...}}, ... }
    """

    def __getitem__(self, item: str) -> StreamVFP | None:
        return self.root.get(item, None)

    def __iter__(self) -> Iterator[str]:
        return iter(self.root)

    def __len__(self) -> int:
        return len(self.root)

    def keys(self) -> KeysView[str]:
        return self.root.keys()

    def values(self) -> ValuesView[StreamVFP]:
        return self.root.values()

    def items(self) -> ItemsView[str, StreamVFP]:
        return self.root.items()

    @model_validator(mode="after")
    def table_number_must_be_unique(self) -> dict[str, StreamVFP]:
        table_numbers_vfpprod = set()
        table_numbers_vfpinj = set()

        for well, stream_vfp in self.root.items():
            if stream_vfp.VFPPROD.table_number in table_numbers_vfpprod:
                raise ValueError(
                    f"Duplicate table number {stream_vfp.VFPPROD.table_number} for well {well} (VFPPROD)"
                )
            table_numbers_vfpprod.add(stream_vfp.VFPPROD.table_number)

            if stream_vfp.VFPINJ.table_number in table_numbers_vfpinj:
                raise ValueError(
                    f"Duplicate table number {stream_vfp.VFPINJ.table_number} for well {well} (VFPINJ)"
                )
            table_numbers_vfpinj.add(stream_vfp.VFPINJ.table_number)

        return self
