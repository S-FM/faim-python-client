from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

T = TypeVar("T", bound="DatasetListItemResponse")


@_attrs_define
class DatasetListItemResponse:
    """Dataset list item for overview responses.

    Attributes:
        id (str):
        name (str):
        created_timestamp (int):
        updated_timestamp (int):
        size_bytes (int):
    """

    id: str
    name: str
    created_timestamp: int
    updated_timestamp: int
    size_bytes: int
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        id = self.id

        name = self.name

        created_timestamp = self.created_timestamp

        updated_timestamp = self.updated_timestamp

        size_bytes = self.size_bytes

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "id": id,
                "name": name,
                "created_timestamp": created_timestamp,
                "updated_timestamp": updated_timestamp,
                "size_bytes": size_bytes,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        id = d.pop("id")

        name = d.pop("name")

        created_timestamp = d.pop("created_timestamp")

        updated_timestamp = d.pop("updated_timestamp")

        size_bytes = d.pop("size_bytes")

        dataset_list_item_response = cls(
            id=id,
            name=name,
            created_timestamp=created_timestamp,
            updated_timestamp=updated_timestamp,
            size_bytes=size_bytes,
        )

        dataset_list_item_response.additional_properties = d
        return dataset_list_item_response

    @property
    def additional_keys(self) -> list[str]:
        return list(self.additional_properties.keys())

    def __getitem__(self, key: str) -> Any:
        return self.additional_properties[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.additional_properties[key] = value

    def __delitem__(self, key: str) -> None:
        del self.additional_properties[key]

    def __contains__(self, key: str) -> bool:
        return key in self.additional_properties
