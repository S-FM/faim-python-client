from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

T = TypeVar("T", bound="DatasetInfoResponse")


@_attrs_define
class DatasetInfoResponse:
    """Dataset information response model.

    Attributes:
        id (str):
        size_bytes (int):
        status (str):
        gcs_path (str):
        created_timestamp (int):
        updated_timestamp (int):
    """

    id: str
    size_bytes: int
    status: str
    gcs_path: str
    created_timestamp: int
    updated_timestamp: int
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        id = self.id

        size_bytes = self.size_bytes

        status = self.status

        gcs_path = self.gcs_path

        created_timestamp = self.created_timestamp

        updated_timestamp = self.updated_timestamp

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "id": id,
                "size_bytes": size_bytes,
                "status": status,
                "gcs_path": gcs_path,
                "created_timestamp": created_timestamp,
                "updated_timestamp": updated_timestamp,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        id = d.pop("id")

        size_bytes = d.pop("size_bytes")

        status = d.pop("status")

        gcs_path = d.pop("gcs_path")

        created_timestamp = d.pop("created_timestamp")

        updated_timestamp = d.pop("updated_timestamp")

        dataset_info_response = cls(
            id=id,
            size_bytes=size_bytes,
            status=status,
            gcs_path=gcs_path,
            created_timestamp=created_timestamp,
            updated_timestamp=updated_timestamp,
        )

        dataset_info_response.additional_properties = d
        return dataset_info_response

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
