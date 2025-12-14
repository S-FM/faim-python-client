from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

if TYPE_CHECKING:
    from ..models.dataset_list_item_response import DatasetListItemResponse


T = TypeVar("T", bound="DatasetListResponse")


@_attrs_define
class DatasetListResponse:
    """Dataset list response model.

    Attributes:
        dataset (list[DatasetListItemResponse]):
    """

    dataset: list[DatasetListItemResponse]
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        dataset = []
        for dataset_item_data in self.dataset:
            dataset_item = dataset_item_data.to_dict()
            dataset.append(dataset_item)

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "dataset": dataset,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.dataset_list_item_response import DatasetListItemResponse

        d = dict(src_dict)
        dataset = []
        _dataset = d.pop("dataset")
        for dataset_item_data in _dataset:
            dataset_item = DatasetListItemResponse.from_dict(dataset_item_data)

            dataset.append(dataset_item)

        dataset_list_response = cls(
            dataset=dataset,
        )

        dataset_list_response.additional_properties = d
        return dataset_list_response

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
