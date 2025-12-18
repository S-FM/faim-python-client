from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.tabular_start_upload_metadata import TabularStartUploadMetadata


T = TypeVar("T", bound="StartUploadMetadata")


@_attrs_define
class StartUploadMetadata:
    """Metadata attached to a dataset upload request.

    Attributes:
        tabular (None | TabularStartUploadMetadata | Unset): Metadata for tabular pretrain datasets.
    """

    tabular: None | TabularStartUploadMetadata | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        from ..models.tabular_start_upload_metadata import TabularStartUploadMetadata

        tabular: dict[str, Any] | None | Unset
        if isinstance(self.tabular, Unset):
            tabular = UNSET
        elif isinstance(self.tabular, TabularStartUploadMetadata):
            tabular = self.tabular.to_dict()
        else:
            tabular = self.tabular

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if tabular is not UNSET:
            field_dict["tabular"] = tabular

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.tabular_start_upload_metadata import TabularStartUploadMetadata

        d = dict(src_dict)

        def _parse_tabular(data: object) -> None | TabularStartUploadMetadata | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                tabular_type_0 = TabularStartUploadMetadata.from_dict(data)

                return tabular_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(None | TabularStartUploadMetadata | Unset, data)

        tabular = _parse_tabular(d.pop("tabular", UNSET))

        start_upload_metadata = cls(
            tabular=tabular,
        )

        start_upload_metadata.additional_properties = d
        return start_upload_metadata

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
