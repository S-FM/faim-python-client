from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.dataset_type import DatasetType
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.start_upload_metadata import StartUploadMetadata


T = TypeVar("T", bound="StartUploadRequest")


@_attrs_define
class StartUploadRequest:
    """Request payload for starting a dataset upload.

    Attributes:
        name (str): Human readable dataset name.
        size_bytes (int): Expected dataset size in bytes.
        dataset_type (DatasetType): Dataset purpose classification.
        content_type (None | str | Unset): Content type that will be used for the upload. Defaults to application/octet-
            stream.
        dataset_md5 (None | str | Unset): Optional MD5 checksum of the dataset content (base64-encoded, as in GCS
            md5_hash).
        metadata (None | StartUploadMetadata | Unset): Optional dataset metadata. For TABULAR_PRETRAIN datasets,
            metadata.tabular.columns must be provided.
    """

    name: str
    size_bytes: int
    dataset_type: DatasetType
    content_type: None | str | Unset = UNSET
    dataset_md5: None | str | Unset = UNSET
    metadata: None | StartUploadMetadata | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        from ..models.start_upload_metadata import StartUploadMetadata

        name = self.name

        size_bytes = self.size_bytes

        dataset_type = self.dataset_type.value

        content_type: None | str | Unset
        if isinstance(self.content_type, Unset):
            content_type = UNSET
        else:
            content_type = self.content_type

        dataset_md5: None | str | Unset
        if isinstance(self.dataset_md5, Unset):
            dataset_md5 = UNSET
        else:
            dataset_md5 = self.dataset_md5

        metadata: dict[str, Any] | None | Unset
        if isinstance(self.metadata, Unset):
            metadata = UNSET
        elif isinstance(self.metadata, StartUploadMetadata):
            metadata = self.metadata.to_dict()
        else:
            metadata = self.metadata

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "name": name,
                "size_bytes": size_bytes,
                "dataset_type": dataset_type,
            }
        )
        if content_type is not UNSET:
            field_dict["content_type"] = content_type
        if dataset_md5 is not UNSET:
            field_dict["dataset_md5"] = dataset_md5
        if metadata is not UNSET:
            field_dict["metadata"] = metadata

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.start_upload_metadata import StartUploadMetadata

        d = dict(src_dict)
        name = d.pop("name")

        size_bytes = d.pop("size_bytes")

        dataset_type = DatasetType(d.pop("dataset_type"))

        def _parse_content_type(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        content_type = _parse_content_type(d.pop("content_type", UNSET))

        def _parse_dataset_md5(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        dataset_md5 = _parse_dataset_md5(d.pop("dataset_md5", UNSET))

        def _parse_metadata(data: object) -> None | StartUploadMetadata | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                metadata_type_0 = StartUploadMetadata.from_dict(data)

                return metadata_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(None | StartUploadMetadata | Unset, data)

        metadata = _parse_metadata(d.pop("metadata", UNSET))

        start_upload_request = cls(
            name=name,
            size_bytes=size_bytes,
            dataset_type=dataset_type,
            content_type=content_type,
            dataset_md5=dataset_md5,
            metadata=metadata,
        )

        start_upload_request.additional_properties = d
        return start_upload_request

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
