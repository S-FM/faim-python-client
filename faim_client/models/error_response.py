from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.error_code import ErrorCode
from ..types import UNSET, Unset

T = TypeVar("T", bound="ErrorResponse")


@_attrs_define
class ErrorResponse:
    """Canonical error response structure for the entire FAIM system.

    This model is used by:
    1. Triton Python backends (serialized to JSON string in error metadata)
    2. Proxy Server (returned as HTTP JSON response body)
    3. Client SDK (deserialized from HTTP response)

    Attributes:
        error_code: Machine-readable error code for programmatic handling
        message: Human-readable error message
        detail: Optional detailed explanation (backward compatible with SDK)
        request_id: Request identifier for tracing

        Attributes:
            error_code (ErrorCode): Canonical error codes used across the entire system.

                These codes are stable identifiers that clients can use for
                programmatic error handling (retries, fallbacks, user messaging).
            message (str): Human-readable error message
            detail (None | str | Unset): Detailed error explanation (backward compatible with SDK ErrorResponse.detail)
            request_id (None | str | Unset): Request identifier for distributed tracing
    """

    error_code: ErrorCode
    message: str
    detail: None | str | Unset = UNSET
    request_id: None | str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        error_code = self.error_code.value

        message = self.message

        detail: None | str | Unset
        if isinstance(self.detail, Unset):
            detail = UNSET
        else:
            detail = self.detail

        request_id: None | str | Unset
        if isinstance(self.request_id, Unset):
            request_id = UNSET
        else:
            request_id = self.request_id

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "error_code": error_code,
                "message": message,
            }
        )
        if detail is not UNSET:
            field_dict["detail"] = detail
        if request_id is not UNSET:
            field_dict["request_id"] = request_id

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        error_code = ErrorCode(d.pop("error_code"))

        message = d.pop("message")

        def _parse_detail(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        detail = _parse_detail(d.pop("detail", UNSET))

        def _parse_request_id(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        request_id = _parse_request_id(d.pop("request_id", UNSET))

        error_response = cls(
            error_code=error_code,
            message=message,
            detail=detail,
            request_id=request_id,
        )

        error_response.additional_properties = d
        return error_response

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
