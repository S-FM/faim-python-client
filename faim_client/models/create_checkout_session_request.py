from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

T = TypeVar("T", bound="CreateCheckoutSessionRequest")


@_attrs_define
class CreateCheckoutSessionRequest:
    """Request model for creating a checkout session.

    Attributes:
        amount_usd_cents (int): Amount in USD cents
    """

    amount_usd_cents: int
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        amount_usd_cents = self.amount_usd_cents

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "amount_usd_cents": amount_usd_cents,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        amount_usd_cents = d.pop("amount_usd_cents")

        create_checkout_session_request = cls(
            amount_usd_cents=amount_usd_cents,
        )

        create_checkout_session_request.additional_properties = d
        return create_checkout_session_request

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
