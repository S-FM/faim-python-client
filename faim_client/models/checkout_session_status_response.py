from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="CheckoutSessionStatusResponse")


@_attrs_define
class CheckoutSessionStatusResponse:
    """Response model for checkout session status from Stripe API.

    Attributes:
        status (str): Stripe checkout session status
        payment_status (str): Stripe checkout session payment status
        payment_intent_id (None | str | Unset): Stripe payment intent ID
        payment_intent_status (None | str | Unset): Stripe payment intent status
    """

    status: str
    payment_status: str
    payment_intent_id: None | str | Unset = UNSET
    payment_intent_status: None | str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        status = self.status

        payment_status = self.payment_status

        payment_intent_id: None | str | Unset
        if isinstance(self.payment_intent_id, Unset):
            payment_intent_id = UNSET
        else:
            payment_intent_id = self.payment_intent_id

        payment_intent_status: None | str | Unset
        if isinstance(self.payment_intent_status, Unset):
            payment_intent_status = UNSET
        else:
            payment_intent_status = self.payment_intent_status

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "status": status,
                "payment_status": payment_status,
            }
        )
        if payment_intent_id is not UNSET:
            field_dict["payment_intent_id"] = payment_intent_id
        if payment_intent_status is not UNSET:
            field_dict["payment_intent_status"] = payment_intent_status

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        status = d.pop("status")

        payment_status = d.pop("payment_status")

        def _parse_payment_intent_id(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        payment_intent_id = _parse_payment_intent_id(d.pop("payment_intent_id", UNSET))

        def _parse_payment_intent_status(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        payment_intent_status = _parse_payment_intent_status(d.pop("payment_intent_status", UNSET))

        checkout_session_status_response = cls(
            status=status,
            payment_status=payment_status,
            payment_intent_id=payment_intent_id,
            payment_intent_status=payment_intent_status,
        )

        checkout_session_status_response.additional_properties = d
        return checkout_session_status_response

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
