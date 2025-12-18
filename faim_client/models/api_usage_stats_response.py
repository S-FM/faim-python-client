from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

T = TypeVar("T", bound="APIUsageStatsResponse")


@_attrs_define
class APIUsageStatsResponse:
    """Response model for API usage statistics.

    Attributes:
        balance_usd (float): Current account balance in USD
        total_spent_1d_usd (float): Total amount spent in the last 1 day (USD)
        total_spent_7d_usd (float): Total amount spent in the last 7 days (USD)
        total_spent_30d_usd (float): Total amount spent in the last 30 days (USD)
    """

    balance_usd: float
    total_spent_1d_usd: float
    total_spent_7d_usd: float
    total_spent_30d_usd: float
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        balance_usd = self.balance_usd

        total_spent_1d_usd = self.total_spent_1d_usd

        total_spent_7d_usd = self.total_spent_7d_usd

        total_spent_30d_usd = self.total_spent_30d_usd

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "balance_usd": balance_usd,
                "total_spent_1d_usd": total_spent_1d_usd,
                "total_spent_7d_usd": total_spent_7d_usd,
                "total_spent_30d_usd": total_spent_30d_usd,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        balance_usd = d.pop("balance_usd")

        total_spent_1d_usd = d.pop("total_spent_1d_usd")

        total_spent_7d_usd = d.pop("total_spent_7d_usd")

        total_spent_30d_usd = d.pop("total_spent_30d_usd")

        api_usage_stats_response = cls(
            balance_usd=balance_usd,
            total_spent_1d_usd=total_spent_1d_usd,
            total_spent_7d_usd=total_spent_7d_usd,
            total_spent_30d_usd=total_spent_30d_usd,
        )

        api_usage_stats_response.additional_properties = d
        return api_usage_stats_response

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
