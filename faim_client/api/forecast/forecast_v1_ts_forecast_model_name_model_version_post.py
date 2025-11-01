from http import HTTPStatus
from io import BytesIO
from typing import Any, cast

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.model_name import ModelName
from ...types import File, Response


def _get_kwargs(
    model_name: ModelName,
    model_version: str,
    *,
    body: File,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": f"/v1/ts/forecast/{model_name}/{model_version}",
    }

    _kwargs["content"] = body.payload

    headers["Content-Type"] = "application/octet-stream"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Any | File | None:
    if response.status_code == 200:
        response_200 = File(payload=BytesIO(response.content))

        return response_200

    if response.status_code == 401:
        response_401 = cast(Any, None)
        return response_401

    if response.status_code == 404:
        response_404 = cast(Any, None)
        return response_404

    if response.status_code == 413:
        response_413 = cast(Any, None)
        return response_413

    if response.status_code == 422:
        response_422 = cast(Any, None)
        return response_422

    if response.status_code == 500:
        response_500 = cast(Any, None)
        return response_500

    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Response[Any | File]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    model_name: ModelName,
    model_version: str,
    *,
    client: AuthenticatedClient | Client,
    body: File,
) -> Response[Any | File]:
    r"""Generate model forecast

     Generate time series forecasts using the specified Triton model.

        **Authentication**: Requires valid API key in Authorization header as `Bearer <api_key>`

        **Request Format**: Apache Arrow IPC Stream (`application/vnd.apache.arrow.stream`)
        - Large arrays (x, padding_mask, etc.) sent as Arrow columns
        - Small parameters (horizon, quantiles, etc.) sent in schema metadata

        **Required Inputs**:
        - `x`: Time series data (numpy array)
        - `horizon`: Forecast horizon length (integer, in metadata)
        - `output_type`: Output type - \"point\", \"quantiles\", or \"samples\" (string, in metadata)

        **Model-Specific Inputs**:

        *FlowState*:
        - `scale_factor` (optional): Scaling factor (float)
        - `prediction_type` (optional): Prediction type (string)

        *ToTo*:
        - `padding_mask`: Mask for padded values (numpy array)
        - `id_mask`: Identifier mask (numpy array)
        - `num_samples` (optional): Number of samples (integer)
        - `quantiles` (optional): Quantile levels (list of floats)

        **Response**: Arrow IPC stream with predictions and metadata

    Args:
        model_name (ModelName): Available model names for inference.
        model_version (str):
        body (File): Apache Arrow IPC stream containing input arrays and metadata

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[Any, File]]
    """

    kwargs = _get_kwargs(
        model_name=model_name,
        model_version=model_version,
        body=body,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    model_name: ModelName,
    model_version: str,
    *,
    client: AuthenticatedClient | Client,
    body: File,
) -> Any | File | None:
    r"""Generate model forecast

     Generate time series forecasts using the specified Triton model.

        **Authentication**: Requires valid API key in Authorization header as `Bearer <api_key>`

        **Request Format**: Apache Arrow IPC Stream (`application/vnd.apache.arrow.stream`)
        - Large arrays (x, padding_mask, etc.) sent as Arrow columns
        - Small parameters (horizon, quantiles, etc.) sent in schema metadata

        **Required Inputs**:
        - `x`: Time series data (numpy array)
        - `horizon`: Forecast horizon length (integer, in metadata)
        - `output_type`: Output type - \"point\", \"quantiles\", or \"samples\" (string, in metadata)

        **Model-Specific Inputs**:

        *FlowState*:
        - `scale_factor` (optional): Scaling factor (float)
        - `prediction_type` (optional): Prediction type (string)

        *ToTo*:
        - `padding_mask`: Mask for padded values (numpy array)
        - `id_mask`: Identifier mask (numpy array)
        - `num_samples` (optional): Number of samples (integer)
        - `quantiles` (optional): Quantile levels (list of floats)

        **Response**: Arrow IPC stream with predictions and metadata

    Args:
        model_name (ModelName): Available model names for inference.
        model_version (str):
        body (File): Apache Arrow IPC stream containing input arrays and metadata

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[Any, File]
    """

    return sync_detailed(
        model_name=model_name,
        model_version=model_version,
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    model_name: ModelName,
    model_version: str,
    *,
    client: AuthenticatedClient | Client,
    body: File,
) -> Response[Any | File]:
    r"""Generate model forecast

     Generate time series forecasts using the specified Triton model.

        **Authentication**: Requires valid API key in Authorization header as `Bearer <api_key>`

        **Request Format**: Apache Arrow IPC Stream (`application/vnd.apache.arrow.stream`)
        - Large arrays (x, padding_mask, etc.) sent as Arrow columns
        - Small parameters (horizon, quantiles, etc.) sent in schema metadata

        **Required Inputs**:
        - `x`: Time series data (numpy array)
        - `horizon`: Forecast horizon length (integer, in metadata)
        - `output_type`: Output type - \"point\", \"quantiles\", or \"samples\" (string, in metadata)

        **Model-Specific Inputs**:

        *FlowState*:
        - `scale_factor` (optional): Scaling factor (float)
        - `prediction_type` (optional): Prediction type (string)

        *ToTo*:
        - `padding_mask`: Mask for padded values (numpy array)
        - `id_mask`: Identifier mask (numpy array)
        - `num_samples` (optional): Number of samples (integer)
        - `quantiles` (optional): Quantile levels (list of floats)

        **Response**: Arrow IPC stream with predictions and metadata

    Args:
        model_name (ModelName): Available model names for inference.
        model_version (str):
        body (File): Apache Arrow IPC stream containing input arrays and metadata

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[Any, File]]
    """

    kwargs = _get_kwargs(
        model_name=model_name,
        model_version=model_version,
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    model_name: ModelName,
    model_version: str,
    *,
    client: AuthenticatedClient | Client,
    body: File,
) -> Any | File | None:
    r"""Generate model forecast

     Generate time series forecasts using the specified Triton model.

        **Authentication**: Requires valid API key in Authorization header as `Bearer <api_key>`

        **Request Format**: Apache Arrow IPC Stream (`application/vnd.apache.arrow.stream`)
        - Large arrays (x, padding_mask, etc.) sent as Arrow columns
        - Small parameters (horizon, quantiles, etc.) sent in schema metadata

        **Required Inputs**:
        - `x`: Time series data (numpy array)
        - `horizon`: Forecast horizon length (integer, in metadata)
        - `output_type`: Output type - \"point\", \"quantiles\", or \"samples\" (string, in metadata)

        **Model-Specific Inputs**:

        *FlowState*:
        - `scale_factor` (optional): Scaling factor (float)
        - `prediction_type` (optional): Prediction type (string)

        *ToTo*:
        - `padding_mask`: Mask for padded values (numpy array)
        - `id_mask`: Identifier mask (numpy array)
        - `num_samples` (optional): Number of samples (integer)
        - `quantiles` (optional): Quantile levels (list of floats)

        **Response**: Arrow IPC stream with predictions and metadata

    Args:
        model_name (ModelName): Available model names for inference.
        model_version (str):
        body (File): Apache Arrow IPC stream containing input arrays and metadata

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[Any, File]
    """

    return (
        await asyncio_detailed(
            model_name=model_name,
            model_version=model_version,
            client=client,
            body=body,
        )
    ).parsed
