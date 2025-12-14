from http import HTTPStatus
from io import BytesIO
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.error_response import ErrorResponse
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
        "url": "/v1/ts/forecast/{model_name}/{model_version}".format(
            model_name=quote(str(model_name), safe=""),
            model_version=quote(str(model_version), safe=""),
        ),
    }

    _kwargs["content"] = body.payload

    headers["Content-Type"] = "application/octet-stream"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(*, client: AuthenticatedClient | Client, response: httpx.Response) -> ErrorResponse | File | None:
    if response.status_code == 200:
        response_200 = File(payload=BytesIO(response.content))

        return response_200

    if response.status_code == 401:
        response_401 = ErrorResponse.from_dict(response.json())

        return response_401

    if response.status_code == 402:
        response_402 = ErrorResponse.from_dict(response.json())

        return response_402

    if response.status_code == 404:
        response_404 = ErrorResponse.from_dict(response.json())

        return response_404

    if response.status_code == 413:
        response_413 = ErrorResponse.from_dict(response.json())

        return response_413

    if response.status_code == 422:
        response_422 = ErrorResponse.from_dict(response.json())

        return response_422

    if response.status_code == 500:
        response_500 = ErrorResponse.from_dict(response.json())

        return response_500

    if response.status_code == 503:
        response_503 = ErrorResponse.from_dict(response.json())

        return response_503

    if response.status_code == 504:
        response_504 = ErrorResponse.from_dict(response.json())

        return response_504

    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Response[ErrorResponse | File]:
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
) -> Response[ErrorResponse | File]:
    r"""Generate time series forecast

     Generate time series forecasts using the specified Triton model.

        ## Authentication
        Requires valid API key in Authorization header as `Bearer <api_key>`
        - API keys can be created via `/v1/user/api-keys/create` endpoint
        - Keys must follow the format: `sk-<public_id>-<secret>`
        - Invalid or expired keys return 401 INVALID_API_KEY

        ## Request Format
        Apache Arrow IPC Stream (`application/vnd.apache.arrow.stream`)
        - Large arrays (x, padding_mask, etc.) sent as Arrow columns
        - Small parameters (horizon, quantiles, etc.) sent in schema metadata
        - Request timeout: 20 seconds (configurable on server)
        - Maximum request size: 100MB (configurable on server)

        ## Required Inputs
        - `x`: Time series data (numpy array, 1D or 2D)
        - `horizon`: Forecast horizon length (integer, positive, in metadata)
        - `output_type`: Output type (string, in metadata)
          - Valid values: `\"point\"`, `\"quantiles\"`, `\"samples\"`
          - Determines output format in response

        ## Model-Specific Inputs

        ### FlowState Model
        - `scale_factor` (float, optional): Scaling factor for input time series (default: None)
        - `prediction_type` (string, optional): Type of prediction (default: None)

        ### Chronos2 Model
        - `quantiles` (float array, optional): Quantile levels for quantiles output (list of floats,
    0.0-1.0)

        ### TiRex Model

        ## Response Format
        Successful responses return Apache Arrow IPC stream with:

        ### Common Response Metadata (All Models)
        - `model_name` (string): Name of model used
        - `model_version` (string): Version of model used
        - `transaction_id` (string): Unique identifier for billing tracking
        - `cost_amount` (string): Amount charged for this inference (in currency units)
        - `cost_currency` (string): Currency code (e.g., \"USD\")

        ### Response Output Arrays by Model & output_type

        **FlowState:**
        - `output_type=\"point\"`: Single point forecast array (shape: batch x horizon)
        - `output_type=\"quantiles\"`: Quantile forecasts array (shape: batch x horizon x num_quantiles)

        **Chronos2:**
        - `output_type=\"point\"`: Point forecast array (shape: batch x horizon)
        - `output_type=\"quantiles\"`: Quantile predictions array (shape: batch x horizon x
    num_quantiles)

        **TiRex:**
        - `output_type=\"point\"`: Point forecast array (shape: batch x horizon)
        - `output_type=\"quantiles\"`: Quantile predictions array (shape: batch x horizon x
    num_quantiles)

        ## Error Handling
        All errors return `ErrorResponse` (HTTP JSON) with:
        - `error_code`: Machine-readable code for programmatic handling (see ErrorCode enum)
        - `message`: Human-readable error message
        - `detail`: Optional detailed explanation
        - `request_id`: Request identifier for debugging

        ### Error Categories

        **Authentication & Authorization (401, 403)**
        - AUTHENTICATION_REQUIRED: Missing or malformed Authorization header
        - AUTHENTICATION_FAILED: Bearer format invalid
        - INVALID_API_KEY: API key not found, expired, or revoked
        - AUTHORIZATION_FAILED: Valid credentials but insufficient permissions

        **Validation Errors (422)**
        - MISSING_REQUIRED_FIELD: Required input missing (x, horizon, output_type)
        - INVALID_PARAMETER: Parameter out of valid range or invalid type
        - INVALID_SHAPE: Input tensor shape incompatible with model
        - INVALID_DTYPE: Input data type incorrect for model
        - INVALID_VALUE_RANGE: Input values outside acceptable range
        - VALIDATION_ERROR: Invalid Arrow format or other validation failure

        **Resource Errors (404, 402)**
        - MODEL_NOT_FOUND: Model or version doesn't exist
        - PRICING_NOT_FOUND: Pricing configuration missing for model/version
        - INSUFFICIENT_FUNDS: User billing account balance insufficient
        - REQUEST_TOO_LARGE: Request payload exceeds 100MB limit

        **Inference Errors (500, 503, 504)**
        - INFERENCE_ERROR: Model inference failed (check detail for reason)
        - TIMEOUT_ERROR: Inference exceeded 20 second deadline
        - OUT_OF_MEMORY: GPU/CPU memory exhausted during inference
        - RESOURCE_EXHAUSTED: Compute resources temporarily unavailable
        - MODEL_INITIALIZATION_ERROR: Model failed to load or initialize

        **System Errors (500, 503)**
        - TRITON_CONNECTION_ERROR: Cannot connect to Triton server
        - DATABASE_ERROR: Database operation failed
        - INTERNAL_SERVER_ERROR: Unexpected server error
        - BILLING_TRANSACTION_FAILED: Billing system error (insufficient funds, etc.)

        ### Retry Strategy
        **Retryable Errors** (transient, safe to retry with exponential backoff):
        - TIMEOUT_ERROR (504): Retry after 2-5 seconds
        - OUT_OF_MEMORY (503): Retry after 5-10 seconds or reduce batch size
        - RESOURCE_EXHAUSTED (503): Retry after 5-10 seconds
        - TRITON_CONNECTION_ERROR (503): Retry after 5-10 seconds
        - DATABASE_ERROR (500): Retry after 2-5 seconds

        **Non-Retryable Errors** (permanent failures, don't retry):
        - All 401/403 auth errors: Fix credentials and resubmit
        - All 404 errors: Resource doesn't exist, fix request
        - All 422 validation errors: Fix input data and resubmit
        - INSUFFICIENT_FUNDS (402): Add credit to account
        - REQUEST_TOO_LARGE (413): Reduce payload size
        - INFERENCE_ERROR (500): Check error detail, may require model update

    Args:
        model_name (ModelName): Available model names for inference.
        model_version (str):
        body (File): Apache Arrow IPC stream containing input arrays and metadata

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ErrorResponse | File]
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
) -> ErrorResponse | File | None:
    r"""Generate time series forecast

     Generate time series forecasts using the specified Triton model.

        ## Authentication
        Requires valid API key in Authorization header as `Bearer <api_key>`
        - API keys can be created via `/v1/user/api-keys/create` endpoint
        - Keys must follow the format: `sk-<public_id>-<secret>`
        - Invalid or expired keys return 401 INVALID_API_KEY

        ## Request Format
        Apache Arrow IPC Stream (`application/vnd.apache.arrow.stream`)
        - Large arrays (x, padding_mask, etc.) sent as Arrow columns
        - Small parameters (horizon, quantiles, etc.) sent in schema metadata
        - Request timeout: 20 seconds (configurable on server)
        - Maximum request size: 100MB (configurable on server)

        ## Required Inputs
        - `x`: Time series data (numpy array, 1D or 2D)
        - `horizon`: Forecast horizon length (integer, positive, in metadata)
        - `output_type`: Output type (string, in metadata)
          - Valid values: `\"point\"`, `\"quantiles\"`, `\"samples\"`
          - Determines output format in response

        ## Model-Specific Inputs

        ### FlowState Model
        - `scale_factor` (float, optional): Scaling factor for input time series (default: None)
        - `prediction_type` (string, optional): Type of prediction (default: None)

        ### Chronos2 Model
        - `quantiles` (float array, optional): Quantile levels for quantiles output (list of floats,
    0.0-1.0)

        ### TiRex Model

        ## Response Format
        Successful responses return Apache Arrow IPC stream with:

        ### Common Response Metadata (All Models)
        - `model_name` (string): Name of model used
        - `model_version` (string): Version of model used
        - `transaction_id` (string): Unique identifier for billing tracking
        - `cost_amount` (string): Amount charged for this inference (in currency units)
        - `cost_currency` (string): Currency code (e.g., \"USD\")

        ### Response Output Arrays by Model & output_type

        **FlowState:**
        - `output_type=\"point\"`: Single point forecast array (shape: batch x horizon)
        - `output_type=\"quantiles\"`: Quantile forecasts array (shape: batch x horizon x num_quantiles)

        **Chronos2:**
        - `output_type=\"point\"`: Point forecast array (shape: batch x horizon)
        - `output_type=\"quantiles\"`: Quantile predictions array (shape: batch x horizon x
    num_quantiles)

        **TiRex:**
        - `output_type=\"point\"`: Point forecast array (shape: batch x horizon)
        - `output_type=\"quantiles\"`: Quantile predictions array (shape: batch x horizon x
    num_quantiles)

        ## Error Handling
        All errors return `ErrorResponse` (HTTP JSON) with:
        - `error_code`: Machine-readable code for programmatic handling (see ErrorCode enum)
        - `message`: Human-readable error message
        - `detail`: Optional detailed explanation
        - `request_id`: Request identifier for debugging

        ### Error Categories

        **Authentication & Authorization (401, 403)**
        - AUTHENTICATION_REQUIRED: Missing or malformed Authorization header
        - AUTHENTICATION_FAILED: Bearer format invalid
        - INVALID_API_KEY: API key not found, expired, or revoked
        - AUTHORIZATION_FAILED: Valid credentials but insufficient permissions

        **Validation Errors (422)**
        - MISSING_REQUIRED_FIELD: Required input missing (x, horizon, output_type)
        - INVALID_PARAMETER: Parameter out of valid range or invalid type
        - INVALID_SHAPE: Input tensor shape incompatible with model
        - INVALID_DTYPE: Input data type incorrect for model
        - INVALID_VALUE_RANGE: Input values outside acceptable range
        - VALIDATION_ERROR: Invalid Arrow format or other validation failure

        **Resource Errors (404, 402)**
        - MODEL_NOT_FOUND: Model or version doesn't exist
        - PRICING_NOT_FOUND: Pricing configuration missing for model/version
        - INSUFFICIENT_FUNDS: User billing account balance insufficient
        - REQUEST_TOO_LARGE: Request payload exceeds 100MB limit

        **Inference Errors (500, 503, 504)**
        - INFERENCE_ERROR: Model inference failed (check detail for reason)
        - TIMEOUT_ERROR: Inference exceeded 20 second deadline
        - OUT_OF_MEMORY: GPU/CPU memory exhausted during inference
        - RESOURCE_EXHAUSTED: Compute resources temporarily unavailable
        - MODEL_INITIALIZATION_ERROR: Model failed to load or initialize

        **System Errors (500, 503)**
        - TRITON_CONNECTION_ERROR: Cannot connect to Triton server
        - DATABASE_ERROR: Database operation failed
        - INTERNAL_SERVER_ERROR: Unexpected server error
        - BILLING_TRANSACTION_FAILED: Billing system error (insufficient funds, etc.)

        ### Retry Strategy
        **Retryable Errors** (transient, safe to retry with exponential backoff):
        - TIMEOUT_ERROR (504): Retry after 2-5 seconds
        - OUT_OF_MEMORY (503): Retry after 5-10 seconds or reduce batch size
        - RESOURCE_EXHAUSTED (503): Retry after 5-10 seconds
        - TRITON_CONNECTION_ERROR (503): Retry after 5-10 seconds
        - DATABASE_ERROR (500): Retry after 2-5 seconds

        **Non-Retryable Errors** (permanent failures, don't retry):
        - All 401/403 auth errors: Fix credentials and resubmit
        - All 404 errors: Resource doesn't exist, fix request
        - All 422 validation errors: Fix input data and resubmit
        - INSUFFICIENT_FUNDS (402): Add credit to account
        - REQUEST_TOO_LARGE (413): Reduce payload size
        - INFERENCE_ERROR (500): Check error detail, may require model update

    Args:
        model_name (ModelName): Available model names for inference.
        model_version (str):
        body (File): Apache Arrow IPC stream containing input arrays and metadata

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ErrorResponse | File
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
) -> Response[ErrorResponse | File]:
    r"""Generate time series forecast

     Generate time series forecasts using the specified Triton model.

        ## Authentication
        Requires valid API key in Authorization header as `Bearer <api_key>`
        - API keys can be created via `/v1/user/api-keys/create` endpoint
        - Keys must follow the format: `sk-<public_id>-<secret>`
        - Invalid or expired keys return 401 INVALID_API_KEY

        ## Request Format
        Apache Arrow IPC Stream (`application/vnd.apache.arrow.stream`)
        - Large arrays (x, padding_mask, etc.) sent as Arrow columns
        - Small parameters (horizon, quantiles, etc.) sent in schema metadata
        - Request timeout: 20 seconds (configurable on server)
        - Maximum request size: 100MB (configurable on server)

        ## Required Inputs
        - `x`: Time series data (numpy array, 1D or 2D)
        - `horizon`: Forecast horizon length (integer, positive, in metadata)
        - `output_type`: Output type (string, in metadata)
          - Valid values: `\"point\"`, `\"quantiles\"`, `\"samples\"`
          - Determines output format in response

        ## Model-Specific Inputs

        ### FlowState Model
        - `scale_factor` (float, optional): Scaling factor for input time series (default: None)
        - `prediction_type` (string, optional): Type of prediction (default: None)

        ### Chronos2 Model
        - `quantiles` (float array, optional): Quantile levels for quantiles output (list of floats,
    0.0-1.0)

        ### TiRex Model

        ## Response Format
        Successful responses return Apache Arrow IPC stream with:

        ### Common Response Metadata (All Models)
        - `model_name` (string): Name of model used
        - `model_version` (string): Version of model used
        - `transaction_id` (string): Unique identifier for billing tracking
        - `cost_amount` (string): Amount charged for this inference (in currency units)
        - `cost_currency` (string): Currency code (e.g., \"USD\")

        ### Response Output Arrays by Model & output_type

        **FlowState:**
        - `output_type=\"point\"`: Single point forecast array (shape: batch x horizon)
        - `output_type=\"quantiles\"`: Quantile forecasts array (shape: batch x horizon x num_quantiles)

        **Chronos2:**
        - `output_type=\"point\"`: Point forecast array (shape: batch x horizon)
        - `output_type=\"quantiles\"`: Quantile predictions array (shape: batch x horizon x
    num_quantiles)

        **TiRex:**
        - `output_type=\"point\"`: Point forecast array (shape: batch x horizon)
        - `output_type=\"quantiles\"`: Quantile predictions array (shape: batch x horizon x
    num_quantiles)

        ## Error Handling
        All errors return `ErrorResponse` (HTTP JSON) with:
        - `error_code`: Machine-readable code for programmatic handling (see ErrorCode enum)
        - `message`: Human-readable error message
        - `detail`: Optional detailed explanation
        - `request_id`: Request identifier for debugging

        ### Error Categories

        **Authentication & Authorization (401, 403)**
        - AUTHENTICATION_REQUIRED: Missing or malformed Authorization header
        - AUTHENTICATION_FAILED: Bearer format invalid
        - INVALID_API_KEY: API key not found, expired, or revoked
        - AUTHORIZATION_FAILED: Valid credentials but insufficient permissions

        **Validation Errors (422)**
        - MISSING_REQUIRED_FIELD: Required input missing (x, horizon, output_type)
        - INVALID_PARAMETER: Parameter out of valid range or invalid type
        - INVALID_SHAPE: Input tensor shape incompatible with model
        - INVALID_DTYPE: Input data type incorrect for model
        - INVALID_VALUE_RANGE: Input values outside acceptable range
        - VALIDATION_ERROR: Invalid Arrow format or other validation failure

        **Resource Errors (404, 402)**
        - MODEL_NOT_FOUND: Model or version doesn't exist
        - PRICING_NOT_FOUND: Pricing configuration missing for model/version
        - INSUFFICIENT_FUNDS: User billing account balance insufficient
        - REQUEST_TOO_LARGE: Request payload exceeds 100MB limit

        **Inference Errors (500, 503, 504)**
        - INFERENCE_ERROR: Model inference failed (check detail for reason)
        - TIMEOUT_ERROR: Inference exceeded 20 second deadline
        - OUT_OF_MEMORY: GPU/CPU memory exhausted during inference
        - RESOURCE_EXHAUSTED: Compute resources temporarily unavailable
        - MODEL_INITIALIZATION_ERROR: Model failed to load or initialize

        **System Errors (500, 503)**
        - TRITON_CONNECTION_ERROR: Cannot connect to Triton server
        - DATABASE_ERROR: Database operation failed
        - INTERNAL_SERVER_ERROR: Unexpected server error
        - BILLING_TRANSACTION_FAILED: Billing system error (insufficient funds, etc.)

        ### Retry Strategy
        **Retryable Errors** (transient, safe to retry with exponential backoff):
        - TIMEOUT_ERROR (504): Retry after 2-5 seconds
        - OUT_OF_MEMORY (503): Retry after 5-10 seconds or reduce batch size
        - RESOURCE_EXHAUSTED (503): Retry after 5-10 seconds
        - TRITON_CONNECTION_ERROR (503): Retry after 5-10 seconds
        - DATABASE_ERROR (500): Retry after 2-5 seconds

        **Non-Retryable Errors** (permanent failures, don't retry):
        - All 401/403 auth errors: Fix credentials and resubmit
        - All 404 errors: Resource doesn't exist, fix request
        - All 422 validation errors: Fix input data and resubmit
        - INSUFFICIENT_FUNDS (402): Add credit to account
        - REQUEST_TOO_LARGE (413): Reduce payload size
        - INFERENCE_ERROR (500): Check error detail, may require model update

    Args:
        model_name (ModelName): Available model names for inference.
        model_version (str):
        body (File): Apache Arrow IPC stream containing input arrays and metadata

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ErrorResponse | File]
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
) -> ErrorResponse | File | None:
    r"""Generate time series forecast

     Generate time series forecasts using the specified Triton model.

        ## Authentication
        Requires valid API key in Authorization header as `Bearer <api_key>`
        - API keys can be created via `/v1/user/api-keys/create` endpoint
        - Keys must follow the format: `sk-<public_id>-<secret>`
        - Invalid or expired keys return 401 INVALID_API_KEY

        ## Request Format
        Apache Arrow IPC Stream (`application/vnd.apache.arrow.stream`)
        - Large arrays (x, padding_mask, etc.) sent as Arrow columns
        - Small parameters (horizon, quantiles, etc.) sent in schema metadata
        - Request timeout: 20 seconds (configurable on server)
        - Maximum request size: 100MB (configurable on server)

        ## Required Inputs
        - `x`: Time series data (numpy array, 1D or 2D)
        - `horizon`: Forecast horizon length (integer, positive, in metadata)
        - `output_type`: Output type (string, in metadata)
          - Valid values: `\"point\"`, `\"quantiles\"`, `\"samples\"`
          - Determines output format in response

        ## Model-Specific Inputs

        ### FlowState Model
        - `scale_factor` (float, optional): Scaling factor for input time series (default: None)
        - `prediction_type` (string, optional): Type of prediction (default: None)

        ### Chronos2 Model
        - `quantiles` (float array, optional): Quantile levels for quantiles output (list of floats,
    0.0-1.0)

        ### TiRex Model

        ## Response Format
        Successful responses return Apache Arrow IPC stream with:

        ### Common Response Metadata (All Models)
        - `model_name` (string): Name of model used
        - `model_version` (string): Version of model used
        - `transaction_id` (string): Unique identifier for billing tracking
        - `cost_amount` (string): Amount charged for this inference (in currency units)
        - `cost_currency` (string): Currency code (e.g., \"USD\")

        ### Response Output Arrays by Model & output_type

        **FlowState:**
        - `output_type=\"point\"`: Single point forecast array (shape: batch x horizon)
        - `output_type=\"quantiles\"`: Quantile forecasts array (shape: batch x horizon x num_quantiles)

        **Chronos2:**
        - `output_type=\"point\"`: Point forecast array (shape: batch x horizon)
        - `output_type=\"quantiles\"`: Quantile predictions array (shape: batch x horizon x
    num_quantiles)

        **TiRex:**
        - `output_type=\"point\"`: Point forecast array (shape: batch x horizon)
        - `output_type=\"quantiles\"`: Quantile predictions array (shape: batch x horizon x
    num_quantiles)

        ## Error Handling
        All errors return `ErrorResponse` (HTTP JSON) with:
        - `error_code`: Machine-readable code for programmatic handling (see ErrorCode enum)
        - `message`: Human-readable error message
        - `detail`: Optional detailed explanation
        - `request_id`: Request identifier for debugging

        ### Error Categories

        **Authentication & Authorization (401, 403)**
        - AUTHENTICATION_REQUIRED: Missing or malformed Authorization header
        - AUTHENTICATION_FAILED: Bearer format invalid
        - INVALID_API_KEY: API key not found, expired, or revoked
        - AUTHORIZATION_FAILED: Valid credentials but insufficient permissions

        **Validation Errors (422)**
        - MISSING_REQUIRED_FIELD: Required input missing (x, horizon, output_type)
        - INVALID_PARAMETER: Parameter out of valid range or invalid type
        - INVALID_SHAPE: Input tensor shape incompatible with model
        - INVALID_DTYPE: Input data type incorrect for model
        - INVALID_VALUE_RANGE: Input values outside acceptable range
        - VALIDATION_ERROR: Invalid Arrow format or other validation failure

        **Resource Errors (404, 402)**
        - MODEL_NOT_FOUND: Model or version doesn't exist
        - PRICING_NOT_FOUND: Pricing configuration missing for model/version
        - INSUFFICIENT_FUNDS: User billing account balance insufficient
        - REQUEST_TOO_LARGE: Request payload exceeds 100MB limit

        **Inference Errors (500, 503, 504)**
        - INFERENCE_ERROR: Model inference failed (check detail for reason)
        - TIMEOUT_ERROR: Inference exceeded 20 second deadline
        - OUT_OF_MEMORY: GPU/CPU memory exhausted during inference
        - RESOURCE_EXHAUSTED: Compute resources temporarily unavailable
        - MODEL_INITIALIZATION_ERROR: Model failed to load or initialize

        **System Errors (500, 503)**
        - TRITON_CONNECTION_ERROR: Cannot connect to Triton server
        - DATABASE_ERROR: Database operation failed
        - INTERNAL_SERVER_ERROR: Unexpected server error
        - BILLING_TRANSACTION_FAILED: Billing system error (insufficient funds, etc.)

        ### Retry Strategy
        **Retryable Errors** (transient, safe to retry with exponential backoff):
        - TIMEOUT_ERROR (504): Retry after 2-5 seconds
        - OUT_OF_MEMORY (503): Retry after 5-10 seconds or reduce batch size
        - RESOURCE_EXHAUSTED (503): Retry after 5-10 seconds
        - TRITON_CONNECTION_ERROR (503): Retry after 5-10 seconds
        - DATABASE_ERROR (500): Retry after 2-5 seconds

        **Non-Retryable Errors** (permanent failures, don't retry):
        - All 401/403 auth errors: Fix credentials and resubmit
        - All 404 errors: Resource doesn't exist, fix request
        - All 422 validation errors: Fix input data and resubmit
        - INSUFFICIENT_FUNDS (402): Add credit to account
        - REQUEST_TOO_LARGE (413): Reduce payload size
        - INFERENCE_ERROR (500): Check error detail, may require model update

    Args:
        model_name (ModelName): Available model names for inference.
        model_version (str):
        body (File): Apache Arrow IPC stream containing input arrays and metadata

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ErrorResponse | File
    """

    return (
        await asyncio_detailed(
            model_name=model_name,
            model_version=model_version,
            client=client,
            body=body,
        )
    ).parsed
