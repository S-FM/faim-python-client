from http import HTTPStatus
from typing import Any, cast
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.model_name import ModelName
from ...types import Response


def _get_kwargs(
    model_name: ModelName,
    model_version: str,
) -> dict[str, Any]:
    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/tabular/predict/{model_name}/{model_version}".format(
            model_name=quote(str(model_name), safe=""),
            model_version=quote(str(model_version), safe=""),
        ),
    }

    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Any | HTTPValidationError | None:
    if response.status_code == 200:
        response_200 = cast(Any, None)
        return response_200

    if response.status_code == 422:
        response_422 = HTTPValidationError.from_dict(response.json())

        return response_422

    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Response[Any | HTTPValidationError]:
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
) -> Response[Any | HTTPValidationError]:
    r"""Generate tabular predictions

     Generate predictions for tabular data using the specified model.

        ## Authentication
        Requires valid API key in Authorization header as `Bearer <api_key>`

        ## Request Format
        Apache Arrow IPC Stream (`application/vnd.apache.arrow.stream`)
        - Large arrays (X_train, y_train, X_test) sent as Arrow columns
        - Small parameters (task_type, use_retrieval) sent in schema metadata

        ## Required Inputs
        **Arrays:**
        - `X_train`: Training features, shape (n_train_samples, n_features)
        - `y_train`: Training labels, shape (n_train_samples,) or (n_train_samples, n_targets)
        - `X_test`: Test features, shape (n_test_samples, n_features)

        **Metadata:**
        - `task_type`: Task type - \"Classification\" or \"Regression\" (required)
        - `use_retrieval`: Whether to use retrieval mechanism - boolean (optional, default: False)
        - `compression`: Response compression - \"zstd\" or null (optional, default: \"zstd\")

        ## Response Format
        Apache Arrow IPC Stream with:
        - `predictions`: Model predictions
        - `probabilities`: Class probabilities (Classification only)
        - Billing metadata (transaction_id, cost_amount, cost_currency, token_count)

        ## Supported Models
        - `limix`: LimiX foundation model for tabular data

    Args:
        model_name (ModelName): Available model names for inference.
        model_version (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Any | HTTPValidationError]
    """

    kwargs = _get_kwargs(
        model_name=model_name,
        model_version=model_version,
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
) -> Any | HTTPValidationError | None:
    r"""Generate tabular predictions

     Generate predictions for tabular data using the specified model.

        ## Authentication
        Requires valid API key in Authorization header as `Bearer <api_key>`

        ## Request Format
        Apache Arrow IPC Stream (`application/vnd.apache.arrow.stream`)
        - Large arrays (X_train, y_train, X_test) sent as Arrow columns
        - Small parameters (task_type, use_retrieval) sent in schema metadata

        ## Required Inputs
        **Arrays:**
        - `X_train`: Training features, shape (n_train_samples, n_features)
        - `y_train`: Training labels, shape (n_train_samples,) or (n_train_samples, n_targets)
        - `X_test`: Test features, shape (n_test_samples, n_features)

        **Metadata:**
        - `task_type`: Task type - \"Classification\" or \"Regression\" (required)
        - `use_retrieval`: Whether to use retrieval mechanism - boolean (optional, default: False)
        - `compression`: Response compression - \"zstd\" or null (optional, default: \"zstd\")

        ## Response Format
        Apache Arrow IPC Stream with:
        - `predictions`: Model predictions
        - `probabilities`: Class probabilities (Classification only)
        - Billing metadata (transaction_id, cost_amount, cost_currency, token_count)

        ## Supported Models
        - `limix`: LimiX foundation model for tabular data

    Args:
        model_name (ModelName): Available model names for inference.
        model_version (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Any | HTTPValidationError
    """

    return sync_detailed(
        model_name=model_name,
        model_version=model_version,
        client=client,
    ).parsed


async def asyncio_detailed(
    model_name: ModelName,
    model_version: str,
    *,
    client: AuthenticatedClient | Client,
) -> Response[Any | HTTPValidationError]:
    r"""Generate tabular predictions

     Generate predictions for tabular data using the specified model.

        ## Authentication
        Requires valid API key in Authorization header as `Bearer <api_key>`

        ## Request Format
        Apache Arrow IPC Stream (`application/vnd.apache.arrow.stream`)
        - Large arrays (X_train, y_train, X_test) sent as Arrow columns
        - Small parameters (task_type, use_retrieval) sent in schema metadata

        ## Required Inputs
        **Arrays:**
        - `X_train`: Training features, shape (n_train_samples, n_features)
        - `y_train`: Training labels, shape (n_train_samples,) or (n_train_samples, n_targets)
        - `X_test`: Test features, shape (n_test_samples, n_features)

        **Metadata:**
        - `task_type`: Task type - \"Classification\" or \"Regression\" (required)
        - `use_retrieval`: Whether to use retrieval mechanism - boolean (optional, default: False)
        - `compression`: Response compression - \"zstd\" or null (optional, default: \"zstd\")

        ## Response Format
        Apache Arrow IPC Stream with:
        - `predictions`: Model predictions
        - `probabilities`: Class probabilities (Classification only)
        - Billing metadata (transaction_id, cost_amount, cost_currency, token_count)

        ## Supported Models
        - `limix`: LimiX foundation model for tabular data

    Args:
        model_name (ModelName): Available model names for inference.
        model_version (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Any | HTTPValidationError]
    """

    kwargs = _get_kwargs(
        model_name=model_name,
        model_version=model_version,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    model_name: ModelName,
    model_version: str,
    *,
    client: AuthenticatedClient | Client,
) -> Any | HTTPValidationError | None:
    r"""Generate tabular predictions

     Generate predictions for tabular data using the specified model.

        ## Authentication
        Requires valid API key in Authorization header as `Bearer <api_key>`

        ## Request Format
        Apache Arrow IPC Stream (`application/vnd.apache.arrow.stream`)
        - Large arrays (X_train, y_train, X_test) sent as Arrow columns
        - Small parameters (task_type, use_retrieval) sent in schema metadata

        ## Required Inputs
        **Arrays:**
        - `X_train`: Training features, shape (n_train_samples, n_features)
        - `y_train`: Training labels, shape (n_train_samples,) or (n_train_samples, n_targets)
        - `X_test`: Test features, shape (n_test_samples, n_features)

        **Metadata:**
        - `task_type`: Task type - \"Classification\" or \"Regression\" (required)
        - `use_retrieval`: Whether to use retrieval mechanism - boolean (optional, default: False)
        - `compression`: Response compression - \"zstd\" or null (optional, default: \"zstd\")

        ## Response Format
        Apache Arrow IPC Stream with:
        - `predictions`: Model predictions
        - `probabilities`: Class probabilities (Classification only)
        - Billing metadata (transaction_id, cost_amount, cost_currency, token_count)

        ## Supported Models
        - `limix`: LimiX foundation model for tabular data

    Args:
        model_name (ModelName): Available model names for inference.
        model_version (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Any | HTTPValidationError
    """

    return (
        await asyncio_detailed(
            model_name=model_name,
            model_version=model_version,
            client=client,
        )
    ).parsed
