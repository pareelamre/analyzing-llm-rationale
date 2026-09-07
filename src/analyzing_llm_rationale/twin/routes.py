"""Owner-derived control API for autonomous mandate lifecycle."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from typing import Any, Callable, Mapping, Optional

from fastapi import APIRouter, HTTPException, Request
from opentelemetry import metrics, trace
from pydantic import BaseModel, ConfigDict, Field

from .mandates import (
    Mandate,
    MandateConflict,
    MandateError,
    MandateStore,
    approve,
    revise,
    revoke,
)
from .models import AccountScope

tracer = trace.get_tracer(__name__)
mandate_operations = metrics.get_meter(__name__).create_counter("twin.mandate.operations", unit="1")


class MandateDraftRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    client_request_id: str = Field(min_length=8, max_length=128)
    account_scope_id: str = Field(min_length=1, max_length=256)
    strategy_version: str = Field(min_length=1, max_length=128)
    expires_at: datetime
    live: bool = False
    allowed_actions: tuple[str, ...] = ("BUY_YES", "BUY_NO")
    max_capital: str = "0"
    max_loss: str = "0"
    max_model_usd: str = "0"
    max_model_tokens: int = Field(default=0, ge=0)
    max_model_requests: int = Field(default=0, ge=0)


class MandateApprovalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_hash: str = Field(min_length=64, max_length=64)
    idempotency_key: str = Field(min_length=8, max_length=128)


class MandateTransitionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    idempotency_key: str = Field(min_length=8, max_length=128)


class MandateRevisionRequest(MandateDraftRequest):
    pass


@dataclass(frozen=True)
class MandateRuntime:
    identity_hash: str
    release_hash: str
    config_hash: str
    model_hash: str
    readiness_hash: Optional[str]
    readiness_artifact: Optional[Mapping[str, Any]]


OwnerResolver = Callable[[Request], str]
ScopeResolver = Callable[[str], AccountScope]
RuntimeResolver = Callable[[str, AccountScope], MandateRuntime]
Clock = Callable[[], datetime]


class MandateService:
    def __init__(
        self, store: MandateStore, *, resolve_scope: ScopeResolver,
        resolve_runtime: RuntimeResolver, clock: Clock,
    ) -> None:
        self.store = store
        self.resolve_scope = resolve_scope
        self.resolve_runtime = resolve_runtime
        self.clock = clock

    def _scope(self, owner_id: str, scope_id: str) -> AccountScope:
        scope = self.resolve_scope(scope_id)
        if scope.owner_id != owner_id:
            raise PermissionError("account scope does not belong to the authenticated owner")
        return scope

    @staticmethod
    def _id(owner_id: str, request_id: str) -> str:
        return "mandate-" + sha256(f"{owner_id}:{request_id}".encode()).hexdigest()[:24]

    @tracer.start_as_current_span("twin.mandate.create")
    def create(self, owner_id: str, request: MandateDraftRequest) -> Mandate:
        now = self.clock()
        scope = self._scope(owner_id, request.account_scope_id)
        if request.expires_at.tzinfo is None or request.expires_at <= now:
            raise MandateError("mandate expiry must be in the future")
        if request.live != (scope.environment == "live"):
            raise MandateError("mandate environment must match the registered account scope")
        runtime = self.resolve_runtime(owner_id, scope)
        mandate = Mandate(
            self._id(owner_id, request.client_request_id), owner_id, scope.id,
            request.strategy_version, request.expires_at, live=request.live,
            account_epoch=scope.account_epoch, venue=scope.venue,
            allowed_actions=request.allowed_actions, max_capital=request.max_capital,
            max_loss=request.max_loss, model_hash=runtime.model_hash,
            config_hash=runtime.config_hash, readiness_hash=runtime.readiness_hash,
            release_hash=runtime.release_hash, max_model_usd=request.max_model_usd,
            max_model_tokens=request.max_model_tokens,
            max_model_requests=request.max_model_requests,
            identity_hash=runtime.identity_hash, created_at=now,
        )
        result = self.store.create(mandate)
        mandate_operations.add(1, {"operation": "draft", "environment": scope.environment})
        return result

    @tracer.start_as_current_span("twin.mandate.revise")
    def revise(self, owner_id: str, mandate_id: str, request: MandateRevisionRequest) -> Mandate:
        current = self.require(owner_id, mandate_id)
        scope = self._scope(owner_id, request.account_scope_id)
        now = self.clock()
        if request.expires_at.tzinfo is None or request.expires_at <= now:
            raise MandateError("mandate expiry must be in the future")
        if request.live != (scope.environment == "live"):
            raise MandateError("mandate environment must match the registered account scope")
        runtime = self.resolve_runtime(owner_id, scope)
        updated = revise(
            current, account_scope_id=scope.id, account_epoch=scope.account_epoch,
            venue=scope.venue, strategy_version=request.strategy_version,
            expires_at=request.expires_at, live=request.live,
            allowed_actions=request.allowed_actions, max_capital=request.max_capital,
            max_loss=request.max_loss, max_model_usd=request.max_model_usd,
            max_model_tokens=request.max_model_tokens, max_model_requests=request.max_model_requests,
            model_hash=runtime.model_hash, config_hash=runtime.config_hash,
            readiness_hash=runtime.readiness_hash, release_hash=runtime.release_hash,
            identity_hash=runtime.identity_hash, created_at=now,
        )
        result = self.store.save_transition(current, updated, idempotency_key=request.client_request_id)
        mandate_operations.add(1, {"operation": "revise", "environment": scope.environment})
        return result

    def require(self, owner_id: str, mandate_id: str) -> Mandate:
        mandate = self.store.get(owner_id, mandate_id)
        if mandate is None:
            raise LookupError("mandate not found")
        return mandate

    @tracer.start_as_current_span("twin.mandate.approve")
    def approve(self, owner_id: str, mandate_id: str, request: MandateApprovalRequest) -> Mandate:
        current = self.require(owner_id, mandate_id)
        scope = self._scope(owner_id, current.account_scope_id)
        if scope.account_epoch != current.account_epoch:
            raise MandateConflict("registered account epoch changed after review")
        runtime = self.resolve_runtime(owner_id, scope)
        activated = approve(
            current, owner_id=owner_id, expected_hash=request.expected_hash, now=self.clock(),
            identity_hash=runtime.identity_hash, release_hash=runtime.release_hash,
            config_hash=runtime.config_hash, model_hash=runtime.model_hash,
            readiness_hash=runtime.readiness_hash, readiness_artifact=runtime.readiness_artifact,
        )
        result = self.store.save_transition(current, activated, idempotency_key=request.idempotency_key)
        mandate_operations.add(1, {"operation": "approve", "environment": scope.environment})
        return result

    @tracer.start_as_current_span("twin.mandate.revoke")
    def revoke(self, owner_id: str, mandate_id: str, request: MandateTransitionRequest) -> Mandate:
        current = self.require(owner_id, mandate_id)
        revoked = revoke(current, owner_id=owner_id, now=self.clock())
        if revoked == current:
            return current
        result = self.store.save_transition(current, revoked, idempotency_key=request.idempotency_key)
        mandate_operations.add(1, {"operation": "revoke", "environment": "live" if current.live else "shadow"})
        return result


def create_mandate_router(service: MandateService, *, resolve_owner: OwnerResolver) -> APIRouter:
    router = APIRouter(prefix="/twin/mandates", tags=["Autonomous Twin"])

    def call(request: Request, operation: Callable[[str], Mandate]) -> dict[str, Any]:
        try:
            owner_id = resolve_owner(request)
            if not owner_id:
                raise PermissionError("owner authentication is required")
            result = operation(owner_id)
            return {**result.to_storage(), "authority_hash": result.digest()}
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except MandateConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except MandateError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail="Autonomous mandate storage is unavailable.") from exc

    @router.post("")
    async def create_route(body: MandateDraftRequest, request: Request):
        return call(request, lambda owner: service.create(owner, body))

    @router.get("/{mandate_id}")
    async def read_route(mandate_id: str, request: Request):
        return call(request, lambda owner: service.require(owner, mandate_id))

    @router.post("/{mandate_id}/revisions")
    async def revise_route(mandate_id: str, body: MandateRevisionRequest, request: Request):
        return call(request, lambda owner: service.revise(owner, mandate_id, body))

    @router.post("/{mandate_id}/approve")
    async def approve_route(mandate_id: str, body: MandateApprovalRequest, request: Request):
        return call(request, lambda owner: service.approve(owner, mandate_id, body))

    @router.post("/{mandate_id}/revoke")
    async def revoke_route(mandate_id: str, body: MandateTransitionRequest, request: Request):
        return call(request, lambda owner: service.revoke(owner, mandate_id, body))

    return router
