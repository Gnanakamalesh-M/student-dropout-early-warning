"""Authentication, password hashing, and role-based access control.

Three roles, from ETHICS.md:

* ``admin`` — everything, including threshold configuration.
* ``counsellor`` — individual student profiles and intervention assignment. The
  only role that may see a named individual's risk.
* ``analyst`` — **aggregates only**. Explicitly denied individual profiles, so
  cohort analysis does not require access to identifiable risk scores.

The analyst restriction is the substantive one. Data minimisation is not only
about which columns exist; it is about who can see a person.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

# `enum.StrEnum` is 3.11+; the project is pinned to 3.10 (ADR-0004), so the
# str mixin is used instead. It gives the same behaviour for our purposes:
# members compare equal to their string value and serialise as strings.
from enum import Enum
from typing import Any

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext

from dropout_ews.config.settings import get_settings

ALGORITHM = "HS256"

# argon2 rather than bcrypt: bcrypt silently truncates passwords at 72 bytes,
# which turns a long passphrase into a weaker secret without telling anyone.
password_context = CryptContext(schemes=["argon2"], deprecated="auto")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/token", auto_error=False)


class Role(str, Enum):
    """Access role.

    Note the mixin gotcha this replaced ``StrEnum`` with: ``str(Role.ADMIN)``
    returns ``"Role.ADMIN"``, not ``"admin"``. Only ``.value`` and f-string
    formatting give the wire value, so ``.value`` is used everywhere the role is
    serialised. ``StrEnum`` would have made ``str()`` safe, but it is 3.11+.
    """

    ADMIN = "admin"
    COUNSELLOR = "counsellor"
    ANALYST = "analyst"


@dataclass(frozen=True)
class User:
    username: str
    role: Role
    hashed_password: str = ""

    @property
    def may_view_individuals(self) -> bool:
        """Whether this user may see an identifiable student's risk."""
        return self.role in (Role.ADMIN, Role.COUNSELLOR)

    @property
    def may_assign_interventions(self) -> bool:
        return self.role in (Role.ADMIN, Role.COUNSELLOR)

    @property
    def may_configure(self) -> bool:
        return self.role is Role.ADMIN


def hash_password(password: str) -> str:
    # passlib is untyped, so its return arrives as Any.
    hashed: str = password_context.hash(password)
    return hashed


def verify_password(plain: str, hashed: str) -> bool:
    try:
        matches: bool = password_context.verify(plain, hashed)
        return matches
    except ValueError:
        # A malformed stored hash must read as "wrong password", not crash the
        # login endpoint.
        return False


def create_access_token(username: str, role: Role, expires_minutes: int | None = None) -> str:
    settings = get_settings()
    minutes = (
        expires_minutes if expires_minutes is not None else settings.access_token_expire_minutes
    )
    now = datetime.now(timezone.utc)
    payload: dict[str, Any] = {
        "sub": username,
        "role": role.value,
        "iat": now,
        "exp": now + timedelta(minutes=minutes),
    }
    token: str = jwt.encode(payload, settings.secret_key, algorithm=ALGORITHM)
    return token


def decode_access_token(token: str) -> tuple[str, Role]:
    """Decode a token, or raise 401.

    Every failure mode returns the same message. Distinguishing "expired" from
    "malformed" from "unknown role" tells an attacker which of those they
    achieved.
    """
    settings = get_settings()
    credentials_error = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[ALGORITHM])
        username = payload.get("sub")
        role_value = payload.get("role")
        if not username or not role_value:
            raise credentials_error
        return str(username), Role(str(role_value))
    except (JWTError, ValueError) as exc:
        raise credentials_error from exc


async def get_current_user(token: str | None = Depends(oauth2_scheme)) -> User:
    """Resolve the authenticated user, or raise 401."""
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    username, role = decode_access_token(token)
    return User(username=username, role=role)


def require_roles(*roles: Role) -> Callable[..., Awaitable[User]]:
    """Dependency factory enforcing that the caller holds one of ``roles``."""

    async def dependency(user: User = Depends(get_current_user)) -> User:
        if user.role not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    f"Role '{user.role}' may not access this resource. "
                    f"Required: {', '.join(sorted(r.value for r in roles))}."
                ),
            )
        return user

    return dependency


# Named dependencies, so a router declares intent rather than repeating a role
# list that could drift between endpoints.
require_individual_access = require_roles(Role.ADMIN, Role.COUNSELLOR)
require_admin = require_roles(Role.ADMIN)
require_any_role = require_roles(Role.ADMIN, Role.COUNSELLOR, Role.ANALYST)
