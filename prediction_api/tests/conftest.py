import pytest

from app.auth import AuthenticatedUser, optional_authenticated_user, require_authenticated_user
from app.main import app


@pytest.fixture
def authenticated_api():
    """Replace remote token validation for endpoint behavior tests only."""
    app.dependency_overrides[require_authenticated_user] = lambda: AuthenticatedUser(
        id="test-authenticated-user"
    )
    app.dependency_overrides[optional_authenticated_user] = lambda: AuthenticatedUser(
        id="test-authenticated-user"
    )
    try:
        yield
    finally:
        app.dependency_overrides.pop(require_authenticated_user, None)
        app.dependency_overrides.pop(optional_authenticated_user, None)
