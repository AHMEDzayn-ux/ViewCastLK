import pytest

from app.auth import AuthenticatedUser, require_authenticated_user
from app.main import app


@pytest.fixture
def authenticated_api():
    """Replace remote token validation for endpoint behavior tests only."""
    app.dependency_overrides[require_authenticated_user] = lambda: AuthenticatedUser(
        id="test-authenticated-user"
    )
    try:
        yield
    finally:
        app.dependency_overrides.pop(require_authenticated_user, None)
