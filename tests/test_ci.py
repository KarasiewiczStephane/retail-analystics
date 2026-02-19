"""Tests for CI/CD configuration files."""

from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent


class TestCIWorkflow:
    """Tests for GitHub Actions CI workflow."""

    def test_workflow_file_exists(self) -> None:
        """CI workflow YAML exists."""
        assert (ROOT / ".github" / "workflows" / "ci.yml").exists()

    def test_workflow_valid_yaml(self) -> None:
        """CI workflow is valid YAML."""
        path = ROOT / ".github" / "workflows" / "ci.yml"
        data = yaml.safe_load(path.read_text())
        assert isinstance(data, dict)

    def test_workflow_has_name(self) -> None:
        """Workflow has a name."""
        path = ROOT / ".github" / "workflows" / "ci.yml"
        data = yaml.safe_load(path.read_text())
        assert "name" in data
        assert data["name"] == "CI"

    def test_workflow_triggers_on_push(self) -> None:
        """Workflow triggers on push to main/master."""
        path = ROOT / ".github" / "workflows" / "ci.yml"
        data = yaml.safe_load(path.read_text())
        # PyYAML parses bare 'on' as boolean True
        triggers = data.get("on") or data.get(True)
        assert "push" in triggers
        branches = triggers["push"]["branches"]
        assert "main" in branches

    def test_workflow_triggers_on_pr(self) -> None:
        """Workflow triggers on pull requests."""
        path = ROOT / ".github" / "workflows" / "ci.yml"
        data = yaml.safe_load(path.read_text())
        triggers = data.get("on") or data.get(True)
        assert "pull_request" in triggers

    def test_workflow_has_lint_job(self) -> None:
        """Workflow has a lint job."""
        path = ROOT / ".github" / "workflows" / "ci.yml"
        data = yaml.safe_load(path.read_text())
        assert "lint" in data["jobs"]

    def test_workflow_has_test_job(self) -> None:
        """Workflow has a test job."""
        path = ROOT / ".github" / "workflows" / "ci.yml"
        data = yaml.safe_load(path.read_text())
        assert "test" in data["jobs"]

    def test_workflow_has_docker_build_job(self) -> None:
        """Workflow has a Docker build job."""
        path = ROOT / ".github" / "workflows" / "ci.yml"
        data = yaml.safe_load(path.read_text())
        assert "build-docker" in data["jobs"]

    def test_test_job_depends_on_lint(self) -> None:
        """Test job runs after lint."""
        path = ROOT / ".github" / "workflows" / "ci.yml"
        data = yaml.safe_load(path.read_text())
        assert data["jobs"]["test"]["needs"] == "lint"

    def test_docker_depends_on_test(self) -> None:
        """Docker build runs after test."""
        path = ROOT / ".github" / "workflows" / "ci.yml"
        data = yaml.safe_load(path.read_text())
        assert data["jobs"]["build-docker"]["needs"] == "test"

    def test_coverage_threshold_enforced(self) -> None:
        """CI enforces coverage threshold."""
        path = ROOT / ".github" / "workflows" / "ci.yml"
        content = path.read_text()
        assert "--fail-under=80" in content

    def test_uses_python_311(self) -> None:
        """CI uses Python 3.11."""
        path = ROOT / ".github" / "workflows" / "ci.yml"
        content = path.read_text()
        assert "3.11" in content

    def test_uses_pip_caching(self) -> None:
        """CI uses pip dependency caching."""
        path = ROOT / ".github" / "workflows" / "ci.yml"
        content = path.read_text()
        assert "actions/cache@v4" in content

    def test_uses_docker_buildx(self) -> None:
        """CI uses Docker Buildx for efficient builds."""
        path = ROOT / ".github" / "workflows" / "ci.yml"
        content = path.read_text()
        assert "docker/setup-buildx-action" in content


class TestPytestConfig:
    """Tests for pytest configuration."""

    def test_pytest_ini_exists(self) -> None:
        """pytest.ini exists in project root."""
        assert (ROOT / "pytest.ini").exists()

    def test_pytest_ini_has_testpaths(self) -> None:
        """pytest.ini configures test paths."""
        content = (ROOT / "pytest.ini").read_text()
        assert "testpaths = tests" in content

    def test_pytest_ini_has_markers(self) -> None:
        """pytest.ini defines custom markers."""
        content = (ROOT / "pytest.ini").read_text()
        assert "integration" in content
        assert "slow" in content
