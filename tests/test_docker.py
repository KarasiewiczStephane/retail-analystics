"""Tests for Docker configuration files."""

from pathlib import Path

import yaml


ROOT = Path(__file__).parent.parent


class TestDockerfile:
    """Tests for Dockerfile structure."""

    def test_dockerfile_exists(self) -> None:
        """Dockerfile exists in project root."""
        assert (ROOT / "Dockerfile").exists()

    def test_dockerfile_has_multistage_build(self) -> None:
        """Dockerfile uses multi-stage build."""
        content = (ROOT / "Dockerfile").read_text()
        assert "AS builder" in content
        assert "COPY --from=builder" in content

    def test_dockerfile_exposes_streamlit_port(self) -> None:
        """Dockerfile exposes port 8501 for Streamlit."""
        content = (ROOT / "Dockerfile").read_text()
        assert "EXPOSE 8501" in content

    def test_dockerfile_has_healthcheck(self) -> None:
        """Dockerfile includes a healthcheck."""
        content = (ROOT / "Dockerfile").read_text()
        assert "HEALTHCHECK" in content

    def test_dockerfile_installs_opencv_deps(self) -> None:
        """Dockerfile installs OpenCV runtime dependencies."""
        content = (ROOT / "Dockerfile").read_text()
        assert "libgl1-mesa-glx" in content
        assert "ffmpeg" in content

    def test_dockerfile_copies_source(self) -> None:
        """Dockerfile copies source code and configs."""
        content = (ROOT / "Dockerfile").read_text()
        assert "COPY src/ src/" in content
        assert "COPY configs/ configs/" in content

    def test_dockerfile_sets_unbuffered(self) -> None:
        """Dockerfile sets PYTHONUNBUFFERED for logging."""
        content = (ROOT / "Dockerfile").read_text()
        assert "PYTHONUNBUFFERED=1" in content


class TestDockerCompose:
    """Tests for docker-compose.yml."""

    def test_compose_file_exists(self) -> None:
        """docker-compose.yml exists in project root."""
        assert (ROOT / "docker-compose.yml").exists()

    def test_compose_valid_yaml(self) -> None:
        """docker-compose.yml is valid YAML."""
        content = (ROOT / "docker-compose.yml").read_text()
        data = yaml.safe_load(content)
        assert isinstance(data, dict)

    def test_compose_has_dashboard_service(self) -> None:
        """docker-compose.yml defines a dashboard service."""
        content = (ROOT / "docker-compose.yml").read_text()
        data = yaml.safe_load(content)
        assert "dashboard" in data["services"]

    def test_compose_has_cli_service(self) -> None:
        """docker-compose.yml defines a CLI service."""
        content = (ROOT / "docker-compose.yml").read_text()
        data = yaml.safe_load(content)
        assert "cli" in data["services"]

    def test_compose_dashboard_port(self) -> None:
        """Dashboard service maps port 8501."""
        content = (ROOT / "docker-compose.yml").read_text()
        data = yaml.safe_load(content)
        ports = data["services"]["dashboard"]["ports"]
        assert "8501:8501" in ports

    def test_compose_volumes_mounted(self) -> None:
        """Services have data and config volumes."""
        content = (ROOT / "docker-compose.yml").read_text()
        data = yaml.safe_load(content)
        dash_volumes = data["services"]["dashboard"]["volumes"]
        assert any("data" in v for v in dash_volumes)
        assert any("configs" in v for v in dash_volumes)


class TestDockerIgnore:
    """Tests for .dockerignore."""

    def test_dockerignore_exists(self) -> None:
        """dockerignore file exists."""
        assert (ROOT / ".dockerignore").exists()

    def test_dockerignore_excludes_git(self) -> None:
        """dockerignore excludes .git directory."""
        content = (ROOT / ".dockerignore").read_text()
        assert ".git" in content

    def test_dockerignore_excludes_tests(self) -> None:
        """dockerignore excludes tests directory."""
        content = (ROOT / ".dockerignore").read_text()
        assert "tests/" in content

    def test_dockerignore_excludes_env(self) -> None:
        """dockerignore excludes .env file."""
        content = (ROOT / ".dockerignore").read_text()
        assert ".env" in content

    def test_dockerignore_excludes_cache(self) -> None:
        """dockerignore excludes cache directories."""
        content = (ROOT / ".dockerignore").read_text()
        assert "__pycache__" in content
        assert ".pytest_cache" in content


class TestMakefileDocker:
    """Tests for Docker-related Makefile targets."""

    def test_makefile_has_docker_build(self) -> None:
        """Makefile has docker-build target."""
        content = (ROOT / "Makefile").read_text()
        assert "docker-build:" in content

    def test_makefile_has_docker_run(self) -> None:
        """Makefile has docker-run target."""
        content = (ROOT / "Makefile").read_text()
        assert "docker-run:" in content

    def test_makefile_has_docker_cli(self) -> None:
        """Makefile has docker-cli target."""
        content = (ROOT / "Makefile").read_text()
        assert "docker-cli:" in content

    def test_makefile_has_docker_compose(self) -> None:
        """Makefile has docker compose targets."""
        content = (ROOT / "Makefile").read_text()
        assert "docker-up:" in content
        assert "docker-down:" in content
