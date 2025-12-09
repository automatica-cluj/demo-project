# GitHub Actions CI/CD Setup Guide

This guide explains how to set up and use the GitHub Actions workflows for the Fake News Reporter application.

## Table of Contents
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Initial Setup](#initial-setup)
- [Workflows](#workflows)
- [Semantic Versioning](#semantic-versioning)
- [Using the Workflows](#using-the-workflows)
- [Docker Images](#docker-images)
- [Troubleshooting](#troubleshooting)

## Overview

The CI/CD pipeline consists of three main components:

1. **Continuous Integration (CI)** - Automated testing on every push/PR
2. **Build and Push** - Build Docker images and push to GitHub Container Registry
3. **Semantic Versioning** - Automatic version management based on commit messages

### Pipeline Flow

```
┌─────────────┐
│  Git Push   │
└──────┬──────┘
       │
       ├─────────────────────┐
       │                     │
       ▼                     ▼
┌─────────────┐      ┌─────────────────┐
│   CI Tests  │      │  Feature Branch │
│  (All PRs)  │      │   Development   │
└─────────────┘      └─────────────────┘
       │
       │ (on main branch)
       ▼
┌──────────────────┐
│ Semantic Version │
│   Detection      │
└────────┬─────────┘
         │
         ├──────────────┐
         ▼              ▼
┌──────────────┐  ┌──────────────┐
│ Build Docker │  │Create Release│
│ Push to GHCR │  │ (if new ver) │
└──────────────┘  └──────────────┘
```

## Prerequisites

### Required
- GitHub repository for the project
- GitHub account with repository admin access

### Recommended
- Familiarity with Git and GitHub
- Understanding of Docker basics
- Knowledge of conventional commits

## Initial Setup

### 1. Enable GitHub Actions

1. Go to your repository on GitHub
2. Navigate to **Settings** → **Actions** → **General**
3. Under "Actions permissions", select:
   - ✅ **Allow all actions and reusable workflows**

### 2. Configure Package Permissions

Enable GitHub Container Registry (GHCR) access:

1. Go to **Settings** → **Actions** → **General**
2. Scroll to "Workflow permissions"
3. Select:
   - ✅ **Read and write permissions**
   - ✅ **Allow GitHub Actions to create and approve pull requests**
4. Click **Save**

### 3. Configure Dependabot (Optional but Recommended)

The `.github/dependabot.yml` file is already configured with the repository owner as reviewer.

### 4. Verify Workflows

The workflow files are located in `.github/workflows/`:
- `ci.yml` - Continuous Integration workflow
- `build-push.yml` - Build and push Docker images workflow

## Workflows

### 1. CI Workflow (`ci.yml`)

**Triggers:** All pushes and pull requests to `main` and `develop` branches

**Jobs:**
- **test**: Runs Maven tests with Spring Boot test profile
- **code-quality**: Performs code quality checks and dependency analysis
- **build-verification**: Builds the JAR file to verify packaging

**What it does:**
- ✅ Compiles the application
- ✅ Runs all unit and integration tests
- ✅ Generates test reports
- ✅ Verifies JAR packaging
- ✅ Uploads test artifacts

**Status:** Required to pass before merging PRs

### 2. Build and Push Workflow (`build-push.yml`)

**Triggers:** Pushes to `main` branch, manual workflow dispatch

**Jobs:**
- **semantic-version**: Determines next version based on commits
- **build-and-push**: Builds multi-arch Docker image and pushes to GHCR
- **create-release**: Creates GitHub release with changelog (if new version)

**What it does:**
- 🔍 Analyzes commits for semantic versioning
- 🐳 Builds Docker images (amd64 + arm64)
- 📦 Pushes to `ghcr.io/automatica-cluj/demo-project`
- 🏷️ Tags images with version, SHA, and branch name
- 📝 Generates release notes and changelog
- 🚀 Creates GitHub release

### 3. Dependabot (`dependabot.yml`)

**Automatic dependency updates:**
- Maven dependencies (weekly on Mondays)
- Docker base images (weekly)
- GitHub Actions (weekly)

**Benefits:**
- 🔒 Security vulnerability patches
- 📦 Keep dependencies up to date
- 🤖 Automated pull requests

## Semantic Versioning

### How It Works

The project uses commit message analysis to automatically determine versions based on the [Conventional Commits](https://www.conventionalcommits.org/) specification.

### Commit Message Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Commit Types and Version Bumps

| Commit Type | Description | Version Impact | Example |
|-------------|-------------|----------------|---------|
| `feat:` | New feature | **Minor** (1.0.0 → 1.1.0) | `feat: add user dashboard` |
| `fix:` | Bug fix | **Patch** (1.0.0 → 1.0.1) | `fix: resolve login timeout` |
| `perf:` | Performance improvement | **Patch** (1.0.0 → 1.0.1) | `perf: optimize database queries` |
| `refactor:` | Code refactoring | **Patch** (1.0.0 → 1.0.1) | `refactor: simplify auth logic` |
| `BREAKING CHANGE:` | Breaking change | **Major** (1.0.0 → 2.0.0) | `feat!: redesign API structure` |
| `docs:` | Documentation | **None** | `docs: update API documentation` |
| `chore:` | Maintenance | **None** | `chore: update dependencies` |
| `test:` | Tests only | **None** | `test: add service tests` |
| `ci:` | CI/CD changes | **None** | `ci: update GitHub Actions` |

### Examples

#### Feature Addition (Minor Version)
```bash
git commit -m "feat: add report filtering by category

Implemented new UI for filtering reports including:
- Category dropdown filter
- Real-time filtering
- URL search capability"
```
Result: `1.2.3` → `1.3.0`

#### Bug Fix (Patch Version)
```bash
git commit -m "fix: resolve report approval issue

The report status wasn't updating correctly after
admin approval. Fixed by adding proper state handling."
```
Result: `1.2.3` → `1.2.4`

#### Breaking Change (Major Version)
```bash
git commit -m "feat!: redesign authentication API

BREAKING CHANGE: Authentication endpoints have been restructured.
All clients must update to use the new /api/v2/auth endpoints."
```
Result: `1.2.3` → `2.0.0`

#### No Release
```bash
git commit -m "docs: update deployment guide"
git commit -m "chore: update Maven dependencies"
git commit -m "test: add report service tests"
```
Result: No version change

### Scopes (Optional)

Add scopes for better organization:

```bash
git commit -m "feat(auth): add OAuth2 support"
git commit -m "fix(reports): resolve date formatting"
git commit -m "docs(api): document report endpoints"
```

## Using the Workflows

### Development Workflow

1. **Create a feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make changes and commit with conventional commits:**
   ```bash
   git add .
   git commit -m "feat: add new feature"
   ```

3. **Push and create a Pull Request:**
   ```bash
   git push origin feature/your-feature-name
   ```
   - CI tests will run automatically
   - Review test results in the PR checks

4. **Merge to main:**
   - Once approved and tests pass, merge the PR
   - Build & Push workflow triggers automatically
   - Version is determined and Docker image is built
   - If new version: GitHub release is created

### Manual Docker Build

You can manually trigger a Docker build:

1. Go to **Actions** tab in GitHub
2. Select "Build and Push to GHCR" workflow
3. Click **Run workflow**
4. Select branch and click **Run workflow**

### Viewing Build Results

After each push to main:

1. Go to **Actions** tab
2. Click on the latest workflow run
3. Review the summary:
   - ✅ Test results
   - 🐳 Docker image tags
   - 📦 Version information
   - 🔗 Pull command

## Docker Images

### Image Location

All Docker images are pushed to GitHub Container Registry:

```
ghcr.io/automatica-cluj/demo-project
```

### Image Tags

Each build creates multiple tags:

| Tag Format | Example | Description |
|------------|---------|-------------|
| `latest` | `ghcr.io/automatica-cluj/demo-project:latest` | Latest build from main branch |
| `v{version}` | `ghcr.io/automatica-cluj/demo-project:v1.2.3` | Semantic version |
| `sha-{short}` | `ghcr.io/automatica-cluj/demo-project:sha-abc1234` | Git commit SHA |
| `{branch}` | `ghcr.io/automatica-cluj/demo-project:main` | Branch name |
| `{timestamp}` | `ghcr.io/automatica-cluj/demo-project:20250112-143022` | Build timestamp |

### Pulling Images

#### Latest Version
```bash
docker pull ghcr.io/automatica-cluj/demo-project:latest
```

#### Specific Version
```bash
docker pull ghcr.io/automatica-cluj/demo-project:v1.2.3
```

#### By Commit SHA
```bash
docker pull ghcr.io/automatica-cluj/demo-project:sha-abc1234
```

### Making Images Public

By default, GHCR images are private. To make them public:

1. Go to your repository on GitHub
2. Click **Packages** (right sidebar)
3. Click on the package name
4. Go to **Package settings**
5. Scroll to "Danger Zone"
6. Click **Change visibility** → **Public**

### Using Images in Production

Update your `docker-compose.yml` to use the GHCR image:

```yaml
services:
  app:
    image: ghcr.io/automatica-cluj/demo-project:v1.2.3
    # ... rest of configuration
```

Or use environment variable:

```bash
export APP_IMAGE=ghcr.io/automatica-cluj/demo-project:v1.2.3
docker-compose up -d
```

### Authenticating with GHCR

To pull private images:

```bash
# Create a GitHub Personal Access Token (PAT)
# Go to Settings → Developer settings → Personal access tokens → Tokens (classic)
# Create token with 'read:packages' scope

# Login to GHCR
echo $GITHUB_TOKEN | docker login ghcr.io -u your-github-username --password-stdin

# Pull the image
docker pull ghcr.io/automatica-cluj/demo-project:latest
```

## Troubleshooting

### CI Tests Failing

**Problem:** Tests fail in GitHub Actions but pass locally

**Solutions:**
1. Check test profile: Ensure tests use `SPRING_PROFILES_ACTIVE=test`
2. Database issues: Verify H2 configuration in `application-test.yml`
3. Check logs in the Actions tab → Failed job → Test step

### Docker Build Fails

**Problem:** Docker build fails with dependency errors

**Solutions:**
1. Test Dockerfile locally:
   ```bash
   docker build -t test-build .
   ```
2. Check Maven cache: Clear and retry
3. Verify `pom.xml` is committed

### No New Release Created

**Problem:** Push to main but no release created

**Possible reasons:**
- No conventional commit messages since last release
- Only commit types that don't trigger releases (`docs`, `chore`, `test`, `ci`)
- Commit messages don't follow conventional format

**Solution:** Use proper commit message format:
```bash
git commit -m "feat: add new feature"  # Creates release
```

### Cannot Push to GHCR

**Problem:** "Permission denied" when pushing to GHCR

**Solutions:**
1. Verify workflow permissions (see [Initial Setup](#initial-setup))
2. Check repository settings → Actions → Workflow permissions
3. Ensure "Read and write permissions" is enabled

### Semantic Version Not Detected

**Problem:** Version shows as `v0.0.0` or incorrect

**Solutions:**
1. Create initial tag:
   ```bash
   git tag v0.1.0
   git push origin v0.1.0
   ```
2. Use conventional commits going forward
3. Check commit history for proper formatting

### Dependabot PRs Failing

**Problem:** Dependabot creates PRs but CI fails

**Solutions:**
1. Review the dependency update
2. Run tests locally with new version
3. May need to update code for breaking changes
4. Close PR if incompatible, pin version in `pom.xml`

## Best Practices

### Commit Messages
- ✅ Use clear, descriptive commit messages
- ✅ Follow conventional commits format
- ✅ Include scope when relevant: `feat(auth): add OAuth2`
- ✅ Use body to explain "why", not "what"
- ❌ Avoid vague messages: `fix stuff`, `update code`

### Branching Strategy
- `main` - Production-ready code, protected branch
- `develop` - Development branch (optional, creates beta releases)
- `feature/*` - Feature branches, merged to main via PR
- `hotfix/*` - Emergency fixes, fast-track to main

### Version Management
- Let semantic versioning handle versions automatically
- Don't manually edit version numbers in `pom.xml`
- Use tags for important milestones
- Document breaking changes in commit messages

### Docker Images
- Use specific version tags in production, not `latest`
- Test images before deploying:
  ```bash
  docker run --rm ghcr.io/automatica-cluj/demo-project:v1.2.3 java -version
  ```
- Keep old versions available for rollback
- Clean up old images periodically

## Additional Resources

- [Conventional Commits Specification](https://www.conventionalcommits.org/)
- [Semantic Versioning](https://semver.org/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [GitHub Container Registry](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry)
- [Docker Multi-Platform Builds](https://docs.docker.com/build/building/multi-platform/)

## Support

For issues or questions:
1. Check this documentation first
2. Review [Troubleshooting](#troubleshooting) section
3. Check GitHub Actions logs in the Actions tab
4. Create an issue in the repository with:
   - Workflow name
   - Error message
   - Relevant logs
   - Steps to reproduce
