"""
AgentFlow — GitHub Integration
Provides REST endpoints for GitHub API: repos, files, PRs, issues, commits.
Token stored in .env as GITHUB_TOKEN.
"""
from __future__ import annotations
import asyncio
import base64
import os
import time
from typing import Any, Dict, List, Optional

import httpx
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

router = APIRouter(prefix="/github", tags=["github"])

# ── Auth ───────────────────────────────────────────────────────────────────────
def _token() -> str:
    return os.getenv("GITHUB_TOKEN", "")

def _headers(token: str | None = None) -> Dict[str, str]:
    t = token or _token()
    h = {"Accept": "application/vnd.github.v3+json", "X-GitHub-Api-Version": "2022-11-28"}
    if t:
        h["Authorization"] = f"Bearer {t}"
    return h

GH_API = "https://api.github.com"
_cache: Dict[str, tuple[float, Any]] = {}  # url → (ts, data)
CACHE_TTL = 60  # seconds

async def _gh(path: str, method: str = "GET", body: Any = None, token: str | None = None) -> Any:
    """Make a GitHub API request with simple in-memory caching for GETs."""
    url = path if path.startswith("https://") else f"{GH_API}{path}"
    cache_key = f"{method}:{url}"

    if method == "GET" and cache_key in _cache:
        ts, data = _cache[cache_key]
        if time.time() - ts < CACHE_TTL:
            return data

    async with httpx.AsyncClient(timeout=20.0) as client:
        kwargs: Dict[str, Any] = {"headers": _headers(token)}
        if body:
            kwargs["json"] = body
        r = await client.request(method, url, **kwargs)
        if r.status_code == 404:
            raise HTTPException(404, f"GitHub: not found — {path}")
        if r.status_code == 401:
            raise HTTPException(401, "GitHub: invalid token — set GITHUB_TOKEN in .env")
        if r.status_code == 403:
            detail = r.json().get("message", "Forbidden")
            raise HTTPException(403, f"GitHub: {detail}")
        if r.status_code >= 400:
            raise HTTPException(r.status_code, r.text[:200])
        data = r.json() if r.text else {}

    if method == "GET":
        _cache[cache_key] = (time.time(), data)
    return data


# ── Status / Auth ──────────────────────────────────────────────────────────────
@router.get("/status")
async def github_status():
    token = _token()
    if not token:
        return {"configured": False, "message": "No GITHUB_TOKEN set"}
    try:
        user = await _gh("/user")
        return {
            "configured": True,
            "login": user.get("login"),
            "name": user.get("name"),
            "avatar_url": user.get("avatar_url"),
            "public_repos": user.get("public_repos"),
            "private_repos": user.get("total_private_repos", 0),
            "plan": user.get("plan", {}).get("name", "free"),
        }
    except HTTPException as e:
        return {"configured": False, "message": str(e.detail)}

@router.post("/token")
async def save_github_token(token: str = Query(...)):
    """Save GitHub Personal Access Token to .env."""
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    try:
        with open(env_path) as f:
            lines = f.readlines()
        new_lines = [l for l in lines if not l.startswith("GITHUB_TOKEN=")]
        new_lines.append(f"GITHUB_TOKEN={token}\n")
        with open(env_path, "w") as f:
            f.writelines(new_lines)
        os.environ["GITHUB_TOKEN"] = token
        _cache.clear()
        # Validate immediately
        user = await _gh("/user", token=token)
        return {"saved": True, "login": user.get("login"), "name": user.get("name")}
    except HTTPException as e:
        raise
    except Exception as e:
        raise HTTPException(400, str(e))

@router.delete("/token")
async def remove_github_token():
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    try:
        with open(env_path) as f:
            lines = f.readlines()
        new_lines = [l for l in lines if not l.startswith("GITHUB_TOKEN=")]
        with open(env_path, "w") as f:
            f.writelines(new_lines)
        os.environ.pop("GITHUB_TOKEN", None)
        _cache.clear()
        return {"removed": True}
    except Exception as e:
        raise HTTPException(400, str(e))


# ── Repositories ───────────────────────────────────────────────────────────────
@router.get("/repos")
async def list_repos(
    type: str = Query("all", description="all|owner|public|private|member"),
    sort: str = Query("updated"),
    per_page: int = Query(30, ge=1, le=100),
    page: int = Query(1),
):
    data = await _gh(f"/user/repos?type={type}&sort={sort}&per_page={per_page}&page={page}")
    return [_repo_summary(r) for r in data]

@router.get("/repos/search")
async def search_repos(q: str = Query(...), per_page: int = Query(20)):
    data = await _gh(f"/search/repositories?q={q}&per_page={per_page}&sort=updated")
    return {
        "total": data.get("total_count", 0),
        "items": [_repo_summary(r) for r in data.get("items", [])],
    }

@router.get("/repos/{owner}/{repo}")
async def get_repo(owner: str, repo: str):
    data = await _gh(f"/repos/{owner}/{repo}")
    langs = await _gh(f"/repos/{owner}/{repo}/languages")
    return {**_repo_summary(data), "languages": langs}

def _repo_summary(r: dict) -> dict:
    return {
        "id": r.get("id"),
        "name": r.get("name"),
        "full_name": r.get("full_name"),
        "description": r.get("description") or "",
        "private": r.get("private"),
        "fork": r.get("fork"),
        "stars": r.get("stargazers_count", 0),
        "forks": r.get("forks_count", 0),
        "open_issues": r.get("open_issues_count", 0),
        "language": r.get("language") or "—",
        "default_branch": r.get("default_branch", "main"),
        "pushed_at": r.get("pushed_at"),
        "updated_at": r.get("updated_at"),
        "html_url": r.get("html_url"),
        "clone_url": r.get("clone_url"),
        "topics": r.get("topics", []),
        "archived": r.get("archived", False),
    }


# ── File Tree & Contents ───────────────────────────────────────────────────────
@router.get("/repos/{owner}/{repo}/tree")
async def get_tree(owner: str, repo: str, path: str = Query(""), ref: str = Query("HEAD")):
    """List files at a path (directory listing)."""
    url = f"/repos/{owner}/{repo}/contents/{path}?ref={ref}"
    data = await _gh(url)
    if isinstance(data, list):
        return sorted(
            [_file_entry(f) for f in data],
            key=lambda x: (0 if x["type"] == "dir" else 1, x["name"])
        )
    # Single file
    return [_file_entry(data)]

@router.get("/repos/{owner}/{repo}/file")
async def get_file(owner: str, repo: str, path: str = Query(...), ref: str = Query("HEAD")):
    """Get file content (decoded from base64)."""
    data = await _gh(f"/repos/{owner}/{repo}/contents/{path}?ref={ref}")
    if data.get("type") != "file":
        raise HTTPException(400, "Path is not a file")
    content_b64 = data.get("content", "").replace("\n", "")
    try:
        content = base64.b64decode(content_b64).decode("utf-8", errors="replace")
    except Exception:
        content = "[Binary file]"
    return {
        "path": data.get("path"),
        "name": data.get("name"),
        "sha": data.get("sha"),
        "size": data.get("size"),
        "content": content,
        "encoding": data.get("encoding"),
        "html_url": data.get("html_url"),
        "download_url": data.get("download_url"),
    }

def _file_entry(f: dict) -> dict:
    return {
        "name": f.get("name"),
        "path": f.get("path"),
        "type": "dir" if f.get("type") == "dir" else "file",
        "size": f.get("size", 0),
        "sha": f.get("sha"),
        "html_url": f.get("html_url"),
    }


# ── Commits ───────────────────────────────────────────────────────────────────
@router.get("/repos/{owner}/{repo}/commits")
async def list_commits(
    owner: str, repo: str,
    branch: str = Query(""),
    path: str = Query(""),
    per_page: int = Query(20, ge=1, le=100),
    page: int = Query(1),
):
    params = f"per_page={per_page}&page={page}"
    if branch:
        params += f"&sha={branch}"
    if path:
        params += f"&path={path}"
    data = await _gh(f"/repos/{owner}/{repo}/commits?{params}")
    return [_commit_summary(c) for c in data]

@router.get("/repos/{owner}/{repo}/commits/{sha}")
async def get_commit(owner: str, repo: str, sha: str):
    data = await _gh(f"/repos/{owner}/{repo}/commits/{sha}")
    files = data.get("files", [])
    return {
        **_commit_summary(data),
        "stats": data.get("stats", {}),
        "files": [
            {
                "filename": f.get("filename"),
                "status": f.get("status"),
                "additions": f.get("additions"),
                "deletions": f.get("deletions"),
                "patch": f.get("patch", "")[:5000],  # cap large patches
            }
            for f in files[:50]
        ],
    }

def _commit_summary(c: dict) -> dict:
    commit = c.get("commit", {})
    author = commit.get("author", {})
    gh_author = c.get("author") or {}
    return {
        "sha": c.get("sha"),
        "sha_short": (c.get("sha") or "")[:7],
        "message": commit.get("message", "").split("\n")[0][:120],
        "message_full": commit.get("message", ""),
        "author_name": author.get("name"),
        "author_email": author.get("email"),
        "author_login": gh_author.get("login"),
        "author_avatar": gh_author.get("avatar_url"),
        "date": author.get("date"),
        "html_url": c.get("html_url"),
    }


# ── Branches ──────────────────────────────────────────────────────────────────
@router.get("/repos/{owner}/{repo}/branches")
async def list_branches(owner: str, repo: str, per_page: int = Query(30)):
    data = await _gh(f"/repos/{owner}/{repo}/branches?per_page={per_page}")
    return [{"name": b.get("name"), "sha": b.get("commit", {}).get("sha"), "protected": b.get("protected")} for b in data]


# ── Pull Requests ──────────────────────────────────────────────────────────────
@router.get("/repos/{owner}/{repo}/pulls")
async def list_pulls(
    owner: str, repo: str,
    state: str = Query("open"),
    per_page: int = Query(20),
    page: int = Query(1),
):
    data = await _gh(f"/repos/{owner}/{repo}/pulls?state={state}&per_page={per_page}&page={page}&sort=updated")
    return [_pr_summary(p) for p in data]

@router.get("/repos/{owner}/{repo}/pulls/{number}")
async def get_pull(owner: str, repo: str, number: int):
    pr = await _gh(f"/repos/{owner}/{repo}/pulls/{number}")
    files = await _gh(f"/repos/{owner}/{repo}/pulls/{number}/files?per_page=50")
    reviews = await _gh(f"/repos/{owner}/{repo}/pulls/{number}/reviews?per_page=20")
    comments = await _gh(f"/repos/{owner}/{repo}/issues/{number}/comments?per_page=20")
    return {
        **_pr_summary(pr),
        "body": pr.get("body") or "",
        "diff_url": pr.get("diff_url"),
        "mergeable": pr.get("mergeable"),
        "mergeable_state": pr.get("mergeable_state"),
        "files": [
            {
                "filename": f.get("filename"),
                "status": f.get("status"),
                "additions": f.get("additions"),
                "deletions": f.get("deletions"),
                "patch": (f.get("patch") or "")[:8000],
            }
            for f in files
        ],
        "reviews": [
            {
                "user": r.get("user", {}).get("login"),
                "avatar": r.get("user", {}).get("avatar_url"),
                "state": r.get("state"),
                "body": r.get("body") or "",
                "submitted_at": r.get("submitted_at"),
            }
            for r in reviews
        ],
        "comments": [
            {
                "user": c.get("user", {}).get("login"),
                "avatar": c.get("user", {}).get("avatar_url"),
                "body": c.get("body") or "",
                "created_at": c.get("created_at"),
            }
            for c in comments
        ],
    }

class CreatePRBody(BaseModel):
    title: str
    body: str = ""
    head: str  # branch to merge from
    base: str = "main"
    draft: bool = False

@router.post("/repos/{owner}/{repo}/pulls")
async def create_pull(owner: str, repo: str, pr: CreatePRBody):
    data = await _gh(f"/repos/{owner}/{repo}/pulls", "POST", pr.dict())
    return _pr_summary(data)

@router.patch("/repos/{owner}/{repo}/pulls/{number}")
async def update_pull(owner: str, repo: str, number: int, body: dict):
    data = await _gh(f"/repos/{owner}/{repo}/pulls/{number}", "PATCH", body)
    return _pr_summary(data)

@router.put("/repos/{owner}/{repo}/pulls/{number}/merge")
async def merge_pull(owner: str, repo: str, number: int, merge_method: str = Query("squash")):
    data = await _gh(
        f"/repos/{owner}/{repo}/pulls/{number}/merge", "PUT",
        {"merge_method": merge_method}
    )
    return data

def _pr_summary(p: dict) -> dict:
    return {
        "number": p.get("number"),
        "title": p.get("title"),
        "state": p.get("state"),
        "draft": p.get("draft"),
        "user_login": p.get("user", {}).get("login"),
        "user_avatar": p.get("user", {}).get("avatar_url"),
        "head_ref": p.get("head", {}).get("ref"),
        "base_ref": p.get("base", {}).get("ref"),
        "additions": p.get("additions"),
        "deletions": p.get("deletions"),
        "changed_files": p.get("changed_files"),
        "commits": p.get("commits"),
        "created_at": p.get("created_at"),
        "updated_at": p.get("updated_at"),
        "merged_at": p.get("merged_at"),
        "html_url": p.get("html_url"),
        "labels": [l.get("name") for l in p.get("labels", [])],
        "requested_reviewers": [r.get("login") for r in p.get("requested_reviewers", [])],
        "mergeable_state": p.get("mergeable_state"),
        "review_state": p.get("state"),  # open/closed/merged
    }


# ── Issues ────────────────────────────────────────────────────────────────────
@router.get("/repos/{owner}/{repo}/issues")
async def list_issues(
    owner: str, repo: str,
    state: str = Query("open"),
    labels: str = Query(""),
    per_page: int = Query(20),
    page: int = Query(1),
):
    params = f"state={state}&per_page={per_page}&page={page}&sort=updated"
    if labels:
        params += f"&labels={labels}"
    data = await _gh(f"/repos/{owner}/{repo}/issues?{params}")
    # Filter out PRs (GitHub returns PRs in issues endpoint)
    issues = [i for i in data if not i.get("pull_request")]
    return [_issue_summary(i) for i in issues]

@router.get("/repos/{owner}/{repo}/issues/{number}")
async def get_issue(owner: str, repo: str, number: int):
    issue = await _gh(f"/repos/{owner}/{repo}/issues/{number}")
    comments = await _gh(f"/repos/{owner}/{repo}/issues/{number}/comments?per_page=30")
    return {
        **_issue_summary(issue),
        "body": issue.get("body") or "",
        "comments_list": [
            {
                "user": c.get("user", {}).get("login"),
                "avatar": c.get("user", {}).get("avatar_url"),
                "body": c.get("body") or "",
                "created_at": c.get("created_at"),
            }
            for c in comments
        ],
    }

class CreateIssueBody(BaseModel):
    title: str
    body: str = ""
    labels: List[str] = []
    assignees: List[str] = []

@router.post("/repos/{owner}/{repo}/issues")
async def create_issue(owner: str, repo: str, issue: CreateIssueBody):
    data = await _gh(f"/repos/{owner}/{repo}/issues", "POST", issue.dict())
    return _issue_summary(data)

@router.post("/repos/{owner}/{repo}/issues/{number}/comments")
async def create_comment(owner: str, repo: str, number: int, body: str = Query(...)):
    data = await _gh(f"/repos/{owner}/{repo}/issues/{number}/comments", "POST", {"body": body})
    return data

def _issue_summary(i: dict) -> dict:
    return {
        "number": i.get("number"),
        "title": i.get("title"),
        "state": i.get("state"),
        "user_login": i.get("user", {}).get("login"),
        "user_avatar": i.get("user", {}).get("avatar_url"),
        "labels": [{"name": l.get("name"), "color": l.get("color")} for l in i.get("labels", [])],
        "assignees": [a.get("login") for a in i.get("assignees", [])],
        "comments": i.get("comments", 0),
        "created_at": i.get("created_at"),
        "updated_at": i.get("updated_at"),
        "html_url": i.get("html_url"),
        "body_preview": (i.get("body") or "").strip()[:150],
    }


# ── AI Actions ────────────────────────────────────────────────────────────────
class AIReviewBody(BaseModel):
    owner: str
    repo: str
    pr_number: int
    model: Optional[str] = None

@router.post("/ai/review-pr")
async def ai_review_pr(body: AIReviewBody):
    """Generate an AI code review for a PR using the best available model."""
    pr = await get_pull(body.owner, body.repo, body.pr_number)
    files_summary = "\n".join(
        f"**{f['filename']}** (+{f['additions']}/-{f['deletions']}):\n```diff\n{f['patch'][:2000]}\n```"
        for f in pr.get("files", [])[:10]
        if f.get("patch")
    )
    prompt = f"""Review this pull request and provide constructive feedback.

**PR #{pr['number']}: {pr['title']}**
Base: `{pr['base_ref']}` ← Head: `{pr['head_ref']}`
Changed files: {pr['changed_files']}, +{pr['additions']}/-{pr['deletions']} lines

{files_summary}

Provide:
1. **Summary** — what this PR does
2. **Strengths** — what's done well
3. **Issues** — bugs, security concerns, logic errors (be specific with line references)
4. **Suggestions** — improvements, style, best practices
5. **Verdict** — Approve / Request Changes / Needs Discussion
"""
    from server import _build_llm
    from langchain_core.messages import HumanMessage, SystemMessage
    llm = _build_llm(body.model or "", temperature=0.3)
    result = await llm.ainvoke([
        SystemMessage(content="You are a senior software engineer performing a thorough code review. Be specific, constructive, and actionable."),
        HumanMessage(content=prompt),
    ])
    return {"review": result.content, "pr_number": body.pr_number, "pr_title": pr["title"]}

@router.post("/ai/summarize-commits")
async def ai_summarize_commits(
    owner: str = Query(...), repo: str = Query(...),
    branch: str = Query("main"), limit: int = Query(10),
    model: Optional[str] = Query(None),
):
    """AI summary of recent commits."""
    commits = await list_commits(owner, repo, branch=branch, per_page=limit, page=1)
    commit_list = "\n".join(
        f"- {c['sha_short']} {c['author_name']}: {c['message']}" for c in commits
    )
    prompt = f"Summarize the following {len(commits)} recent commits in the `{branch}` branch of `{owner}/{repo}`. What's been worked on? Any notable patterns?\n\n{commit_list}"
    from server import _build_llm
    from langchain_core.messages import HumanMessage
    llm = _build_llm(model or "", temperature=0.5)
    result = await llm.ainvoke([HumanMessage(content=prompt)])
    return {"summary": result.content, "commit_count": len(commits)}

@router.post("/ai/draft-pr")
async def ai_draft_pr(
    owner: str = Query(...), repo: str = Query(...),
    head: str = Query(...), base: str = Query("main"),
    model: Optional[str] = Query(None),
):
    """AI-generated PR title and body from branch diff."""
    try:
        commits = await list_commits(owner, repo, branch=head, per_page=10, page=1)
        commit_msgs = "\n".join(f"- {c['message']}" for c in commits[:10])
    except Exception:
        commit_msgs = "(could not fetch commits)"
    prompt = f"""Draft a professional pull request for merging `{head}` into `{base}` in `{owner}/{repo}`.

Recent commits on `{head}`:
{commit_msgs}

Output JSON with keys:
- "title": concise PR title (conventional commits style)
- "body": full PR description with ## Summary, ## Changes, ## Testing sections"""
    from server import _build_llm
    from langchain_core.messages import HumanMessage, SystemMessage
    llm = _build_llm(model or "", temperature=0.4)
    result = await llm.ainvoke([
        SystemMessage(content="You are a senior engineer. Output only valid JSON, no markdown fences."),
        HumanMessage(content=prompt),
    ])
    import json as _json
    text = result.content.strip().strip("```json").strip("```").strip()
    try:
        return _json.loads(text)
    except Exception:
        return {"title": f"Merge {head} into {base}", "body": result.content}
