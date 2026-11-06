"""Send existing Alps results to a report-only GitHub Actions workflow."""

import argparse
import base64
import csv
import gzip
import json
import re
import subprocess
import tomllib
from collections import Counter, defaultdict
from pathlib import Path
from xml.etree import ElementTree

REPO_ROOT = Path(__file__).resolve().parents[3]
POLICY_PATH = "tests/test_utils/recipes/gh200/known-failures.json"


def load_policy(ref):
    revision = subprocess.run(
        ["git", "rev-parse", "--verify", "--end-of-options", f"{ref}^{{commit}}"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
        timeout=15,
    )
    sha = revision.stdout.strip()
    contents = subprocess.run(
        ["git", "show", f"{sha}:{POLICY_PATH}"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
        timeout=15,
    )
    return sha, json.loads(contents.stdout)


def result_environment(directory):
    paths = (directory / "environment.toml", directory.parent / "environment.toml")
    for path in paths:
        if not path.is_file():
            continue
        image = tomllib.loads(path.read_text()).get("image")
        if image:
            return f"gh200:{image}"
    return "unknown"


def error_signature(element):
    text = element.get("message", "") + "\n" + (element.text or "")
    exceptions = []
    for line in text.splitlines():
        match = re.match(r"^\s*(?:E\s+)?([\w.]*(?:Error|Exception|Failure):.*)$", line)
        if match:
            exceptions.append(match.group(1).strip())

    if exceptions:
        signature = exceptions[-1]
    else:
        message = element.get("message") or "Unclassified failure"
        signature = message.splitlines()[0]

    signature = re.sub(r"(?<= at )0x[0-9a-fA-F]+", "<address>", signature)
    signature = re.sub(r"\(/[^)\n]+\)", "(<path>)", signature)
    signature = re.sub(r"\[(?:\d{1,3}\.){3}\d{1,3}\]:\d+", "[<peer>]", signature)
    return re.sub(
        r"(['\"])(/[^'\"\n]+)\1",
        lambda match: f"{match[1]}<path>/{Path(match[2]).name}{match[1]}",
        signature,
    )


def gh_json(*arguments):
    completed = subprocess.run(
        ["gh", *arguments], text=True, capture_output=True, check=True, cwd=REPO_ROOT, timeout=60
    )
    return json.loads(completed.stdout)


def git_output(*arguments, optional=False):
    completed = subprocess.run(
        ["git", *arguments], cwd=REPO_ROOT, text=True, capture_output=True, timeout=15
    )
    if completed.returncode and not (optional and completed.returncode == 1):
        completed.check_returncode()
    return completed.stdout.strip()


def configured_repository(branch):
    keys = ["remote.pushDefault"]
    if branch:
        keys.insert(0, f"branch.{branch}.pushRemote")
        keys.append(f"branch.{branch}.remote")
    remote = ""
    for key in keys:
        remote = git_output("config", "--get", key, optional=True)
        if remote:
            break
    if not remote or remote == ".":
        remote = "origin"
    return git_output("remote", "get-url", "--push", remote)


def launch_value(directory, name):
    path = directory / "launch.txt"
    if not path.is_file():
        return None
    prefix = f"{name}: "
    for line in path.read_text().splitlines():
        if line.startswith(prefix):
            return line[len(prefix):].strip() or None
    return None


def tested_sha(directory, override):
    if override:
        sha = override
    elif (directory / "launch.txt").is_file():
        sha = launch_value(directory, "Commit") or ""
    else:
        raise ValueError("No launch.txt: supply --sha with the recorded tested commit")
    if not re.fullmatch(r"[0-9a-fA-F]{7,40}", sha):
        raise ValueError("Invalid tested commit SHA")
    return sha


def count_range(values):
    if not values:
        return "—"
    minimum, maximum = min(values), max(values)
    if minimum == maximum:
        return str(minimum)
    return f"{minimum}–{maximum}"


def encode_observations(observations, selected_ids):
    evidence = []
    for test_id in sorted(selected_ids):
        records = observations[test_id]
        signatures = sorted({error for record in records.values() for error in record["errors"]})
        ranks = {}
        for rank, record in records.items():
            ranks[rank] = {
                "statuses": sorted(record["statuses"]),
                "failures": record["failures"],
                "error_indexes": [signatures.index(error) for error in sorted(record["errors"])],
            }
        evidence.append({"test_id": test_id, "signatures": signatures, "ranks": ranks})
    return evidence


def coverage_complete(directory, observations, rank_ids):
    path = directory / "expected-tests.json"
    if not path.is_file():
        return False

    plan = json.loads(path.read_text())
    expected = Counter(plan["test_ids"])
    if not expected or not rank_ids or plan.get("collection_complete") is not True:
        return False

    for rank in rank_ids:
        executed = Counter()
        for test_id, ranks in observations.items():
            if rank in ranks:
                executed[test_id] = ranks[rank]["executions"]
        if executed != expected:
            return False
    return True


def junit_results(directory, known_ids=()):
    counts = defaultdict(Counter)
    observations = defaultdict(dict)
    problems = []
    for path in sorted(directory.rglob("*.xml")):
        try:
            root = ElementTree.parse(path).getroot()
        except ElementTree.ParseError:
            problems.append(f"Malformed JUnit: {path.name}")
            continue

        match = re.search(r"\.rank-(\d+)\.xml$", path.name)
        rank = match.group(1) if match else "0"
        for suite in root.iter("testsuite"):
            if suite.find("testsuite") is not None:
                continue
            counts[rank]["tests"] += int(suite.get("tests", 0))
            counts[rank]["failed"] += int(suite.get("failures", 0))
            counts[rank]["failed"] += int(suite.get("errors", 0))
            counts[rank]["skipped"] += int(suite.get("skipped", 0))

        for case in root.iter("testcase"):
            test_id = f'{case.get("classname", "")}::{case.get("name", "")}'
            record = observations[test_id].setdefault(
                rank, {"statuses": set(), "executions": 0, "failures": 0, "errors": set()}
            )
            record["executions"] += 1
            errors = case.findall("failure") + case.findall("error")
            if errors:
                status = "failed"
            elif case.find("skipped") is not None:
                status = "skipped"
            else:
                status = "passed"
            record["statuses"].add(status)
            record["failures"] += len(errors)
            record["errors"].update(error_signature(error) for error in errors)

    failing_ids = {
        test_id
        for test_id, ranks in observations.items()
        if any(record["failures"] for record in ranks.values())
    }
    selected_ids = failing_ids | (set(known_ids) & observations.keys())
    for known_id in known_ids:
        if not known_id.startswith("::"):
            continue
        module = known_id[2:]
        for rank in counts:
            for test_id, records in observations.items():
                in_module = test_id.startswith(module + ".") or test_id.startswith(module + "::")
                record = records.get(rank)
                if in_module and record and record["statuses"] == {"passed"}:
                    selected_ids.add(test_id)
                    break
    return {
        "tests_per_rank": count_range([count["tests"] for count in counts.values()]),
        "failures_errors_per_rank": count_range([count["failed"] for count in counts.values()]),
        "skipped_per_rank": count_range([count["skipped"] for count in counts.values()]),
        "reporting_ranks": len(counts),
        "rank_ids": sorted(counts, key=int),
        "tests_by_rank": {rank: count["tests"] for rank, count in counts.items()},
        "skips_by_rank": {rank: count["skipped"] for rank, count in counts.items()},
        "failures_by_rank": {rank: count["failed"] for rank, count in counts.items()},
        "failing_tests": sorted(failing_ids)[:5],
        "has_failures": bool(failing_ids) or any(count["failed"] for count in counts.values()),
        "observations": encode_observations(observations, selected_ids),
        "coverage_confirmed": coverage_complete(directory, observations, counts.keys()),
        "problems": problems,
    }


def build_report(directory, results, suite, sha, policy=None, policy_sha=None, environment=None):
    policy = policy or {"scopes": []}
    known_ids = defaultdict(set)
    for scope in policy["scopes"]:
        if scope["suite"] == suite:
            for failure in scope["failures"]:
                known_ids[failure["selection"]].add(failure["test_id"])

    with results.open() as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if not rows:
        raise ValueError("Results ledger is empty")

    selections = []
    for row in rows:
        name = row.get("group") or row.get("selection") or row.get("case")
        if not name or "result" not in row:
            raise ValueError("Expected group/selection/case and result columns")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
            raise ValueError(f"Unsupported selection name: {name}")

        group_dir = directory / name
        if not group_dir.is_dir():
            group_dir = directory / "run" / suite / name
        ranks = row.get("ranks")
        inner_ledger = group_dir / "results.tsv"
        if not ranks and inner_ledger.is_file():
            with inner_ledger.open() as stream:
                inner_rows = csv.DictReader(stream, delimiter="\t")
                ranks = next(
                    (item.get("ranks") for item in inner_rows if item.get("group") == name), None
                )
        expected_ranks = int(ranks) if ranks else None
        if suite == "functional":
            expected_ranks = 1

        selections.append(
            {
                "name": name,
                "outcome": row["result"],
                "reason": row.get("reason", ""),
                "expected_reporting_ranks": expected_ranks,
                "junit": junit_results(group_dir, known_ids[name]),
            }
        )
    return {
        "version": 2,
        "suite": suite,
        "sha": sha,
        "environment": environment or result_environment(directory),
        "policy_sha": policy_sha,
        "tested_branch": launch_value(directory, "Branch"),
        "tested_pr": None,
        "selections": selections,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Directory containing the results ledger")
    parser.add_argument("--suite", choices=("unit", "functional"), required=True)
    parser.add_argument(
        "--sha", help="Recorded tested commit; defaults to launch.txt, never current HEAD"
    )
    parser.add_argument(
        "--results", default="results.tsv", help="Ledger filename relative to directory"
    )
    parser.add_argument("--repo", help="OWNER/REPO; defaults to the branch's configured remote")
    parser.add_argument(
        "--ref",
        help="Workflow branch; defaults to the current branch, or repository default if detached",
    )
    parser.add_argument(
        "--policy-ref",
        default="HEAD",
        help="Trusted publisher revision containing the policy, not the tested PR revision",
    )
    parser.add_argument(
        "--environment", help="Environment identity; otherwise read from environment.toml"
    )
    parser.add_argument("--publish", action="store_true", help="Dispatch the report-only workflow")
    args = parser.parse_args()
    directory = args.directory.resolve()
    sha = tested_sha(directory, args.sha)
    policy_sha, policy = load_policy(args.policy_ref)
    report = build_report(
        directory, directory / args.results, args.suite, sha, policy, policy_sha, args.environment
    )
    if not args.publish:
        print(json.dumps(report, indent=2))
        print("Preview only. No GitHub writes; use --publish to dispatch the reporting workflow.")
        return

    branch = git_output("branch", "--show-current")
    repository = args.repo or configured_repository(branch)
    repo_info = gh_json("repo", "view", repository, "--json", "nameWithOwner,defaultBranchRef")
    repo = repo_info["nameWithOwner"]
    report["sha"] = gh_json("api", f"repos/{repo}/commits/{sha}")["sha"]
    report["repository"] = repo
    if report["tested_branch"]:
        pulls = gh_json("api", f"repos/{repo}/commits/{report['sha']}/pulls?per_page=100")
        matches = [
            pr
            for pr in pulls
            if pr["head"]["sha"] == report["sha"]
            and pr["head"]["ref"] == report["tested_branch"]
        ]
        if len(matches) == 1:
            report["tested_pr"] = matches[0]["number"]
    workflow_ref = args.ref or branch or repo_info["defaultBranchRef"]["name"]
    inputs = {
        "suite": report["suite"],
        "tested_sha": report["sha"],
        "tested_branch": report["tested_branch"] or "",
        "tested_pr": str(report["tested_pr"] or ""),
        "report": json.dumps(report, separators=(",", ":")),
    }
    if len(inputs["report"].encode()) > 5_000_000:
        raise ValueError("Report exceeds the 5 MB uncompressed limit")
    payload = json.dumps(inputs)
    if len(payload.encode()) > 60000:
        compressed = gzip.compress(inputs["report"].encode(), mtime=0)
        inputs["report"] = "gzip:" + base64.b64encode(compressed).decode()
        payload = json.dumps(inputs)
    if len(payload.encode()) > 60000:
        raise ValueError("Compressed report exceeds workflow input size")
    subprocess.run(
        [
            "gh",
            "workflow",
            "run",
            "alps-test-results.yml",
            "--repo",
            repo,
            "--ref",
            workflow_ref,
            "--json",
        ],
        input=payload,
        text=True,
        check=True,
        cwd=REPO_ROOT,
        timeout=60,
    )
    print(f"Report submitted: https://github.com/{repo}/actions/workflows/alps-test-results.yml")


if __name__ == "__main__":
    main()
