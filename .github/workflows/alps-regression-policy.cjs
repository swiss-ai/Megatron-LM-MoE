function hasOnlyStatus(result, status) {
  return result?.statuses?.length === 1 && result.statuses[0] === status;
}

function evaluateSelection(selection, policy, environment, suite) {
  const {name, junit} = selection;
  if (!junit || !Array.isArray(junit.observations) || !Array.isArray(junit.rank_ids)) {
    throw new Error('Missing full JUnit evidence');
  }
  const expected = policy.required_selections[suite][name];
  const ranks = Array.from({length: expected || 0}, (_, index) => String(index));
  const problems = [...(junit.problems || [])];
  const unexpected = [];
  const known = [];
  const fixed = [];

  if (!expected || selection.expected_reporting_ranks !== expected) {
    problems.push('Unexpected selection or rank plan');
  }
  if (
    junit.rank_ids.length !== ranks.length ||
    ranks.some(rank => !junit.rank_ids.includes(rank)) ||
    junit.reporting_ranks !== ranks.length
  ) {
    problems.push('Missing or unexpected rank reports');
  }
  if (!/^[1-9]/.test(String(junit.tests_per_rank))) {
    problems.push('No executed tests');
  }
  if (!['COMPLETED', 'FAILED'].includes(selection.outcome)) {
    problems.push(`Outcome: ${selection.outcome}`);
  }
  if (
    (selection.outcome === 'FAILED' || junit.has_failures) &&
    junit.coverage_confirmed !== true
  ) {
    problems.push('Coverage unconfirmed after a failed selection');
  }

  const rules = new Map();
  for (const scope of policy.scopes) {
    if (scope.suite !== suite || scope.environment !== environment) {
      continue;
    }
    for (const rule of scope.failures) {
      if (rule.selection !== name) {
        continue;
      }
      if (
        !rule.reason ||
        !rule.test_id ||
        !Array.isArray(rule.error_signatures) ||
        !rule.error_signatures.length ||
        !rule.error_signatures.every(error => typeof error === 'string' && error.length) ||
        rules.has(rule.test_id)
      ) {
        throw new Error('Invalid known-failure rule');
      }
      rules.set(rule.test_id, rule);
    }
  }

  const observed = new Set();
  const failureCounts = Object.create(null);
  for (const test of junit.observations) {
    if (
      !test.test_id ||
      observed.has(test.test_id) ||
      !test.ranks ||
      !Array.isArray(test.signatures) ||
      !test.signatures.every(signature => typeof signature === 'string')
    ) {
      throw new Error('Invalid test evidence');
    }
    observed.add(test.test_id);

    let failures = 0;
    const errors = new Set();
    for (const [rank, result] of Object.entries(test.ranks)) {
      if (
        !junit.rank_ids.includes(rank) ||
        !Array.isArray(result.statuses) ||
        !result.statuses.every(status => ['passed', 'failed', 'skipped'].includes(status)) ||
        !Array.isArray(result.error_indexes) ||
        !Number.isInteger(result.failures) ||
        result.failures < 0
      ) {
        throw new Error('Invalid rank evidence');
      }
      for (const index of result.error_indexes) {
        if (!Number.isInteger(index) || index < 0 || index >= test.signatures.length) {
          throw new Error('Invalid error signature index');
        }
        errors.add(test.signatures[index]);
      }
      failures += result.failures;
      failureCounts[rank] = (failureCounts[rank] || 0) + result.failures;
      if ((result.failures > 0) !== result.statuses.includes('failed')) {
        problems.push(`Inconsistent failure record: ${test.test_id}`);
      }
      if (result.statuses.length !== 1) {
        problems.push(`Mixed outcomes: ${test.test_id}`);
      }
    }

    const rule = rules.get(test.test_id);
    if (failures) {
      if (!errors.size) {
        problems.push(`Missing error signature: ${test.test_id}`);
      }
      if (test.test_id.startsWith('::')) {
        problems.push(`Collection failed: ${test.test_id}`);
      }
      const item = {
        selection: name,
        test_id: test.test_id,
        errors: [...errors],
      };
      const expectedError = rule && errors.size && [...errors].every(
        error => rule.error_signatures.includes(error)
      );
      if (!expectedError) {
        unexpected.push(item);
        continue;
      }

      known.push({...item, reason: rule.reason});
      if (rule.blocking === true) {
        problems.push(`Pre-existing blocker: ${test.test_id}`);
      }
      const allRanksFailed = Object.keys(test.ranks).length === ranks.length && ranks.every(
        rank => hasOnlyStatus(test.ranks[rank], 'failed')
      );
      if (!allRanksFailed) {
        problems.push(`Rank-inconsistent known failure: ${test.test_id}`);
      }
      continue;
    }

    if (rule) {
      const allRanksPassed = Object.keys(test.ranks).length === ranks.length && ranks.every(
        rank => hasOnlyStatus(test.ranks[rank], 'passed')
      );
      if (allRanksPassed) {
        fixed.push({selection: name, test_id: test.test_id});
      } else {
        problems.push(`Known test skipped or missing ranks: ${test.test_id}`);
      }
    }
  }

  for (const rank of junit.rank_ids) {
    const tests = junit.tests_by_rank[rank];
    const skipped = junit.skips_by_rank[rank];
    if (
      !Number.isInteger(tests) ||
      !Number.isInteger(skipped) ||
      tests < 0 ||
      skipped < 0 ||
      skipped > tests
    ) {
      throw new Error('Invalid per-rank test counts');
    }
    if (tests === skipped) {
      problems.push(`No executed tests on rank ${rank}`);
    }
    if ((failureCounts[rank] || 0) !== junit.failures_by_rank[rank]) {
      problems.push(`Unattributed JUnit failures on rank ${rank}`);
    }
  }
  for (const id of rules.keys()) {
    if (observed.has(id)) {
      continue;
    }
    const moduleName = id.startsWith('::') ? id.slice(2) : null;
    const collectionFixed = moduleName && ranks.length && ranks.every(rank => {
      return junit.observations.some(test => {
        const inModule = test.test_id.startsWith(moduleName + '.') ||
          test.test_id.startsWith(moduleName + '::');
        return inModule && hasOnlyStatus(test.ranks[rank], 'passed');
      });
    });
    if (collectionFixed) {
      fixed.push({selection: name, test_id: id});
    } else {
      problems.push(`Known test not observed: ${id}`);
    }
  }
  const hasDeclaredFailures = Object.values(junit.failures_by_rank).some(count => count > 0);
  if (['FAILED', 'TIMED_OUT'].includes(selection.outcome) && !hasDeclaredFailures) {
    unexpected.push({
      selection: name,
      test_id: `${name}::<runner>`,
      errors: [`Runner outcome: ${selection.outcome}`],
    });
  }
  if (junit.has_failures && !hasDeclaredFailures) {
    problems.push('JUnit totals contradict failure records');
  }

  let result = 'PASS';
  if (unexpected.length) {
    result = 'NEW FAILURE';
  } else if (problems.length) {
    result = 'INCOMPLETE';
  } else if (known.length) {
    result = 'KNOWN FAILURES';
  } else if (fixed.length) {
    result = 'FIXED';
  }
  return {name, result, unexpected, known, fixed, problems};
}

function evaluateReport(report, policy) {
  if (
    policy.version !== 1 ||
    !Array.isArray(report.selections) ||
    !report.selections.length ||
    !Array.isArray(policy.scopes) ||
    !policy.required_selections[report.suite]
  ) {
    throw new Error('Invalid known-failure policy');
  }

  const unexpected = [];
  const known = [];
  const fixed = [];
  const incomplete = [];
  const warnings = [];
  const selections = [];
  const seen = new Set();
  for (const selection of report.selections) {
    if (!/^[A-Za-z0-9_.-]+$/.test(selection.name) || seen.has(selection.name)) {
      throw new Error('Invalid or duplicate selection');
    }
    seen.add(selection.name);
    const result = evaluateSelection(selection, policy, report.environment, report.suite);
    unexpected.push(...result.unexpected);
    known.push(...result.known);
    fixed.push(...result.fixed);
    incomplete.push(...result.problems.map(problem => `${selection.name}: ${problem}`));
    selections.push({
      name: result.name,
      result: result.result,
      unexpected: result.unexpected.length,
      known: result.known.length,
      fixed: result.fixed.length,
    });
  }

  const required = Object.keys(policy.required_selections[report.suite]);
  const missing = required.filter(name => !seen.has(name));
  const scope = missing.length ? 'selected' : 'full';
  if (!policy.environments.includes(report.environment)) {
    (scope === 'selected' ? warnings : incomplete).push('Unknown test environment');
  }
  if (missing.length) {
    warnings.push(`${missing.length} selections outside this report were not validated`);
  }
  let state = 'success';
  if (unexpected.length) {
    state = 'failure';
  } else if (incomplete.length) {
    state = 'error';
  }
  return {
    state,
    scope,
    warnings,
    unexpected,
    known,
    fixed,
    incomplete: [...new Set(incomplete)],
    missing_selections: missing,
    selections,
  };
}

function renderReport(report, gate) {
  const statusLabels = {
    PASS: '🟢 **PASS**',
    'NEW FAILURE': '🔴 **NEW FAILURE**',
    'KNOWN FAILURES': '🟠 **KNOWN FAILURES**',
    INCOMPLETE: '🟡 **INCOMPLETE**',
    FIXED: '🟢 **FIXED**',
  };
  const escape = value => String(value).replace(/[|<>`\r\n]/g, ' ');
  let heading = gate.scope === 'selected' && !gate.known.length
    ? 'Selected tests passed' : 'No new regressions';
  if (gate.unexpected.length) {
    heading = `NEW / UNEXPECTED FAILURES: ${gate.unexpected.length}`;
  } else if (gate.incomplete.length) {
    heading = 'Test coverage incomplete';
  }
  const repository = report.repository || '';
  const pr = report.tested_pr;
  const prLabel = Number.isSafeInteger(pr) && pr > 0 &&
    /^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/.test(repository)
    ? `[#${pr}](https://github.com/${repository}/pull/${pr})`
    : 'not identified';
  const lines = [
    `# ${heading}`,
    '',
    `${gate.unexpected.length} new · ${gate.known.length} known · ${gate.fixed.length} fixed`,
    '',
    `Tested commit: ${report.sha}`,
    `Tested branch: ${report.tested_branch ? `\`${escape(report.tested_branch)}\`` : 'unknown (not recorded)'}`,
    `Pull request: ${prLabel}`,
    `Reporting workflow ref: ${escape(report.workflow_ref || 'unknown')}`,
    `Report scope: ${gate.scope === 'selected' ? 'selected tests only; not full-suite validation' : 'full suite'}`,
    '',
    '| Selection | Result | Tests/rank | New | Known | Fixed | Skips/rank | Ranks |',
    '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |',
  ];
  for (const [index, selection] of report.selections.entries()) {
    const result = gate.selections[index];
    const junit = selection.junit;
    const row = [
      escape(selection.name),
      statusLabels[result.result],
      escape(junit.tests_per_rank),
      result.unexpected,
      result.known,
      result.fixed,
      escape(junit.skipped_per_rank),
      `${junit.reporting_ranks}/${selection.expected_reporting_ranks ?? '?'}`,
    ];
    lines.push(`| ${row.join(' | ')} |`);
  }
  const blocked = report.selections.filter(selection => selection.outcome === 'BLOCKED');
  if (blocked.length) {
    lines.push('', `## Blocked selections (${blocked.length})`, '');
    for (const selection of blocked) {
      lines.push(`- ${escape(selection.name)} — ${escape(selection.reason || 'See run logs')}`);
    }
  }
  if (gate.unexpected.length) {
    lines.push('', '## New or changed failures', '');
    for (const test of gate.unexpected) {
      const errors = escape(test.errors.join('; ')).slice(0, 300);
      lines.push(`- ${escape(test.test_id)} — ${errors}`);
    }
  }
  if (gate.known.length) {
    lines.push('', '## Known failures', '');
    for (const test of gate.known) {
      lines.push(`- ${escape(test.test_id)} — ${escape(test.reason)}`);
    }
  }
  if (gate.fixed.length) {
    lines.push('', '## Fixed', '');
    for (const test of gate.fixed) {
      lines.push(`- ${escape(test.test_id)}`);
    }
  }
  if (gate.warnings?.length) {
    lines.push('', '## Scope and baseline warnings', '');
    for (const warning of gate.warnings) {
      lines.push(`- ${escape(warning)}`);
    }
  }
  if (gate.incomplete.length) {
    lines.push('', `## Incomplete coverage (${gate.incomplete.length})`, '');
    for (const reason of gate.incomplete.slice(0, 10)) {
      lines.push(`- ${escape(reason)}`);
    }
  }
  return lines.join('\n') + '\n';
}

function functionalSelections(tree) {
  if (tree.truncated || !Array.isArray(tree.tree)) {
    throw new Error('Incomplete trusted functional inventory');
  }
  const required = Object.create(null);
  for (const entry of tree.tree) {
    const match = entry.path.match(
      /^tests\/functional_tests\/test_cases\/[^/]+\/([^/]+)\/model_config\.yaml$/,
    );
    if (match) {
      if (Object.hasOwn(required, match[1])) {
        throw new Error(`Duplicate functional selection: ${match[1]}`);
      }
      required[match[1]] = 1;
    }
    if (entry.path === 'tests/functional_tests/test_cases/common/moe_perf/__main__.py') {
      required.moe_perf = 1;
    }
  }
  if (!Object.keys(required).length) {
    throw new Error('No trusted functional selections found');
  }
  return required;
}

module.exports = {evaluateReport, renderReport, functionalSelections};
