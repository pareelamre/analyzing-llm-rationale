const workflow = 'https://api.github.com/repos/pareelamre/analyzing-llm-rationale/actions/workflows/metaculus-futureeval.yml';
const active = ['queued', 'in_progress', 'waiting', 'pending', 'requested'];

export async function dispatch(token, fetcher = fetch) {
  try {
    if (typeof token !== 'string' || !token || /\s/.test(token)) throw new Error();
    const headers = {
      Authorization: `Bearer ${token}`,
      Accept: 'application/vnd.github+json',
      'X-GitHub-Api-Version': '2022-11-28',
      'User-Agent': 'foresea-metaculus-dispatch',
    };
    for (const status of active) {
      const response = await fetcher(`${workflow}/runs?branch=main&status=${status}&per_page=1`, {
        headers, redirect: 'error', signal: AbortSignal.timeout(10000),
      });
      if (response.status !== 200) throw new Error();
      const payload = await response.json();
      if (!payload || !Array.isArray(payload.workflow_runs)) throw new Error();
      if (payload.workflow_runs.length) return 'skipped';
    }
    const response = await fetcher(`${workflow}/dispatches`, {
      method: 'POST', headers: { ...headers, 'Content-Type': 'application/json' },
      body: JSON.stringify({ ref: 'main', inputs: { submit: 'true' } }),
      redirect: 'error', signal: AbortSignal.timeout(10000),
    });
    if (response.status !== 204) throw new Error();
    return 'dispatched';
  } catch {
    // Never expose fetch exceptions: they may contain credentials or response data.
    throw new Error('Tournament dispatch failed.');
  }
}

export default {
  fetch() { return new Response('Not found', { status: 404 }); },
  async scheduled(_event, env, ctx) {
    // Native runtime tracing supplies invocation/fetch spans and RED metrics.
    return ctx.tracing.enterSpan('metaculus.workflow.dispatch', async span => {
      const started = Date.now();
      let outcome = 'failure';
      span.setAttribute('service.name', 'foresea-metaculus-dispatch');
      span.setAttribute('deployment.environment.name', 'production');
      span.setAttribute('vcs.repository.url.full', 'https://github.com/pareelamre/analyzing-llm-rationale');
      try {
        outcome = await dispatch(env.METACULUS_GITHUB_DISPATCH_TOKEN);
      } catch {
        const error = new Error('Tournament dispatch failed.');
        span.recordException(error);
        // enterSpan marks thrown failures on the native invocation trace.
        throw error;
      } finally {
        span.setAttribute('outcome', outcome);
        const record = { operation: 'metaculus.workflow.dispatch', outcome, duration_ms: Date.now() - started };
        (outcome === 'failure' ? console.error : console.log)(JSON.stringify(record));
      }
    });
  },
};
