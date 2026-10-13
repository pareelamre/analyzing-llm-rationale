import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { Miniflare } from 'miniflare';

test('native Workers Request accepts every fixed dispatch request without network writes', async () => {
  const source = await readFile(new URL('./worker.mjs', import.meta.url), 'utf8');
  const harness = `import worker, { dispatch } from './worker.mjs';
    export default { async fetch(_request, _env, ctx) {
      let calls = 0;
      const fetcher = async (url, options) => {
        new Request(url, options); // Real workerd validation, unlike Node mocks.
        calls++;
        return options.method === 'POST' ? new Response(null, { status: 204 })
          : Response.json({ workflow_runs: [] });
      };
      const outcome = await dispatch('synthetic_secret_do_not_log', fetcher);
      const original = globalThis.fetch;
      try {
        globalThis.fetch = fetcher;
        await worker.scheduled({}, { METACULUS_GITHUB_DISPATCH_TOKEN: 'synthetic_secret_do_not_log' }, ctx);
      } finally { globalThis.fetch = original; }
      return Response.json({ outcome, calls });
    }};`;
  const mf = new Miniflare({ workers: [{ config: {
    name: 'contract-fixture', compatibilityDate: '2026-10-13',
    observability: { enabled: true, traces: { enabled: true } },
    manifest: { mainModule: 'fixture.mjs', modules: {
      'fixture.mjs': { type: 'esm', contents: harness },
      'worker.mjs': { type: 'esm', contents: source },
    } },
  } }] });
  try {
    const response = await mf.dispatchFetch('https://fixture.invalid');
    assert.equal(response.status, 200, await response.clone().text());
    assert.deepEqual(await response.json(), { outcome: 'dispatched', calls: 12 });
  } finally { await mf.dispose(); }
});
