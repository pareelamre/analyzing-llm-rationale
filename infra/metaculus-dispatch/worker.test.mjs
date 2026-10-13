import test from 'node:test';
import assert from 'node:assert/strict';
import worker, { dispatch } from './worker.mjs';

const token = 'synthetic_secret_do_not_log';
const base = 'https://api.github.com/repos/pareelamre/analyzing-llm-rationale/actions/workflows/metaculus-futureeval.yml';
const active = ['queued', 'in_progress', 'waiting', 'pending', 'requested'];
function fixture(responder = () => new Response(JSON.stringify({ workflow_runs: [] }))) {
  const calls = [];
  return { calls, fetcher: async (url, options) => {
    calls.push({ url: String(url), options });
    return options.method === 'POST' ? new Response(null, { status: 204 }) : responder(url);
  } };
}
test('idle workflow dispatches fixed main submit target after every active-state check', async () => {
  const f = fixture();
  assert.equal(await dispatch(token, f.fetcher), 'dispatched');
  assert.equal(f.calls.length, 6);
  assert.deepEqual(f.calls.slice(0, 5).map(c => new URL(c.url).searchParams.get('status')), active);
  for (const { url, options } of f.calls) {
    assert.equal(new URL(url).origin, 'https://api.github.com');
    assert.equal(options.redirect, 'manual');
    assert.equal(options.headers.Authorization, `Bearer ${token}`);
    assert.ok(options.signal instanceof AbortSignal);
  }
  assert.equal(f.calls[5].url, `${base}/dispatches`);
  assert.deepEqual(JSON.parse(f.calls[5].options.body), { ref: 'main', inputs: { submit: 'true' } });
});
for (const status of active) test(`coalesces ${status}`, async () => {
  const f = fixture(url => new Response(JSON.stringify({ workflow_runs:
    new URL(url).searchParams.get('status') === status ? [{ id: 123 }] : [] })));
  assert.equal(await dispatch(token, f.fetcher), 'skipped');
  assert.ok(f.calls.every(c => c.options.method !== 'POST'));
});
for (const value of [undefined, '', ' ', 'a b', 'a\nb', 42]) test(`invalid credential ${JSON.stringify(value)}`, async () => {
  const f = fixture();
  await assert.rejects(dispatch(value, f.fetcher), /Tournament dispatch failed/);
  assert.equal(f.calls.length, 0);
});
for (const payload of [null, {}, { workflow_runs: null }, { workflow_runs: {} }]) test(`invalid response ${JSON.stringify(payload)}`, async () => {
  const f = fixture(() => new Response(JSON.stringify(payload)));
  await assert.rejects(dispatch(token, f.fetcher), /Tournament dispatch failed/);
  assert.equal(f.calls.length, 1);
});
for (const status of [302, 401, 403, 429, 500]) test(`rejects HTTP ${status}`, async () => {
  const f = fixture(() => new Response(token, { status }));
  await assert.rejects(dispatch(token, f.fetcher), err => {
    assert.equal(err.message, 'Tournament dispatch failed.');
    assert.equal(err.cause, undefined);
    assert.ok(!String(err).includes(token));
    return true;
  });
  assert.equal(f.calls.length, 1);
});
test('network and body errors are sanitized without retries', async () => {
  for (const fetcher of [async () => { throw new Error(token); }, async () => new Response(token)]) {
    await assert.rejects(dispatch(token, fetcher), err => err.message === 'Tournament dispatch failed.' && !err.cause);
  }
});
test('ambiguous or rejected POST is not repeated', async () => {
  let posts = 0;
  const f = fixture();
  await assert.rejects(dispatch(token, async (url, options) => {
    if (options.method === 'POST') { posts++; throw new Error(token); }
    return f.fetcher(url, options);
  }), /Tournament dispatch failed/);
  assert.equal(posts, 1);
});
test('public HTTP cannot trigger forecasts', async () => {
  assert.equal((await worker.fetch(new Request('https://test.invalid'))).status, 404);
});

test('scheduled handler records bounded outcome telemetry and awaits dispatch', async () => {
  const oldFetch = globalThis.fetch;
  const oldLog = console.log;
  const records = [];
  const attributes = {};
  const f = fixture();
  const ctx = { tracing: { enterSpan: async (name, callback) => {
    assert.equal(name, 'metaculus.workflow.dispatch');
    return callback({ setAttribute(key, value) { attributes[key] = value; } });
  } } };
  try {
    globalThis.fetch = f.fetcher;
    console.log = line => records.push(JSON.parse(line));
    await worker.scheduled({}, { METACULUS_GITHUB_DISPATCH_TOKEN: token }, ctx);
    assert.equal(f.calls.length, 6);
    assert.equal(attributes.outcome, 'dispatched');
    assert.equal(records[0].outcome, 'dispatched');
    assert.ok(Number.isFinite(records[0].duration_ms));
    assert.ok(!JSON.stringify({ records, attributes }).includes(token));
  } finally { globalThis.fetch = oldFetch; console.log = oldLog; }
});
test('scheduled failure records only sanitized exception and rejects', async () => {
  const oldError = console.error;
  const records = [];
  const errors = [];
  const attributes = {};
  const ctx = { tracing: { enterSpan: async (_name, callback) => callback({
    setAttribute(key, value) { attributes[key] = value; },
    recordException(error) { errors.push(error.message); },
  }) } };
  try {
    console.error = line => records.push(JSON.parse(line));
    await assert.rejects(worker.scheduled({}, {}, ctx), /Tournament dispatch failed/);
    assert.equal(attributes.outcome, 'failure');
    assert.deepEqual(errors, ['Tournament dispatch failed.']);
    assert.equal(records[0].outcome, 'failure');
    assert.ok(!JSON.stringify({ records, attributes, errors }).includes(token));
  } finally { console.error = oldError; }
});
