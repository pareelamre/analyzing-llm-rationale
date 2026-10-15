import { bindings, defineConfig, triggers } from 'cf/config';

export default defineConfig({
  accountId: 'bc1ed489cc61299392eaedf43595f990',
  worker: {
    name: 'foresea-metaculus-dispatch',
    entrypoint: './worker.mjs',
    compatibilityDate: '2026-10-13',
    workersDev: false,
    previewUrls: false,
    env: { METACULUS_GITHUB_DISPATCH_TOKEN: bindings.secret() },
    triggers: [triggers.scheduled({ schedule: '3,13,23,33,43,53 * * * *' })],
    observability: {
      enabled: true,
      logs: { enabled: true, headSamplingRate: 1 },
      traces: { enabled: true, headSamplingRate: 1 },
    },
  },
});
