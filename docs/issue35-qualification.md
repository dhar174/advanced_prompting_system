# Issue #35 qualification after review

Issue #35 remains open pending human acceptance. Tree of Thoughts (#36) remains gated; #20 and #22 remain outside this qualification.

## Historical results and corrected claims

The original Windows report recorded 243 backend tests, 126 frontend tests and six browser scenarios. These were historical results, not clean Linux/container certification. Linux run [43436229893](https://github.com/dhar174/advanced_prompting_system/actions/runs/43436229893) failed with one non-browser pass and five browser setup errors because installation and guarded execution used different Chromium caches. The previous answer-copy change could also stop after one plan step; the updated upstream engine retains step-local answers and requires plan completion.

## Updated implementation and evidence

PR #39 head `17cba78282d56ce19eeb6cc2929484f30715272a` passes [exact-head CI](https://github.com/dhar174/advanced_prompting_system/actions/runs/43469704838). Local Python 3.11 and 3.12 each pass 248 backend tests. Repairs include portable path rejection, opaque artifact owners, active-scope foreign-owner denial, authoritative explicit model allowlists, shared provider limits, revision acceptance and honest completion.

PR #40 head `26122f3666cede50a070d4f060461d5cc2e997a6` passes [exact-head CI](https://github.com/dhar174/advanced_prompting_system/actions/runs/43469708884). Local Python 3.11 passes 259 backend tests and the frontend passes 147 tests, lint and build. Authentication recovery clears stale 401 state; token entry clears and returns to masking on application. Legacy execution uses the shared dispatcher.

PR #41 merges those upstream repairs and adds real HTTP coverage for both mutation paths, restrictive model/call/deadline policies, actual escaping writer calls and distinct registered artifact ownership. Two-step solo/team scenarios require a real spaCy model installed during setup; the production-image smoke separately exercises the missing-model fallback. Test execution never downloads assets.

The expanded Linux Chromium suite passes eight scenarios locally, including wrong/missing token rejection, authorized catalogue and both execution modes, logout, complete-response/Event Stream error safety, and registered-owner access controls. The SPA has no feedback form: authenticated feedback is exercised through browser HTTP using the application token entered through the UI. This limitation is explicit, not a claim of feedback-form coverage.

## Reproduction and retained evidence

Install `.[dev,e2e,server]`, download `en_core_web_sm`, and install Chromium/OS dependencies during setup. Set `PLAYWRIGHT_BROWSERS_PATH` to the same absolute directory for installation and execution. Build the frontend, then run the guarded wrappers:

```sh
bash tools/run-offline.sh -q
bash tools/run-frontend-offline.sh guard
bash tools/run-frontend-offline.sh test
bash tools/run-frontend-offline.sh lint
bash tools/run-frontend-offline.sh build
```

Each browser invocation retains a unique `.qualification/logs/browser-e2e/session_*/manifest.json`. Scenario records reference separate ledgers, redacted HTTP/server logs, trace ZIPs, screenshots and artifact hashes. Counts/hashes and required scenario coverage are checked; teardown failures update recorded outcomes. CI uploads evidence even on failure. Provider call counts are fixture dispatch counts; they do not imply available token usage or reasoning-quality certification.

The production-image job builds a tracked-files-only archive, installs runtime dependencies (no dev extras or optional NLP assets), launches the shipped Gunicorn entry point as the non-root image user and probes health/readiness/SPA and both modes. The test-only provider/guard is mounted at qualification time, with container networking disabled; no production test endpoint is added.

A local Docker build was attempted but its npm dependency-download stage could not reach the registry from the Docker network. This does not establish an image pass. Production-image qualification must be judged from the exact-head CI job, independently of dev-environment results. No deployment or paid provider calls are part of these checks.
