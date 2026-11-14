# Issue 51 implementation plan

Approved in the conversation. Base: merged main `a85a913f673fe65afb22485b1e919f7fcaedebe1`.

## Required contract

- Keep `AdvancedPromptEngineer.main(task) -> Interaction` supported. Establish one finite validated provider scope before retrieval/rephrasing/planning when standalone; preserve artifact ownership and lazy imports.
- Move provider-only configuration/environment resolution/model validation into provider_client; keep execution_modes compatibility imports. Canonical defaults:40calls/120seconds. Add optional constructor max_calls/timeout_seconds; only tighten environment ceilings; zero calls dispatchnothing, timeout positivefinite; rejectinvalid/excessive values.
- Distinguish omitted and explicit model. Validate explicit selection; do not infer omission by the default model string.
- User chose caller-policy-wins: reuse a valid finite parent scope unchanged, including ledgeridentity/consumedcalls/deadline/model/cancellation; constructor limits governstandalone roots only. Reject incomplete/unboundedparentpolicy and conflicting explicit model beforedispatch.
- All reachable planning/embedding/generation/revision/selection/judging/complexity/nestedcalls shareledger, chargedonce. Embeddings sharecounts without chat-model allowlisting. PreservezeroSDKretries and boundedexplicitretries/waits.
- Normalize earlypolicytermination to safe non-successInteraction withunknownquality; preserve acceptedpartials on later stops. Checkdeadline/cancelbeforepublishingreturn, without rejecting exact-final-call success. Clean up ownedcontexts and preserveparentcontexts.
- Cooperative deadlines enforceSDKremainingtimeouts and rejectlatereturns; no hardPythonpreemptionclaim.

## Delivery and qualification

- [x] Configuration contract with RED→GREEN regressions; commit.
- [x] Standalone/main composition, early/late termination and cleanup; RED→GREEN; commit.
- [ ] Real provider-only direct, nested, revision, exact-cap, timeout/retry, cancellation, concurrent-run and HTTP compatibility qualification; README literalexample; commit.
- [ ] Guarded fullPython3.11/3.12, frontendguard/tests/lint/build, Chromium, productionimageCI, independentwholebranchreview andCRLFdiffchecks.
- [ ] FocusedPR, exact-source qualification/evidence and documentedlimitations.

## Boundaries

Preserve #48/#49result,ownership,revision,qualityandpartialdiagnostics. No paidcalls,deployment,merges,#35closure,newstreaming/shared-engineconcurrency/hardkill/durableresume. #50deliverables,#56/#57broaderconfig/setup,#59/#60/#61acceptance remainseparate. Useexistingoutboundguards anddeterministicfixtures;rootsoleproductionwriter, specialistsread-only.
