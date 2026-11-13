# Historical Verification Scripts (Archived)

These files are legacy, one-off diagnostic artifacts and historical verification scripts from earlier iterations of the repository:
- `final_verification.py`
- `verify_bug_fix.py`
- `simple_test.py`
- `minimal_test_bug_fix.py`
- `test_fixes.py`

They reference outdated environments or APIs that no longer represent production (such as a legacy `ConversationManager` class or `/workspaces/` paths).

### Authoritative Test Suite
Do not use scripts in this directory for validation. Use the authoritative production test commands documented in the root `README.md`:

```bash
pytest -q
```
