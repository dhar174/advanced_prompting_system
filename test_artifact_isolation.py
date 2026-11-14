"""Dedicated regression tests for PR #39 artifact isolation and identity separation."""

from __future__ import annotations

import os
import threading
from pathlib import Path
from unittest.mock import Mock, patch

from test_collaboration_results import typed_fixture
import pytest

import conversation_manager
import output_generator
from execution_modes import RunRequest, execute_run


@pytest.fixture(autouse=True)
def isolated_artifacts_env(monkeypatch, tmp_path):
    """Ensure every test runs with clean registry state and isolated artifact base dir."""
    artifacts_root = tmp_path / "aps_artifacts"
    monkeypatch.setenv("APS_ARTIFACTS_DIR", str(artifacts_root))
    output_generator.clear_tracked_artifacts()
    yield artifacts_root
    output_generator.clear_tracked_artifacts()


class TestArtifactIdentityAndCollidingLabels:
    """Verify colliding correlation labels never alias directories or ownership."""

    def test_colliding_labels_isolated_in_run_scopes(self):
        with output_generator.artifact_run_scope("tenant/a") as owner_a:
            dir_a = output_generator.get_run_artifact_dir("tenant/a")
            path_a = output_generator.generate_text_file_output("alpha secret", "report.txt")

            assert owner_a != "tenant/a"
            assert output_generator.get_active_run_id() == owner_a
            assert os.path.commonpath([dir_a, path_a]) == dir_a

            ref_a = output_generator.get_tracked_artifact_ref(path_a, run_id="tenant/a")
            assert ref_a is not None
            assert ref_a.run_id == owner_a
            assert ref_a.correlation_id == "tenant/a"
            assert output_generator.read_tracked_artifact(path_a, run_id="tenant/a") == "alpha secret"
            assert output_generator.read_tracked_artifact(path_a, run_id=owner_a) == "alpha secret"
            assert len(output_generator.list_run_artifacts("tenant/a")) == 1

            # Foreign colliding label cannot list/read while tenant/a is active
            assert output_generator.list_run_artifacts("tenant:a") == []
            assert output_generator.is_tracked_artifact(path_a, run_id="tenant:a") is False
            assert output_generator.read_tracked_artifact(path_a, run_id="tenant:a") is None

        with output_generator.artifact_run_scope("tenant:a") as owner_b:
            dir_b = output_generator.get_run_artifact_dir("tenant:a")
            assert owner_b != owner_a
            assert dir_b != dir_a

            # Foreign colliding scope cannot see or read tenant/a's artifact
            assert output_generator.list_run_artifacts() == []
            assert output_generator.list_run_artifacts("tenant:a") == []
            assert output_generator.is_tracked_artifact(path_a) is False
            assert output_generator.is_tracked_artifact(path_a, run_id="tenant:a") is False
            assert output_generator.read_tracked_artifact(path_a) is None
            assert output_generator.read_tracked_artifact(path_a, run_id="tenant:a") is None

            # Writing the same filename in tenant:a must not overwrite tenant/a's file
            path_b = output_generator.generate_text_file_output("beta secret", "report.txt")
            assert path_b != path_a
            assert os.path.commonpath([dir_b, path_b]) == dir_b
            assert Path(path_a).read_text(encoding="utf-8") == "alpha secret"
            assert Path(path_b).read_text(encoding="utf-8") == "beta secret"

            refs_b = output_generator.list_run_artifacts("tenant:a")
            assert [r.path for r in refs_b] == [path_b]
            assert refs_b[0].correlation_id == "tenant:a"

        # After both scopes exit, each owner reads only its own artifact
        assert output_generator.read_tracked_artifact(path_a, run_id="tenant/a") == "alpha secret"
        assert output_generator.read_tracked_artifact(path_b, run_id="tenant/a") is None
        assert output_generator.read_tracked_artifact(path_b, run_id="tenant:a") == "beta secret"
        assert output_generator.read_tracked_artifact(path_a, run_id="tenant:a") is None

    def test_direct_unscoped_colliding_run_ids_never_alias(self):
        labels = ["tenant/a", "tenant:a", "tenant_a", "tenant?a", "..", "///", "default"]
        dirs = {label: output_generator.get_run_artifact_dir(label) for label in labels}
        assert len(set(dirs.values())) == len(labels)

        p_slash = output_generator.resolve_output_filepath("shared.txt", run_id="tenant/a")
        Path(p_slash).write_text("slash-owner", encoding="utf-8")
        output_generator.track_generated_artifact(p_slash, run_id="tenant/a")

        assert output_generator.read_tracked_artifact(p_slash, run_id="tenant/a") == "slash-owner"
        slash_refs = output_generator.list_run_artifacts("tenant/a")
        assert len(slash_refs) == 1
        assert slash_refs[0].correlation_id == "tenant/a"
        assert output_generator.read_tracked_artifact(p_slash, run_id=slash_refs[0].run_id) == "slash-owner"

        for foreign in ("tenant:a", "tenant_a", "tenant?a", "..", "default"):
            assert output_generator.list_run_artifacts(foreign) == []
            assert output_generator.is_tracked_artifact(p_slash, run_id=foreign) is False
            assert output_generator.read_tracked_artifact(p_slash, run_id=foreign) is None

        # Foreign colliding label writing same filename writes to a distinct path
        p_colon = output_generator.resolve_output_filepath("shared.txt", run_id="tenant:a")
        assert p_colon != p_slash
        Path(p_colon).write_text("colon-owner", encoding="utf-8")
        output_generator.track_generated_artifact(p_colon, run_id="tenant:a")

        assert Path(p_slash).read_text(encoding="utf-8") == "slash-owner"
        assert output_generator.read_tracked_artifact(p_colon, run_id="tenant:a") == "colon-owner"
        assert output_generator.read_tracked_artifact(p_colon, run_id="tenant/a") is None


class TestReusedAndParallelRunIsolation:
    """Verify reused correlation IDs and parallel runs remain strictly isolated."""

    def test_reused_correlation_id_isolated_across_sequential_runs(self):
        with output_generator.artifact_run_scope("reused_corr") as owner_1:
            p1 = output_generator.generate_text_file_output("first-run-secret", "summary.txt")
            assert output_generator.read_tracked_artifact(p1, run_id="reused_corr") == "first-run-secret"

        with output_generator.artifact_run_scope("reused_corr") as owner_2:
            assert owner_2 != owner_1
            # Reused label starts with an empty artifact view and cannot read Run 1's file
            assert output_generator.list_run_artifacts("reused_corr") == []
            assert output_generator.list_run_artifacts() == []
            assert output_generator.is_tracked_artifact(p1, run_id="reused_corr") is False
            assert output_generator.read_tracked_artifact(p1, run_id="reused_corr") is None
            assert output_generator.read_tracked_artifact(p1) is None

            p2 = output_generator.generate_text_file_output("second-run-secret", "summary.txt")
            assert p2 != p1
            assert Path(p1).read_text(encoding="utf-8") == "first-run-secret"
            assert Path(p2).read_text(encoding="utf-8") == "second-run-secret"
            assert [r.path for r in output_generator.list_run_artifacts("reused_corr")] == [p2]
            assert output_generator.read_tracked_artifact(p2, run_id="reused_corr") == "second-run-secret"

        # Original opaque owner handle retains isolated access to Run 1 only
        assert output_generator.read_tracked_artifact(p1, run_id=owner_1) == "first-run-secret"
        assert output_generator.read_tracked_artifact(p2, run_id=owner_1) is None
        assert output_generator.read_tracked_artifact(p1, run_id=owner_2) is None
        assert output_generator.read_tracked_artifact(p2, run_id=owner_2) == "second-run-secret"
        assert output_generator.read_tracked_artifact(p1, run_id="reused_corr") is None

    def test_parallel_runs_with_identical_correlation_id_stay_isolated(self):
        barrier = threading.Barrier(2)
        results = {}
        errors = []

        def worker(idx: int) -> None:
            try:
                with output_generator.artifact_run_scope("shared_parallel_label") as owner_id:
                    barrier.wait(timeout=5)
                    path = output_generator.generate_text_file_output(
                        f"payload-{idx}",
                        "parallel.txt",
                    )
                    barrier.wait(timeout=5)
                    listed = output_generator.list_run_artifacts("shared_parallel_label")
                    content = output_generator.read_tracked_artifact(
                        path, run_id="shared_parallel_label"
                    )
                    results[idx] = (owner_id, path, [r.path for r in listed], content)
            except Exception as exc:  # pragma: no cover - diagnostic capture
                errors.append(exc)

        t1 = threading.Thread(target=worker, args=(1,))
        t2 = threading.Thread(target=worker, args=(2,))
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        assert errors == []
        owner_1, path_1, listed_1, content_1 = results[1]
        owner_2, path_2, listed_2, content_2 = results[2]

        assert owner_1 != owner_2
        assert path_1 != path_2
        assert listed_1 == [path_1]
        assert listed_2 == [path_2]
        assert content_1 == "payload-1"
        assert content_2 == "payload-2"
        assert output_generator.read_tracked_artifact(path_2, run_id=owner_1) is None
        assert output_generator.read_tracked_artifact(path_1, run_id=owner_2) is None

    def test_execute_run_colliding_and_reused_correlation_ids_cannot_cross_read(self, monkeypatch):
        def run_owner_creating_artifact(*args, **kwargs):
            path = output_generator.generate_json_output({"secret": "tenant-a-only"}, "out.json")
            return typed_fixture(([{"role": "assistant", "content": "Done"}], [], path), artifact=True)

        monkeypatch.setattr(conversation_manager, "run_conversation_result", run_owner_creating_artifact)
        res_a = execute_run(
            RunRequest(
                mode="collaborative",
                task_prompt="Produce report",
                correlation_id="tenant/a",
            )
        )
        assert res_a.status == "completed"
        assert "tenant-a-only" in res_a.final_output
        assert len(res_a.artifacts) == 1
        owner_path = res_a.artifacts[0].path
        assert res_a.artifacts[0].correlation_id == "tenant/a"
        assert res_a.artifacts[0].run_id != "tenant/a"

        # Probe 1: colliding label "tenant:a" returns owner_path as output string
        monkeypatch.setattr(
            conversation_manager,
            "run_conversation_result",
            Mock(return_value=typed_fixture(([{"role": "assistant", "content": "Done"}], [], owner_path))),
        )
        res_colon = execute_run(
            RunRequest(
                mode="collaborative",
                task_prompt="Probe",
                correlation_id="tenant:a",
            )
        )
        assert "tenant-a-only" not in res_colon.final_output
        assert res_colon.final_output == owner_path
        assert res_colon.artifacts == []

        # Probe 2: reused exact correlation_id "tenant/a" cannot read or list prior run's artifact
        res_reused = execute_run(
            RunRequest(
                mode="collaborative",
                task_prompt="Probe reused",
                correlation_id="tenant/a",
            )
        )
        assert "tenant-a-only" not in res_reused.final_output
        assert res_reused.final_output == owner_path
        assert res_reused.artifacts == []


class TestTraversalSymlinkAndGeneratorErrors:
    """Verify traversal, symlink escape rejection, and all 7 generators error handling."""

    def test_traversal_and_invalid_paths_rejected_in_run_scope(self, tmp_path):
        outside = tmp_path / "outside.txt"
        with output_generator.artifact_run_scope("tenant/safe"):
            for bad_rel in ("../escape.txt", "sub/../../escape.txt", "..\\escape.txt", ".", "/abs_root.txt"):
                with pytest.raises(PermissionError):
                    output_generator.resolve_output_filepath(bad_rel)

            with pytest.raises(PermissionError):
                output_generator.resolve_output_filepath(str(outside))

            with pytest.raises(ValueError):
                output_generator.resolve_output_filepath("   ")
            with pytest.raises(ValueError):
                output_generator.resolve_output_filepath("bad\x00name.txt")
            with pytest.raises(ValueError):
                output_generator.artifact_run_scope("bad\x00run").__enter__()

            if os.name == "nt":
                with pytest.raises(PermissionError):
                    output_generator.resolve_output_filepath("C:drive_rel.txt")

    def test_symlink_escapes_rejected_for_files_and_parent_dirs(self, tmp_path):
        outside_file = tmp_path / "host_secret.txt"
        outside_file.write_text("TOP_SECRET", encoding="utf-8")
        outside_dir = tmp_path / "outside_dir"
        outside_dir.mkdir()

        with output_generator.artifact_run_scope("tenant/sym"):
            run_dir = output_generator.get_run_artifact_dir("tenant/sym")
            real_path = output_generator.generate_text_file_output("owned", "legit.txt")

            link_file = os.path.join(run_dir, "escape_file.txt")
            link_dir = os.path.join(run_dir, "escape_dir")
            link_to_legit = os.path.join(run_dir, "alias_legit.txt")
            try:
                os.symlink(str(outside_file), link_file)
                os.symlink(str(outside_dir), link_dir, target_is_directory=True)
                os.symlink(real_path, link_to_legit)
            except (OSError, NotImplementedError):
                pytest.skip("OS does not permit creating symlinks in this environment")

            with pytest.raises(PermissionError):
                output_generator.resolve_output_filepath("escape_file.txt")
            with pytest.raises(PermissionError):
                output_generator.resolve_output_filepath(link_file)
            with pytest.raises(PermissionError):
                output_generator.resolve_output_filepath("escape_dir/pwned.txt")
            with pytest.raises(PermissionError):
                output_generator.track_generated_artifact(link_file)

            assert output_generator.is_tracked_artifact(link_file) is False
            assert output_generator.read_tracked_artifact(link_file) is None
            assert output_generator.is_tracked_artifact(link_to_legit) is False
            assert output_generator.read_tracked_artifact(link_to_legit) is None

    def test_symlink_detection_enforced_even_without_os_symlink_privilege(self, monkeypatch):
        with output_generator.artifact_run_scope("tenant/sym_mock"):
            real_path = output_generator.generate_text_file_output("owned", "legit.txt")
            monkeypatch.setattr(os.path, "islink", lambda p: str(p).endswith("legit.txt"))
            with pytest.raises(PermissionError):
                output_generator.resolve_output_filepath("legit.txt")
            with pytest.raises(PermissionError):
                output_generator.track_generated_artifact(real_path)
            assert output_generator.is_tracked_artifact(real_path) is False
            assert output_generator.read_tracked_artifact(real_path) is None

    def test_all_seven_generators_succeed_and_retain_io_error_propagation(self):
        with output_generator.artifact_run_scope("tenant/all7") as owner_id:
            paths = [
                output_generator.generate_json_output({"ok": True}, "a.json"),
                output_generator.generate_pdf_output("PDF text", "b.pdf"),
                output_generator.generate_text_file_output("plain", "c.txt"),
                output_generator.generate_html_output("<b>hi</b>", "d.html"),
                output_generator.generate_python_script("```python\nx = 1\n```", "e.py"),
                output_generator.generate_code_snippet("y = 2", "f.py"),
                output_generator.generate_csv_output([{"k": "v"}], "g.csv"),
            ]
            assert len(paths) == 7
            refs = output_generator.list_run_artifacts("tenant/all7")
            assert len(refs) == 7
            assert all(r.run_id == owner_id and r.correlation_id == "tenant/all7" for r in refs)
            # PDF returns None on read_tracked_artifact; text formats return content
            assert output_generator.read_tracked_artifact(paths[1], run_id="tenant/all7") is None
            assert output_generator.read_tracked_artifact(paths[2], run_id="tenant/all7") == "plain"

        with output_generator.artifact_run_scope("tenant/ioerr"):
            with patch("builtins.open", side_effect=PermissionError("denied")):
                with pytest.raises(PermissionError):
                    output_generator.generate_json_output({"a": 1}, "fail.json")
                with pytest.raises(PermissionError):
                    output_generator.generate_pdf_output("x", "fail.pdf")
                with pytest.raises(PermissionError):
                    output_generator.generate_text_file_output("x", "fail.txt")
                with pytest.raises(PermissionError):
                    output_generator.generate_html_output("x", "fail.html")
                with pytest.raises(PermissionError):
                    output_generator.generate_python_script("x = 1", "fail_s.py")
                with pytest.raises(PermissionError):
                    output_generator.generate_code_snippet("x = 1", "fail_c.py")
                with pytest.raises(PermissionError):
                    output_generator.generate_csv_output("a,b\n1,2", "fail.csv")


def test_active_scope_cannot_select_known_foreign_owner_or_label():
    with output_generator.artifact_run_scope('owner-a') as owner:
        path = output_generator.generate_text_file_output('private', 'report.txt')
        assert output_generator.read_tracked_artifact(path) == 'private'
    with output_generator.artifact_run_scope('owner-b'):
        for foreign in (owner, 'owner-a'):
            assert output_generator.list_run_artifacts(foreign) == []
            assert output_generator.read_tracked_artifact(path, run_id=foreign) is None
            with pytest.raises(PermissionError):
                output_generator.resolve_output_filepath('report.txt', run_id=foreign)
    assert Path(path).read_text() == 'private'


class TestWindowsPathNormalizationAndConcurrency:
    """Verify mixed-case containment and race-free concurrent artifact root creation."""

    def test_mixed_case_path_containment_and_traversal_denial(self, monkeypatch, tmp_path):
        import ntpath

        mixed_base = tmp_path / "MixedCaseArtifacts"
        mixed_base.mkdir()
        effective_base = str(mixed_base).swapcase() if os.name == "nt" else str(mixed_base)
        monkeypatch.setenv("APS_ARTIFACTS_DIR", effective_base)

        with output_generator.artifact_run_scope("tenant/case") as owner_id:
            run_dir = output_generator.get_run_artifact_dir("tenant/case")
            assert output_generator._is_within_directory(run_dir, effective_base) is True

            generated = output_generator.generate_text_file_output("case-ok", "nested/note.txt")
            assert Path(generated).read_text(encoding="utf-8") == "case-ok"
            assert output_generator.read_tracked_artifact(generated, run_id=owner_id) == "case-ok"

            if os.name == "nt":
                assert output_generator._is_within_directory(run_dir.swapcase(), run_dir) is True
                # Swapped-case absolute path inside run_dir resolves cleanly; outside path is still denied.
                resolved = output_generator.resolve_output_filepath(
                    os.path.join(run_dir.swapcase(), "nested", "second.txt")
                )
                assert os.path.normcase(resolved) == os.path.normcase(
                    os.path.join(run_dir, "nested", "second.txt")
                )

            outside = str(tmp_path / "outside_escape.txt")
            bad_paths = ["../escape.txt", ".", outside]
            if os.name == "nt":
                bad_paths.append(outside.swapcase())
            for bad in bad_paths:
                with pytest.raises(PermissionError):
                    output_generator.resolve_output_filepath(bad)

        # Verify _is_within_directory / _paths_equal honor Windows normcase semantics on all platforms.
        monkeypatch.setattr(output_generator.os.path, "normcase", ntpath.normcase)
        monkeypatch.setattr(output_generator.os.path, "normpath", ntpath.normpath)
        monkeypatch.setattr(output_generator.os.path, "commonpath", ntpath.commonpath)
        assert output_generator._is_within_directory(r"c:\APS\Run_1\file.txt", r"C:/aps/run_1") is True
        assert output_generator._paths_equal(r"c:\APS\Run_1", r"C:/aps/run_1/") is True
        assert output_generator._is_within_directory(r"D:\APS\Run_1\file.txt", r"C:\APS\Run_1") is False

    def test_uncreated_artifacts_dir_concurrent_threads_zero_permission_errors(self, monkeypatch, tmp_path):
        uncreated_root = tmp_path / "fresh_parent" / "uncreated_aps_artifacts"
        assert not uncreated_root.exists()
        monkeypatch.setenv("APS_ARTIFACTS_DIR", str(uncreated_root))

        num_threads = 16
        barrier = threading.Barrier(num_threads)
        errors = []
        outputs = {}

        def scoped_worker(idx: int) -> None:
            try:
                barrier.wait(timeout=5)
                with output_generator.artifact_run_scope(f"concurrent-{idx}") as owner_id:
                    run_dir = output_generator.get_run_artifact_dir()
                    path = output_generator.generate_text_file_output(
                        f"thread-{idx}",
                        f"sub/item_{idx}.txt",
                    )
                    outputs[idx] = (
                        owner_id,
                        run_dir,
                        path,
                        output_generator.read_tracked_artifact(path),
                    )
            except Exception as exc:  # pragma: no cover - diagnostic capture
                errors.append((idx, exc))

        threads = [threading.Thread(target=scoped_worker, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)

        assert errors == []
        assert len(outputs) == num_threads
        assert uncreated_root.is_dir()
        seen_dirs = set()
        for idx, (owner_id, run_dir, path, content) in outputs.items():
            assert content == f"thread-{idx}"
            assert os.path.isdir(run_dir)
            assert os.path.isfile(path)
            seen_dirs.add(os.path.normcase(run_dir))
        assert len(seen_dirs) == num_threads

    def test_import_output_generator_does_not_create_artifacts_dir(self, tmp_path):
        import subprocess
        import sys

        uncreated = tmp_path / "import_only_no_create"
        env = os.environ.copy()
        env["APS_ARTIFACTS_DIR"] = str(uncreated)
        proc = subprocess.run(
            [sys.executable, "-c", "import output_generator"],
            cwd=os.path.dirname(__file__),
            env=env,
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stderr
        assert not uncreated.exists()
