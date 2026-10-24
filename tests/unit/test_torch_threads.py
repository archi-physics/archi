"""The CPU thread count compose gives torch in the embedding containers.

Unset, torch starts one thread per visible core, and a container sees every host
core. On the 32-core dev host that made each embedding call ~3 s instead of
~16 ms, so the default is a bounded 8.
"""

import os

import pytest

from src.cli.utils import torch_threads
from src.cli.utils.torch_threads import (
    DEFAULT_TORCH_THREADS,
    chart_torch_threads,
    configured_torch_threads,
    deployment_env_values,
    resolve_torch_threads,
    visible_cpu_count,
)


@pytest.fixture
def cpus(monkeypatch):
    """Set the CPU count this process sees, for the default's cap."""

    def _set(count):
        monkeypatch.setattr(torch_threads, "visible_cpu_count", lambda: count)

    return _set


class TestDefault:
    def test_the_default_is_eight(self):
        assert DEFAULT_TORCH_THREADS == 8

    @pytest.mark.parametrize(
        "config",
        [
            None,
            {},
            {"data_manager": None},
            {"data_manager": {}},
            {"data_manager": {"torch_threads": None}},
        ],
    )
    def test_absent_key_gives_the_default(self, cpus, config):
        cpus(32)

        assert resolve_torch_threads(config) == DEFAULT_TORCH_THREADS


class TestDefaultIsCappedAtVisibleCpus:
    """A default of 8 on a host with fewer CPUs would oversubscribe it."""

    @pytest.mark.parametrize(("visible", "expected"), [(1, 1), (4, 4), (7, 7)])
    def test_fewer_than_eight_cpus_caps_the_default(self, cpus, visible, expected):
        cpus(visible)

        assert resolve_torch_threads({}) == expected

    @pytest.mark.parametrize("visible", [8, 9, 32])
    def test_eight_or_more_cpus_keeps_eight(self, cpus, visible):
        cpus(visible)

        assert resolve_torch_threads({}) == DEFAULT_TORCH_THREADS

    def test_an_unknown_cpu_count_keeps_eight(self, cpus):
        """os.cpu_count() returns None when the count cannot be determined."""
        cpus(None)

        assert resolve_torch_threads({}) == DEFAULT_TORCH_THREADS

    def test_an_explicit_value_is_not_capped(self, cpus):
        cpus(4)

        assert resolve_torch_threads({"data_manager": {"torch_threads": 16}}) == 16


class TestVisibleCpuCount:
    def test_the_affinity_mask_counts_when_present(self, monkeypatch):
        monkeypatch.setattr(
            os, "sched_getaffinity", lambda pid: {0, 1, 2}, raising=False
        )
        monkeypatch.setattr(os, "cpu_count", lambda: 64)

        assert visible_cpu_count() == 3

    def test_the_core_count_counts_without_an_affinity_mask(self, monkeypatch):
        monkeypatch.delattr(os, "sched_getaffinity", raising=False)
        monkeypatch.setattr(os, "cpu_count", lambda: 6)

        assert visible_cpu_count() == 6


class TestEnvFileValueIsKept:
    """An OMP_NUM_THREADS the deployment's env file sets reaches the container.

    Compose's ``environment`` entry takes precedence over ``env_file``, so
    rendering one would silently replace the operator's value. None means
    "render no entry".
    """

    @pytest.mark.parametrize("value", ["4", "16", "4,2"])
    def test_absent_key_and_env_value_renders_nothing(self, cpus, value):
        cpus(32)

        assert resolve_torch_threads({}, {"OMP_NUM_THREADS": value}) is None

    def test_an_env_mkl_value_alone_renders_nothing(self, cpus):
        """Torch sizes its pool from MKL_NUM_THREADS ahead of OMP_NUM_THREADS."""
        cpus(32)

        assert resolve_torch_threads({}, {"MKL_NUM_THREADS": "12"}) is None

    @pytest.mark.parametrize("name", ["OMP_NUM_THREADS", "MKL_NUM_THREADS"])
    def test_an_explicit_value_wins_over_the_env_file(self, cpus, name):
        cpus(32)
        config = {"data_manager": {"torch_threads": 2}}

        assert resolve_torch_threads(config, {name: "4"}) == 2

    @pytest.mark.parametrize("name", ["OMP_NUM_THREADS", "MKL_NUM_THREADS"])
    @pytest.mark.parametrize("value", [None, "", "  "])
    def test_an_empty_env_value_counts_as_unset(self, cpus, name, value):
        cpus(32)

        assert resolve_torch_threads({}, {name: value}) == 8

    def test_other_env_values_do_not_count(self, cpus):
        cpus(32)

        assert resolve_torch_threads({}, {"PG_PASSWORD": "x"}) == 8


class TestDeploymentEnvValues:
    def test_reads_the_env_file_in_the_deployment_directory(self, tmp_path):
        (tmp_path / ".env").write_text("PG_PASSWORD=x\nOMP_NUM_THREADS=4\n")

        assert deployment_env_values(tmp_path)["OMP_NUM_THREADS"] == "4"

    def test_no_env_file_gives_no_values(self, tmp_path):
        assert deployment_env_values(tmp_path) == {}


class TestOverride:
    @pytest.mark.parametrize("value", [1, 4, 16, 64])
    def test_a_positive_integer_is_used(self, cpus, value):
        cpus(32)

        assert (
            resolve_torch_threads({"data_manager": {"torch_threads": value}}) == value
        )


class TestRefusal:
    @pytest.mark.parametrize("value", [0, -1, True, False, 8.0, "8", [8]])
    def test_anything_but_a_positive_integer_is_refused(self, value):
        with pytest.raises(ValueError, match=r"data_manager\.torch_threads"):
            resolve_torch_threads({"data_manager": {"torch_threads": value}})

    @pytest.mark.parametrize("value", [0, "8"])
    def test_the_config_check_alone_refuses_it_too(self, value):
        """``validate_configs`` calls this before create or restart writes a file."""
        with pytest.raises(ValueError, match=r"data_manager\.torch_threads"):
            configured_torch_threads({"data_manager": {"torch_threads": value}})

    @pytest.mark.parametrize(
        ("config", "expected"),
        [({}, None), ({"data_manager": {"torch_threads": 4}}, 4)],
    )
    def test_the_config_check_returns_the_explicit_value(self, config, expected):
        assert configured_torch_threads(config) == expected


class TestChartTorchThreads:
    """The Helm chart's count: the pods run on cluster nodes, not on this machine.

    Neither this machine's CPU count nor its deployment ``.env`` says anything
    about a node, so the chart gets the explicit value or a plain 8.
    """

    @pytest.mark.parametrize("visible", [1, 4, 32, None])
    def test_absent_key_gives_eight_whatever_this_machine_has(self, cpus, visible):
        cpus(visible)

        assert chart_torch_threads({}) == DEFAULT_TORCH_THREADS

    def test_an_explicit_value_is_used(self, cpus):
        cpus(4)

        assert chart_torch_threads({"data_manager": {"torch_threads": 16}}) == 16

    @pytest.mark.parametrize("value", [0, "8"])
    def test_an_invalid_value_is_refused(self, value):
        with pytest.raises(ValueError, match=r"data_manager\.torch_threads"):
            chart_torch_threads({"data_manager": {"torch_threads": value}})
