from dataclasses import FrozenInstanceError

import pytest
import yaml

import src.evaluation.qa.profile as qa_profile


class TestProfileValidation:
    def test_builtin_profile_uses_the_profile_schema(self):
        profile = qa_profile.load_profile(None)

        assert profile.to_dict() == {
            "version": 1,
            "qa": {
                "atoms_extractor": {
                    "provider": "openai",
                    "model": "gpt-5.6-terra",
                },
                "evaluator": {
                    "provider": "openai",
                    "model": "gpt-5.6-terra",
                },
            },
        }

    def test_builtin_profile_is_immutable(self):
        profile = qa_profile.load_profile(None)

        with pytest.raises(FrozenInstanceError):
            profile.atoms_extractor.model = "changed"

    def test_accepts_yaml_profile_regardless_of_filename_suffix(self, tmp_path):
        profile_path = tmp_path / "profile.txt"
        profile_path.write_text(
            "version: 1\n"
            "qa:\n"
            "  atoms_extractor: {provider: openai, model: extractor}\n"
            "  evaluator: {provider: openai, model: evaluator}\n"
        )

        assert qa_profile.load_profile(profile_path).to_dict() == {
            "version": 1,
            "qa": {
                "atoms_extractor": {
                    "provider": "openai",
                    "model": "extractor",
                },
                "evaluator": {
                    "provider": "openai",
                    "model": "evaluator",
                },
            },
        }

    def test_resolves_optional_timeout_into_dataclass(self, tmp_path):
        profile_path = tmp_path / "profile.yaml"
        profile_path.write_text(
            "version: 1\n"
            "qa:\n"
            "  atoms_extractor:\n"
            "    provider: ' openai '\n"
            "    model: ' extractor '\n"
            "    timeout: 12.5\n"
            "  evaluator: {provider: openai, model: evaluator}\n"
        )

        descriptor = qa_profile.load_profile(profile_path).atoms_extractor

        assert descriptor == qa_profile.ModelDescriptor(
            provider="openai",
            model="extractor",
            timeout=12.5,
        )
        assert descriptor.provider_kwargs() == {
            "temperature": 0,
            "timeout": 12.5,
        }

    def test_rejects_unknown_profile_fields(self, tmp_path):
        profile_path = tmp_path / "profile.yaml"
        profile_path.write_text(
            "version: 1\n"
            "unexpected: true\n"
            "qa:\n"
            "  atoms_extractor: {provider: openai, model: extractor}\n"
            "  evaluator: {provider: openai, model: evaluator}\n"
        )

        with pytest.raises(ValueError, match=r"unknown field\(s\): unexpected"):
            qa_profile.load_profile(profile_path)

    @pytest.mark.parametrize(
        "field,value",
        [
            ("temperature", 0),
            ("unexpected", True),
        ],
    )
    def test_rejects_unsupported_descriptor_fields(self, tmp_path, field, value):
        path = tmp_path / "profile.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "version": 1,
                    "qa": {
                        "atoms_extractor": {
                            "provider": "openai",
                            "model": "x",
                            field: value,
                        },
                        "evaluator": {"provider": "openai", "model": "x"},
                    },
                }
            )
        )

        with pytest.raises(ValueError, match=rf"unknown field\(s\): {field}"):
            qa_profile.load_profile(path)

    @pytest.mark.parametrize(
        "value",
        [
            float("inf"),
            True,
            0,
            "slow",
        ],
    )
    def test_rejects_invalid_timeout(self, tmp_path, value):
        path = tmp_path / "profile.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "version": 1,
                    "qa": {
                        "atoms_extractor": {
                            "provider": "openai",
                            "model": "x",
                            "timeout": value,
                        },
                        "evaluator": {"provider": "openai", "model": "x"},
                    },
                }
            )
        )

        with pytest.raises(ValueError, match="timeout must be a positive number"):
            qa_profile.load_profile(path)

    def test_serialized_profile_round_trip(self, tmp_path):
        profile = qa_profile.load_profile(None)
        profile_path = tmp_path / "evaluator_profile.resolved.yaml"
        profile_path.write_text(yaml.safe_dump(profile.to_dict()))

        assert qa_profile.load_profile(profile_path) == profile
