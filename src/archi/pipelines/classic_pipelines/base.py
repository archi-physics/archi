"""Base pipeline definition used by all concrete pipelines."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.archi.pipelines.classic_pipelines.utils.prompt_utils import read_prompt
from src.archi.pipelines.classic_pipelines.utils.prompt_validator import ValidatedPromptTemplate
from src.archi.providers import get_model
from src.archi.utils.output_dataclass import PipelineOutput
from src.utils.logging import get_logger

logger = get_logger(__name__)


class BasePipeline:
    """
    BasePipeline provides a foundational structure for building pipeline classes that
    process user queries using configurable language models and prompts.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        *args,
        default_provider: Optional[str] = None,
        default_model: Optional[str] = None,
        prompt_overrides: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> None:
        self.config = config
        self.archi_config = self.config.get("archi") or {}
        self.dm_config = self.config.get("data_manager", {})
        pipeline_map = self.archi_config.get("pipeline_map", {}) if isinstance(self.archi_config, dict) else {}
        self.pipeline_config = pipeline_map.get(self.__class__.__name__, {}) if isinstance(pipeline_map, dict) else {}
        self.default_provider = default_provider
        self.default_model = default_model
        self.prompt_overrides = prompt_overrides or {}
        self._init_llms()
        self._init_prompts()

    def update_retriever(self, vectorstore):
        self.retriever = None

    def invoke(self, *args, **kwargs) -> PipelineOutput:
        return PipelineOutput(
            answer="Stat rosa pristina nomine, nomina nuda tenemus.",
            source_documents=[],
            intermediate_steps=[],
        )

    def _init_llms(self) -> None:
        """Initialise language models declared for the pipeline."""

        models_config = self.pipeline_config.get("models", {}) if isinstance(self.pipeline_config, dict) else {}
        self.llms: Dict[str, Any] = {}

        all_models = dict(models_config.get("required", {}), **models_config.get("optional", {}))
        initialised_models: Dict[str, Any] = {}

        if not all_models and self.default_provider and self.default_model:
            model_ref = f"{self.default_provider}/{self.default_model}"
            default_keys = self._default_model_keys()
            if default_keys:
                instance = get_model(self.default_provider, self.default_model)
                for key in default_keys:
                    self.llms[key] = instance
            else:
                self.llms["chat_model"] = get_model(self.default_provider, self.default_model)
            return

        for model_name, model_class_name in all_models.items():
            if model_class_name in initialised_models:
                self.llms[model_name] = initialised_models[model_class_name]
                logger.debug(
                    "Reusing initialised model '%s' of class '%s'",
                    model_name,
                    model_class_name,
                )
                continue

            provider, model_id = self._parse_provider_model(model_class_name)
            instance = get_model(provider, model_id)
            self.llms[model_name] = instance
            initialised_models[model_class_name] = instance

    @staticmethod
    def _parse_provider_model(model_ref: str) -> (str, str):
        """Expect model_ref as 'provider/model'. Raise if malformed."""
        if not isinstance(model_ref, str) or "/" not in model_ref:
            raise ValueError(f"Model reference must be 'provider/model', got '{model_ref}'")
        provider, model_id = model_ref.split("/", 1)
        if not provider or not model_id:
            raise ValueError(f"Invalid model reference '{model_ref}'")
        return provider, model_id

    def _init_prompts(self) -> None:
        """Initialise prompts defined in pipeline configuration."""

        prompts_config = self.pipeline_config.get("prompts", {}) if isinstance(self.pipeline_config, dict) else {}
        required = prompts_config.get("required", {})
        optional = prompts_config.get("optional", {})
        all_prompts = {**optional, **required}

        if self.prompt_overrides:
            all_prompts.update({k: v for k, v in self.prompt_overrides.items() if v})

        if not all_prompts:
            defaults = self._default_prompt_paths()
            if defaults:
                required = defaults
                all_prompts = defaults

        self.prompts: Dict[str, ValidatedPromptTemplate] = {}
        for name, path in all_prompts.items():
            if not path:
                continue
            try:
                prompt_template = read_prompt(path)
            except FileNotFoundError as exc:
                if name in required:
                    raise FileNotFoundError(
                        f"Required prompt file '{path}' for '{name}' not found: {exc}"
                    ) from exc
                logger.warning(
                    "Optional prompt file '%s' for '%s' not found or unreadable: %s",
                    path,
                    name,
                    exc,
                )
                continue
            self.prompts[name] = ValidatedPromptTemplate(
                name=name,
                prompt_template=prompt_template,
            )

    def _default_model_keys(self) -> List[str]:
        return {
            "QAPipeline": ["condense_model", "chat_model"],
            "GradingPipeline": ["final_grade_model", "summary_model", "analysis_model"],
            "ImageProcessingPipeline": ["image_processing_model"],
        }.get(self.__class__.__name__, [])

    def _default_prompt_paths(self) -> Dict[str, str]:
        if self.__class__.__name__ == "QAPipeline":
            return {
                "condense_prompt": "/root/archi/data/prompts/condense/default.prompt",
                "chat_prompt": "/root/archi/data/prompts/chat/default.prompt",
            }
        return {}
