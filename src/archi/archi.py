import src.archi.pipelines as archiPipelines 
from src.utils.config_access import get_full_config
from src.utils.logging import get_logger
from src.archi.utils.output_dataclass import PipelineOutput
from src.archi.utils.vectorstore_connector import VectorstoreConnector

logger = get_logger(__name__)

class archi():
    """
    Central class of the archi framework.
    Connects your database with the Pipeline, 
    creates and executes your Pipeline.
    """

    def __init__(
            self,
            pipeline,
            *args,
            **kwargs
        ):
        self.pipeline_kwargs = dict(kwargs)
        self.pipeline_kwargs.pop("config_name", None)
        self.update(pipeline, config_name=kwargs.get("config_name", None))
        self.pipeline_name = pipeline
        self.vs_connector = VectorstoreConnector(self.config)

    def update(self, pipeline=None, config_name = None):
        """
        Read relevant configuration settings.
        Initialize the Pipeline: either passed as argument or from config file.
        """
        logger.debug("Loading config")
        # config_name kept for compatibility; currently single active config
        self.config = get_full_config()
        if pipeline:
            self.pipeline_name=pipeline
        self.pipeline = self._create_pipeline_instance(
            self.pipeline_name,
            config=self.config,
            **self.pipeline_kwargs,
        )

    def _create_pipeline_instance(self, class_name, *args, **kwargs):
        """
        Initialize the Pipeline chosen by the config.
        """
        logger.debug(f"Initializing Pipeline: {class_name}.")
        logger.debug("With args:")
        logger.debug(f"{args}")
        logger.debug("and kwargs:")
        logger.debug(f"{kwargs}")
        try:
            cls = getattr(archiPipelines, class_name)
            return cls(*args, **kwargs)
        except AttributeError:
            raise ValueError(f"Class '{class_name}' not found in module")
        except Exception as e:
            raise RuntimeError(f"Error creating instance of '{class_name}': {e}")

    def _prepare_call_kwargs(self, kwargs):
        """Attach a freshly initialised vectorstore to the call kwargs."""
        call_kwargs = dict(kwargs)
        call_kwargs["vectorstore"] = self.vs_connector.get_vectorstore() # TODO this probably should just be moved to the specific tool that uses it
        return call_kwargs

    def _ensure_pipeline_output(self, result) -> PipelineOutput:
        """Validate that pipelines return the standard PipelineOutput object."""
        if isinstance(result, PipelineOutput):
            return result
        raise TypeError(
            f"Pipeline '{self.pipeline_name}' returned '{type(result).__name__}' instead of PipelineOutput."
        )

    def supports_stream(self) -> bool:
        """Return True when the active pipeline exposes a synchronous stream."""
        return callable(getattr(self.pipeline, "stream", None))

    def supports_astream(self) -> bool:
        """Return True when the active pipeline exposes an async stream."""
        return callable(getattr(self.pipeline, "astream", None))

    def invoke(self, *args, **kwargs) -> PipelineOutput:
        """
        Updates the vectorstore connection,
        passes it to the Pipeline's retriever,
        and then invokes the Pipeline.
        """
        call_kwargs = self._prepare_call_kwargs(kwargs)
        result = self.pipeline.invoke(*args, **call_kwargs)
        return self._ensure_pipeline_output(result)

    def stream(self, *args, **kwargs):
        """
        Stream the pipeline output if the underlying pipeline supports it.
        """
        if not self.supports_stream():
            raise AttributeError(f"Pipeline '{self.pipeline_name}' does not expose a 'stream' method.")
        call_kwargs = self._prepare_call_kwargs(kwargs)
        for event in self.pipeline.stream(*args, **call_kwargs):
            yield self._ensure_pipeline_output(event)

    async def astream(self, *args, **kwargs):
        """
        Asynchronously stream the pipeline output if supported.
        """
        if not self.supports_astream():
            raise AttributeError(f"Pipeline '{self.pipeline_name}' does not expose an 'astream' method.")
        call_kwargs = self._prepare_call_kwargs(kwargs)
        async for event in self.pipeline.astream(*args, **call_kwargs):
            yield self._ensure_pipeline_output(event)

    def __call__(self, *args, **kwargs) -> PipelineOutput:
        return self.invoke(*args, **kwargs)

    


    
