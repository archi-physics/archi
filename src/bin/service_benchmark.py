import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
from urllib import error as url_error
from urllib import request as url_request

import pandas as pd
import yaml
from datasets import Dataset
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from ragas import RunConfig, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (answer_relevancy, context_precision, context_recall,
                           faithfulness)

from src.archi.archi import archi
from src.archi.pipelines.agents.agent_spec import AgentSpecError, load_agent_spec
from src.archi.providers import get_model
from src.utils.env import read_secret
from src.utils.logging import get_logger, setup_logging
from src.utils.generate_benchmark_report import parse_benchmark_results, format_html_output
from src.utils.postgres_service_factory import PostgresServiceFactory

CONFIG_PATH = "/root/archi/config.yaml"
OUTPUT_PATH = "/root/archi/benchmarks"
EXTRA_METADATA_PATH = "/root/archi/git_info.yaml"
OUTPUT_DIR = Path(OUTPUT_PATH)

setup_logging()
logger = get_logger(__name__)

os.environ['OPENAI_API_KEY'] = read_secret("OPENAI_API_KEY")
os.environ['ANTHROPIC_API_KEY'] = read_secret("ANTHROPIC_API_KEY")
os.environ['HUGGING_FACE_HUB_TOKEN'] = read_secret("HUGGING_FACE_HUB_TOKEN")

factory = PostgresServiceFactory.from_env(password_override=os.environ.get("PG_PASSWORD"))
PostgresServiceFactory.set_instance(factory)


class ResultHandler:
    results = [] # store the results for each config
    metadata = {} # store the metadata about the benchmark run 

    @staticmethod
    def map_prompts(config: Dict[str, Any]):
        prompts = config.get("services", {}).get("benchmarking", {}).get("prompts")
        if not isinstance(prompts, dict):
            return
        for _, section in prompts.items():
            if not isinstance(section, dict):
                continue
            for prompt_name, file_path in section.items():
                if not file_path:
                    continue
                path = Path(file_path)
                if not path.exists():
                    continue
                with open(path, "r") as f:
                    prompt_str = f.read()
                section[prompt_name] = prompt_str


    @staticmethod
    def handle_results(config_path: Path, results: Dict, total_results: Dict):
        with open(config_path, "r") as f: 
            config = yaml.load(f, Loader=yaml.FullLoader)

        ResultHandler.map_prompts(config)

        current_results = { 
            "single_question_results": results, 
            "total_results": total_results, 
            "configuration_file": str(config_path),
            "configuration": config, 
        }

        ResultHandler.results.append(current_results)

    @staticmethod
    def add_metadata():
        with open(EXTRA_METADATA_PATH, "r") as f: 
            additional_info = yaml.safe_load(f)

        meta_data = {
            "time": str(datetime.now()),
            "git_info": additional_info, 
        }

        ResultHandler.metadata.update(meta_data)


    @staticmethod 
    def dump_html(benchmark_name: Path):

        config_data, config_name, timestamp, questions, total_results = parse_benchmark_results(ResultHandler.results, ResultHandler.metadata)

        logger.info(config_data)

        html_content = format_html_output(config_data, config_name, timestamp,questions, total_results)

        filename = f"{benchmark_name}-{datetime.now().strftime('%Y%m%d_%H%M%S')}_report.html"
        file_path = OUTPUT_DIR / filename

        logger.info(f"Dumping results to {file_path}")

        with open(file_path, 'w') as f:
            f.write(html_content)

        logger.info(f"✅ HTML report generated: {file_path}")

            

    @staticmethod 
    def dump(benchmark_name: Path):
        filename = f"{benchmark_name}-{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        file_path = OUTPUT_DIR / filename
        logger.info(f"Dumping results to {file_path}")
        logger.debug(f"Full results: {ResultHandler.results}")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output = {
            "benchmarking_results": ResultHandler.results,
            "metadata": ResultHandler.metadata,
        }
        with open(file_path, "w") as f:
            json.dump(output, f, indent=4)


class Benchmarker: 

    def __init__(self, configs: Path, q_to_a: dict[str, str]):
        self.queries_to_answers = q_to_a 
        self.required_fields = ['question']
        self.benchmark_name = os.environ['container_name']
        self.all_config_files = self.get_all_configs(configs)
        self.all_config_files.append('FINISHED')
        self.previous_input_list = []
        self.chain = None 
        self.config = None 
        self.current_config = None 

        self.load_new_configuration()
        self.data_path = self.config["global"]["DATA_PATH"]
    
    def get_all_configs(self, configs_dir):
        all_paths = []
        for root, _, filenames in os.walk(configs_dir):
            for file in filenames: 
                full_path = os.path.join(root, file)
                all_paths.append(full_path)
        return all_paths

    def load_new_configuration(self):
        self.current_config = self.all_config_files.pop(0)
        if self.current_config == 'FINISHED': return
        with open(self.current_config, "r") as f:
            config = yaml.safe_load(f)

        with open(CONFIG_PATH, 'w') as f: 
            yaml.dump(config, stream=f)

        del self.chain
        self.config = config 
        self.benchmarking_configs = config['services']['benchmarking']
        if 'SOURCES' in self.benchmarking_configs:
            self.required_fields += ['sources']
        elif 'RAGAS' in self.benchmarking_configs:
            self.required_fields += ['answer']

        # for now it only uses one pipeline (the first one) but maybe later we make this work for mulitple
        logger.info(f"loaded new configuration: {self.current_config}")
        benchmark_cfg = config.get("services", {}).get("benchmarking", {}) if isinstance(config, dict) else {}
        pipeline = benchmark_cfg.get("agent_class")
        provider = benchmark_cfg.get("provider")
        model = benchmark_cfg.get("model")
        agent_md_file = benchmark_cfg.get("agent_md_file")
        ollama_url = benchmark_cfg.get("ollama_url")
        missing = [k for k, v in {
            "agent_class": pipeline,
            "provider": provider,
            "model": model,
            "agent_md_file": agent_md_file,
        }.items() if not v]
        if missing:
            raise ValueError(
                f"Missing required benchmarking runtime fields in services.benchmarking: {', '.join(missing)}"
            )
        if str(provider).lower() == "local" and not ollama_url:
            raise ValueError(
                "Missing required benchmarking runtime field in services.benchmarking: ollama_url (required when provider is local)"
            )
        if ollama_url:
            os.environ["OLLAMA_HOST"] = str(ollama_url)

        agent_spec = None
        try:
            agent_spec = load_agent_spec(Path(str(agent_md_file)))
        except AgentSpecError as exc:
            raise ValueError(f"Failed to load benchmark agent spec '{agent_md_file}': {exc}") from exc
        self.chain = archi(
            pipeline,
            agent_spec=agent_spec,
            default_provider=provider,
            default_model=model,
            prompt_overrides={},
        )


    def get_ragas_llm_evaluator(self):
        ragas_configs = self.config['services']['benchmarking']['mode_settings']['ragas_settings']
        benchmark_cfg = self.config.get("services", {}).get("benchmarking", {})
        provider = benchmark_cfg.get("provider")
        model_name = benchmark_cfg.get("model")
        ollama_url = benchmark_cfg.get("ollama_url")

        match str(provider).lower():
            case "openai":
                return ChatOpenAI(model=model_name)
            case "ollama":
                from langchain_ollama import ChatOllama
                base_url = ollama_url
                return ChatOllama(model=model_name, base_url=base_url,num_predict=-2,model_kwargs={'format': 'json'})
            case "local":
                from langchain_ollama import ChatOllama
                base_url = ollama_url
                return ChatOllama(model=model_name, base_url=base_url,num_predict=-2,model_kwargs={'format': 'json'})
            case "huggingface":
                base_url = ollama_url or "http://localhost:8000/v1"
                return get_model("local", model_name, base_url=base_url, local_mode="openai_compat")
            case "anthropic":
                from langchain_anthropic import ChatAnthropic
                return ChatAnthropic(model=model_name)
            case _:
                return ChatOpenAI(model=model_name)


    def get_ragas_embedding_model(self):
        ragas_configs = self.config['services']['benchmarking']['mode_settings']['ragas_settings']
        embedding_model = ragas_configs['embedding_model']

        match embedding_model.lower():
            case "openai":
                return OpenAIEmbeddings()
            case "huggingface":
                return HuggingFaceEmbeddings()
            case _:
                return OpenAIEmbeddings()
            

    def prepare_match_fields(self, question_item):

        # either grab the match field(s) from the question item or use the default
        match_fields = question_item.get('source_match_field')
        if not match_fields:
            match_fields = self.benchmarking_configs['mode_settings']['sources_settings']['default_match_field']

        # make it to a list if it's passed as a string
        if isinstance(match_fields, str):
            match_fields = [match_fields] if match_fields else []

        n_sources = len(question_item.get('sources', []))
        if not match_fields:
            # hardcode a default if nothing is provided
            match_fields = ['file_name'] * n_sources
        elif len(match_fields) == 1 and n_sources > 1:
            # expand single field to all sources
            match_fields = match_fields * n_sources
        elif len(match_fields) != n_sources:
            logger.error(
                "Number of match fields (%s) does not align with number of reference sources (%s); reusing the last field for the remaining references.",
                len(match_fields),
                n_sources,
            )
            raise ValueError("Mismatch between number of match fields and reference sources.")
        
        return match_fields


    def prepare_reference_sources(self, reference_sources, match_fields):

        # Clean and prepare reference sources
        raw_references: List[str] = []
        if isinstance(reference_sources, str):
            cleaned = reference_sources.strip()
            if cleaned and cleaned != 'N/A':
                raw_references = [reference_sources]
        elif isinstance(reference_sources, list):
            raw_references = [ref for ref in reference_sources if ref not in (None, '')]
        elif reference_sources is None:
            raw_references = []
        else:
            raw_references = [reference_sources]
        reference_sources_list: List[str] = []
        for ref in raw_references:
            ref_str = str(ref).strip()
            if ref_str and ref_str != 'N/A':
                reference_sources_list.append(ref_str)

        formatted_reference_sources = []
        for field, reference in zip(match_fields, reference_sources_list):
            formatted_reference_sources.append({field: reference})

        return formatted_reference_sources


    def prepare_messages(self, raw_messages):
        """Format the langchain Messages into something we can store and view later."""
        formatted_messages = []
        for msg in raw_messages:
            if type(msg) is AIMessage:
                # there are two types of AI messages, content and tool calls
                # e.g. tool_calls=[{'name': 'search_vectorstore', 'args': {'query': 'CMSTRANSF-1078'}, 'id': '4a73724f-db40-41eb-9843-7f325df76f58', 'type': 'tool_call'}]
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        formatted_messages.append({
                            'type': 'tool_call',
                            'tool_name': tool_call.get('name'),
                            'tool_args': tool_call.get('args',{}).get('query', 'No query found.'),
                            'total_duration': getattr(msg, 'response_metadata', {}).get('total_duration', None),
                        })
                elif hasattr(msg, 'content'):
                    formatted_messages.append({
                        'type': 'ai_message',
                        'content': msg.content,
                        'total_duration': getattr(msg, 'response_metadata', {}).get('total_duration', None),
                    })
            elif type(msg) is HumanMessage:
                # we don't store these...
                pass
            elif type(msg) is ToolMessage:
                # we don't store these?
                logger.debug(msg)
                pass
            else:
                logger.warning(f"Unexpected message type: {type(msg)}")
        return formatted_messages


    def get_source_results(
            self,
            result: Dict,
            formatted_reference_sources: List[Dict[str, str]],
        ) -> List[bool]:
        """
        For each reference source, check the specified metadata field in the retrieved documents.
        The reference sources and match fields are paired one-to-one; a single string field is
        expanded to cover all provided sources. Returns summary information and whether all
        reference sources were found.
        """
        sources = result.get('source_documents', [])
        logger.info("Agent found %s sources.", len(sources))
        
        matches: List[bool] = []
        for source in formatted_reference_sources:
            field, reference = list(source.items())[0]
            logger.debug("Checking for reference source '%s' in field '%s'", reference, field)
            for document in sources:
                metadata = getattr(document, 'metadata', {}) or {}
                value = metadata.get(field)
                if value is None:
                    continue
                if isinstance(value, list):
                    values = [str(v).strip() for v in value if v is not None]
                else:
                    values = [str(value).strip()]
                logger.info("Returned source '%s': %s", field, values)
                logger.debug("Checking reference '%s' against document metadata field '%s': %s", reference, field, values)
                if reference in values:
                    logger.debug("Matched reference source '%s' in document metadata.", reference)
                    matches.append(True)
                    break
            else:
                matches.append(False)

        # match is determined if at least once source is found
        logger.info("Source matching result: %s", matches)
        return matches


    def get_ragas_results(self, data, to_add):
        """WARNING: this method modifies the to_add dictionary to add the relevant scores to the relevant questions"""
        
        all_metrics_dict = {
                'answer_relevancy': answer_relevancy, 
                'faithfulness': faithfulness, 
                'context_precision': context_precision, 
                'context_recall': context_recall
                }

        enabled_metrics = self.benchmarking_configs['mode_settings']['ragas_settings']['enabled_metrics']

        metrics_dict = {k: v for k, v in all_metrics_dict.items() if k in enabled_metrics}
                       
        res = pd.DataFrame()

        ragas_settings = self.config['services']['benchmarking']['mode_settings']['ragas_settings']
        log_tenacity = self.config['global']['verbosity'] >= 4
        timeout = ragas_settings['timeout']
        batch_settings = ragas_settings['batch_size']
        if not batch_settings: 
            batch_settings = None
        
        runconfig = RunConfig(timeout=timeout, log_tenacity=log_tenacity)
        # going one metric at a time prevents errors 
        for metric_name, metric in metrics_dict.items():
            evaluation_results = evaluate(data, 
                                          metrics=[metric],
                                          llm=LangchainLLMWrapper(self.get_ragas_llm_evaluator()),
                                          embeddings=LangchainEmbeddingsWrapper(self.get_ragas_embedding_model()),
                                          run_config=runconfig,
                                          batch_size=batch_settings
                                          )

            metric_results = evaluation_results.to_pandas()
            res[metric_name] = metric_results[metric_name]

        for question_idx, question in enumerate(to_add.values()):
            for metric in metrics_dict.keys():
                question[metric] = res.at[question_idx, metric]

        return res


    def run(self):
        self.wait_for_ingestion_completion()

        modes_being_run = set(self.benchmarking_configs['modes'])

        logger.info("")
        logger.info("====== Starting benchmark: %s ======", self.benchmark_name)
        logger.info("Modes being run: %s", modes_being_run)
        logger.info(f"Processing {len(self.queries_to_answers)} questions and {len(self.all_config_files)} configuration(s).")
        logger.info("")

        while self.all_config_files: 

            question_id = 0

            # results for each question
            question_wise_results = {}

            # results for all of the questions in this config
            total_results = {}

            # RAGAS mode: ragas inputs
            ragas_input = []

            # SOUCES mode: sources accuracy
            relative_source_accuracy = 0.0 
            source_accuracy = 0.0

            for question_item in self.queries_to_answers:

                logger.info("")
                logger.info("====================================")
                logger.info(f"Answering question: {question_id + 1}")

                if type(question_item) is not dict:
                    logger.error(f"Each item in the question to answer list must be a dictionary, but got {type(question_item)}")
                    continue
                if not all(field in question_item for field in self.required_fields):
                    logger.error(f"Each item in the question to answer list must contain the following fields: {self.required_fields}, but got {question_item.keys()}")
                    continue

                question = question_item['question']
                reference_answer = question_item.get('answer', 'N/A')
                reference_sources = question_item.get('sources', 'N/A')

                logger.info(f"Question: {question}")
                logger.info(f"Reference Answer: {reference_answer}")
                logger.info(f"Reference Sources: {reference_sources}")

                question_id +=1
                formatted_question = [("User", question)]
                start = time.perf_counter()
                result = self.chain(history=formatted_question)
                end = time.perf_counter()
                logger.info(f"Finished answering question: {question_id} ({end - start:.2f}s)")
                q_results = {}

                # prepare info to store for this question
                q_results["time_elapsed"] = end - start
                q_results["question"] = question
                q_results["reference_answer"] = reference_answer
                q_results["answer"] = result['answer']

                # format the messages
                q_results['messages'] = self.prepare_messages(result.get("messages", []))

                # format the reference sources
                match_fields_list = self.prepare_match_fields(question_item)
                formatted_reference_sources = self.prepare_reference_sources(reference_sources, match_fields_list)
                q_results["reference_sources_match_fields"] = match_fields_list
                q_results["reference_sources_metadata"] = formatted_reference_sources

                if "RAGAS" in modes_being_run:
                    # we collect the necessary info for ragas evaluation
                    # TODO this is likely broken now
                    contexts = [s.page_content for s in result['source_documents']]
                    dataset_result = {
                            "question": question,
                            "contexts": contexts,
                            "answer": result['answer'],
                            "ground_truth": reference_answer,
                            }
                    ragas_input.append(dataset_result)

                if "SOURCES" in modes_being_run: 
                    # sources evaluation is done on the fly -- check if each of the given sources was found                  
                    matches = self.get_source_results(
                        result,
                        formatted_reference_sources,
                    )
                    # we count accuracy via any of the sources matching
                    if any(matches): 
                        relative_source_accuracy += 1.0
                    if len(matches) == len(formatted_reference_sources) and all(matches):
                        source_accuracy += 1.0
                    # but we still store the match of each reference source in its metadata
                    for idx, source in enumerate(q_results["reference_sources_metadata"]):
                        source['matched'] = matches[idx]
                    logger.info(f"Current relative accuracy: {relative_source_accuracy / question_id if question_id > 0 else 0.0}")
                    logger.info(f"Current strict accuracy: {source_accuracy / question_id if question_id > 0 else 0.0}")

                # store the sources metadata and truncated content
                sources_metadata: List[Dict[str, Any]] = []
                sources_trunc_content: List[str] = []
                for document in result['source_documents']:
                    metadata = getattr(document, 'metadata', {}) or {}
                    sources_metadata.append(metadata)
                    sources_trunc_content.append(getattr(document, 'page_content', '')[:300])  # first 300 chars
                q_results['sources_metadata'] = sources_metadata
                q_results['sources_trunc_content'] = sources_trunc_content
                logger.debug("Sources returned: %s", sources_metadata)

                # store the results for this question
                question_wise_results[f"question_{question_id}"] = q_results
                
                logger.info("====================================")
                logger.info("")

            if "RAGAS" in modes_being_run:
                # TODO this is likely broken now
                logger.info(f"Starting to collect RAGAS results")
                data = Dataset.from_list(ragas_input)
                # were modifying final_addition here to add ragas results by question
                ragas_results = self.get_ragas_results(data, question_wise_results)

                answer_relevancy = ragas_results['answer_relevancy'].mean()
                faithfulness = ragas_results['faithfulness'].mean()
                context_precision = ragas_results['context_precision'].mean()
                context_recall = ragas_results['context_recall'].mean()

                total_results['aggregate_answer_relevancy'] = answer_relevancy
                total_results['aggregate_faithfulness'] = faithfulness
                total_results['aggregate_context_precision'] = context_precision
                total_results['aggregate_context_recall'] = context_recall

            if "SOURCES" in modes_being_run:
                total_results['relative_source_accuracy'] = relative_source_accuracy / len(self.queries_to_answers)
                total_results['source_accuracy'] = source_accuracy / len(self.queries_to_answers)

            ResultHandler.handle_results(Path(self.current_config), question_wise_results, total_results)
            self.load_new_configuration()

        ResultHandler.add_metadata()
        ResultHandler.dump(self.benchmark_name)
        ResultHandler.dump_html(self.benchmark_name)
        return

    def wait_for_ingestion_completion(self):
        timeout_seconds = int(os.environ.get("BENCH_INGEST_WAIT_TIMEOUT", "3600"))
        poll_interval_seconds = int(os.environ.get("BENCH_INGEST_POLL_INTERVAL", "5"))
        status_urls = [
            "http://data-manager:7871/api/ingestion/status",
            "http://localhost:7871/api/ingestion/status",
            "http://host.containers.internal:7871/api/ingestion/status",
        ]
        start_time = time.monotonic()
        attempt = 0

        logger.info("Waiting for data-manager ingestion to complete before benchmarking...")
        while True:
            attempt += 1
            last_error = None
            for status_url in status_urls:
                try:
                    with url_request.urlopen(status_url, timeout=5) as response:
                        payload = json.loads(response.read().decode("utf-8"))
                    state = str(payload.get("state", "")).lower()
                    step = payload.get("step")
                    err = payload.get("error")
                    logger.info(
                        "Ingestion status check #%s via %s -> state=%s step=%s",
                        attempt,
                        status_url,
                        state,
                        step,
                    )
                    if state == "completed":
                        logger.info("Data-manager ingestion completed; starting benchmark.")
                        return
                    if state == "error":
                        raise RuntimeError(f"Data-manager ingestion failed at step '{step}': {err}")
                    break
                except (url_error.URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
                    last_error = exc
                    continue

            elapsed = time.monotonic() - start_time
            if elapsed >= timeout_seconds:
                if last_error:
                    raise TimeoutError(
                        f"Timed out after {timeout_seconds}s waiting for ingestion status endpoint. Last error: {last_error}"
                    )
                raise TimeoutError(f"Timed out after {timeout_seconds}s waiting for ingestion completion.")

            time.sleep(poll_interval_seconds)

if __name__ == "__main__":

    query_file = Path("QandA.txt") 
    configs_folder = Path('configs')

    with open(Path(query_file), "r") as f:
        question_to_answer = json.load(f)

    benchmarker = Benchmarker(configs_folder, question_to_answer)
    benchmarker.run()
    logger.info("\n\nFINISHED RUNNING\n\n")
