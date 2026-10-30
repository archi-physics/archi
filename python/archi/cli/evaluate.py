"""`archi eval [--bundle X] [--instance DIR] [--model M] [--model-url URL] [--out DIR]`.

Runs a bundle's smoke set (`bundles/<bundle>/eval/smoke.yaml`) against the
instance's MCP endpoint with a local model, writes report.md and report.json,
and exits non-zero below the bundle's pass threshold (PACT r3-bundle-evals).

Exit codes: 0 the threshold was met; 1 it was not; 2 the run could not
execute (no MCP tools, the endpoint or the model unreachable, a bad smoke
file). Checks are deterministic: no LLM judge, no paid model call.

`archi eval --context [--questions FILE] [--k N]` scores retrieval alone
(PACT r3-context-quality): it calls the MCP search tool directly, with no
model, in a graph arm and a lexical baseline arm, reports recall@k and MRR
of each question's gold ids (`archi.evaluation.context`), and writes each run
to `<data root>/eval/context/<bundle>/<UTC stamp>/`, where the context gate
reads the newest. Exit codes as above; 1 means the graph arm fell below the
file's `min_recall`; 2 also when any graph-arm search call failed.

`archi eval --context --agents` adds the raw-document baseline (`archi.evaluation.raw_arm`):
a bash agent over the raw corpus in a bubblewrap sandbox, and an agent with the MCP tools,
on one model with one set of caps, scored on gold-evidence recall and MRR, with an optional
answer judge (`--judge`). Every model call goes to `ledger.jsonl` in the run's directory;
the run refuses to start without `--cap` and a price, and stops at the cap (exit 3).
`archi eval --raw-corpus` builds the corpus from the instance's caches
(`archi.evaluation.raw_corpus`).

This module only builds the parser. The harness (`archi.evaluation`, and the
MCP client it loads) is imported when the command runs, so every other
`archi` command stays as fast as before.
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

DEFAULT_MODEL_URL = "http://127.0.0.1:11434/v1"
TOKEN_VAR = "OKG_CHAT_MCP_TOKEN"
#: The context question set's path. An instance's operator questions are private and never
#: committed here, so the set is named at run time.
QUESTIONS_VAR = "ARCHI_CONTEXT_EVAL_QUESTIONS"
MODEL_KEY_VAR = "ARCHI_EVAL_MODEL_KEY_FILE"
JUDGE_KEY_VAR = "ARCHI_EVAL_JUDGE_KEY_FILE"
DEFAULT_JUDGE_URL = "https://api.openai.com/v1"
AGENT_ARM_NAMES = ("agent-bash", "agent-graph")
#: exit code of a run whose agent arms stopped at the spend cap
EXIT_SPEND_CAP = 3
EVAL_HELP = ("ask a bundle's smoke questions through the instance MCP with a local model, "
             "or score retrieval alone with --context")


def add_parser(sub, roots) -> None:
    p = sub.add_parser(
        "eval",
        help=EVAL_HELP,
        description=(
            "Run a bundle's smoke set against the instance MCP endpoint with a local "
            "model (an OpenAI-compatible endpoint such as Ollama). Checks are "
            "deterministic; nothing is paid. Exits 1 below the bundle's pass "
            "threshold and 2 when the run could not execute. With --context, no model is "
            "used: the MCP search tool is called directly and recall@k of gold evidence ids "
            "is reported for the graph arm and a lexical baseline arm."
        ),
    )
    p.add_argument("--bundle", action="append", default=[],
                   help="bundle whose smoke set to run (repeatable; default: the instance's bundles)")
    p.add_argument("--instance", type=Path, default=Path("."), help="instance directory (default: .)")
    p.add_argument("--model", help="model name at the endpoint (default: the bundle's chat.default_model)")
    p.add_argument("--model-url", default=os.environ.get("ARCHI_EVAL_MODEL_URL", DEFAULT_MODEL_URL),
                   help=f"OpenAI-compatible base URL (default: {DEFAULT_MODEL_URL}, a local Ollama)")
    p.add_argument("--out", type=Path, help="report directory (default: <data root>/eval/<bundle>, outside the instance repository)")
    p.add_argument("--mcp-url", help="MCP endpoint (default: the instance's, http://127.0.0.1:<port>/mcp)")
    p.add_argument("--chat-prompt", type=Path,
                   help="the chat system prompt to give the agent (default: the instance deployment's "
                        "skills/chat-system-prompt.md, the one `archi up` gives the chat)")
    p.add_argument("--max-steps", type=int, default=8, help="tool-call rounds allowed per question")
    p.add_argument("--no-think", action="store_true",
                   help="ask a reasoning model to skip its thinking pass (reasoning_effort: none)")
    p.add_argument("--allow-remote-model", action="store_true",
                   help="allow a model endpoint off this host or network (may cost money; off by default)")
    p.add_argument("--context", action="store_true",
                   help="score retrieval instead: call the MCP search tool directly (no model) and report "
                        "recall@k of each question's gold ids, with a lexical baseline arm; each run is kept "
                        "under <data root>/eval/context/<bundle>/<UTC stamp>/")
    p.add_argument("--questions", type=Path,
                   default=Path(os.environ[QUESTIONS_VAR]) if os.environ.get(QUESTIONS_VAR) else None,
                   help=f"with --context: the question file, kept outside this repository (default: "
                        f"${QUESTIONS_VAR}, else the bundle's eval/context.yaml)")
    p.add_argument("--k", type=int, help="with --context: the recall cut-off (default: the file's k, else 10)")
    a = p.add_argument_group(
        "agent arms (with --context)",
        "The raw-document baseline: agent-bash (bash over the raw corpus, in a bubblewrap sandbox with no "
        "network) and agent-graph (the instance's MCP tools), one model, one set of caps, one scorer. These "
        "calls may cost money: every one is written to the run's ledger.jsonl and the run stops at --cap.")
    a.add_argument("--agents", nargs="?", const=",".join(AGENT_ARM_NAMES), metavar="ARMS",
                   help=f"also run the agent arms (default both: {','.join(AGENT_ARM_NAMES)})")
    a.add_argument("--tries", type=int, default=1, help="tries per question and arm (default 1)")
    a.add_argument("--corpus", type=Path,
                   help="the raw corpus (default: the newest under <data root>/eval/raw-corpus/<bundle>/)")
    a.add_argument("--reasoning-effort", help="reasoning_effort sent with each agent call (for example medium)")
    a.add_argument("--model-api", choices=("auto", "chat", "responses"), default="auto",
                   help="the API the agent calls use: responses (/responses, OpenAI's Responses API) or chat "
                        "(/chat/completions); auto, the default, is responses for api.openai.com and chat for "
                        "anything else, such as a local Ollama")
    a.add_argument("--model-key-env", default=MODEL_KEY_VAR,
                   help=f"environment variable holding the PATH of a file with the model API key "
                        f"(default {MODEL_KEY_VAR}); the key is never logged and never enters the sandbox")
    a.add_argument("--model-price", metavar="IN/OUT[/CACHED]",
                   help="the model's price in dollars per million input/output[/cached input] tokens (required)")
    a.add_argument("--max-output-tokens", type=int, default=16000,
                   help="max_completion_tokens per agent call (max_output_tokens on the Responses API), also the "
                        "spend guard's output bound (default 16000)")
    a.add_argument("--max-turns", type=int, default=30, help="model turns per try (default 30)")
    a.add_argument("--max-tool-calls", type=int, default=200, help="tool calls per try (default 200)")
    a.add_argument("--try-seconds", type=float, default=900.0, help="wall time per try (default 900)")
    a.add_argument("--cap", type=float, metavar="DOLLARS",
                   help="spend cap in dollars for the run, judge included (required: there is no default); the "
                        "run stops before a call that could cross it")
    a.add_argument("--judge", action="store_true",
                   help="grade answers against the question file's answer keys with a judge model (off by default)")
    a.add_argument("--judge-model", help="the judge model")
    a.add_argument("--judge-url", default=DEFAULT_JUDGE_URL,
                   help=f"the judge's OpenAI-compatible base URL (default {DEFAULT_JUDGE_URL})")
    a.add_argument("--judge-reasoning-effort", default="high", help="reasoning_effort for the judge (default high)")
    a.add_argument("--judge-max-output-tokens", type=int, default=16000,
                   help="max_completion_tokens per judge call, also its spend bound (default 16000)")
    a.add_argument("--allow-low-price", action="store_true",
                   help="accept a price below $0.01 input or $0.04 output per million tokens (a local model); "
                        "refused by default as a likely typo")
    a.add_argument("--judge-key-env", default=JUDGE_KEY_VAR,
                   help=f"environment variable holding the PATH of a file with the judge API key (default {JUDGE_KEY_VAR})")
    a.add_argument("--judge-price", metavar="IN/OUT[/CACHED]", help="the judge's price, as --model-price")
    a.add_argument("--judge-passes", type=int, choices=(1, 2), default=1, help="judge passes per answer (default 1)")
    p.add_argument("--raw-corpus", action="store_true",
                   help="build the raw-document corpus from the instance's connector caches, under "
                        "<data root>/eval/raw-corpus/<bundle>/<lock sha>/, and exit")
    roots(p)
    p.set_defaults(func=cmd_eval)


def _fail(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return 2


def cmd_eval(args: argparse.Namespace) -> int:
    from archi.bundles import discovery
    from archi.evaluation.agent import ModelError, require_local
    from archi.evaluation.smoke import SmokeError, load_smoke, smoke_path
    from archi.stack.instance import Instance, InstanceError, read_env

    token = os.environ.get(TOKEN_VAR, "")
    bundles = list(args.bundle)
    mcp_url = args.mcp_url
    report_root = Path("eval-reports")
    chat_prompt_file = args.chat_prompt
    instance = None
    # With --mcp-url the endpoint is not the instance's own, so nothing is taken from an
    # instance: its bearer token must never be sent to another server.
    if not mcp_url:
        try:
            instance = Instance.load(args.instance)
        except InstanceError as exc:
            return _fail(str(exc))
    if instance is not None:
        # The data root lies outside the instance repository: `archi up` commits
        # everything inside it, and a report is not configuration.
        report_root = instance.data_root / "eval"
        if chat_prompt_file is None:
            found = instance.deployment_dir / "skills" / "chat-system-prompt.md"
            chat_prompt_file = found if found.is_file() else None
        bundles = bundles or list(instance.bundles)
        mcp_url = f"http://127.0.0.1:{instance.ports.mcp}/mcp"
        try:
            token = read_env(instance.env_path).get(TOKEN_VAR) or token
        except InstanceError as exc:
            if not token:
                return _fail(str(exc))
    if args.raw_corpus:
        return _raw_corpus(args, instance, bundles, mcp_url or "", token)
    if args.context:
        return _context(args, instance, bundles, mcp_url, token, report_root)
    if args.agents is not None:
        return _fail("--agents runs with --context")
    if not bundles:
        return _fail("name a bundle with --bundle")
    if not token:
        return _fail(f"no MCP bearer token: {TOKEN_VAR} is not in the instance .env or the environment")

    catalog = discovery.discover(args.pack_root, args.bundle_root)
    plan = []
    for name in bundles:
        bundle = catalog.bundles.get(name)
        if bundle is None:
            return _fail(f"no bundle named {name!r}")
        path = smoke_path(bundle.dir)
        if not path.is_file():
            if args.bundle:
                return _fail(f"bundle {name!r} ships no smoke set ({path})")
            print(f"note: bundle {name!r} ships no smoke set; skipped", file=sys.stderr)
            continue
        try:
            plan.append((bundle, load_smoke(path)))
        except SmokeError as exc:
            return _fail(str(exc))
    if not plan:
        return _fail("none of the bundles ships a smoke set (bundles/<name>/eval/smoke.yaml)")

    if not args.allow_remote_model:
        try:
            require_local(args.model_url)
        except ModelError as exc:
            return _fail(str(exc))

    worst = 0
    for bundle, smoke in plan:
        model_name = args.model or ((bundle.manifest.get("chat") or {}).get("default_model"))
        if not model_name:
            return _fail(f"bundle {bundle.name!r} sets no chat.default_model; pass --model")
        out = args.out or report_root / bundle.name
        if args.out and len(plan) > 1:
            out = args.out / bundle.name
        code = _run_one(smoke, model_name, args, mcp_url, token, out, chat_prompt_file)
        worst = max(worst, code)
    return worst


def _run_one(smoke, model_name: str, args: argparse.Namespace, mcp_url: str, token: str, out: Path,
             chat_prompt_file=None) -> int:
    import asyncio

    from archi.evaluation.agent import McpToolBox, OpenAIChat
    from archi.evaluation.runner import run_smoke, write_report

    chat_prompt = None
    if chat_prompt_file is not None:
        try:
            chat_prompt = Path(chat_prompt_file).read_text(encoding="utf-8")
        except OSError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
    else:
        print("note: no chat system prompt found for this instance; the agent gets only the MCP "
              "server's instructions", file=sys.stderr)
    model = OpenAIChat(args.model_url, model_name, no_think=args.no_think)

    def progress(result, i: int, total: int) -> None:
        print(f"[{i}/{total}] {result.status:16} {result.id} ({result.seconds:.0f}s)", flush=True)

    print(f"archi eval: {smoke.bundle}, {len(smoke.questions)} questions, model {model_name}, MCP {mcp_url}")
    report = asyncio.run(
        run_smoke(
            smoke,
            model,
            lambda: McpToolBox(mcp_url, token),
            model_url=args.model_url,
            mcp_url=mcp_url,
            chat_prompt=chat_prompt,
            chat_prompt_file=str(chat_prompt_file or ""),
            max_steps=args.max_steps,
            progress=progress,
        )
    )
    md, js = write_report(report, out)
    print(f"{report.status}: {report.passed}/{len(report.questions)} passed "
          f"(threshold {report.pass_threshold}); {md} and {js}")
    if report.error:
        print(f"error: {report.error}", file=sys.stderr)
    return report.exit_code


def _context(args: argparse.Namespace, instance, bundles: list, mcp_url: str, token: str, report_root: Path) -> int:
    """`archi eval --context`: retrieval alone, no model (PACT r3-context-quality)."""
    import asyncio
    import dataclasses

    from archi.bundles import discovery
    from archi.evaluation.agent import McpToolBox
    from archi.evaluation.context import (
        ContextSetError,
        context_path,
        load_context,
        run_context,
        publish_run,
        write_report,
    )
    from archi.ops.context_gate import context_eval_dir

    if not token:
        return _fail(f"no MCP bearer token: {TOKEN_VAR} is not in the instance .env or the environment")
    if args.k is not None and not 1 <= args.k <= 100:
        return _fail("--k must be from 1 to 100")
    sets = []
    try:
        if args.questions is not None:
            if not args.questions.is_file():
                return _fail(f"question file {args.questions} does not exist (from --questions or ${QUESTIONS_VAR})")
            cset = load_context(args.questions)
            if args.bundle and args.bundle != [cset.bundle]:
                return _fail(f"{args.questions} is the set of bundle {cset.bundle!r}, not {', '.join(args.bundle)}")
            sets.append(cset)
        else:
            if not bundles:
                return _fail("name a bundle with --bundle, or a question file with --questions")
            catalog = discovery.discover(args.pack_root, args.bundle_root)
            for name in bundles:
                bundle = catalog.bundles.get(name)
                if bundle is None:
                    return _fail(f"no bundle named {name!r}")
                path = context_path(bundle.dir)
                if path.is_file():
                    sets.append(load_context(path))
                elif args.bundle:
                    return _fail(f"bundle {name!r} ships no context set ({path}); pass --questions FILE")
                else:
                    print(f"note: bundle {name!r} ships no context set; skipped", file=sys.stderr)
    except ContextSetError as exc:
        return _fail(str(exc))
    if not sets:
        return _fail("no context question set: none of the bundles ships eval/context.yaml; pass --questions FILE")
    if instance is not None:
        # A run under the data root is what the context gate reads: only a bundle of this instance may
        # put one there (a fixture or another instance's set must never pass a live gate).
        foreign = [c.bundle for c in sets if c.bundle not in instance.bundles]
        if foreign:
            return _fail(f"the question set is for bundle {', '.join(foreign)}, which this instance does not list "
                         f"({', '.join(instance.bundles) or 'none'}); its run would be read by the context gate, "
                         "so it is refused")
        overridden = [c.bundle for c in sets if c.graph_overridden]
        if overridden and not args.out:
            return _fail(f"the question set of {', '.join(overridden)} overrides the graph arm; such a run is "
                         "informational only and would be written where the context gate reads, so it is refused "
                         "(write it elsewhere with --out)")

    setup = None
    if args.agents is not None:
        setup = _prepare_agents(args)
        if isinstance(setup, str):
            return _fail(setup)
        args._several_sets = len(sets) > 1
    worst = 0
    for cset in sets:
        if args.k is not None:
            cset = dataclasses.replace(cset, k=args.k)
        if setup is not None:
            code = _context_with_agents(args, instance, cset, mcp_url, token, report_root, setup)
            worst = max(worst, code)
            continue

        def progress(result, i: int, total: int) -> None:
            cells = " ".join(f"{name}={'error' if r.error else f'{len(r.found)}/{len(result.gold)}'}"
                             for name, r in result.arms.items())
            print(f"[{i}/{total}] {result.id}: {cells}", flush=True)

        print(f"archi eval --context: {cset.bundle}, {len(cset.scored)} scored questions, k={cset.k}, MCP {mcp_url}")
        report = asyncio.run(run_context(cset, lambda: McpToolBox(mcp_url, token), mcp_url=mcp_url, progress=progress))
        if args.out:
            md, js = write_report(report, args.out / cset.bundle if len(sets) > 1 else args.out)
        else:
            # one directory per run, so the history is kept: <root>/context/<bundle>/<UTC stamp>[-n]/,
            # built in a hidden directory and renamed into place when complete
            base = (context_eval_dir(instance.data_root, cset.bundle) if instance is not None
                    else report_root / "context" / cset.bundle)
            md, js = publish_run(report, base)
        data = report.to_dict()
        arms = ", ".join(f"{name} {s['recall_at_k']:.3f}" for name, s in data["arms"].items())
        print(f"{report.status}: recall@{cset.k} {arms}; {data['errors']} failed calls; {md} and {js}")
        if report.error:
            print(f"error: {report.error}", file=sys.stderr)
        if report.graph_errors:
            print(f"error: {report.graph_errors} graph-arm search calls failed; the run has no score",
                  file=sys.stderr)
        if instance is None:
            print("note: --mcp-url names an endpoint outside the instance; no context gate reads this run",
                  file=sys.stderr)
        elif args.out:
            print("note: --out moves the report away from <data root>/eval/context/, where the context gate "
                  "reads it", file=sys.stderr)
        worst = max(worst, report.exit_code)
    return worst




def _published_generation(mcp_url: str, token: str) -> tuple[Optional[str], str]:
    """The instance's published generation, read through MCP `inspect(graph)`; (None, why) when it
    cannot be read."""
    import asyncio
    import json

    from archi.evaluation import agent as agent_mod

    if not token:
        return None, f"no MCP bearer token ({TOKEN_VAR})"

    async def go():
        async with agent_mod.McpToolBox(mcp_url, token) as tools:
            return await tools.call("inspect", {"target": {"kind": "graph"}}, meta={"okg/result_format": "json"})

    try:
        text, is_error = asyncio.run(go())
        payload = json.loads(text) if not is_error else None
    except Exception as exc:  # noqa: BLE001 - the corpus is still built; the report says the generation is unknown
        return None, f"the MCP endpoint could not be read ({type(exc).__name__}: {exc})"
    if not isinstance(payload, dict):
        return None, f"inspect(graph) did not answer in JSON: {text.strip()[:200]}"
    generation = payload.get("pinned_generation_id") or payload.get("generation_id")
    return (str(generation), "") if generation else (None, "inspect(graph) named no generation")


def _raw_corpus(args: argparse.Namespace, instance, bundles: list, mcp_url: str = "", token: str = "") -> int:
    """`archi eval --raw-corpus`: pack the instance's caches into the agent-bash arm's corpus."""
    from archi.bundles.lock import deployment_answers
    from archi.evaluation.raw_corpus import CorpusError, build_corpus, corpus_root, plan_sources, read_registry

    if instance is None:
        return _fail("--raw-corpus reads an instance's caches; run it with --instance, not --mcp-url")
    names = list(args.bundle) or list(instance.bundles)
    if len(names) != 1:
        return _fail(f"name the bundle the corpus is for with --bundle (the instance has {', '.join(names) or 'none'})")
    bundle = names[0]
    if bundle not in instance.bundles:
        return _fail(f"bundle {bundle!r} is not a bundle of this instance ({', '.join(instance.bundles)})")
    generation, why = _published_generation(mcp_url, token)
    if generation is None:
        print(f"warning: the published generation is unknown ({why}); the corpus records none, and a run will say "
              "it cannot check the corpus against the graph", file=sys.stderr)
    try:
        registry = read_registry(instance.deployment_dir)
        try:
            answers = deployment_answers(instance.deployment_dir)
        except Exception:  # noqa: BLE001 - no recorded answers: only ${archi_data_root} is filled in
            answers = {}
        plan = plan_sources(registry, data_root=instance.data_root, answers=answers,
                            deployment_dir=instance.deployment_dir)
        print(f"archi eval --raw-corpus: {bundle}: packing {', '.join(p.source for p in plan.included) or 'nothing'}")
        corpus = build_corpus(plan, bundle=bundle, out_root=corpus_root(instance.data_root, bundle),
                              generation=generation)
    except CorpusError as exc:
        return _fail(str(exc))
    print(f"corpus {corpus.path}")
    print(f"digest {corpus.digest}; generation {corpus.generation or 'unknown'}; "
          f"{sum(s['files'] for s in corpus.sources.values())} record files")
    print(f"coverage: {corpus.coverage}")
    for name, reason in corpus.left_out.items():
        print(f"left out: {name}: {reason}")
    return 0


@dataclass
class _AgentSetup:
    """What one invocation's agent arms share across its question sets: one model, one ledger, one cap."""

    arms: tuple
    price: Any
    key: Optional[str]
    caps: Any
    judge: Optional[tuple] = None  # (OpenAIChat, Price)
    ledger: Any = None
    bounds_printed: bool = False


def _prepare_agents(args: argparse.Namespace) -> "_AgentSetup | str":
    """Check every agent option before anything runs; the reason when one is unusable."""
    from archi.evaluation import agent as agent_mod
    from archi.evaluation import raw_arm

    arms = tuple(dict.fromkeys(a.strip() for a in args.agents.split(",") if a.strip()))
    unknown = sorted(set(arms) - set(AGENT_ARM_NAMES))
    if not arms or unknown:
        return f"--agents takes {', '.join(AGENT_ARM_NAMES)} (got {args.agents!r})"
    if not args.model:
        return "the agent arms need --model (and --model-url for an endpoint other than a local Ollama)"
    if args.tries < 1 or args.max_turns < 1 or args.max_tool_calls < 1 or args.try_seconds <= 0 \
            or args.max_output_tokens < 1 or args.judge_max_output_tokens < 1:
        return ("--tries, --max-turns, --max-tool-calls, --try-seconds, --max-output-tokens and "
                "--judge-max-output-tokens must be positive")
    try:
        if args.cap is None or args.cap <= 0:
            raise raw_arm.SpendError("the agent arms refuse to start without a positive spend cap: pass --cap DOLLARS"
                                     + ("" if args.cap is None else f" (got {args.cap!r})"))
        if not args.model_price:
            raise raw_arm.SpendError("the agent arms need --model-price IN/OUT[/CACHED] to keep the spend ledger")
        price = raw_arm.Price.parse(args.model_price)
        price.check_plausible(allow_low=args.allow_low_price)
        urls = [args.model_url] + ([args.judge_url] if args.judge else [])
        if not args.allow_remote_model:
            for url in urls:
                agent_mod.require_local(url)
        key = raw_arm.read_key_file(args.model_key_env)
        if key is None and not agent_mod.is_local_url(args.model_url):
            raise raw_arm.SpendError(f"no API key for {args.model_url}: set ${args.model_key_env} to the path of a "
                                     "file holding it")
        judge = None
        if args.judge:
            if not args.judge_model or not args.judge_price:
                raise raw_arm.SpendError("--judge needs --judge-model and --judge-price")
            judge_price = raw_arm.Price.parse(args.judge_price)
            judge_price.check_plausible(allow_low=args.allow_low_price, what="--judge-price")
            judge_key = raw_arm.read_key_file(args.judge_key_env)
            if judge_key is None and not agent_mod.is_local_url(args.judge_url):
                raise raw_arm.SpendError(f"no API key for the judge: set ${args.judge_key_env} to the path of a "
                                         "file holding it")
            judge = (agent_mod.OpenAIChat(args.judge_url, args.judge_model, api_key=judge_key or "none",
                                          temperature=None, reasoning_effort=args.judge_reasoning_effort,
                                          max_completion_tokens=args.judge_max_output_tokens), judge_price)
    except (raw_arm.SpendError, agent_mod.ModelError) as exc:
        return str(exc)
    caps = raw_arm.Caps(max_turns=args.max_turns, max_tool_calls=args.max_tool_calls, try_seconds=args.try_seconds)
    return _AgentSetup(arms, price, key, caps, judge)


def _context_with_agents(args: argparse.Namespace, instance, cset, mcp_url: str, token: str, report_root: Path,
                         setup: _AgentSetup) -> int:
    """One context run with the agent arms beside the no-model ones (PACT r3-context-quality)."""
    import asyncio
    import shutil
    import time

    from archi.evaluation import agent as agent_mod
    from archi.evaluation import raw_arm
    from archi.evaluation.context import publish_run, run_context, write_report
    from archi.evaluation.raw_corpus import CorpusError, newest_corpus, verify_corpus
    from archi.ops.context_gate import context_eval_dir

    caps = setup.caps
    try:
        corpus_dir = args.corpus
        if corpus_dir is None:
            if instance is None:
                raise CorpusError("name the raw corpus with --corpus (there is no instance data root to look in)")
            corpus_dir = newest_corpus(instance.data_root, cset.bundle)
        corpus = verify_corpus(corpus_dir, bundle=cset.bundle)
        sandbox = (raw_arm.BashSandbox(corpus.records, command_seconds=caps.command_seconds,
                                       output_bytes=caps.output_bytes, max_procs=caps.max_procs,
                                       memory_bytes=caps.memory_bytes, file_bytes=caps.file_bytes)
                   if raw_arm.AGENT_BASH in setup.arms else None)
    except (raw_arm.SandboxError, CorpusError) as exc:
        return _fail(str(exc))
    if sandbox is not None and sandbox.limit_scope != raw_arm.SCOPE_COMMAND:
        print(f"note: the sandbox's memory and process limits hold per process, not per command: "
              f"{sandbox.scope_note}", file=sys.stderr)
    if args.judge and not any(q.answer for q in cset.scored):
        print("note: --judge: the question file has no answer keys, so nothing is judged (the report says "
              f"\"{raw_arm.NO_ANSWER_KEYS}\")", file=sys.stderr)

    # The ledger is written while the run goes, in a hidden directory beside the runs, and moved into the
    # run's own directory with the reports (with --out, straight into it). One ledger, and one cap, per
    # invocation: a second question set carries on the same spend in its own run directory's file.
    if args.out:
        out_dir = args.out / cset.bundle if getattr(args, "_several_sets", False) else args.out
        work = None
        ledger_path = out_dir / raw_arm.LEDGER
    else:
        base = (context_eval_dir(instance.data_root, cset.bundle) if instance is not None
                else report_root / "context" / cset.bundle)
        work = base / f".agents-{os.getpid()}-{int(time.time())}"
        ledger_path = work / raw_arm.LEDGER
    try:
        if setup.ledger is None:
            setup.ledger = raw_arm.Ledger(ledger_path, args.cap)
        else:
            setup.ledger.redirect(ledger_path)
    except raw_arm.SpendError as exc:
        return _fail(str(exc))
    ledger = setup.ledger
    calls_before = ledger.calls
    api = agent_mod.model_api(args.model_url, args.model_api)
    model = agent_mod.make_model(args.model_url, args.model, api=api, api_key=setup.key or "ollama",
                                 temperature=None if args.reasoning_effort else 0.0,
                                 reasoning_effort=args.reasoning_effort,
                                 max_completion_tokens=args.max_output_tokens)
    metered = raw_arm.Metered(model, setup.price, ledger, args.max_output_tokens)
    judge = None
    if setup.judge is not None and any(q.answer for q in cset.scored):
        judge = raw_arm.Judge(raw_arm.Metered(setup.judge[0], setup.judge[1], ledger, args.judge_max_output_tokens),
                              passes=args.judge_passes)
    plan = raw_arm.AgentPlan(setup.arms, metered, caps, args.tries, corpus, sandbox, judge, judge_requested=args.judge,
                             reasoning_effort=args.reasoning_effort, model_url=args.model_url, model_api=api)
    manifest = corpus.manifest_paths()

    def progress(result, i: int, total: int) -> None:
        ev = result.evidence
        recall = "unscored" if ev is None or ev.recall is None else f"{len(ev.found)}/{ev.available}"
        print(f"[{i}/{total}] {result.arm} {result.question} try {result.attempt}: {result.status} {recall} "
              f"{result.tool_calls} calls ${result.cost:.4f}", flush=True)

    def context_progress(result, i: int, total: int) -> None:
        print(f"[{i}/{total}] {result.id}: " + " ".join(
            f"{n}={'error' if r.error else f'{len(r.found)}/{len(result.gold)}'}" for n, r in result.arms.items()),
              flush=True)

    def print_bounds() -> None:
        if setup.bounds_printed:
            return
        setup.bounds_printed = True
        bounds = raw_arm.one_call_bounds(plan, cset.scored)
        print("spend: one call is bounded at " + "; ".join(f"${v:.4f} ({k})" for k, v in bounds.items())
              + f"; the cap is ${ledger.cap:.2f}", flush=True)

    async def go():
        async with agent_mod.McpToolBox(mcp_url, token) as tools:
            gold = await raw_arm.resolve_gold(tools, cset.scored, corpus.keys(), manifest, corpus.records,
                                              drop_shared_text=False)
        print(f"gold: checking {len(manifest)} corpus files for text the gold records share with others", flush=True)
        started = time.monotonic()
        gold.records = await asyncio.to_thread(raw_arm.drop_shared, gold.records, corpus.records, manifest)
        dropped = sum(g.shared for golds in gold.records.values() for g in golds)
        print(f"gold: dropped {dropped} shared fingerprints in {time.monotonic() - started:.0f} s", flush=True)
        report = await run_context(cset, lambda: agent_mod.McpToolBox(mcp_url, token), mcp_url=mcp_url,
                                   progress=context_progress)
        print_bounds()
        async with agent_mod.McpToolBox(mcp_url, token) as tools:
            agents = await raw_arm.run_agent_arms(cset.scored, plan, tools, gold, manifest=manifest,
                                                  evaluated_generations=report.generations(), progress=progress)
        report.agents = agents.to_dict()
        return report, agents

    print(f"archi eval --context --agents: {cset.bundle}, {len(cset.scored)} scored questions, arms "
          f"{', '.join(setup.arms)}, model {args.model}, {args.tries} tries, cap ${args.cap:.2f}, corpus "
          f"{corpus.digest} (generation {corpus.generation or 'unknown'})")
    try:
        report, agents = asyncio.run(go())
    except (raw_arm.GoldError, agent_mod.ToolError) as exc:
        # Refused before any model call: leave nothing behind (a ledger with rows is kept).
        if ledger.calls == calls_before:
            if work is not None:
                shutil.rmtree(work, ignore_errors=True)
            elif ledger_path.exists() and not ledger_path.read_text():
                ledger_path.unlink()
        if isinstance(exc, agent_mod.ToolError):
            return _fail(f"MCP endpoint not usable: {exc}")
        return _fail(str(exc))
    if work is None:
        md, js = write_report(report, out_dir)
    else:
        md, js = publish_run(report, base, carry={raw_arm.LEDGER: ledger_path})
        shutil.rmtree(work, ignore_errors=True)
    data = report.to_dict()
    print(f"{report.status}: recall@{cset.k} " + ", ".join(f"{n} {s['recall_at_k']:.3f}" for n, s in data["arms"].items()))
    for name, s in data["agent_arms"]["arms"].items():
        print(f"{name}: evidence recall {s['evidence_recall']}, MRR {s['mrr']}, answer {s['answer_score']}, "
              f"${s['cost_usd']:.4f} over {s['tries']} tries, {s['failures']} errors, {s['capped']} capped")
    print(f"spent ${ledger.spent:.4f} of ${ledger.cap:.2f} this invocation; {md} and {js}")
    corpus_data = data["agent_arms"]["corpus"]
    if corpus_data["generation_warning"]:
        print(f"warning: {corpus_data['generation_warning']}", file=sys.stderr)
    for warning in ledger.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    if report.error:
        print(f"error: {report.error}", file=sys.stderr)
    code = report.exit_code
    if agents.aborted:
        print(f"error: the agent arms were aborted: {agents.aborted}; the report holds what completed",
              file=sys.stderr)
        return 2
    if agents.stopped:
        print(f"error: the agent arms stopped at the spend cap: {agents.stopped}; the report holds what completed",
              file=sys.stderr)
        code = max(code, EXIT_SPEND_CAP)
    return code
