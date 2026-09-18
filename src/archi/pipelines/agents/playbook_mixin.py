from __future__ import annotations
from typing import Any, Callable, Dict, List

from src.archi.pipelines.agents.tools import (
    create_playbook_tool,
    create_playbook_listing_middleware,
    create_save_playbook_tool,
    create_update_playbook_tool,
    create_delete_playbook_tool,
    get_playbook_owner,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


class SupportsPlaybooks:
    """Opt-in agent capability: per-user playbooks (authoring tools + load tool + the
    always-in-context listing). Mix in BEFORE BaseReActAgent. Agents that don't inherit
    this carry no playbook code and open no PlaybookService connection."""

    # The hooks this mixin contributes to via cooperative super() chaining. A base
    # agent's implementations are terminal, so any non-mixin class positioned before
    # SupportsPlaybooks in the MRO that defines one of these WITHOUT chaining would
    # swallow the whole feature — silently (its __init__ would not even run).
    _COOPERATIVE_HOOKS = ("_tool_definitions", "_build_static_tools", "_build_static_middleware")

    @staticmethod
    def _is_terminal_hook(func) -> bool:
        """True when a hook implementation ends the builder chain.

        A cooperative hook calls super(), which shows up as the name ``super``
        in its code object; a terminal one (e.g. BaseReActAgent's builders)
        never does. Anything we cannot introspect is treated as cooperative —
        the guard must never reject a valid composition, only the provably
        chain-breaking one.
        """
        code = getattr(func, "__code__", None)
        return code is not None and "super" not in code.co_names

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        mro = cls.__mro__
        mixin_idx = mro.index(SupportsPlaybooks)
        shadower = next(
            (c for c in mro[:mixin_idx]
             if not issubclass(c, SupportsPlaybooks)
             and any(hook in vars(c)
                     and SupportsPlaybooks._is_terminal_hook(vars(c)[hook])
                     for hook in SupportsPlaybooks._COOPERATIVE_HOOKS)),
            None,
        )
        if shadower is not None:
            raise TypeError(
                f"{cls.__name__}: SupportsPlaybooks must come BEFORE "
                f"{shadower.__name__} in the class bases — declare "
                f"`class {cls.__name__}(SupportsPlaybooks, {shadower.__name__})`. "
                f"As written, {shadower.__name__}'s builder methods terminate the "
                "chain and every playbook tool would be silently dropped."
            )

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        playbooks_requested = bool(
            set(self._playbook_tool_definitions()).intersection(self.selected_tool_names)
        )
        self._playbook_service = (
            self._init_playbook_service() if playbooks_requested else None
        )

    def _init_playbook_service(self):
        try:
            from src.utils.env import read_secret
            from src.utils.postgres_service_factory import PostgresServiceFactory
            factory = PostgresServiceFactory.get_instance()
            if factory is None:
                factory = PostgresServiceFactory.from_env(password_override=read_secret("PG_PASSWORD"))
                PostgresServiceFactory.set_instance(factory)
            return factory.playbook_service
        except AttributeError:
            raise
        except Exception as e:
            logger.warning("PlaybookService unavailable; playbooks disabled: %s", e)
            return None

    def _tool_definitions(self) -> Dict[str, Dict[str, Any]]:
        # BaseReActAgent has no _tool_definitions(); only chain to a base that does,
        # so this mixin stays usable above a bare base agent (base stays untouched).
        _base = getattr(super(), "_tool_definitions", None)
        defs = _base() if callable(_base) else {}
        defs.update(self._playbook_tool_definitions())
        return defs

    def _playbook_tool_definitions(self) -> Dict[str, Dict[str, Any]]:
        return {
            "save_playbook": {"builder": self._build_save_playbook_tool,
                "description": "Save a new playbook to the user's personal playbook library."},
            "update_playbook": {"builder": self._build_update_playbook_tool,
                "description": "Modify an existing saved playbook — rename, change its description, or change its body (append or overwrite). Use save_playbook only for brand-new playbooks."},
            "delete_playbook": {"builder": self._build_delete_playbook_tool,
                "description": "Permanently delete a saved playbook by name (only after the user confirms)."},
        }

    def _build_save_playbook_tool(self) -> Callable:
        return create_save_playbook_tool(getattr(self, "_playbook_service", None), get_playbook_owner)

    def _build_update_playbook_tool(self) -> Callable:
        return create_update_playbook_tool(getattr(self, "_playbook_service", None), get_playbook_owner)

    def _build_delete_playbook_tool(self) -> Callable:
        return create_delete_playbook_tool(getattr(self, "_playbook_service", None), get_playbook_owner)

    def _build_static_tools(self) -> List[Callable]:
        tools = list(super()._build_static_tools())
        if getattr(self, "_playbook_service", None) is not None:
            tools.append(create_playbook_tool(self._playbook_service, get_playbook_owner))
        else:
            logger.info("Playbook tool not registered: no PlaybookService available.")
        return tools

    def _build_static_middleware(self) -> List[Callable]:
        mw = list(super()._build_static_middleware())
        if getattr(self, "_playbook_service", None) is not None:
            mw.append(create_playbook_listing_middleware(self._playbook_service, get_playbook_owner))
        return mw
