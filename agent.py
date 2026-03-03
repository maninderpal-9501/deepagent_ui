"""EKYAM - riOS (ekyam-chief-agent) with planning, file management, and subagent integration."""
import os
from typing import Optional

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain_core.language_models import BaseChatModel
from langgraph.graph.state import CompiledStateGraph
from config.middleware import build_main_agent_middleware
from agents.subagents.registry import compile_registered_subagents
from agents.tools import create_main_agent_tools
from config import settings
from utils.ai.llm import get_llm
from agents.prompts.system.system_prompt import SYSTEM_PROMPT, build_system_prompt

import logging
from utils.db.neo4j_db import get_neo4j_db
get_neo4j_db()  # initializes the global neo4j_db singleton

def create_llm(
    *,
    provider_override: Optional[str] = None,
    model_override: Optional[str] = None,
    temperature_override: Optional[float] = None,
) -> BaseChatModel:
    """Create and configure the LLM based on optional per-request overrides."""
    return get_llm(
        provider_override=provider_override,
        model_override=model_override,
        temperature_override=temperature_override,
    )

# ... your imports and defs ...

def build_graph(config: dict = None) -> CompiledStateGraph:
    """Builds the graph for LangGraph CLI/Studio."""
    user_query = config.get("configurable", {}).get("user_query", "") if config else ""
    
    llm = create_llm(
        provider_override='groq-openai',
        model_override='',
        temperature_override=0.4,
    )
    middleware = build_main_agent_middleware(llm)
    
    os.makedirs(settings.context_dir, exist_ok=True)
    backend = FilesystemBackend(root_dir=settings.context_dir, virtual_mode=True)
    
    main_tools = create_main_agent_tools()
    registered_subagents = compile_registered_subagents()
    system_prompt = build_system_prompt(user_query=user_query)
    
    # Do NOT pass checkpointer here — langgraph-runtime-inmem manages its own
    # checkpointing and cannot serialize MemorySaver (contains _thread.RLock).
    agent = create_deep_agent(
        model=llm,
        system_prompt=system_prompt,
        backend=backend,
        tools=main_tools,
        subagents=registered_subagents,
        middleware=middleware,
    )
    return agent

# For CLI export (static fallback)
graph = build_graph()