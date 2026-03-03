"""Tool for synthesizing final answer from all subagent calls and creating answer template."""

from __future__ import annotations

import json
import logging
import re
from typing import Annotated, Any, Dict, List, Optional, Tuple

from langchain_core.runnables import RunnableConfig  # NEW: gives access to LangGraph config (thread_id, etc.)
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.prebuilt import InjectedState  # NEW: injects current graph state directly into the tool

from config.checkpoint import MEMORY_SAVER
from config.runtime import get_current_thread_id
from agents.prompts.common.verbosity import MAXIMAL_VERBOSE_INSTRUCTIONS

LOGGER = logging.getLogger(__name__)


async def _get_state_from_checkpointer(thread_id: str) -> Optional[Dict[str, Any]]:
    """Get the current state from the checkpointer."""
    config = {"configurable": {"thread_id": thread_id}}

    # Try direct get/aget with config (MemorySaver supports this)
    try:
        if hasattr(MEMORY_SAVER, "aget"):
            checkpoint_data = await MEMORY_SAVER.aget(config)
            if checkpoint_data:
                return checkpoint_data.get("channel_values", {})
    except Exception as e:
        LOGGER.debug("Async checkpointer get failed: %s", e)

    try:
        if hasattr(MEMORY_SAVER, "get"):
            checkpoint_data = MEMORY_SAVER.get(config)
            if checkpoint_data:
                return checkpoint_data.get("channel_values", {})
    except Exception as e:
        LOGGER.debug("Sync checkpointer get failed: %s", e)

    # Fallback: try direct access to internal storage
    try:
        if hasattr(MEMORY_SAVER, "_storage"):
            storage = getattr(MEMORY_SAVER, "_storage", {})
            if thread_id in storage:
                thread_data = storage[thread_id]
                if thread_data and isinstance(thread_data, dict):
                    checkpoints = thread_data.get("checkpoints", [])
                    if checkpoints:
                        latest_checkpoint = checkpoints[-1]
                        if isinstance(latest_checkpoint, dict):
                            return latest_checkpoint.get("channel_values", {})
    except Exception as e:
        LOGGER.warning("Direct storage access also failed: %s", e)

    return None


def _extract_user_query(messages: List[BaseMessage]) -> Optional[str]:
    """Extract the original user query from messages."""
    for msg in messages:
        if isinstance(msg, HumanMessage):
            content = msg.content if hasattr(msg, 'content') else str(msg)
            if content and isinstance(content, str) and content.strip():
                return content.strip()
    return None


def _extract_spawn_subagent_calls(messages: List[BaseMessage]) -> List[Dict[str, Any]]:
    """Extract all spawn_subagent tool calls and their outputs from messages."""
    subagent_calls = []
    
    i = 0
    while i < len(messages):
        msg = messages[i]
        
        # Look for tool calls to spawn_subagent
        if isinstance(msg, AIMessage):
            # Get tool_calls - could be a list or attribute
            tool_calls = []
            if hasattr(msg, 'tool_calls'):
                tool_calls = msg.tool_calls or []
            elif hasattr(msg, 'tool_calls') and callable(getattr(msg, 'tool_calls', None)):
                try:
                    tool_calls = msg.tool_calls() or []
                except:
                    pass
            
            for tool_call in tool_calls:
                # Handle both dict and object tool calls
                if isinstance(tool_call, dict):
                    tool_name = tool_call.get('name') or tool_call.get('tool')
                    tool_id = tool_call.get('id') or tool_call.get('tool_call_id')
                    args = tool_call.get('args', {}) or tool_call.get('arguments', {})
                else:
                    tool_name = getattr(tool_call, 'name', None) or getattr(tool_call, 'tool', None)
                    tool_id = getattr(tool_call, 'id', None) or getattr(tool_call, 'tool_call_id', None)
                    args = getattr(tool_call, 'args', {}) or getattr(tool_call, 'arguments', {})
                
                # Parse args if it's a string
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except:
                        args = {}
                
                if tool_name == 'spawn_subagent':
                    # Extract tool call arguments
                    subagent_type = args.get('subagent_type', 'unknown') if isinstance(args, dict) else 'unknown'
                    task_description = args.get('task_description', '') if isinstance(args, dict) else ''
                    inputs = args.get('inputs', []) if isinstance(args, dict) else []
                    
                    # Look for the corresponding tool response
                    tool_output = None
                    for j in range(i + 1, min(i + 20, len(messages))):  # Look ahead up to 20 messages
                        next_msg = messages[j]
                        if isinstance(next_msg, ToolMessage):
                            # Check if this tool message corresponds to our tool call
                            msg_tool_call_id = None
                            if hasattr(next_msg, 'tool_call_id'):
                                msg_tool_call_id = next_msg.tool_call_id
                            elif hasattr(next_msg, 'name') and hasattr(next_msg, 'id'):
                                # Some formats store it differently
                                if next_msg.name == 'spawn_subagent':
                                    msg_tool_call_id = next_msg.id
                            
                            if msg_tool_call_id and msg_tool_call_id == tool_id:
                                tool_output = next_msg.content if hasattr(next_msg, 'content') else str(next_msg)
                                break
                    
                    subagent_calls.append({
                        'subagent_type': str(subagent_type),
                        'task_description': str(task_description),
                        'inputs': inputs if isinstance(inputs, list) else [str(inputs)] if inputs else [],
                        'output': str(tool_output) if tool_output else None,
                        'has_insights': 'insights' in str(subagent_type).lower() or 
                                       (tool_output and '[auto-insights]' in str(tool_output))
                    })
        
        i += 1
    
    return subagent_calls


def _remove_handoffs_and_conclusions(content: str, *, remove_conclusions: bool) -> str:
    """
    Remove handoff blocks and conclusion sections from insights/reporting responses.
    Preserves 90% of original content with minimal changes.
    """
    if not content:
        return content
    
    # Remove handoff blocks (pattern: [handoff→...] ...)
    # Match [handoff→...] followed by content until next section, separator, or end
    # Also handle the separator lines (---) that may appear before/after handoff blocks
    handoff_pattern = r'(?:\n\n---\n\n)?\[handoff→[^\]]+\]\s*\n.*?(?=\n\n---\n\n|\n\n|\n##|\Z)'
    content = re.sub(handoff_pattern, '', content, flags=re.DOTALL | re.MULTILINE)
    
    # Remove standalone separator lines that might be left behind
    content = re.sub(r'\n\n---\n\n+', '\n\n', content)
    
    if remove_conclusions:
        # Remove conclusion sections ONLY when explicitly requested (e.g., merging multiple reports)
        # Look for patterns like "## Conclusion", "## Summary", "## Final Thoughts", etc.
        conclusion_patterns = [
            r'##\s+(Conclusion|Summary|Final\s+Thoughts|Closing|Wrap-up|End)\s*\n.*',
            r'###\s+(Conclusion|Summary|Final\s+Thoughts|Closing|Wrap-up|End)\s*\n.*',
        ]
        
        for pattern in conclusion_patterns:
            content = re.sub(pattern, '', content, flags=re.DOTALL | re.IGNORECASE)
    
    # Clean up multiple consecutive newlines
    content = re.sub(r'\n{3,}', '\n\n', content)
    
    return content.strip()


def _determine_response_type(user_query: Optional[str], subagent_calls: List[Dict[str, Any]]) -> str:
    """
    Determine if user wants a 'summary' or 'report/table' based on query keywords and subagent outputs.
    Returns 'report' or 'summary'.
    """
    if not user_query:
        return 'report'  # Default to report if ambiguous
    
    query_lower = user_query.lower()
    
    # Keywords for "report"
    report_keywords = ['report', 'table', 'monthly', 'yearly', 'breakdown', 'list', 'all', 
                       'detailed', 'complete', 'full', 'comprehensive', 'break down', 
                       'by month', 'by year', 'by department', 'by product']
    
    # Keywords for "summary"
    summary_keywords = ['summary', 'overview', 'brief', 'highlights', 'key points', 
                       'main findings', 'top', 'bottom', 'best', 'worst', 'insights']
    
    # Check for report keywords
    has_report_keywords = any(keyword in query_lower for keyword in report_keywords)
    
    # Check for summary keywords
    has_summary_keywords = any(keyword in query_lower for keyword in summary_keywords)
    
    # If both are present, prioritize report keywords (more specific)
    if has_report_keywords:
        return 'report'
    elif has_summary_keywords:
        return 'summary'
    else:
        # Default to report if ambiguous
        return 'report'


def _format_subagent_summary(subagent_calls: List[Dict[str, Any]]) -> str:
    """Format a summary of all subagent calls."""
    # Filter out visualization_react_subagent calls completely
    filtered_calls = [call for call in subagent_calls if call['subagent_type'] != 'visualization_react_subagent']
    
    if not filtered_calls:
        return "No subagents were called during this session."
    
    summary_lines = [f"## Subagent Activity Summary\n"]
    summary_lines.append(f"Total subagent calls: {len(filtered_calls)}\n")
    
    # Group by subagent type
    by_type: Dict[str, List[Dict[str, Any]]] = {}
    for call in filtered_calls:
        agent_type = call['subagent_type']
        if agent_type not in by_type:
            by_type[agent_type] = []
        by_type[agent_type].append(call)
    
    for agent_type, calls in by_type.items():
        summary_lines.append(f"\n### {agent_type.upper()} Subagent ({len(calls)} call(s))")
        for idx, call in enumerate(calls, 1):
            summary_lines.append(f"\n**Call {idx}:**")
            summary_lines.append(f"- Task: {call['task_description'][:200]}...")
            if call.get('output'):
                output_preview = str(call['output'])[:300]
                summary_lines.append(f"- Output preview: {output_preview}...")
    
    return "\n".join(summary_lines)


def _extract_combined_data(subagent_calls: List[Dict[str, Any]]) -> str:
    """
    Extract and combine all data from subagent outputs.
    For insights/reporting: display verbatim (90% as-is) with only handoff/conclusion removal.
    For other subagents: extract raw data for formatting according to new guidelines.
    """
    if not subagent_calls:
        return "No data collected from subagents."
    
    combined_sections = ["## Combined Data from Subagents\n"]
    
    # Separate insights/reporting from other subagents
    # Filter out visualization_react_subagent completely - do not include it anywhere
    insights_reporting_calls = []
    other_calls = []
    
    for call in subagent_calls:
        subagent_type = call.get('subagent_type', '').lower()
        # Skip visualization_react_subagent completely - do not include in any section
        if subagent_type == 'visualization_react_subagent':
            continue
        if subagent_type in ['insights', 'reporting']:
            insights_reporting_calls.append(call)
        else:
            other_calls.append(call)
    
    # Handle insights and reporting subagents (display verbatim with minimal cleaning)
    if insights_reporting_calls:
        combined_sections.append("\n### Insights and Reporting Subagent Outputs (Display Verbatim)")
        combined_sections.append("**CRITICAL INSTRUCTIONS:**")
        combined_sections.append("- Display these responses EXACTLY as-is (verbatim)")
        combined_sections.append("- Only remove handoff blocks ([handoff→...])")
        combined_sections.append("- Do NOT remove conclusions/insights unless you are merging multiple reporting outputs; then you may remove ONLY repeated conclusion sections")
        combined_sections.append("- Preserve all tables, formatting, and content structure")
        combined_sections.append("- Integrate these into the main response flow (not separate sections)")
        combined_sections.append("")
        
        reporting_calls = [c for c in insights_reporting_calls if c.get("subagent_type", "").lower() == "reporting"]
        remove_conclusions = len(reporting_calls) > 1
        
        for idx, call in enumerate(insights_reporting_calls, 1):
            subagent_type = call.get('subagent_type', 'unknown')
            combined_sections.append(f"\n#### {subagent_type.upper()} Subagent Call {idx}")
            combined_sections.append(f"**Task:** {call.get('task_description', 'N/A')}")
            
            if call.get('output'):
                output_str = str(call['output'])
                # Apply minimal cleaning: always remove handoff blocks; remove conclusions only when merging multiple reporting outputs
                cleaned_output = _remove_handoffs_and_conclusions(
                    output_str,
                    remove_conclusions=remove_conclusions and subagent_type.lower() == "reporting",
                )
                combined_sections.append(f"**Cleaned Output (display verbatim):**\n{cleaned_output}")
    
    # Handle other subagents (extract raw data for formatting)
    # visualization_react_subagent is already filtered out in the loop above, so other_calls won't contain it
    if other_calls:
        combined_sections.append("\n### Other Subagent Outputs (Format According to Guidelines)")
        combined_sections.append("**INSTRUCTIONS:** Format this data according to the comprehensive guidelines below.")
        combined_sections.append("")
        
        for idx, call in enumerate(other_calls, 1):
            # visualization_react_subagent is already filtered out above, so no check needed here
            
            combined_sections.append(f"\n#### Data from {call['subagent_type']} Call {idx}")
            combined_sections.append(f"**Task:** {call['task_description']}")
            
            if call.get('inputs'):
                combined_sections.append(f"**Inputs provided:**")
                for inp in call['inputs']:
                    combined_sections.append(f"- {str(inp)[:200]}...")
            
            if call.get('output'):
                output_str = str(call['output'])
                combined_sections.append(f"**Raw Output (format according to guidelines):**\n{output_str}")
    
    return "\n".join(combined_sections)


def _generate_analysis(subagent_calls: List[Dict[str, Any]], user_query: Optional[str]) -> str:
    """
    Generate analysis guidance.
    For insights/reporting subagents: they provide their own analysis, display verbatim.
    For other subagents: provide analysis guidance.
    """
    insights_calls = [call for call in subagent_calls if call.get('subagent_type', '').lower() == 'insights']
    reporting_calls = [call for call in subagent_calls if call.get('subagent_type', '').lower() == 'reporting']
    other_calls = [call for call in subagent_calls if call.get('subagent_type', '').lower() not in ['insights', 'reporting', 'visualization_react_subagent']]
    
    analysis = ["## Analysis Guidance\n"]
    
    if insights_calls or reporting_calls:
        analysis.append("**For Insights/Reporting Subagents:**")
        analysis.append("- Insights and reporting subagent outputs already contain their own analysis")
        analysis.append("- Display their analysis content verbatim (90% as-is) as part of the main response")
        analysis.append("- Do not add additional analysis for insights/reporting subagents")
        analysis.append("")
    
    
    # Count subagent types (exclude visualization_react_subagent completely)
    cypher_calls = [c for c in subagent_calls if c['subagent_type'] == 'cypher']
    other_calls = [c for c in subagent_calls if c['subagent_type'] != 'cypher' and c['subagent_type'] != 'insights' and c['subagent_type'] != 'visualization_react_subagent']
    
    analysis.append(f"- Executed {len(cypher_calls)} Cypher query(ies) to retrieve data")
    if other_calls:
        analysis.append("**For Other Subagents:**")
        analysis.append("- Perform your own analysis on the data from other subagents")
        analysis.append("- For each major finding, add a brief 'Why this matters' note and concrete 'Actions/Next Steps'")
        analysis.append("- Keep insights rich (do NOT over-condense); favor meaningful detail over brevity")
        analysis.append("- If a product name/sku is null/missing, you may ignore that row in your reasoning and outputs")
        analysis.append("")
        
        # Count subagent types
        cypher_calls = [c for c in other_calls if c.get('subagent_type', '').lower() == 'cypher']
        other_non_cypher = [c for c in other_calls if c.get('subagent_type', '').lower() != 'cypher']
        
        if cypher_calls:
            analysis.append(f"- Executed {len(cypher_calls)} Cypher query(ies) to retrieve data")
        if other_non_cypher:
            analysis.append(f"- Utilized {len(other_non_cypher)} other subagent(s) for specialized tasks")
        analysis.append("")
    
    if not insights_calls and not reporting_calls and not other_calls:
        analysis.append("- No subagent calls found. No analysis needed.")
    
    if user_query:
        analysis.append(f"\n### Query Context:")
        analysis.append(f"Original user request: {user_query[:200]}")
    
    return "\n".join(analysis)


def _create_answer_template(
    user_query: Optional[str],
    subagent_calls: List[Dict[str, Any]],
    has_insights: bool
) -> str:
    """Create a template structure for the final answer with new comprehensive guidelines."""
    template = ["## Final Answer Template Structure\n"]
    template.append("\nUse this structure to format your final response:\n")
    
    # Determine response type
    response_type = _determine_response_type(user_query, subagent_calls)
    
    # Check which subagents were called
    insights_calls = [c for c in subagent_calls if c['subagent_type'].lower() == 'insights']
    reporting_calls = [c for c in subagent_calls if c['subagent_type'].lower() == 'reporting']
    other_calls = [c for c in subagent_calls if c['subagent_type'].lower() not in ['insights', 'reporting', 'visualization_react_subagent']]
    
    has_insights_subagent = len(insights_calls) > 0
    has_reporting_subagent = len(reporting_calls) > 0
    has_other_subagents = len(other_calls) > 0
    
    # Introduction section
    template.append("\n### 1. Introduction (PLACEHOLDER)")
    template.append("   **Instructions:** Provide a brief 1-2 sentence introduction that:")
    template.append("   - Directly addresses the user's query without repeating it verbatim")
    template.append("   - Sets context for the data presented")
    if user_query:
        template.append(f"   - Reference: Original query was about '{user_query[:100]}'")
    
    # Section for insights/reporting subagents (integrated into main flow)
    section_num = 2
    if has_insights_subagent or has_reporting_subagent:
        template.append(f"\n### {section_num}. Insights and Reporting Content (INTEGRATE INTO MAIN FLOW)")
        template.append("   **CRITICAL INSTRUCTIONS FOR INSIGHTS/REPORTING SUBAGENTS:**")
        template.append("   - Display responses from insights and reporting subagents EXACTLY as-is (verbatim)")
        template.append("   - Only remove handoff blocks ([handoff→...])")
        template.append("   - Preserve ALL tables, formatting, markdown structure, data points, and insights")
        template.append("   - Do NOT remove conclusions/insights unless you are merging MULTIPLE reporting outputs; then you may remove ONLY repeated conclusion sections to reduce repetition")
        template.append("   - Integrate these responses naturally into the main response flow")
        template.append("   - Do NOT create separate sections - blend them with other content")
        template.append("   - If multiple insights/reporting calls exist, display each one in sequence")
        if has_insights_subagent:
            template.append("   - Insights subagent output should appear as part of the main narrative")
        if has_reporting_subagent:
            template.append("   - Reporting subagent output should appear as part of the main narrative")
        section_num += 1
    
    # Section for other subagents (apply new comprehensive guidelines)
    if has_other_subagents:
        template.append(f"\n### {section_num}. Data from Other Subagents (APPLY COMPREHENSIVE GUIDELINES)")
        template.append("   **COMPREHENSIVE FORMATTING GUIDELINES FOR NON-INSIGHTS/REPORTING SUBAGENTS:**")
        template.append("")
        template.append("   **1. First, determine what kind of response the user expects:**")
        template.append(f"      - Detected response type: {response_type.upper()}")
        if response_type == 'summary':
            template.append("      - User asked for a SUMMARY → Provide a concise, paragraph-style narrative")
            template.append("        with key highlights, patterns, or anomalies")
        else:
            template.append("      - User asked for a REPORT/TABLE → Return a structured list or grouped report")
            template.append("        with complete coverage (include ALL months, years, or departments)")
        template.append("")
        template.append("   **2. For reports or time-based breakdowns:**")
        template.append("      - List data in CHRONOLOGICAL ORDER (e.g., Jan → Dec for months; earliest to latest for years)")
        template.append("      - Even if some entries (like a month or department) have LOW OR NO SALES,")
        template.append("        they MUST be included with a corresponding value (like 0) — do NOT skip any relevant groups")
        template.append("      - Each row or section should include at least:")
        template.append("        * The grouping key (e.g., month/year/department)")
        template.append("        * Sales figures (e.g., amount_sold, units_sold)")
        template.append("        * Any relevant product/store name if part of the original query")
        template.append("")
        template.append("   **3. When summarizing (instead of reporting):**")
        template.append("      - Highlight top and bottom performers")
        template.append("      - Briefly explain potential reasons or patterns if observable")
        template.append("      - Include names and figures, but avoid exhaustive listings")
        template.append("")
        template.append("   **4. If no results were found:**")
        template.append("      - State that clearly")
        template.append("      - Suggest likely reasons (e.g., wrong date range, no data, misspelled input)")
        template.append("      - Offer tips for improving the query next time")
        template.append("")
        template.append("   **5. Additional critical rules:**")
        template.append("      - **CRITICAL: You MUST include ALL content and tables in your response.**")
        template.append("        Do not omit any information that was provided to you.")
        template.append("      - **Whenever the response involves multiple items** (products, variants, stores, months,")
        template.append("        or any group of records), **always format the response in a clean Markdown table.**")
        template.append("        This applies even if the user does not explicitly request a table.")
        template.append("      - **If you need to include more than one Markdown table**, insert at least one empty line")
        template.append("        (two newline characters) or a short separator line between tables so the UI does not merge them.")
        template.append("      - **For large datasets**: If you see sample data with indicators like 'Showing X of Y items',")
        template.append("        create a table with the sample data and clearly indicate the total count.")
        template.append("        Add a note about how users can get more specific results.")
        template.append("      - **Never show raw JSON data** in your response. Always convert JSON arrays and objects")
        template.append("        into readable tables or formatted text.")
        template.append("      - **For product listings**: Always include key fields like name, SKU, price, and status in a table format.")
        template.append("      - **COMPLETENESS CHECK**: Before finalizing your response, verify that you have included:")
        template.append("        * All tables that were generated")
        template.append("        * All data points mentioned in the results")
        template.append("        * Complete coverage of time periods, categories, or groups")
        template.append("        * Any summary statistics or insights")
        template.append("      - Do **not fabricate or assume** missing values.")
        template.append("      - Keep the tone business-informative but friendly")
        template.append("      - Avoid repetition and fluff — be clear and useful")
        template.append("      - Always double-check: If the user asked for **all months** or a **full report**,")
        template.append("        ensure **no groups are skipped**")
        template.append("      - **FINAL CHECK**: Your response should be a complete, self-contained answer")
        template.append("        that includes everything the user needs to see")
        
        # New Guideline for GCS Links
        template.append("   **6. ⚠️ GCS Download Links (HIGHEST PRIORITY):**")
        template.append("      - If ANY subagent output contains a Google Cloud Storage (GCS) download link (e.g., for CSV export),")
        template.append("        you **MUST** include this link prominently in your response.")
        template.append("      - Place it at the top of the relevant data section or immediately after the preview table.")
        template.append("      - Format it clearly: `[Download Full Dataset (CSV)](URL)`.")
        template.append("      - Do not bury it in text; make it a standalone bullet point or line.")
        
        section_num += 1
    
    # General formatting guidelines
    template.append(f"\n### {section_num}. General Formatting Guidelines")
    template.append("   - Never repeat or paraphrase the user request in your final response")
    template.append("   - Provide one short sentence (≤1 line) before tables naming the entity/metric")
    template.append("   - Use markdown tables with Title Case column names")
    template.append("   - Use $ for all currency values in the tables, and % for all percentage values")
    template.append("   - If multiple tables are needed, organize them by topic/metric")
    template.append("   - If a table is not available, explain why in the analysis section")
    
    # Insights section
    if has_insights:
        template.append("\n### 3. Insights Section (PLACEHOLDER)")
        template.append("   **Instructions:** Include the insights analysis:")
        template.append("   - Use the insights provided by the insights subagent")
        template.append("   - Synthesize drivers, trends, and hypotheses")
        template.append("   - Keep to 3-5 sentences")
        template.append("   - Label clearly as 'Insights' section")
    else:
        template.append("\n### 3. Analysis Section (PLACEHOLDER)")
        template.append("   **Instructions:** Provide your own analysis:")
        template.append("   - Synthesize the key findings from the data")
        template.append("   - Highlight important patterns or trends")
        template.append("   - Keep concise and focused")
    
    # Conclusion section (if needed)
    if len(subagent_calls) > 1 or (user_query and any(word in user_query.lower() for word in ['summary', 'conclusion', 'overview'])):
        template.append("\n### 4. Conclusion (PLACEHOLDER - Optional)")
        template.append("   **Instructions:** If the query requires a conclusion:")
        template.append("   - Summarize the main takeaways")
        template.append("   - Keep it brief (1-2 sentences)")
    
    template.append("\n### Formatting Guidelines:")
    template.append("- ⚠️ CRITICAL: NEVER include visualization code in your final answer. Visualization React code (JSON with `code`, `libraries`, `title` fields) from `visualization_react_subagent` should ONLY appear in the sandbox via `spawn_subagent` output. DO NOT include this code, JSON blocks, or any React/JavaScript code in your final answer text. The visualization code is handled separately by the frontend and should never appear in text responses.")
    template.append("- ⚠️ MANDATORY: If a GCS download link is present in the data, it must be displayed prominently. This is a HIGH PRIORITY requirement.")
    template.append("- Never repeat or paraphrase the user request in your final response")
    template.append("- Provide one short sentence (≤1 line) before tables naming the entity/metric")
    template.append("- Use markdown tables with Title Case column names")
    template.append("- Do not add prose outside tables unless insights/analysis section is included")
    template.append("- If insights were generated, append a clearly labeled 'Insights' section")
    template.append("- Template priority: If any instruction here conflicts with the base prompt, this template wins.")
    template.append("- Verbosity priority: Use the verbosity level specified in the VERBOSE OUTPUT INSTRUCTIONS (below) even if it conflicts with base prompt verbosity.")
    template.append("- High-cardinality guidance: For entity-heavy results (e.g., many customers/orders), present per-entity sections or grouped tables. Default to top ~20 entities, and within each entity include only the most recent 4–5 records. Paginate or batch if needed; do not drop entities due to truncation.")
    template.append("- Layout: insert clear blank lines between sections (Intro, Data, Analysis/Insights, Conclusion) so the answer is easy to scan.")
    template.append("- Presentation: tasteful text highlighting (bold/italics) is allowed. Emojis are allowed to improve readability/section breaks—use sparingly and only if they enhance clarity.")
    template.append("- Insights-first: Anchor 'Why this matters' and 'Actions/Next Steps' to the insights/analysis derived from the data (not to the original user request wording).")
    template.append("- For each major insight/analysis, add a brief 'Why this matters' note and concrete 'Actions/Next Steps' tied to that insight. These can be inline bullets under the analysis/insights content (no separate standalone section needed).")
    template.append("- Do NOT over-condense insights: keep the insights narrative rich enough to convey key implications; favor meaningful detail over brevity.")
    template.append("- If the insights subagent ran, you may reuse relevant sections verbatim (tables/insight paragraphs) from its output to preserve fidelity; integrate them into the template structure.")
    template.append("   - Use N/A for unavailable data")
    template.append("   - Template priority: If any instruction here conflicts with the base prompt, this template wins.")
    template.append("   - Verbosity priority: Use the verbosity level specified in the VERBOSE OUTPUT INSTRUCTIONS (below)")
    template.append("     even if it conflicts with base prompt verbosity.")
    template.append("   - Layout: insert clear blank lines between sections so the answer is easy to scan.")
    template.append("   - Presentation: tasteful text highlighting (bold/italics) is allowed.")
    template.append("     Emojis are allowed to improve readability/section breaks—use sparingly and only if they enhance clarity.")
    
    return "\n".join(template)


@tool
async def draft_answer(
    # NEW: InjectedState injects the live graph state at call time — not exposed to the LLM.
    # This works in both FastAPI and `langgraph dev` because it reads from the graph runtime
    # directly, bypassing MEMORY_SAVER entirely (which is what was failing in langgraph dev).
    state: Annotated[dict, InjectedState()],
    # NEW: RunnableConfig carries the LangGraph configurable dict (thread_id, etc.).
    # In `langgraph dev` this is always populated by the in-memory runtime.
    config: RunnableConfig,
) -> str:
    """
    ⚠️ MANDATORY FINAL STEP - DO NOT SKIP THIS TOOL ⚠️

    You MUST call this tool BEFORE providing ANY final answer to the user. This is not optional.

    This tool automatically:
    1. Accesses the conversation state to find all spawn_subagent tool calls and outputs
    2. Extracts the original user query
    3. Provides a brief summary of what was done
    4. Shows combined data from all subagent outputs
    5. Lists insights analysis if insights subagent was called, otherwise generates analysis
    6. Creates a template structure with placeholders and instructions for the final answer

    WORKFLOW: After completing all subagent calls, call this tool (no parameters needed), then follow the template it provides to write your final response.

    ⚠️ NEVER provide a final answer without calling this tool first. This is a hard requirement.
    """
    # NEW PRIMARY PATH: Read messages directly from the injected graph state.
    # This works in both FastAPI and `langgraph dev` — no checkpointer lookup needed.
    messages = state.get('messages', [])

    if not messages:
        # FALLBACK PATH (FastAPI / edge cases): attempt the original checkpointer approach.
        # Try context variable first (set by chat_streamer.py in the FastAPI app).
        thread_id = get_current_thread_id()
        if thread_id:
            LOGGER.debug("InjectedState empty — falling back to context-variable thread_id: %s", thread_id)

        # If context variable is also missing, try the LangGraph config (always present in langgraph dev).
        if not thread_id:
            thread_id = (config or {}).get("configurable", {}).get("thread_id")
            if thread_id:
                LOGGER.debug("InjectedState empty — falling back to config thread_id: %s", thread_id)

        # Last resort: scan MEMORY_SAVER internal storage (FastAPI only).
        if not thread_id:
            LOGGER.warning("Could not extract thread_id from config or context, trying to find from checkpointer storage")
            try:
                if hasattr(MEMORY_SAVER, '_storage'):
                    storage = getattr(MEMORY_SAVER, '_storage', {})
                    if storage:
                        thread_id = list(storage.keys())[-1]
                        LOGGER.info("Using thread_id from checkpointer storage: %s", thread_id)
            except Exception as e:
                LOGGER.warning("Could not get thread_id from checkpointer storage: %s", e)

        if not thread_id:
            return "Error: Could not extract thread_id from config, context variable, or checkpointer storage. Cannot access conversation state. The tool requires access to the conversation context to work properly."

        # Retrieve state via checkpointer (FastAPI fallback).
        cp_state = await _get_state_from_checkpointer(thread_id)
        if not cp_state:
            return "Error: Could not retrieve conversation state from checkpointer."
        messages = cp_state.get('messages', [])

    if not messages:
        return "Error: No messages found in conversation state."
    
    # Extract user query
    user_query = _extract_user_query(messages)
    
    # Extract all spawn_subagent calls
    subagent_calls = _extract_spawn_subagent_calls(messages)
    
    # Build the synthesis report
    report_sections = []
    
    # Brief summary
    report_sections.append("=" * 80)
    report_sections.append("FINAL ANSWER SYNTHESIS")
    report_sections.append("=" * 80)
    report_sections.append("")
    
    # What was done
    report_sections.append(_format_subagent_summary(subagent_calls))
    report_sections.append("")
    
    # Combined data
    report_sections.append(_extract_combined_data(subagent_calls))
    report_sections.append("")
    
    # Analysis (insights or generated)
    analysis = _generate_analysis(subagent_calls, user_query)
    report_sections.append(analysis)
    report_sections.append("")
    
    # Answer template
    has_insights = any(call['has_insights'] for call in subagent_calls)
    template = _create_answer_template(user_query, subagent_calls, has_insights)
    report_sections.append(template)
    report_sections.append("")
    
    # Add MAXIMAL_VERBOSE_INSTRUCTIONS
    report_sections.append("=" * 80)
    report_sections.append("VERBOSE OUTPUT INSTRUCTIONS")
    report_sections.append("=" * 80)
    report_sections.append("")
    report_sections.append(MAXIMAL_VERBOSE_INSTRUCTIONS)
    report_sections.append("")
    
    report_sections.append("=" * 80)
    report_sections.append("⚠️ MANDATORY INSTRUCTIONS FOR ALL MODELS (GPT, Gemini, Claude) ⚠️")
    report_sections.append("=" * 80)
    report_sections.append("")
    report_sections.append("1. **FOLLOW THIS TEMPLATE EXACTLY** - It overrides ALL base prompt instructions")
    report_sections.append("2. **USE MAXIMAL VERBOSITY** - Write detailed, multi-paragraph responses with analysis")
    report_sections.append("3. **DO NOT BE BRIEF** - Even if base prompt says 'be concise', this template says BE VERBOSE")
    report_sections.append("4. **INCLUDE ALL SECTIONS**:")
    report_sections.append("   - Introduction (1-2 sentences)")
    report_sections.append("   - Data tables with all metrics")
    report_sections.append("   - 'Why This Matters' section")
    report_sections.append("   - 'Actions/Next Steps' section")
    report_sections.append("   - Conclusion")
    report_sections.append("5. **USE FORMATTING**: Bold headers, emojis for sections, blank lines between sections")
    report_sections.append("6. **VERIFY TODOS**: Ensure all todos are completed/cancelled before this point")
    report_sections.append("")
    report_sections.append("=" * 80)
    report_sections.append("END OF SYNTHESIS - NOW WRITE YOUR FINAL ANSWER FOLLOWING THE ABOVE")
    report_sections.append("=" * 80)
    
    return "\n".join(report_sections)


__all__ = ["draft_answer"]

