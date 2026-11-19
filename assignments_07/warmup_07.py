# Amir Daniali
# https://github.com/amirdaniali
# Code The Dream  PyAICC 26.4
# Week 7

import os
import json
from pathlib import Path

import pandas as pd
import numpy as np
from scipy import stats
from dotenv import load_dotenv
from openai import OpenAI

# smolagents imports (used in Lesson 04 section)
from smolagents import (
    ToolCallingAgent,
    CodeAgent,
    OpenAIServerModel,
    Tool,
    tool,
)

BASE_DIR = Path(__file__).parent  # assignments_07/
RESOURCES_DIR = BASE_DIR / "resources"  # assignments_07/resources/
OUTPUTS_DIR = BASE_DIR / "outputs"  # assignments_07/outputs/
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Setup: load API key from .env
# ---------------------------------------------------------------------------
load_dotenv()
client = OpenAI()  # uses OPENAI_API_KEY from environment
MODEL = "gpt-4o-mini"


# ===========================================================================
# --- Lesson 02: Tool Definitions and the ReAct Loop ---
# ===========================================================================


# --- Q1 --------------------------------------------------------------------
def celsius_to_fahrenheit(celsius: float) -> str:
    """Convert a Celsius temperature to Fahrenheit and return it as a formatted string."""
    fahrenheit = (celsius * 9 / 5) + 32
    return f"{celsius}°C is {fahrenheit}°F"


# JSON schema describing celsius_to_fahrenheit to an LLM.
celsius_schema = {
    "name": "celsius_to_fahrenheit",
    "description": (
        "Convert a temperature given in degrees Celsius to degrees Fahrenheit. "
        "Returns a human-readable string like '37°C is 98.6°F'."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "celsius": {
                "type": "number",
                "description": "Temperature in degrees Celsius to convert.",
            }
        },
        "required": ["celsius"],
    },
}


print("=== Q1: Direct calls to celsius_to_fahrenheit ===")
for temp in [0, 100, -40]:
    print(celsius_to_fahrenheit(temp))
# Terminal output:
# === Q1: Direct calls to celsius_to_fahrenheit ===
# 0°C is 32.0°F
# 100°C is 212.0°F
# -40°C is -40.0°F
#
# Analysis:
# The function correctly applies the formula (C * 9/5) + 32. The edge case of -40
# correctly returns -40.0, which is the unique temperature where the Celsius and
# Fahrenheit scales intersect.


# --- Q2 --------------------------------------------------------------------
get_current_time_schema = {
    "name": "get_current_time",
    "description": "Return the current local date and time as a string.",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}


def get_current_time() -> str:
    """Return the current local date/time."""
    from datetime import datetime

    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


TOOLS_V1 = {
    "get_current_time": {
        "schema": get_current_time_schema,
        "fn": get_current_time,
    }
}


def run_agent(query: str, tools: dict = None) -> str:
    """
    Minimal ReAct loop:
      1. Send system + user message to the model.
      2. If the model returns tool_calls, execute them, append results, repeat.
      3. Otherwise return the assistant's final text response.
    """
    if tools is None:
        tools = TOOLS_V1

    system_prompt = (
        "You are a helpful assistant. You have access to a small set of tools. "
        "When the user's request requires one of your tools, call it. "
        "Otherwise answer directly in plain English."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]

    tool_schemas = [t["schema"] for t in tools.values()]

    while True:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=[{"type": "function", "function": s} for s in tool_schemas],
            tool_choice="auto",
        )
        msg = response.choices[0].message
        messages.append(msg.model_dump())

        if msg.tool_calls:
            for tc in msg.tool_calls:
                name = tc.function.name
                args = json.loads(tc.function.arguments)
                fn = tools[name]["fn"]
                result = fn(**args)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": str(result),
                    }
                )
        else:
            return msg.content


# Prediction block for Q2:
# Q: Will calling run_agent("Convert 100 degrees Celsius to Fahrenheit") trigger a tool call?
# A: NO. The only tool available is get_current_time, which cannot convert temperatures.
# Q: How many API calls will be made to answer this query?
# A: 1. Since no tool call is issued, the loop runs exactly once.

print("\n=== Q2: Agent with only get_current_time ===")
result_q2 = run_agent("Convert 100 degrees Celsius to Fahrenheit")
print("Result:", result_q2)
# Terminal output:
# === Q2: Agent with only get_current_time ===
# Result: To convert degrees Celsius to Fahrenheit, you can use the formula:
# \n F = \frac{9}{5}C + 32 \n ... So, 100 degrees Celsius is equal to 212 degrees Fahrenheit.
#
# Analysis:
# The prediction was CORRECT. The output is a plain-English explanation of the
# conversion formula. No tool was called, and no tool result appears in the trace.
# The model answered directly from its pre-training because its only available tool
# (get_current_time) is completely irrelevant to temperature conversion. Exactly 1
# API call was made.


# --- Q3 --------------------------------------------------------------------
TOOLS_V2 = {
    "get_current_time": {
        "schema": get_current_time_schema,
        "fn": get_current_time,
    },
    "celsius_to_fahrenheit": {
        "schema": celsius_schema,
        "fn": celsius_to_fahrenheit,
    },
}


def run_agent_v2(query: str) -> str:
    """ReAct loop with the extended tool set (Q3)."""
    return run_agent(query, tools=TOOLS_V2)


print("\n=== Q3: Extended agent ===")
response_a = run_agent_v2("What is 37 degrees Celsius in Fahrenheit?")
print("Response A:", response_a)
# Terminal output:
# Response A: 37°C is 98.6°F.
#
# Analysis:
# A tool WAS called. The query explicitly asks for a Celsius -> Fahrenheit conversion,
# which perfectly matches the celsius_to_fahrenheit tool. The agent dispatched the tool,
# received "37.0°C is 98.6°F", and returned that as the final answer.

response_b = run_agent_v2("What is the boiling point of water in plain English?")
print("Response B:", response_b)
# Terminal output:
# Response B: The boiling point of water is the temperature at which water changes
# from a liquid to a gas (steam). At sea level, this temperature is 100 degrees
# Celsius (212 degrees Fahrenheit). When water reaches this temperature...
#
# Analysis:
# A tool was NOT called. Even though the agent now has a Celsius->Fahrenheit tool,
# the query asks a general knowledge question rather than asking to convert a specific
# number. The model answered from pre-training (100°C / 212°F) and added context about
# altitude/pressure. This shows the model's judgment: it only reaches for a tool when
# the task clearly requires one, not whenever a tool *could* theoretically apply.


# ===========================================================================
# --- Lesson 03: Multi-Tool Agent ---
# ===========================================================================


# --- Q4 --------------------------------------------------------------------
class CsvManager:
    """Simple CSV workspace for the agent."""

    def __init__(self):
        self.df = None
        self.filename = None

    def load_csv(self, filename: str) -> str:
        """Load a CSV from the resources/ folder next to this script."""
        path = RESOURCES_DIR / filename
        if not path.exists():
            return {"error": f"File not found: {path}"}
        self.df = pd.read_csv(path)
        self.filename = filename
        return f"Loaded '{filename}' with {len(self.df)} rows and columns: {list(self.df.columns)}"

    def describe(self) -> str:
        if self.df is None:
            return {"error": "No CSV loaded. Call load_csv first."}
        return self.df.describe().to_string()

    def head(self, n: int = 5) -> str:
        if self.df is None:
            return {"error": "No CSV loaded."}
        return self.df.head(n).to_string()

    def query(self, expression: str) -> str:
        """Run a pandas query expression on the loaded DataFrame."""
        if self.df is None:
            return {"error": "No CSV loaded."}
        try:
            result = self.df.query(expression)
            return result.to_string()
        except Exception as e:
            return {"error": f"Query failed: {e}"}

    def compute_correlation(self, col1: str, col2: str):
        """
        Compute the Pearson correlation between two columns in the loaded DataFrame.
        Returns the correlation coefficient and p-value.
        """
        if self.df is None:
            return {"error": "No CSV loaded. Call load_csv first."}
        if col1 not in self.df.columns:
            return {"error": f"Column '{col1}' not found in the data."}
        if col2 not in self.df.columns:
            return {"error": f"Column '{col2}' not found in the data."}

        clean = self.df[[col1, col2]].dropna()
        if len(clean) < 3:
            return {"error": "Not enough non-missing data to compute correlation."}

        r, p = stats.pearsonr(clean[col1], clean[col2])
        return {
            "col1": col1,
            "col2": col2,
            "pearson_r": round(float(r), 4),
            "p_value": round(float(p), 4),
        }


csv_tools_schema = [
    {
        "name": "load_csv",
        "description": "Load a CSV file from the resources/ folder into the workspace.",
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "Name of the CSV file, e.g. 'bike_commute.csv'.",
                }
            },
            "required": ["filename"],
        },
    },
    {
        "name": "describe",
        "description": "Return summary statistics for the currently loaded CSV.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "head",
        "description": "Return the first n rows of the loaded CSV.",
        "parameters": {
            "type": "object",
            "properties": {
                "n": {"type": "integer", "description": "Number of rows (default 5)."}
            },
            "required": [],
        },
    },
    {
        "name": "query",
        "description": "Run a pandas-style query expression on the loaded CSV.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A pandas query string, e.g. 'distance_km > 10'.",
                }
            },
            "required": ["expression"],
        },
    },
    {
        "name": "compute_correlation",
        "description": (
            "Compute the Pearson correlation coefficient and p-value between two "
            "columns of the loaded CSV. Returns col1, col2, pearson_r, and p_value."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "col1": {"type": "string", "description": "First column name."},
                "col2": {"type": "string", "description": "Second column name."},
            },
            "required": ["col1", "col2"],
        },
    },
]

csv_manager = CsvManager()
node_tools = {
    "load_csv": csv_manager.load_csv,
    "describe": csv_manager.describe,
    "head": csv_manager.head,
    "query": csv_manager.query,
    "compute_correlation": csv_manager.compute_correlation,
}


SYSTEM_PROMPT = (
    "You are a data analyst assistant. You have tools to load a CSV, describe "
    "it, query it, and compute correlations. Always load the CSV first before "
    "running analyses. Think step-by-step and use the tools that fit the task."
)


def run_agent_cycle(messages, user_query):
    """
    ReAct loop that appends to an existing messages list (so we can inspect
    the full conversation afterward, as Q6 requires).
    """
    messages.append({"role": "user", "content": user_query})
    tool_schemas = csv_tools_schema

    while True:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=[{"type": "function", "function": s} for s in tool_schemas],
            tool_choice="auto",
        )
        msg = response.choices[0].message
        messages.append(msg.model_dump())

        if msg.tool_calls:
            for tc in msg.tool_calls:
                name = tc.function.name
                args = json.loads(tc.function.arguments)
                fn = node_tools.get(name)
                if fn is None:
                    result = {"error": f"Unknown tool: {name}"}
                else:
                    result = fn(**args)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result, default=str),
                    }
                )
        else:
            return msg.content


# --- Q5 --------------------------------------------------------------------
print("\n=== Q5: Correlation task (previously hit the tool-round limit) ===")
messages_q5 = [{"role": "system", "content": SYSTEM_PROMPT}]
result_q5 = run_agent_cycle(
    messages_q5,
    "Load bike_commute.csv and compute the correlation between "
    "avg_traffic_density and avg_speed_kmh.",
)
print("Agent final response:")
print(result_q5)
# Terminal output:
# Agent final response:
# The correlation between average traffic density and average speed (km/h) is as follows:
# - **Pearson correlation coefficient (r)**: -0.5321
# - **p-value**: 0.0
# This indicates a moderate negative correlation, suggesting that as traffic density
# increases, the average speed tends to decrease. The p-value indicates that this
# correlation is statistically significant.
#
# Analysis:
# With the pathlib fix, RESOURCES_DIR correctly resolves to assignments_07/resources/,
# so load_csv finds bike_commute.csv. The agent successfully:
# 1) called load_csv("bike_commute.csv") -> success
# 2) called compute_correlation("avg_traffic_density", "avg_speed_kmh")
# 3) returned a plain-English summary of the Pearson r (-0.5321) and p-value (0.0),
#    correctly interpreting the moderate negative correlation. The tool-round limit
#    issue is permanently fixed.


# --- Q6 --------------------------------------------------------------------
print("\n=== Q6: Full message trace ===")
print(json.dumps(messages_q5, indent=2, default=str))
# Terminal output:
# [
#   { "role": "system", "content": "You are a data analyst assistant..." },
#   { "role": "user", "content": "Load bike_commute.csv and compute..." },
#   { "role": "assistant", "tool_calls": [ { "function": { "name": "load_csv", "arguments": "{\"filename\":\"bike_commute.csv\"}" } } ] },
#   { "role": "tool", "tool_call_id": "...", "content": "\"Loaded 'bike_commute.csv' with 160 rows...\"" },
#   { "role": "assistant", "tool_calls": [ { "function": { "name": "compute_correlation", "arguments": "{\"col1\":\"avg_traffic_density\",\"col2\":\"avg_speed_kmh\"}" } } ] },
#   { "role": "tool", "tool_call_id": "...", "content": "{\"col1\": \"avg_traffic_density\", \"col2\": \"avg_speed_kmh\", \"pearson_r\": -0.5321, \"p_value\": 0.0}" },
#   { "role": "assistant", "content": "The correlation between average traffic density..." }
# ]
#
# Analysis of ReAct loop roles:
# - "system": Initial instructions defining the agent's persona and rules (e.g., "always load CSV first").
# - "user": The human's request that kicks off the task.
# - "assistant": The model's reply. It either provides a final text answer OR requests
#   to call one or more tools (indicated by the presence of the `tool_calls` array).
# - "tool": The result of executing a tool call, fed back to the model so it can
#   decide the next step or produce the final answer.


# ===========================================================================
# --- Lesson 04: smolagents ---
# ===========================================================================


# --- Q7 --------------------------------------------------------------------
@tool
def compute_correlation_smol(col1: str, col2: str) -> dict:
    """Compute the Pearson correlation between two columns of the currently loaded CSV.

    Args:
        col1 (str): Name of the first column.
        col2 (str): Name of the second column.

    Returns:
        dict: A dictionary with keys col1, col2, pearson_r, and p_value
            (or an 'error' key if the columns are missing or no CSV is loaded).
    """
    return csv_manager.compute_correlation(col1, col2)


print("\n=== Q7: smolagents auto-generated description ===")
print(compute_correlation_smol.description)
# Terminal output:
# === Q7: smolagents auto-generated description ===
# Compute the Pearson correlation between two columns of the currently loaded CSV.
#
# Analysis:
# Comparison to the manual JSON schema in Q4:
# - smolagents pulls the description strictly from the first line of the docstring
#   (the one-line summary), NOT the full docstring.
# - Parameter types and names are inferred directly from Python type hints
#   (col1: str, col2: str) — no separate "properties" dict to maintain manually.
# - The `Args:` section of the docstring becomes each parameter's "description"
#   field in the schema. If you forget to document even one parameter, smolagents
#   raises a DocstringParsingException.
# What smolagents needs from you (the developer) to produce a good schema:
# 1) A clear one-line summary at the top of the docstring.
# 2) Accurate type hints on every parameter.
# 3) A Google-style "Args:" section with a description for EACH parameter.
# 4) A "Returns:" section describing the return value.


# --- Q8 --------------------------------------------------------------------
@tool
def load_csv_smol(filename: str) -> str:
    """Load a CSV file from the resources/ folder into the shared workspace.

    Args:
        filename (str): Name of the CSV file, e.g. 'bike_commute.csv'.

    Returns:
        str: A message confirming the load (row count + column names), or
            an error string if the file was not found.
    """
    return csv_manager.load_csv(filename)


@tool
def describe_smol() -> str:
    """Return summary statistics for the currently loaded CSV.

    Args:

    Returns:
        str: A pandas describe() table as a string, or an error if no CSV is loaded.
    """
    return csv_manager.describe()


@tool
def head_smol(n: int = 5) -> str:
    """Return the first n rows of the loaded CSV as a string.

    Args:
        n (int): Number of rows to return. Defaults to 5.

    Returns:
        str: The first n rows of the DataFrame as a formatted string.
    """
    return csv_manager.head(n)


SMOL_TOOLS = [load_csv_smol, describe_smol, head_smol, compute_correlation_smol]

model = OpenAIServerModel(model_id=MODEL)

tool_agent = ToolCallingAgent(tools=SMOL_TOOLS, model=model)
code_agent = CodeAgent(tools=SMOL_TOOLS, model=model)

prompt_q8 = (
    "Load bike_commute.csv. Plot avg_heart_rate vs duration_min as a scatter "
    "plot with green dots."
)

print("\n=== Q8a: ToolCallingAgent response ===")
response_tool = tool_agent.run(prompt_q8)
print(response_tool)
# Terminal output:
# === Q8a: ToolCallingAgent response ===
# The scatter plot of avg_heart_rate vs duration_min will be created with green dots.
# This can be done using a plotting library, but it's not supported here. However,
# the correlation between avg_heart_rate and duration_min is -0.0341, indicating a
# very weak negative correlation.
#
# Analysis:
# The ToolCallingAgent inspected the data, realized it has no plotting tool available,
# and honestly reported: "This can be done using a plotting library, but it's not
# supported here." It then gracefully pivoted to providing the correlation as a fallback.
# No plot was generated, and the dot color was not changed, but it avoided hallucinating
# a fake success.

print("\n=== Q8b: CodeAgent response ===")
response_code = code_agent.run(prompt_q8, additional_args={"csv_manager": csv_manager})
print(response_code)
# Terminal output:
# === Q8b: CodeAgent response ===
# The average heart rate is approximately 116.98 bpm, and the average duration of
# the rides is approximately 42.57 minutes.
#
# Analysis:
# The CodeAgent attempted to write Python code to plot using matplotlib, but failed
# with an InterpreterError because 'matplotlib.pyplot' is not in the default
# `additional_authorized_imports`. It then tried to reference an undefined `dataframe`
# variable (failing again), and finally fell back to just reporting the summary
# statistics (average heart rate and duration) from the describe tool. No plot was
# generated.
#
# What this reveals:
# - ToolCallingAgent is constrained and safe. When faced with an impossible request,
#   it can gracefully admit the limitation and pivot to a related task, avoiding
#   hallucination.
# - CodeAgent is more flexible and will attempt to write the exact code needed, but
#   it requires careful sandbox configuration (e.g., adding 'matplotlib' to authorized
#   imports) and robust error handling, as it can get stuck in loops trying to fix
#   undefined variables or hitting import blocks.


# --- Q9 --------------------------------------------------------------------
# Conceptual reflection.
#
# 1) A task where ToolCallingAgent beats CodeAgent:
#    Example: "Book a meeting room for 3pm tomorrow and send a Slack notification
#    to the #ops channel." This task is a sequence of discrete, well-defined API
#    calls with strict side effects. A tool-based approach is a good fit because:
#      - The action space is small and enumerable (book_room, send_slack).
#      - Each action has clear inputs/outputs that fit a JSON schema.
#      - You want tight control over what the agent is allowed to do — you do NOT
#        want the agent writing arbitrary code that could, say, call the wrong
#        endpoint or leak credentials.
#    The property that makes it a good fit: a BOUNDED, WELL-SPECIFIED action space
#    where safety and auditability matter more than flexibility.
#
# 2) A meaningful risk of CodeAgent that doesn't apply to ToolCallingAgent:
#    CodeAgent generates and EXECUTES arbitrary Python code in your process. That
#    means a confused or adversarial prompt can cause it to:
#      - delete files, overwrite data, or exfiltrate secrets
#      - enter infinite loops or consume unbounded resources
#      - install unexpected packages or open network connections
#    ToolCallingAgent, by contrast, can only invoke the specific tools you registered,
#    with the specific arguments the schema allows. The risk surface is the union of
#    your tool implementations — not "anything Python can do."
