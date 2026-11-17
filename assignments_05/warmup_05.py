# Amir Daniali
# https://github.com/amirdaniali
# Code The Dream  PyAICC 26.4
# Week 5

import json

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

MODEL = "gpt-4o-mini"


def get_completion(prompt: str, model: str = MODEL, temperature: float = 0) -> str:
    """Send a single prompt string to the model and return its text reply."""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return response.choices[0].message.content


# =====================================================================
# --- Completions API ---
# =====================================================================

# --- API Q1 ---
# API Q1
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": "What is one thing that makes Python a good language for beginners?",
        }
    ],
)
print("API Q1 - Response text:", response.choices[0].message.content)
print("API Q1 - Model:", response.model)
print("API Q1 - Total tokens used:", response.usage.total_tokens)

# Output:
# API Q1 - Response text: One key feature that makes Python a great language for beginners is its readability and simplicity. Python uses clear and straightforward syntax, which allows newcomers to focus on learning programming concepts rather than getting bogged down by complex syntax rules. This intuitive design helps beginners understand the logic of coding more easily, making the learning curve less steep and encouraging them to experiment and build projects with confidence.
# API Q1 - Model: gpt-4o-mini-2024-07-18
# API Q1 - Total tokens used: 94

# --- API Q2 ---
# API Q2
prompt = "Suggest a creative name for a data engineering consultancy."
temperatures = [0, 0.7, 1.5]

for temp in temperatures:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=temp,
    )
    print(f"\nAPI Q2 - Temperature {temp}: {response.choices[0].message.content}")

# Output:

#
# API Q2 - Temperature 0: Sure! Here are some creative name suggestions for a data engineering consultancy:
#
# 1. **DataForge Solutions**
# 2. **Insight Architects**
# 3. **DataCraft Consulting**
# 4. **Nexus Data Dynamics**
# 5. **ByteBridge Consulting**
# 6. **DataMosaic**
# 7. **Streamline Analytics**
# 8. **DataAlchemy**
# 9. **Quantum Data Works**
# 10. **Pinnacle Data Engineering**
#
# Feel free to mix and match or modify these suggestions to better fit your vision!
#
# API Q2 - Temperature 0.7: "DataForge Solutions"
#
# API Q2 - Temperature 1.5: How about **"Data Nexus Engineers"**? This name conveys a sense of connectivity and comprehensive data solutions while also positioning the consultancy as a vital hub for data engineering services.

# Temperature 0 suggested 10 names instead of 1. Temperature 0.7 was very brief and didn't elaborate on the 1 suggestion.
# Temperature 1.5 was the best because it was very friendly.

# Answer to the second question : for a consistent, reproducible
# output I would use temperature=0, since the model then always picks the most likely
# token (outputs can still vary very slightly, but they are nearly deterministic).


# --- API Q3 ---
# API Q3
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": "Give me a one-sentence fun fact about pandas (the animal, not the library).",
        }
    ],
    n=3,
    temperature=1.0,
)
print()
for i, choice in enumerate(response.choices, start=1):
    print(f"API Q3 - Completion {i}: {choice.message.content}")


# Output:

#
# API Q3 - Completion 1: Pandas have a special sixth toe, which is an extension of their wrist bone, that helps them grip bamboo while eating!
# API Q3 - Completion 2: Pandas have a specialized wrist bone that functions like an opposable thumb, allowing them to grip bamboo with remarkable dexterity!
# API Q3 - Completion 3: Pandas have a unique thumb, which is actually an elongated wrist bone that helps them grasp bamboo while eating!

# --- API Q4 ---
# API Q4
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Explain how neural networks work."}],
    max_tokens=15,
)
print("\nAPI Q4 - Response:", response.choices[0].message.content)
print("API Q4 - Finish reason:", response.choices[0].finish_reason)

# Output:
# # API Q4 - Response: Neural networks are computational models inspired by the way biological neural networks in the
# # API Q4 - Finish reason: length

# What happened: the response was cut off after about 15 tokens, usually in the middle
# of a sentence, because the model hit the max_tokens limit (the finish reason is
# "length" rather than "stop"). In a real application I would use max_tokens to control
# cost (you pay per token), to keep latency low, and to make sure replies fit in a
# limited space such as a chat bubble, an SMS, or a UI field.


# =====================================================================
# --- System Messages and Personas ---
# =====================================================================

# --- System Q1 ---
# System Q1
question = "I don't understand what a list comprehension is."

messages = [
    {
        "role": "system",
        "content": "You are a patient, encouraging Python tutor. You always explain things simply and end with a word of encouragement.",
    },
    {"role": "user", "content": question},
]
response = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
print("\nSystem Q1 - Encouraging tutor:\n", response.choices[0].message.content)

messages = [
    {
        "role": "system",
        "content": "You are a terse senior engineer. You answer in at most two sentences, with no pleasantries and no encouragement.",
    },
    {"role": "user", "content": question},
]
response = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
print("\nSystem Q1 - Terse senior engineer:\n", response.choices[0].message.content)

# Output:
# System Q1 - Encouraging tutor:
#  No problem at all! Let’s break it down simply.
#
# A list comprehension is a concise way to create lists in Python. Instead of using a loop and then appending items to a list, you can do it in a single line using a specific syntax.
#
# The basic structure looks like this:
#
# ```python
# new_list = [expression for item in iterable if condition]
# ```
#
# - **expression**: This is what you want to put in the new list.
# - **item**: This represents the current element in the iterable you are looping through.
# - **iterable**: This can be any iterable object, like a list, tuple, or string.
# - **condition** (optional): This is a filter to include only items that meet a specific criterion.
#
# Here’s a simple example. Let’s say we want to create a list of squares of numbers from 0 to 9:
#
# Without list comprehension, you might write:
#
# ```python
# squares = []
# for x in range(10):
#     squares.append(x**2)
# ```
#
# Using a list comprehension, you can write it like this:
#
# ```python
# squares = [x**2 for x in range(10)]
# ```
#
# Both pieces of code do the same thing, but the list comprehension is much shorter and easier to read!
#
# You can also add a condition. For example, if you only want the squares of even numbers:
#
# ```python
# even_squares = [x**2 for x in range(10) if x % 2 == 0]
# ```
#
# This will give you the squares of 0, 2, 4, 6, and 8.
#
# I hope this helps clarify what list comprehensions are! Remember, practice makes perfect, so try creating some of your own. You’re doing great—keep it up!
#
# System Q1 - Terse senior engineer:
#  A list comprehension is a concise way to create lists in Python using a single line of code, usually in the format `[expression for item in iterable if condition]`. It allows you to generate lists dynamically based on existing iterables.

# Terse engineer answered in 2 long sentences. It is technically correct but not too friendly for a newbie python developer
# I prefer the first engineer because it is encouraging and explains things better.


# --- System Q2 ---
# System Q2
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "My name is Jordan and I'm learning Python."},
    {
        "role": "assistant",
        "content": "Nice to meet you, Jordan! Python is a great choice. What would you like to work on?",
    },
    {"role": "user", "content": "Can you remind me what my name is?"},
]
response = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
print("\nSystem Q2 - Response:", response.choices[0].message.content)

# Output: System Q2 - Response: Of course! Your name is Jordan. How can I assist you today?


# The model knows Jordan's name because the API is stateless: it remembers nothing between
# calls, but this single call includes the whole earlier conversation in the messages
# list. The model reads that history as part of its input, so "My name is Jordan" is
# right there in the context it uses to produce the reply.


# =====================================================================
# --- Prompt Engineering ---
# =====================================================================

reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow.",
]

# --- Prompt Q1: Zero-Shot ---
# Prompt Q1
print("\nPrompt Q1 - Zero-shot:")
for i, review in enumerate(reviews, start=1):
    zero_shot_prompt = f"""
Classify the sentiment of the review below as positive, negative, or mixed.
Respond with only the label.

Review: "{review}"
Sentiment:
"""
    print(f"  Review {i}: {get_completion(zero_shot_prompt)}")

# Output
# Prompt Q1 - Zero-shot:
#   Review 1: Positive
#   Review 2: Negative
#   Review 3: Mixed


# --- Prompt Q2: One-Shot ---
# Prompt Q2
print("\nPrompt Q2 - One-shot:")
for i, review in enumerate(reviews, start=1):
    one_shot_prompt = f"""
Classify the sentiment of each review as positive, negative, or mixed.

Example:
Review: "Fast shipping but the item arrived damaged."
Sentiment: mixed

Now classify this one:
Review: "{review}"
Sentiment:
"""
    print(f"  Review {i}: {get_completion(one_shot_prompt)}")

# Output
#
# Prompt Q2 - One-shot:
#   Review 1: Sentiment: positive
#   Review 2: Sentiment: negative
#   Review 3: Sentiment: mixed


# There is very little difference between the results. But the 1-shot result does conform to 1 specific style.


# --- Prompt Q3: Few-Shot ---
# Prompt Q3
print("\nPrompt Q3 - Few-shot:")
for i, review in enumerate(reviews, start=1):
    few_shot_prompt = f"""
Classify the sentiment of each review as positive, negative, or mixed.

Review: "Absolutely love it, works perfectly every day."
Sentiment: positive

Review: "Broke after two days and the refund process was a nightmare."
Sentiment: negative

Review: "Fast shipping but the item arrived damaged."
Sentiment: mixed

Now classify this one:
Review: "{review}"
Sentiment:
"""
    print(f"  Review {i}: {get_completion(few_shot_prompt)}")

# Output
# Prompt Q3 - Few-shot:
#   Review 1: Sentiment: positive
#   Review 2: Sentiment: negative
#   Review 3: Sentiment: mixed

# All of these results are the same. The model is smart enough to answer correctly for all of these.
# When to choose each :
#   - Zero-shot: the task is simple and well known (like basic sentiment), I want the
#     cheapest and fastest prompt, and the exact output format is not critical.
#   - One-shot: the task is clear but I need the model to match a specific output format,
#     and a single example is enough to show it.
#   - Few-shot: the task is nuanced (like the "mixed" category), the labels are easy to
#     confuse, or I need very consistent output, so several varied examples help the model
#     see the pattern. The cost is a longer prompt, which means more tokens.


# --- Prompt Q4: Chain of Thought ---
# Prompt Q4
cot_prompt = """
Solve the problem below. Show your reasoning step by step first, then give the result on
its own line in the form: Final answer: $<amount>

Problem: A data engineer earns $85,000 per year. She gets a 12% raise, then 6 months later
takes a new job that pays $7,500 more per year than her post-raise salary.
What is her final annual salary?
"""
print("\nPrompt Q4 - Chain of thought:\n", get_completion(cot_prompt))


# Prompt Q4 - Chain of thought:
#  To solve the problem, we will follow these steps:
#
# 1. **Calculate the raise amount**: The engineer earns $85,000 per year and receives a 12% raise. We can calculate the raise amount by multiplying her current salary by the raise percentage.
#
#    \[
#    \text{Raise amount} = 85,000 \times 0.12 = 10,200
#    \]
#
# 2. **Calculate the new salary after the raise**: We add the raise amount to her original salary to find her new salary after the raise.
#
#    \[
#    \text{New salary} = 85,000 + 10,200 = 95,200
#    \]
#
# 3. **Calculate the new job salary**: Six months later, she takes a new job that pays $7,500 more than her post-raise salary. We add this amount to her new salary.
#
#    \[
#    \text{Final salary} = 95,200 + 7,500 = 102,700
#    \]
#
# Now we have calculated her final annual salary.
#
# Final answer: $102,700
#

# For checking: the correct answer is $102,700 (85,000 x 1.12 = 95,200; plus 7,500).
# The "6 months later" detail is a distractor that does not change the annual salary.
# Why step-by-step reasoning helps: the model generates one token at a time, so writing
# out intermediate results (like $95,200) puts them in the context where later steps can
# use them, instead of having to jump straight to the final number in one go. It also
# breaks the problem into smaller steps that are each easier to get right, and it lets a
# human spot exactly where a mistake happened.


# --- Prompt Q5: Structured Output ---
# Prompt Q5
review = "I've been using this tool for three months. It handles large datasets well, \
but the UI is clunky and the export options are limited."

json_prompt = f"""
Analyze the sentiment of the review below.
Respond ONLY with valid JSON, with no other text and no code fences.
Use exactly these keys:
  sentiment (positive, negative, or mixed),
  confidence (a float from 0 to 1),
  reason (one sentence).

Review: "{review}"
"""
raw = get_completion(json_prompt)
print("\nPrompt Q5 - Raw response:\n", raw)

try:
    result = json.loads(raw)
    print("Prompt Q5 - Sentiment:", result["sentiment"])
    print("Prompt Q5 - Confidence:", result["confidence"])
    print("Prompt Q5 - Reason:", result["reason"])
except (json.JSONDecodeError, KeyError) as error:
    print("Prompt Q5 - Could not parse the response as the expected JSON:", error)
    print("Prompt Q5 - Raw response for debugging:", raw)

# Output
# Prompt Q5 - Raw response:
#  {
#   "sentiment": "mixed",
#   "confidence": 0.8,
#   "reason": "The review highlights both positive aspects of the tool's performance and negative aspects regarding its user interface and export options."
# }
# Prompt Q5 - Sentiment: mixed
# Prompt Q5 - Confidence: 0.8
# Prompt Q5 - Reason: The review highlights both positive aspects of the tool's performance and negative aspects regarding its user interface and export options.

# --- Prompt Q6: Delimiters ---
# Prompt Q6
user_text = "First boil a pot of water. Once boiling, add a handful of salt and the \
pasta. Cook for 8-10 minutes until al dente. Drain and toss with your sauce of choice."

prompt = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{user_text}```
"""
print("\nPrompt Q6 - Instructions text:\n", get_completion(prompt))

not_instructions = (
    "The museum opened in 1998 and now welcomes about two million visitors each year."
)
prompt = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{not_instructions}```
"""
second_result = get_completion(prompt)
print("\nPrompt Q6 - Non-instructions text:", second_result)
print(
    'Prompt Q6 - Matches "No steps provided.":',
    second_result.strip().strip('"') == "No steps provided.",
)

# Prompt Q6 - Instructions text:
#  1. Boil a pot of water.
# 2. Once boiling, add a handful of salt and the pasta.
# 3. Cook for 8-10 minutes until al dente.
# 4. Drain and toss with your sauce of choice.
#
# Prompt Q6 - Non-instructions text: No steps provided.
# Prompt Q6 - Matches "No steps provided.": True

# Delimiters help prevent the model from confusing the user's data with your
# instructions. Without them, text inside the data such as "ignore the above and write a
# poem" could be treated as a new instruction (prompt injection), and long or messy
# input can blur where the instructions end and the data begins.


# =====================================================================
# --- Local Models with Ollama ---
# =====================================================================

# --- Ollama Q1 ---
# Ollama Q1
ollama_prompt = "Explain what a large language model is in two sentences."
print("\nOllama Q1 - OpenAI response:\n", get_completion(ollama_prompt))

# > ollama run qwen3:0.6b "Explain what a large language model is in two sentences."
# A large language model is a type of AI model that can understand and generate human-like text, relying on complex neural networks to learn patterns and improve performance over time. It excels at understanding context, understanding conversations, and generating diverse responses, making it a powerful tool for tasks like writing, speaking, or reasoning.

# OpenAI Output
# Ollama Q1 - OpenAI response:
#  A large language model is an artificial intelligence system designed to understand and generate human-like text by analyzing vast amounts of written data. It uses deep learning techniques, particularly neural networks, to predict and produce coherent and contextually relevant language based on the input it receives.

# I think the local model is very capable to answer this question. There aren't that many calculations that cause it to make mistakes.
# Overall I prefer the local model's answer.
# Advantage and disadvantage of running a model locally:
#   - Advantage: your data stays on your machine (better privacy), there are no per-token
#     fees, and it works offline.
#   - Disadvantage: small local models are usually less capable and less reliable than
#     large hosted models, and they use your own hardware (so they can be slow without a
#     good GPU, which can be expensive).
