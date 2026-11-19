# Amir Daniali
# https://github.com/amirdaniali
# Code The Dream  PyAICC 26.4
# Week 6 Project: Groundwork Coffee Co. Q&A Assistant (RAG with LlamaIndex)

# =====================================================================
# Step 1: Setup
# =====================================================================

from pathlib import Path

from dotenv import load_dotenv
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")
# Result: API key loaded successfully.

# Paths are resolved relative to this file, so the script works from any directory.
# Copy the lesson's groundwork_docs folder into assignments_06/resources/ before running.
BASE_DIR = Path(__file__).parent
docs_dir = BASE_DIR / "resources" / "groundwork_docs"
assert docs_dir.exists(), (
    f"Document directory not found: {docs_dir}. "
    "Copy the lesson's groundwork_docs folder into assignments_06/resources/."
)


# =====================================================================
# Step 2: Load the Documents
# =====================================================================

docs = SimpleDirectoryReader(str(docs_dir)).load_data()

print(f"\nLoaded {len(docs)} document(s):")
for doc in docs:
    print(" -", doc.metadata.get("file_name", "unknown"))
# Result:
#   Loaded 5 document(s):
#    - faq.txt
#    - menu.txt
#    - our_story.txt
#    - seasonal_specials.txt
#    - wholesale_catering.txt
# Note: there is no separate hours or loyalty file, so those answers have to come from
# one of these five documents (most likely faq.txt).


# =====================================================================
# Step 3: Build the Index and Query Engine
# =====================================================================

Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
# We changed the LLM to gpt-4o-mini because LlamaIndex's default model (GPT-3.5) is
# deprecated.
Settings.llm = OpenAI(model="gpt-4o-mini")

index = VectorStoreIndex.from_documents(docs)
query_engine = index.as_query_engine(similarity_top_k=3)

print("\nIndex built successfully. Ready to answer questions.")
# Result: Index built successfully. Ready to answer questions.
# (The terminal also printed an INFO line "HTTP Request: POST https://api.openai.com/v1/embeddings"
# here and before every call; those log lines are left out of the results in this file.)


# =====================================================================
# Step 4: Query the Assistant
# =====================================================================


def show_result(question, response):
    """Print the question, the model's answer, and the top retrieved source node."""
    top_node = response.source_nodes[0]
    file_name = top_node.node.metadata.get("file_name", "unknown")
    snippet = top_node.node.get_content()[:200].replace("\n", " ")
    print(f"\nQuestion: {question}")
    print(f"Answer: {response}")
    print(f"Top source: {file_name} | score={top_node.score:.4f}")
    print(f"  {snippet}...")


questions = [
    "What are Groundwork's hours on weekends?",
    "Do you offer any dairy-free milk options?",
    "How does the loyalty program work?",
    "How did Groundwork Coffee get started?",
    "Do you offer catering or wholesale orders?",
]

for question in questions:
    show_result(question, query_engine.query(question))
# Result:
#   Question: What are Groundwork's hours on weekends?
#   Answer: Groundwork's hours on weekends are Saturday and Sunday from 8:00 AM to 5:00 PM.
#   Top source: our_story.txt | score=0.4775
#    Groundwork Coffee Co. was founded in 2018 by two college friends, Maya Torres and Sam Okafor, in Asheville, North Carolina. Maya had spent two years working on a coffee farm in Guatemala....
#
#   Question: Do you offer any dairy-free milk options?
#   Answer: Yes, there are dairy-free milk options available, including oat milk, almond milk, and soy milk, at no extra charge.
#   Top source: seasonal_specials.txt | score=0.4272
#    Freshly squeezed lemonade with lavender syrup and a splash of cold brew. Dair...
#
#   Question: How does the loyalty program work?
#   Answer: The loyalty program is free to join, and you earn one point for every dollar spent. Once you reach 100 points, you can redeem them for any free drink on the menu. You can sign up at the register or on the website.
#   Top source: faq.txt | score=0.2839
#    - Down...closed on Thanksgiving Day and Christmas Day.
#
#   Question: How did Groundwork Coffee get started?
#   Answer: Groundwork Coffee Co. was founded in 2018 by two college friends, Maya Torres and Sam Okafor, in Asheville, North Carolina. Maya had experience working on a coffee farm in Guatemala, while Sam managed a community center in his hometown. They aimed to combine good coffee with strong community values. Their first shop featured six tables and one espresso machine, focusing on sourcing fair-trade, sustainably grown beans directly from small farms. The name "Groundwork" reflects their belief that strong communities are built through local engagement and support.
#   Top source: our_story.txt | score=0.7313
#    Groundwork Coffee Co. was founded in 2018 by two college friends, Maya Torres and Sam Okafor, in Asheville, North Carolina. Maya had spent two years working on a coffee farm in Guatemala....
#
#   Question: Do you offer catering or wholesale orders?
#   Answer: Yes, catering services are available for events of 20 people or more, and wholesale orders can be placed for bulk purchases of coffee starting at 5 pounds.
#   Top source: wholesale_catering.txt | score=0.6291
#    We sell our house blends and single-origin beans in bulk to local restaurants, offices, and retailers. Wholesale pricing is available for orders of 5 pounds...

# Reflection on the five responses:
#   - Confident and accurate? All five answers sounded confident and specific, with no
#     hedging, and they were direct and complete (hours, milk types, point values, founders,
#     minimum order sizes). The two I can check against the printed top source are the
#     founding story (the snippet matches the answer: 2018, Maya Torres, Sam Okafor,
#     Asheville, the Guatemala coffee farm) and wholesale (the snippet confirms the
#     5-pound minimum). For the other three I could only see the top retrieved chunk, so I
#     compared and verified the answer's accuracy against the menu.txt and faq.txt resources.
#   - Was the top source the expected document? Only for the story (our_story.txt, 0.7313,
#     the highest score) and catering (wholesale_catering.txt, 0.6291). For the weekend
#     hours question the top source was our_story.txt (0.4775), the founding story, which
#     is not where hours would be listed. For the dairy-free question it was
#     seasonal_specials.txt (a lemonade chunk) rather than menu.txt or the FAQ.
#   - Surprises: (1) the weekend hours answer was specific and sounded right even though
#     the top-ranked chunk was about the founding of the company. The hours most likely came
#     from one of the other two retrieved chunks, which shows why retrieving k=3 helps,
#     and that I only printed the top node, so I cannot see where the answer came from.
#     (2) The loyalty question had the lowest top score of all (0.2839) and a top-source
#     snippet that does not mention loyalty at all, yet the answer was detailed (free to
#     join, one point per dollar, 100 points for a free drink, sign up at the register or
#     website). So a low similarity score does not by itself mean a bad answer, and a
#     high one does not guarantee a right one (see Step 5).


# =====================================================================
# Step 5: Find a Failure
# =====================================================================

hard_question = (
    "Does Groundwork have a drive-through, and how many calories are "
    "in a large oat milk latte?"
)
hard_response = query_engine.query(hard_question)

print("\n" + "=" * 60)
print(f"Failure test question: {hard_question}")
print(f"Answer: {hard_response}")
for i, node in enumerate(hard_response.source_nodes, start=1):
    file_name = node.node.metadata.get("file_name", "unknown")
    snippet = node.node.get_content()[:200].replace("\n", " ")
    print(f"  Source {i} | {file_name} | score={node.score:.4f}")
    print(f"    {snippet}...")
# Result:
#   ============================================================
#   Failure test question: Does Groundwork have a drive-through, and how many calories are in a large oat milk latte?
#   Answer: There is no information provided about a drive-through at Groundwork Coffee Co. Additionally, the calorie content for a large oat milk latte is not specified.
#     Source 1 | menu.txt | score=0.5718
#    - Pour-over (rotating s...50): $2.50 / $3.00
#     Source 2 | seasonal_specials.txt | score=0.4712
#    Freshly squeezed lemonade with lavender syrup and a splash of cold brew. Dair...
#     Source 3 | our_story.txt | score=0.4387
#    Groundwork Coffee Co. was founded in 2018 by two college friends, Maya Torres and Sam Okafor, in Asheville, North Carolina. Maya had spent two years working on a coffee farm in Guatemala....

# What I asked and why I expected it to be hard : I combined two
# questions, one about a drive-through and one about calories in a specific drink. I
# expected this to be hard because a coffee shop's documents about its menu, story, hours,
# loyalty program and catering are unlikely to contain either fact, and because the
# question has two separate parts that may need chunks from different documents.
# What went wrong: nothing, in the sense that the system did the right thing. The
# information is missing from the documents, and the model said so for both parts of the
# question ("no information provided about a drive-through" and the calorie content "is not
# specified") instead of guessing. The model, gpt-4o-mini, behaved the same way as in the
# Brightleaf warmup. Retrieval did find the closest topics: menu.txt (0.5718) for the
# latte, plus the seasonal specials chunk (0.4712) and the founding story (0.4387), which
# are only loosely related. Notice that the menu chunk's score (0.5718) was higher than the
# top score of three of the five Step 4 questions (0.4775, 0.4272 and 0.2839), even though
# it has no calorie information. The similarity score measures how close the topic is, not whether the
# answer is actually in the text, so a high score cannot tell me the question is answerable.
# Tone: it did not turn uncertain or hedging ("I'm not sure", "might"); it stated flatly
# and clearly that the information is not provided. That is the good outcome here, because
# the model did not invent a calorie number or a drive-through. But the tone alone would not
# have warned me if it had guessed, since the model sounds equally calm either way (the
# Step 4 answers read the same way). This suggests that I should not trust an answer
# because it sounds confident or even because it sounds careful: I should check it against
# the source chunks, and this good behavior may depend on the model and prompt settings.


# What I would change : make the refusal explicit with a custom prompt
# (answer only from the retrieved context and say "I don't know" otherwise) instead of
# relying on LlamaIndex's default prompt, split multi-part questions into separate searches,
# and show the source chunks with each answer so a person can verify it. A fixed similarity
# cutoff alone is unreliable: in the Brightleaf warmup, irrelevant chunks scored as high as
# 0.55 to 0.62, close to the scores of some relevant matches.


# =====================================================================
# Step 6: Reflection
# =====================================================================
"""
1. Lines of code and the value of a framework.
The core of my LlamaIndex pipeline is about 5 lines: setting the embedding model,
SimpleDirectoryReader(...).load_data(), VectorStoreIndex.from_documents(docs),
index.as_query_engine(similarity_top_k=3), and one line to set the LLM (needed because
LlamaIndex's default model is deprecated). Everything else in the file is path checks,
loops and printing. The keyword RAG lesson needed three hand-written functions (PDF text
extraction, retrieval with stopwords and scoring, and the prompt-building ask_llm call),
and that was still only a crude, document-level approach. Writing chunking, embedding
calls, a vector store, cosine-similarity search and prompt assembly myself would take
many more lines and plenty of room for bugs. A framework wraps those solved problems
behind a few well-tested calls, so I can spend my time on the parts that are specific
to my problem, such as which documents to use and how to evaluate the answers. The
deprecated default model is also a reminder that a framework's defaults can go stale and
are worth checking.

2. A different use case.
A hospital's or university's HR department could use the same approach as an internal
assistant over its policy handbook, benefits guides and onboarding documents. Employees
could ask "how many days of parental leave do I get?" and receive an answer with the
source passage, which saves HR staff from answering the same questions repeatedly. When
a policy changes, they only need to update the document and re-index, with no retraining.

3. A failure mode RAG cannot fully prevent.
RAG cannot fix problems in the documents themselves: if the source text is outdated,
wrong or contradicts another document, the model will faithfully repeat it. Even with
perfect retrieval, the model can also still misread or over-generalize the retrieved
text, or blend it with its own background knowledge in a confident-sounding answer, so
a person still needs to be able to check the sources. The evaluation in the warmup showed
another limit: an LLM judge can score an honest "the information is not provided" answer
as 0.0 faithfulness, so automated scores also need a human sanity check.
"""
