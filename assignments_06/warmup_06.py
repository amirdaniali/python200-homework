# Amir Daniali
# https://github.com/amirdaniali
# Code The Dream  PyAICC 26.4
# Week 6

from dotenv import load_dotenv
import os

if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")
# Result: API key loaded successfully.

import string
from pathlib import Path

from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

# Paths are resolved relative to this file, so the script works from any directory.
# Copy the lesson's brightleaf_pdfs folder into assignments_06/resources/ before running.
BASE_DIR = Path(__file__).parent
BRIGHTLEAF_DIR = BASE_DIR / "resources" / "brightleaf_pdfs"


# =====================================================================
# --- RAG Concepts ---
# =====================================================================

# --- Concepts Q1 ---
# Concepts Q1
"""
Scenario A (legal team, hundreds of policy PDFs updated every quarter): RAG.
The knowledge is large and changes every quarter, so retrieving the relevant passages
at question time keeps answers current without retraining; re-indexing the new PDFs is
far cheaper and faster than fine-tuning again.

Scenario B (dry, minimalist brand voice, 3,000 writing examples): fine-tuning.
The goal is a style, not facts, and a style is hard to describe in a prompt but easy for
a model to absorb from thousands of examples, which is exactly what fine-tuning does.

Scenario C (one two-page report, one-off use): prompt engineering.
The report is short enough to paste straight into the prompt, so there is no reason to
build a retrieval system or train anything for a single document.
"""

# --- Concepts Q2 ---
# Concepts Q2
"""
A confidently wrong answer is more harmful than "I am not sure" because the user has no
signal that anything is wrong. "I am not sure" tells the user to double-check, ask an
expert, or look for another source, while a confident answer invites them to act on it
right away. Tone matters because people use fluency and certainty as a shortcut for
reliability: an answer written in a calm, authoritative, well-structured way feels
trustworthy even when the content is invented, and LLMs produce that polished tone
whether they are right or wrong.
Example: a person asks a chatbot whether two medications are safe to take together and
the model confidently says yes, inventing a reassuring explanation. The person takes
both and has a dangerous interaction they would likely have avoided if the model had
said it was unsure and told them to ask a pharmacist.
"""

# --- Concepts Q3 ---
# Concepts Q3
"""
Correct order of the RAG pipeline steps:

1. Extract text from source documents
   - Read the raw text out of the PDFs, text files, or other sources.
2. Split text into chunks
   - Break the long text into smaller overlapping pieces so only relevant sections get retrieved.
3. Convert text chunks into embeddings
   - An embedding model turns each chunk into a vector of numbers that represents its meaning.
4. Receive the user's query
   - The user types the question the system must answer.
5. Embed the user's query
   - The same embedding model turns the question into a vector so it can be compared with the chunks.
6. Retrieve the most relevant chunks
   - The system finds the chunk vectors closest (by cosine similarity) to the query vector.
7. Inject retrieved chunks into the prompt
   - The chosen chunks are added to the prompt as context next to the user's question.
8. Generate a response from the LLM
   - The LLM writes an answer grounded in the supplied context.

(Steps 1 to 3 normally happen once ahead of time when the index is built; steps 4 to 8
happen every time a user asks a question.)
"""


# =====================================================================
# --- Keyword RAG ---
# =====================================================================


def simple_keyword_retrieval(query, documents, verbose=True):
    """Keyword retrieval using token overlap scoring."""
    stopwords = {
        "a",
        "an",
        "the",
        "and",
        "or",
        "in",
        "on",
        "of",
        "for",
        "to",
        "is",
        "are",
        "was",
        "were",
        "by",
        "with",
        "at",
        "from",
        "that",
        "this",
        "as",
        "be",
        "it",
        "its",
        "their",
        "they",
        "we",
        "you",
        "our",
    }
    translator = str.maketrans("", "", string.punctuation)

    query_words = {
        w.translate(translator) for w in query.lower().split() if w not in stopwords
    }
    if verbose:
        print(f"\nQuery tokens (filtered): {sorted(query_words)}")

    scores = []
    for name, content in documents.items():
        content_words = {
            w.translate(translator)
            for w in content.lower().split()
            if w not in stopwords
        }
        overlap = query_words & content_words
        score = len(overlap)
        scores.append((score, name, content))
        if verbose:
            print(f"[{name}] overlap={score} -> {sorted(overlap)}")

    scores.sort(reverse=True)
    best = next(((name, content) for score, name, content in scores if score > 0), None)
    if best:
        if verbose:
            print(f"\nSelected best match: {best[0]}")
        return [best]
    else:
        if verbose:
            print("\nNo overlapping keywords found.")
        return [("None found", "No relevant content.")]


documents = {
    "menu.txt": "We serve espresso, lattes, cappuccinos, and cold brew. Pastries include croissants and muffins baked fresh daily. Oat milk and almond milk are available.",
    "hours.txt": "We are open Monday through Friday from 7am to 7pm. On weekends we open at 8am and close at 5pm. We are closed on Thanksgiving and Christmas Day.",
    "hiring.txt": "We are currently hiring baristas and shift supervisors. Send your resume to jobs@groundworkcoffee.com.",
    "loyalty.txt": "Join our loyalty program to earn one point per dollar spent. Redeem 100 points for a free drink of your choice.",
}

# --- Keyword Q1 ---
# Keyword Q1
query = "What are your hours on weekends?"
results = simple_keyword_retrieval(query, documents, verbose=True)
print("Keyword Q1 - Selected document:", results[0][0])
# Result:
#   Query tokens (filtered): ['hours', 'weekends', 'what', 'your']
#   [menu.txt] overlap=0 -> []
#   [hours.txt] overlap=1 -> ['weekends']
#   [hiring.txt] overlap=1 -> ['your']
#   [loyalty.txt] overlap=1 -> ['your']
#
#   Selected best match: loyalty.txt
#   Keyword Q1 - Selected document: loyalty.txt

# Selected document: loyalty.txt, which is the WRONG one (the answer is in hours.txt).
# Why: after removing stopwords the query tokens are "hours", "weekends", "what" and
# "your". The word "hours" never appears inside hours.txt (only in its file name), so
# hours.txt matched only "weekends" (score 1). hiring.txt and loyalty.txt each matched
# only the filler word "your" (score 1 each), giving a three-way tie at 1. The function
# sorts the (score, name, content) tuples in reverse, so ties are broken by file name in
# reverse alphabetical order, and "loyalty.txt" comes first. The keyword method cannot
# tell that "weekends" is far more meaningful than "your", so a tie let an accident of
# alphabetical order pick the wrong document.


# --- Keyword Q2 ---
# Keyword Q2
query = "Do you have anything without caffeine?"
results = simple_keyword_retrieval(query, documents, verbose=True)
print("Keyword Q2 - Selected document:", results[0][0])
# Result:
#   Query tokens (filtered): ['anything', 'caffeine', 'do', 'have', 'without']
#   [menu.txt] overlap=0 -> []
#   [hours.txt] overlap=0 -> []
#   [hiring.txt] overlap=0 -> []
#   [loyalty.txt] overlap=0 -> []
#
#   No overlapping keywords found.
#   Keyword Q2 - Selected document: None found

# Selected document: none ("None found"), because no query word ("do", "have",
# "anything", "without", "caffeine") appears in any of the four documents, so every
# overlap score was 0.
# Did keyword RAG get it right? No. menu.txt is the document a person would check for
# drink options, but it never uses the word "caffeine" (it lists espresso, lattes, cold
# brew, oat milk and almond milk), so exact word matching cannot connect the question to
# it. Notice also that menu.txt does not actually say whether a caffeine-free drink exists.
# What would do better: semantic retrieval with embeddings, which compares meaning rather
# than exact words and would likely rank menu.txt (drinks and milk options) as most
# related to a question about caffeine-free drinks. Even then, the LLM should say the
# documents do not clearly answer the question rather than guess.


# --- Keyword Q3 ---
# Keyword Q3
# I expect the system to suggest the loyalty.txt for rewards because it makes the most sense.
# elected document will be ___ because ___."
query = "How do I sign up for rewards?"
results = simple_keyword_retrieval(query, documents, verbose=True)
print("Keyword Q3 - Selected document:", results[0][0])
# Result:
#   Query tokens (filtered): ['do', 'how', 'i', 'rewards', 'sign', 'up']
#   [menu.txt] overlap=0 -> []
#   [hours.txt] overlap=0 -> []
#   [hiring.txt] overlap=0 -> []
#   [loyalty.txt] overlap=0 -> []
#
#   No overlapping keywords found.
#   Keyword Q3 - Selected document: None found

# What actually happened (confirmed by the run above): "None found". The query tokens
# are "do", "how", "i", "rewards", "sign" and "up", and none of them appears in any
# document, so all scores were 0. A person would expect loyalty.txt, but it talks about a
# "loyalty program" and "points", never "rewards", so the exact-word match fails even
# though the meaning is the same. This is the synonym problem: keyword retrieval cannot
# treat "rewards" and "loyalty program" as related.
# My predictaion was not correct and that is surprising. Turns out our AI is not very smart yet.


# =====================================================================
# --- Semantic RAG Concepts ---
# =====================================================================

# --- Semantic Q1 ---
# Semantic Q1
"""
1. A vector embedding is a long list of numbers that an embedding model produces to
   represent the meaning of a piece of text, so that texts with similar meaning end up
   with similar lists of numbers (they sit close together as points in a space).

2. The chunk with the 0.85 score is more relevant. Cosine similarity measures how close
   the directions of two vectors are, so 0.85 means the chunk and the query are very
   similar in meaning, while 0.30 means they are only weakly related.

3. The embedding model captures meaning, not spelling. A query about "fuel efficiency"
   and a chunk about "miles per gallon" share no words, but both texts are about the same
   idea, so their embeddings point in nearly the same direction and the chunk scores high.
"""

# --- Semantic Q2 ---
# Semantic Q2
"""
| Feature                    | Keyword RAG                       | Semantic RAG                                          |
|----------------------------|-----------------------------------|-------------------------------------------------------|
| What is compared?          | Exact word overlap                | Meaning, by comparing the embedding of the query with the embeddings of the chunks |
| What is retrieved?         | Full document                     | The top-k most similar chunks (small pieces of documents) |
| Can it handle synonyms?    | No                                | Yes (similar meanings give similar embeddings)        |
| Storage format             | Plain text dictionary             | A vector store / index of embeddings (plus the chunk text) |
| Relevance score            | Number of overlapping keywords    | Cosine similarity between query and chunk embeddings  |
"""


# =====================================================================
# --- LlamaIndex ---
# =====================================================================

assert BRIGHTLEAF_DIR.exists(), (
    f"Document directory not found: {BRIGHTLEAF_DIR}. "
    "Copy the lesson's brightleaf_pdfs folder into assignments_06/resources/."
)

Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
# We changed the LLM to gpt-4o-mini because LlamaIndex's default model (GPT-3.5) is
# deprecated.
Settings.llm = OpenAI(model="gpt-4o-mini")


docs = SimpleDirectoryReader(str(BRIGHTLEAF_DIR)).load_data()
index = VectorStoreIndex.from_documents(docs)
query_engine = index.as_query_engine(similarity_top_k=3)
# Note: while loading, the terminal also printed six WARNING lines of the form
# "incorrect startxref pointer(1)" and "parsing for Object Streams" (one pair per PDF).
# They come from pypdf reading the PDFs and did not stop anything from working. LlamaIndex
# also prints "HTTP Request: POST https://api.openai.com/..." INFO lines for every API
# call; those log lines are left out of the results below.


def show_response(question, response, preview_chars=150):
    """Print a question, the model's answer, and each retrieved source node."""
    print(f"\nQuestion: {question}")
    print(f"Answer: {response}")
    for i, node in enumerate(response.source_nodes, start=1):
        text = node.node.get_content()[:preview_chars].replace("\n", " ")
        print(f"  Source {i} | score={node.score:.4f} | {text}...")


# --- LlamaIndex Q1 ---
# LlamaIndex Q1
questions = [
    "What employee benefits does BrightLeaf offer?",
    "What are BrightLeaf's security policies?",
]
for question in questions:
    show_response(question, query_engine.query(question), preview_chars=150)
# Result:
#   Question: What employee benefits does BrightLeaf offer?
#   Answer: BrightLeaf offers a comprehensive benefits program that includes health, vision, and wellness support, financial security and retirement benefits, parental leave, work flexibility, and opportunities for learning and career growth.
#
#   Health benefits include comprehensive medical insurance through Blue Cross, vision benefits with annual exams and allowances for lenses and frames, and access to wellness programs such as fitness memberships and nutrition counseling. Employees can also utilize a Wellness Reimbursement Plan for approved health-related activities.
#
#   For financial security, BrightLeaf provides life insurance starting on the first day of employment, short- and long-term disability insurance at no cost, and a 401(k) retirement plan with a dollar-for-dollar company match up to five percent of base salary. Employees are encouraged to participate in financial-planning seminars.
#
#   Parental leave is available for all new parents, offering twelve weeks of paid leave, with additional unpaid leave options. The company promotes work flexibility through flexible scheduling and hybrid work arrangements.
#
#   To support professional development, each employee receives an annual stipend for certifications or coursework, and there is an internal mentorship network. BrightLeaf also emphasizes diversity, equity, and inclusion in its hiring and promotion practices, alongside providing free access to online courses through its Learning Hub.
#     Source 1 | score=0.7667 | Introduction BrightLeaf Solar views employee well-being as inseparable from long-term innovation. Our benefits program is designed to help each team m...
#     Source 2 | score=0.5882 | EcoVolt Energy (2022 Partnership) BrightLeaf's collaboration with EcoVolt Energy, established in 2022, focused on delivering microgrid solutions to ru...
#     Source 3 | score=0.5627 | Overview BrightLeaf Solar was founded on the belief that renewable energy should be a right, not a privilege. Our mission is to make solar power pract...
#
#   Question: What are BrightLeaf's security policies?
#   Answer: BrightLeaf's security policies include maintaining layered defenses for production and corporate networks, requiring multi-factor authentication and VPN access for critical systems, and rotating credentials every 90 days. Customer data is encrypted both in transit and at rest, with strict access controls enforced through perimeter firewalls and cloud security groups. Incident response follows a structured plan based on NIST guidelines, with a 24/7 on-call rotation for alert triage and regular tabletop exercises to improve response strategies.
#
#   Employee training is mandatory, covering topics such as phishing, password hygiene, and secure coding practices. Access governance is managed through role-based access control, with regular audits to ensure permissions align with job functions. Vendors must meet security standards and undergo assessments, while physical security measures are in place at facilities.
#
#   Compliance with ISO 27001 practices is emphasized, with quarterly risk reviews and tracking of security metrics to foster a culture of accountability and continuous improvement.
#     Source 1 | score=0.6694 | Network and Data Security BrightLeaf maintains layered defenses for all production and corporate networks. Access to critical systems requires multi■f...
#     Source 2 | score=0.6181 | Introduction BrightLeaf Solar views employee well-being as inseparable from long-term innovation. Our benefits program is designed to help each team m...
#     Source 3 | score=0.5831 | Overview BrightLeaf Solar was founded on the belief that renewable energy should be a right, not a privilege. Our mission is to make solar power pract...

# Query 1, "What employee benefits does BrightLeaf offer?":
#   - Relevant chunks? Only partly. Source 1 (score 0.7667) is the employee benefits
#     policy, which is exactly right. Sources 2 and 3 (0.5882 and 0.5627) are the EcoVolt
#     partnership chunk and the company mission overview, which have nothing to do with
#     benefits.
#   - Tone: confident and specific. The answer names concrete details (Blue Cross
#     medical insurance, life insurance from day one, a 401(k) with a dollar-for-dollar
#     match up to five percent, twelve weeks of paid parental leave) with no hedging such
#     as "I'm not sure" or "it may be".
#   - Unexpected: the two unrelated chunks. They appear because similarity_top_k=3 always
#     returns three chunks even when only one is a good match, and the big score drop
#     (0.77 to 0.59) shows the last two were weak matches. None of their content showed
#     up in the answer.
# Query 2, "What are BrightLeaf's security policies?":
#   - Relevant chunks? Source 1 (0.6694), "Network and Data Security", is right on
#     target. Source 2 (0.6181) is the employee benefits introduction and Source 3
#     (0.5831) is the mission overview, neither of which is about security.
#   - Tone: confident and detailed again. It lists multi-factor authentication, VPN
#     access, credentials rotated every 90 days, encryption in transit and at rest, NIST
#     based incident response and ISO 27001 practices, all stated as plain fact.
#   - Unexpected: the employee benefits chunk ranked second, with a score close to the
#     correct chunk (0.62 vs 0.67). Probably the shared "policy" and company language
#     makes the two documents look similar to the embedding model even though their topics
#     differ. Again the irrelevant chunks did not leak into the answer.


# --- LlamaIndex Q2 ---
# LlamaIndex Q2
question = "What employee benefits does BrightLeaf offer?"
for k in (1, 5):
    engine_k = index.as_query_engine(similarity_top_k=k)
    print(f"\n--- similarity_top_k={k} ---")
    show_response(question, engine_k.query(question), preview_chars=150)
# Result:
#   --- similarity_top_k=1 ---
#
#   Question: What employee benefits does BrightLeaf offer?
#   Answer: BrightLeaf offers a comprehensive benefits program that includes:
#
#   1. **Health, Vision, and Wellness**: Medical insurance through Blue Cross, covering preventive care, hospitalization, telemedicine, and mental health services. Vision benefits include annual exams and allowances for lenses and frames. Employees can participate in wellness programs and utilize a Wellness Reimbursement Plan for up to $600 annually for approved health activities.
#
#   2. **Financial Security and Retirement**: Life insurance coverage starting on the first day of employment, short- and long-term disability insurance at no cost, and a 401(k) retirement plan with a dollar-for-dollar company match up to five percent of base salary. Employees are encouraged to attend financial-planning seminars.
#
#   3. **Parental Leave and Work Flexibility**: Twelve weeks of paid leave for all new parents, with options for additional unpaid leave. Flexible scheduling and hybrid work options are available to help employees balance personal and professional responsibilities.
#
#   4. **Learning, Inclusion, and Career Growth**: An annual professional development stipend for certifications or coursework, an internal mentorship network, and a Diversity, Equity, and Inclusion Council to promote equitable opportunities. Access to curated online courses is provided through the Learning Hub.
#     Source 1 | score=0.7667 | Introduction BrightLeaf Solar views employee well-being as inseparable from long-term innovation. Our benefits program is designed to help each team m...
#
#   --- similarity_top_k=5 ---
#
#   Question: What employee benefits does BrightLeaf offer?
#   Answer: BrightLeaf offers a comprehensive benefits program that includes health, vision, and wellness support, financial security and retirement options, parental leave, work flexibility, and opportunities for learning and career growth.
#
#   Health benefits include comprehensive medical insurance through Blue Cross, vision benefits with annual exams and allowances for lenses and frames, and access to wellness programs such as fitness memberships and nutrition counseling. Employees can also utilize a Wellness Reimbursement Plan for approved health-related activities.
#
#   For financial security, BrightLeaf provides life insurance starting on the first day of employment, short- and long-term disability insurance at no cost, and a 401(k) retirement plan with a dollar-for-dollar company match up to five percent of base salary. Employees are encouraged to participate in financial-planning seminars.
#
#   Parental leave is available for all new parents, offering twelve weeks of paid leave, with options for additional unpaid leave. The company promotes work flexibility through flexible scheduling and hybrid work options.
#
#   To support professional development, each employee receives an annual stipend for certifications or coursework, and there is an internal mentorship network. BrightLeaf also emphasizes diversity, equity, and inclusion in its hiring and promotion practices, along with access to free online courses through its Learning Hub.
#     Source 1 | score=0.7666 | Introduction BrightLeaf Solar views employee well-being as inseparable from long-term innovation. Our benefits program is designed to help each team m...
#     Source 2 | score=0.5882 | EcoVolt Energy (2022 Partnership) BrightLeaf's collaboration with EcoVolt Energy, established in 2022, focused on delivering microgrid solutions to ru...
#     Source 3 | score=0.5627 | Overview BrightLeaf Solar was founded on the belief that renewable energy should be a right, not a privilege. Our mission is to make solar power pract...
#     Source 4 | score=0.4799 | Overview This report summarizes BrightLeaf Solar's financial performance from 2021 through 2025. The period includes a growth phase, a temporary dip i...
#     Source 5 | score=0.4241 | HelioPanel X5 The HelioPanel X5 is BrightLeaf’s workhorse module and the backbone of many residential and community deployments. It uses high■efficien...

# How the response changed: the two answers cover the same four topics (health, financial
# security, parental leave and flexibility, learning and growth). With top_k=1 the answer
# was a numbered list with bold headings and extra specifics (the $600 annual Wellness
# Reimbursement cap, preventive care / hospitalization / telemedicine / mental health, and a
# Diversity, Equity, and Inclusion Council). With top_k=5 it was plain paragraphs with
# slightly less detail (no $600 figure, no DEI Council). The four extra chunks (EcoVolt
# partnership, mission overview, financial report, HelioPanel X5, with scores from 0.5882
# down to 0.4241) are not about benefits, and none of their content appears in the answer,
# so they added nothing useful. The scores confirm it: only the first chunk scored high.
# Caution: the format and detail differences may not be caused by k alone. LLM answers are
# not fully repeatable from run to run, and the k=3 answer in LlamaIndex Q4 included the
# $600 detail while the k=3 answer in Q1 did not, so the same k gave different levels of
# detail.
# Is more retrieved context always better? no. A larger k can add
# the missing piece of an answer, but it also adds less relevant chunks that cost more
# tokens and can distract the model or dilute the answer, while too small a k can miss
# part of the answer. The right k depends on how spread out the needed information is.
# Here all the needed information was in one chunk, so k=1 was enough.


# --- LlamaIndex Q3 ---
# LlamaIndex Q3
hard_question = "Who is BrightLeaf Solar's CEO, and what is their direct phone number?"
hard_response = query_engine.query(hard_question)
show_response(hard_question, hard_response, preview_chars=300)
# Result:
#   Question: Who is BrightLeaf Solar's CEO, and what is their direct phone number?
#   Answer: The information regarding BrightLeaf Solar's CEO and their direct phone number is not provided.
#     Source 1 | score=0.6089 | Overview BrightLeaf Solar was founded on the belief that renewable energy should be a right, not a privilege. Our mission is to make solar power practical, affordable, and accessible to communities that have historically been left behind in the transition to clean energy. We are not only engineers a...
#     Source 2 | score=0.5792 | EcoVolt Energy (2022 Partnership) BrightLeaf's collaboration with EcoVolt Energy, established in 2022, focused on delivering microgrid solutions to rural communities in Georgia and South Carolina. The initiative combined BrightLeaf's solar generation systems with EcoVolt's battery storage expertise....
#     Source 3 | score=0.5540 | Introduction BrightLeaf Solar views employee well-being as inseparable from long-term innovation. Our benefits program is designed to help each team member thrive both professionally and personally. We aim to attract and retain diverse talent through flexibility, transparency, and care. This policy ...

# What I expected : I expected this to be hard because it is probably
# not in the documents at all, so the best possible behavior is to say the information
# is not available, and a weaker system would retrieve loosely related chunks and guess.
# What actually happened: the system did the right thing. The model said the CEO and phone
# number are "not provided" and did not invent a name or number. The retrieved chunks
# (mission overview 0.6089, EcoVolt partnership 0.5792, benefits introduction 0.5540) are
# all general company text with no leadership or contact information, so retrieval had
# nothing useful to find. The model did not change its tone to sound unsure; it simply
# gave a short, flat statement that the information is missing.
# Why it worked: LlamaIndex's default question-answering prompt asks the model to answer
# from the retrieved context rather than from its prior knowledge, which seems to have
# kept gpt-4o-mini from guessing.
# What I would change: make the refusal explicit in a custom prompt (answer only from the
# provided context and say "I don't know" otherwise) rather than relying on the default.
# A similarity cutoff would help less than I first thought: the irrelevant chunks for this
# question scored 0.61, 0.58 and 0.55, which is as high as the irrelevant chunks in the
# successful queries above (0.5882, 0.5627, 0.6181), so one fixed threshold could not
# cleanly separate "no answer" from "answer found". I would also show the source chunks
# next to each answer so users can verify claims.


# --- LlamaIndex Q4 ---
# LlamaIndex Q4
judge_llm = OpenAI(model="gpt-4o-mini", temperature=0.2)
faithfulness_evaluator = FaithfulnessEvaluator(llm=judge_llm)
relevancy_evaluator = RelevancyEvaluator(llm=judge_llm)


def evaluate_query(q):
    """Run the query through the engine, then print its faithfulness and relevancy scores."""
    response = query_engine.query(q)
    faithfulness = faithfulness_evaluator.evaluate_response(query=q, response=response)
    relevancy = relevancy_evaluator.evaluate_response(query=q, response=response)
    print(f"\nQuery: {q}")
    print(f"Answer: {response}")
    print(f"Faithfulness score: {faithfulness.score}")
    print(f"Relevancy score: {relevancy.score}")


evaluate_query("What employee benefits does BrightLeaf offer?")
# Result:
#   Query: What employee benefits does BrightLeaf offer?
#   Answer: BrightLeaf offers a comprehensive benefits program that includes health, vision, and wellness support, financial security and retirement options, parental leave, work flexibility, and opportunities for learning and career growth.
#
#   For health, employees have access to medical insurance through Blue Cross, which covers preventive care, hospitalization, telemedicine, and mental health services. Vision benefits include annual exams and allowances for lenses and frames. Wellness programs, fitness memberships, mindfulness sessions, and nutrition counseling are also available, along with a Wellness Reimbursement Plan for up to $600 annually for approved health-related activities.
#
#   In terms of financial security, BrightLeaf provides life insurance from the first day of employment, short- and long-term disability insurance at no cost, and a 401(k) retirement plan with a dollar-for-dollar company match up to five percent of base salary. Employees can attend financial-planning seminars to enhance their financial literacy.
#
#   Parental leave is offered to all new parents, providing twelve weeks of paid leave, with options for additional unpaid leave. The company promotes work flexibility through flexible scheduling and hybrid work options to help employees balance their personal and professional lives.
#
#   For career growth, each employee receives an annual professional development stipend for certifications or coursework, and there is an internal mentorship network. BrightLeaf also emphasizes diversity, equity, and inclusion in its hiring and promotion practices, and provides access to free online courses through its Learning Hub.
#   Faithfulness score: 1.0
#   Relevancy score: 1.0

evaluate_query(
    "Who won the 2022 FIFA World Cup?"
)  # clearly not in the Brightleaf documents
# Result:
#   Query: Who won the 2022 FIFA World Cup?
#   Answer: The information provided does not include details about the 2022 FIFA World Cup or its winner.
#   Faithfulness score: 0.0
#   Relevancy score: 0.0

"""
Did the scores change between the two queries? Yes, from 1.0 / 1.0 for the benefits
question to 0.0 / 0.0 for the World Cup question.
Why (my interpretation): the relevancy score of 0.0 is expected, because the retrieved
chunks come from a solar company's documents and have nothing to do with soccer, so the
judge decided the context does not match the question. The faithfulness score of 0.0 is
more surprising: the model did not hallucinate. It honestly said the information is not
included. The judge still answered NO, apparently because an answer like "the information
does not include..." is not a statement the retrieved context supports. So a faithfulness
of 0.0 does not always mean the model made something up; I would need to read the answer
(or print the evaluator's feedback attribute) to know why the judge failed it. The benefits
question scored 1.0 on both because the answer's details came from a retrieved benefits chunk
and the chunk matched the question.

Answers that do not depend on the output :

- A faithfulness score of 1.0 means the judge LLM decided the response is fully
  supported by the retrieved context, with no hallucinated claims. A score of 0.0 means
  the judge decided the response is not supported by the retrieved chunks, which is
  usually a sign of hallucination (but see the World Cup result above for an exception).

- The relevancy score measures whether the response, together with the retrieved
  context, actually addresses the user's query (is it on topic and useful for what was
  asked). Faithfulness asks "is the answer grounded in the sources?", while relevancy
  asks "does the answer and its context match the question?". An answer can be faithful
  but irrelevant, or relevant but unfaithful.

- LLM-as-a-judge means using a separate LLM (here gpt-4o-mini) with a fixed prompt to
  grade a response, answering YES or NO, which becomes a score of 1.0 or 0.0. It is used
  because RAG answers are free-form natural language, so there is no single correct
  string or number to compare against with a simple accuracy metric, and judging many
  answers with an LLM is faster and cheaper than having human experts read each one.
"""
