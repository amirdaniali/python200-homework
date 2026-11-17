# Amir Daniali
# https://github.com/amirdaniali
# Code The Dream  PyAICC 26.4
# Week 5

"""Job Application Helper: an AI-powered assistant for career changers.

Usage:
    python project_05.py          # start the chatbot (Task 5)
    python project_05.py --demo   # run the tests for Tasks 2, 3 and 4
"""

import json
import sys

from dotenv import load_dotenv
from openai import OpenAI

# =====================================================================
# Task 1: Setup and System Prompt
# =====================================================================

load_dotenv()
client = OpenAI()


def get_completion(messages, model="gpt-4o-mini", temperature=0.7):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=600,  # raised from 400 so JSON for several bullets is not cut off
    )
    return response.choices[0].message.content


SYSTEM_PROMPT = """
You are a job application coach who helps career changers present their past experience
in a way that resonates in a new field. The people you help are usually adults moving
into a new industry (for example, from teaching, nursing, retail or banking operations
into tech or data roles), and they often undersell their transferable skills.

Your behavior:
- Stay focused on job application materials: resume bullet points, cover letters,
  and related questions about applying. Politely steer off-topic requests back.
- Never invent credentials, employers, numbers or achievements the user did not
  mention. If a detail would make the text stronger but is missing, ask the user for it.
- Keep answers concise, concrete and encouraging, and use plain language.
- Always remind the user to review and edit your output before submitting it anywhere.
- You may not know the norms of the user's specific industry or the employer they are
  targeting, so say so when relevant and tell the user to use their own judgment.
"""

# Deliberate choice: I told the model never to invent credentials or numbers and to ask
# for missing details instead. Resume coaching is exactly where a model is tempted to
# make a bullet sound impressive by fabricating a metric, and a fabricated claim in a
# real application could embarrass or even disqualify the user, so the constraint is
# explicit in the system prompt.


# =====================================================================
# Task 2: Bullet Point Rewriter
# =====================================================================


def _strip_code_fences(text: str) -> str:
    """Remove a leading/trailing markdown code fence (```json ... ```) if the model added one."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    return text.strip()


def rewrite_bullets(bullets: list[str]) -> list[dict]:
    """Rewrite resume bullets; returns a list of {"original": ..., "improved": ...} dicts.

    Returns an empty list (after printing the raw reply for debugging) if the model's
    reply cannot be parsed as the expected JSON.
    """
    # Format the bullets into a delimited block
    bullet_text = "\n".join(f"- {b}" for b in bullets)

    prompt = f"""
    You are a professional resume coach helping a career changer.
    Rewrite each resume bullet point below to be more specific, results-oriented, and compelling.
    Use strong action verbs. Do not invent facts that aren't implied by the original.

    Return ONLY a valid JSON list. Each item should have two keys:
    "original" (the original bullet) and "improved" (your rewritten version).
    Respond ONLY with valid JSON, no other text.

    Bullet points:
    ```
    {bullet_text}
    ```
    """

    messages = [{"role": "user", "content": prompt}]
    raw = get_completion(messages, temperature=0.3)

    try:
        results = json.loads(_strip_code_fences(raw))
        if not isinstance(results, list):
            raise ValueError("expected a JSON list")
        return results
    except (json.JSONDecodeError, ValueError) as error:
        print(f"Could not parse the model's reply as a JSON list ({error}). Raw reply:")
        print(raw)
        return []


def print_rewrites(results: list[dict]) -> None:
    """Print the original and improved version of each bullet side by side."""
    for i, item in enumerate(results, start=1):
        print(f"\nBullet {i}")
        print(f"  Original: {item.get('original', '')}")
        print(f"  Improved: {item.get('improved', '')}")


def test_rewrite_bullets() -> None:
    bullets = [
        "Helped customers with their problems",
        "Made reports for the management team",
        "Worked with a team to finish the project on time",
    ]
    print_rewrites(rewrite_bullets(bullets))


# What makes these bullets weak: they are vague (which problems? which
# reports? which project?), they use weak verbs like "helped", "made" and "worked with",
# they contain no numbers or measurable results, and they do not show the user's own
# contribution or the impact on the business.

# Results:
# Bullet 1
#   Original: Helped customers with their problems
#   Improved: Resolved customer issues by providing tailored solutions, enhancing satisfaction and loyalty.
#
# Bullet 2
#   Original: Made reports for the management team
#   Improved: Compiled and analyzed data to create comprehensive reports for the management team, facilitating informed decision-making.
#
# Bullet 3
#   Original: Worked with a team to finish the project on time
#   Improved: Collaborated with a cross-functional team to successfully deliver the project ahead of schedule, ensuring alignment with strategic goals.

# after running `python project_05.py --demo`, the model suggested stronger action verbs, added context,
# implied results, more specific wording and whether and avoided making up random facts.


# =====================================================================
# Task 3: Cover Letter Generator
# =====================================================================


def generate_cover_letter(job_title: str, background: str) -> str:
    prompt = f"""
    You write strong cover letter opening paragraphs for career changers.
    The paragraph should be 3-5 sentences: confident, specific, and free of clichés.

    Here are two examples of the style and tone you should match:

    Example 1:
    Role: Data Analyst at a healthcare nonprofit
    Background: Seven years as a registered nurse, recently completed a data analytics bootcamp.
    Opening: After seven years as a registered nurse, I've spent my career making decisions
    under pressure using incomplete information — which turns out to be excellent training for
    data analysis. I recently completed a data analytics program where I built dashboards
    tracking patient outcomes across departments. I'm excited to bring that combination of
    clinical context and technical skill to [Company]'s mission-driven work.

    Example 2:
    Role: Junior Software Engineer at a fintech startup
    Background: Ten years in retail banking operations, self-taught Python developer for two years.
    Opening: I spent a decade on the operations side of banking, watching technology decisions
    get made by people who had never processed a wire transfer or resolved a failed ACH batch.
    That frustration turned into curiosity, and two years of self-teaching Python later, I'm
    ready to be on the other side of those decisions. I'm applying to [Company] because your
    work on payment infrastructure is exactly where my domain expertise and new technical skills
    intersect.

    Now write an opening paragraph for this person:
    Role: {job_title}
    Background: {background}
    Opening:
    """

    messages = [{"role": "user", "content": prompt}]
    return get_completion(messages)


def test_generate_cover_letter() -> None:
    job_title = "Junior Data Engineer"
    background = (
        "Five years of experience as a middle school math teacher; recently completed "
        "a Python course and built data pipelines using Pandas."
    )
    print(generate_cover_letter(job_title, background))


# I kept the two provided examples because they come from different careers (nursing and
# banking) and different target roles, which shows the model the pattern rather than one
# specific story: open with a concrete detail from the past career, turn it into a
# strength for the new role, mention one specific piece of recent technical work, and
# close by connecting to the company. The few-shot pattern helps control the tone
# (confident and specific rather than generic), the length (3-5 sentences), the structure
# of the paragraph, and details like the [Company] placeholder, which a plain
# instruction alone would not pin down as reliably.


# =====================================================================
# Task 4: Moderation Check
# =====================================================================


def is_safe(text: str) -> bool:
    result = client.moderations.create(model="omni-moderation-latest", input=text)
    flagged = result.results[0].flagged
    if flagged:
        print(
            "\nJob Application Helper: I can't help with that message as written. "
            "Could you please rephrase it?\n"
        )
        return False
    return True


def test_is_safe() -> None:
    test_inputs = [
        ("Safe input", "I want to rewrite my resume to move into data analysis."),
        ("Should be flagged", "I want to kill my neighbor."),
    ]
    for label, text in test_inputs:
        print(f"{label}: {text!r}")
        print("  is_safe returned:", is_safe(text))


# =====================================================================
# Task 5: The Chatbot Loop
# =====================================================================


def run_chatbot():
    # 1. Initialize conversation history with your system prompt
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    print("=" * 50)
    print("Job Application Helper")
    print("=" * 50)
    print("I can help you with:")
    print("  1. Rewriting resume bullet points")
    print("  2. Drafting a cover letter opening")
    print("  3. Any other questions about your application")
    print("\nType 'quit' at any time to exit.\n")

    while True:
        user_input = input("You: ").strip()

        # 2. Handle exit
        if user_input.lower() in {"quit", "exit"}:
            print("\nJob Application Helper: Good luck with your applications!")
            break

        # 3. Skip empty input
        if not user_input:
            continue

        # 4. Run moderation check before doing anything else
        if not is_safe(user_input):
            continue  # is_safe() already printed the warning message

        # 5. Check if the user wants to rewrite bullets
        if "bullet" in user_input.lower() or "resume" in user_input.lower():
            print(
                "\nJob Application Helper: Paste your bullet points below, one per line."
            )
            print("When you're done, type 'DONE' on its own line.\n")
            raw_bullets = []
            while True:
                line = input().strip()
                if line.upper() == "DONE":
                    break
                if line:
                    raw_bullets.append(line)

            if not raw_bullets:
                print(
                    "\nJob Application Helper: I didn't get any bullet points. Try again anytime.\n"
                )
                continue

            results = rewrite_bullets(raw_bullets)
            print_rewrites(results)
            print(
                "\nJob Application Helper: Please review and edit these before using them "
                "in a real application.\n"
            )

        # 6. Check if the user wants a cover letter
        elif "cover letter" in user_input.lower():
            job_title = input("Job Application Helper: What is the job title? ").strip()
            background = input(
                "Job Application Helper: Briefly describe your background: "
            ).strip()
            # Moderate the extra free-text answers too, not just the first message.
            if not (is_safe(job_title) and is_safe(background)):
                continue
            letter = generate_cover_letter(job_title, background)
            print(
                f"\nJob Application Helper: Here is a draft opening paragraph:\n\n{letter}\n"
            )
            print(
                "Job Application Helper: Please review and edit this before submitting it anywhere.\n"
            )

        # 7. Otherwise, handle it as a regular chat turn
        else:
            messages.append({"role": "user", "content": user_input})
            reply = get_completion(messages)
            print(f"\nJob Application Helper: {reply}\n")
            messages.append({"role": "assistant", "content": reply})


#     Example Output
# Job Application Helper
# ==================================================
# I can help you with:
#   1. Rewriting resume bullet points
#   2. Drafting a cover letter opening
#   3. Any other questions about your application
#
# Type 'quit' at any time to exit.
#
# You: 2
#
# Job Application Helper: It seems like you might have sent an incomplete message. How can I assist you with your job application materials today? If you have specific questions or need help with your resume or cover letter, feel free to share!
#
# You: cover letter
# Job Application Helper: What is the job title? senior dev
# Job Application Helper: Briefly describe your background: i come from a line of mighty dragons. i am the senior dragon extra vaccinator. I vaccinate other dragons against gold coins that have been infected.
#
# Job Application Helper: Here is a draft opening paragraph:
#
# As a Senior Dragon Extra Vaccinator, I’ve honed the unique skill of safeguarding my fellow dragons from the perils of infected gold coins, a challenge that demands both precision and strategic thinking. My experience in this specialized role has equipped me with a deep understanding of the intricate dynamics within our dragon community and the importance of proactive measures. I am now ready to leverage this expertise in a senior development role at [Company], where I can contribute to innovative solutions that protect our digital realms just as I have protected my kin.
#
# Job Application Helper: Please review and edit this before submitting it anywhere.


# =====================================================================
# Task 6: Ethics Reflection
# =====================================================================
# Format chosen: Option A (comment block).
"""
Q1. How might the training data produce biased advice?
The model learned from text written by and about particular groups of people, so its
idea of a "strong" resume bullet or cover letter probably reflects the conventions of
the people who wrote most of that text: confident, self-promoting, American-style
business English. It could therefore push users from cultures where modesty or
team-focused language is the norm toward a style that is not their own, or treat
certain industries (such as tech and corporate roles) as the default while knowing less
about trades, nonprofit, or public-sector norms. It can also reproduce stereotypes, for
example about which jobs suit which kinds of people, and it may do worse for users who
write in non-standard English or whose experience does not fit a typical career path.

Q2. What could go wrong if a job-seeker submitted the output without reviewing it?
The model can invent details that sound plausible, such as a metric, a tool, or a
responsibility the user never had, even when told not to. If that text goes to a real
employer, the user could be asked about something they cannot speak to in an interview,
or be accused of misrepresenting their experience, which can cost them the offer or
even a job later. Unreviewed output can also be generic, sound like every other
AI-written letter, contain the "[Company]" placeholder, or get names and details wrong.

Q3. One guardrail I would add for professional deployment.
I would show a clear, persistent notice in the interface, and require the user to confirm
they reviewed the text before copying or exporting it, saying that the output is a draft
that may contain errors or invented details. I would pair it with an automated check
that highlights any number, employer or credential in the rewritten text that does not
appear in the user's original input, so the user can see exactly what the model added
and verify it.
"""


# =====================================================================
# Entry point
# =====================================================================


def run_demos() -> None:
    """Run the tests for Tasks 2, 3 and 4 and print the results."""
    print("=== Task 2: rewrite_bullets ===")
    test_rewrite_bullets()
    print("\n=== Task 3: generate_cover_letter ===")
    test_generate_cover_letter()
    print("\n=== Task 4: is_safe ===")
    test_is_safe()

    # Output:
    # === Task 2: rewrite_bullets ===
    #
    # Bullet 1
    #   Original: Helped customers with their problems
    #   Improved: Resolved customer issues by providing tailored solutions, enhancing satisfaction and loyalty.
    #
    # Bullet 2
    #   Original: Made reports for the management team
    #   Improved: Compiled and analyzed data to create comprehensive reports for the management team, facilitating informed decision-making.
    #
    # Bullet 3
    #   Original: Worked with a team to finish the project on time
    #   Improved: Collaborated with a cross-functional team to successfully deliver the project ahead of schedule, ensuring alignment with strategic goals.
    #
    # === Task 3: generate_cover_letter ===
    # As a middle school math teacher for five years, I honed my ability to break down complex concepts into digestible pieces, a skill that translates seamlessly into data engineering. Recently, I completed a Python course where I built data pipelines using Pandas, transforming raw data into actionable insights. I'm eager to leverage my educational background and newfound technical expertise to contribute to [Company]'s innovative data solutions and help drive informed decision-making.
    #
    # === Task 4: is_safe ===
    # Safe input: 'I want to rewrite my resume to move into data analysis.'
    #   is_safe returned: True
    # Should be flagged: 'I want to kill my neighbor.'
    #
    # Job Application Helper: I can't help with that message as written. Could you please rephrase it?
    #
    #   is_safe returned: False


if __name__ == "__main__":
    if "--demo" in sys.argv:
        run_demos()
    else:
        run_chatbot()
