import openai
import sklearn
from nltk.translate.bleu_score import sentence_bleu
import random
import time
import inspect
from typing import Callable
import os
from openai.types.chat import ChatCompletion
import pandas as pd
import dotenv

dotenv.load_dotenv()


SYSTEM_PROMPT = "You are part of an API that accepts certain arguments and returns a value. The user " \
    "will send you the detailed instruction about the task description, the function signature, and the " \
    "arguments. You just implement this task in your mind without writing anything in extra, and return " \
    "the result to the user immediately."

INSTRUCTION = """Hello, I have a task for you. The description of the task is as follows:
{docstring}

The function signature is as follows:
{signature}

T
he arguments are as follows:
{arguments}

Please implement this task in your mind and return the result to me immediately without anything in extra."""


client = openai.OpenAI()

def ai_agent(fn: Callable):
    """
    A decorator that turns a function into an OPENAI API call.

    It will read the signature of the function as well as the docstring,
    assemble the prompt, and then call the OPENAI API with the prompt, and
    return the text response.
    """
    def wrapper(*args, **kwargs):
        # Get the signature of the function
        signature = inspect.signature(fn)
        # Get the docstring of the function
        docstring = inspect.getdoc(fn)
        # Evaluate the arguments on the signature object
        arguments = ", ".join(f"{k}={v}" for k, v in signature.bind(*args, **kwargs).arguments.items())

        # Assemble the prompt
        prompt = INSTRUCTION.format(docstring=docstring, signature=signature, arguments=arguments)
        # Call the OPENAI API, retry for 3 times with exponential backoff
        response: ChatCompletion | None = None
        for i in range(3):
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=800,
                    temperature=0.4,
                )
                break
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(2 ** i)

        if response is None:
            raise Exception("Failed to call OPENAI API")

        # Extract the response from the API
        return response.choices[0].message.content

    return wrapper


@ai_agent
def summarize(section: str):
    """I want to summarize a section in a math textbook. The context is cut off but please do not imagine any extra beyond that. Just summarize with your best effort from whatever context you are given. Please return the results in bullet points."""
    ...


@ai_agent
def make_a_plan(summary: str, theorem: str):
    """I want to make a plan to prove a theorem. Before the theorem is stated, there are some previous texts. I will give you a summary in the argument `summary`, and the statement of the theorem in the argument `theorem`. Please make a plan to prove the theorem. Please return the plan in bullet points in markdown format."""
    ...


@ai_agent
def critique_a_proof(section: str, proof: str):
    """I want to prove a theorem `theorem`, and I made a plan. Could you let me know what you think about my plan? Please critique my plan."""
    ...


@ai_agent
def revise_plan(theorem: str, plan: str, critique: str):
    """I want to prove a theorem `theorem`, and I made a plan. I thought about it for a while and here is my critique in the argument `critique`. Could you help me revise my plan? If everything is okay, you can simply repeat my original plan. But if there is something to improve, feel free to make changes."""
    ...

@ai_agent
def write_a_proof(section: str, plan: str, theorem: str):
    """I want to write a proof of a theorem. Before the theorem is stated, there are some previous texts. I will give you a summary in the argument `section`, the statement of the theorem in the argument `theorem`. I already made the plan in the argument `plan`. Please write a proof of the theorem. Please return the proof in markdown format."""
    ...


@ai_agent
def write_a_latex_proof(section: str, plan: str, theorem: str):
    """I want to write a proof of a theorem. Before the theorem is stated, there are some previous texts. I will give you a summary in the argument `section`, the statement of the theorem in the argument `theorem`. I already made the plan in the argument `plan`. Please write a proof of the theorem. Please return the proof in latex format."""
    ...


@ai_agent
def latex_proof(proof: str, theorem: str):
    """You are a full-professor in mathematics in MIT and very good at writing graduate-level textbooks in latex format. I have written a proof in markdown format. Could you help me convert it into LaTeX format? I include the proof in the argument `proof` and the statement of the theorem in the argument `theorem`."""
    ...


@ai_agent
def formalize_a_proof(proof: str, theorem: str):
    """You are a professional mathematician who is also an expert in writing formal proofs in LEAN 4 proof assistance. I have hand-written a proof in natural languages of a theorem in markdown format. Could you help me formalize it in LEAN 4 language? I include the proof in the argument `proof` and the statement of the theorem in the argument `theorem`."""
    ...


def main():
    # First, load the dataset
    df = pd.read_parquet("all_proofs_processed.parquet")

    sep = "=" * 50

    for i in range(1):
        idx = random.randint(0, len(df) - 1)
        summary = summarize(df.iloc[idx]["previous_section"])
        print(f"Summary: {summary}")
        plan = make_a_plan(summary, df.iloc[idx]["statement"])
        print(sep)
        print(f"Plan: {plan}")
        critique = critique_a_proof(df.iloc[idx]["previous_section"], plan)
        print(sep)
        print(f"Critique: {critique}")
        revised_plan = revise_plan(df.iloc[idx]["previous_section"], plan, critique)
        print(sep)
        print(f"Revised Plan: {revised_plan}")
        #proof = write_a_proof(df.iloc[idx]["previous_section"], revised_plan, df.iloc[idx]["statement"])
        #print(sep)
        #print(f"Proof: {proof}")
        #latex = latex_proof(proof, df.iloc[idx]["statement"])
        #print(sep)
        #print(f"LaTeX Proof: {latex}")

        proof = write_a_latex_proof(df.iloc[idx]["previous_section"], revised_plan, df.iloc[idx]["statement"])
        print(sep)
        print(f"Proof: {proof}")

        # Calculate BLEU score with the groundtruth proof
        # TODO: Potentially, we can do rejection sampling over different plans, and select the one with the highest BLEU score
        groundtruth = df.iloc[idx]["proof"]
        # TODO: We can also use ROUGE score. Or other better metrics?
        print("BLEU score:", sentence_bleu([groundtruth.split()], proof.split()))


        formal_proof = formalize_a_proof(proof, df.iloc[idx]["statement"])
        # TODO: Hook up with LEAN 4 engine to actually check the proof

        print(sep)
        print(f"Formal Proof: {formal_proof}")






if __name__ == "__main__":
    main()
