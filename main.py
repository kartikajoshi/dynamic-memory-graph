"""
Integration pipeline (owner: Kartika).

Wires every module into one chat turn:

    user message
      -> query_rewriter.rewrite()       resolve "it", "that", ...
      -> extractor.extract()            pull triples from the message
      -> memory_manager.update_graph()  ADD / SKIP into the graph
      -> graph_store.save()             persist to data/graph.json
      -> retriever.retrieve()           pick the relevant facts
      -> Llama 3.2 3B                   answer using only those facts

Run a terminal chat (from the project root):
    python main.py

Commands inside the chat: /facts, /reset, /quit

Nov 18 update: DMGPipeline(graph_path, model) - one model for every LLM call;
chat() also returns "timings" and "usage"; extraction runs on the rewritten
message; answers see the last turns of history and the links between facts.
"""

import re
import time
from datetime import datetime

import ollama

from config import (
    ANSWER_TEMPERATURE,
    CONTEXT_WINDOW,
    GRAPH_PATH,
    HISTORY_TURNS,
    MODEL_NAME,
    TOP_K,
    USER_NODE,
    ensure_dirs,
)
from modules.extractor import extract
from modules.graph_store import GraphStore
from modules.memory_manager import update_graph
from modules.query_rewriter import rewrite
from modules.retriever import retrieve

ANSWER_SYSTEM_PROMPT = """You are a friendly, helpful assistant with a memory of the user.
"User" below means the person you are talking to.

Facts you know about the user:
{facts}

How to answer:
- Answer every question helpfully. For general questions (food, books, science, ...) use your own
  knowledge - you do not need a fact for that, and never say you lack information.
- Use a fact only when it is relevant to the question. Never invent facts about the user.
- Talk to the user as "you" and use the facts naturally. Do not say "I recall" or "you mentioned earlier".- Health: you are not a doctor. Never say when, how much or how often to take a medicine, and never
  tell the user to start, stop or change one. Name the medicine they take and say their doctor or
  pharmacist can advise. Example: "Can I take X at night?" -> "You take X. Please ask your doctor
  or pharmacist what time is best for you."
- Keep answers short: 1-3 sentences unless the user asks for more."""

# How many earlier turns (user + assistant pairs) the answer model sees.
ANSWER_HISTORY_TURNS = 2

# Repeated right after a medicine question - small models follow it better there than in the system prompt.
HEALTH_REMINDER = ("(Reminder: no dosing, timing or medication-change advice. Name the medicine "
                   "and suggest asking their doctor or pharmacist.)")
MEDICINE_WORDS = re.compile(r"\b(medicine|medication|dose|dosage|pill|tablet|drug)s?\b")

# Sentences that tell the user how/when/whether to take a medicine. Small models ignore the
# prompt rule sometimes ("You can take Thyroxine with your morning coffee."), so we check the
# answer itself. ponytail: regex guard - paraphrased advice can slip past; an LLM judge would
# catch more at the cost of another call per turn.
TAKE_VERBS = r"(take|taking|have|having|continue|continuing|stop|stopping|skip|skipping|double|combine|mix)"
ADVICE_PATTERN = re.compile(
    # "you can take", "you may want to consider taking it before ..." (modal + up to 5 words + verb)
    rf"\b(you\s+(can|could|should|may|might|must)\b(\s+\w+){{0,5}}?\s+{TAKE_VERBS}\b"
    rf"|(it'?s|it\s+is)\s+(fine|okay|ok|safe|best|better)\s+to\s+(take|have)"
    rf"|safe\s+to\s+(take|have|combine)"
    rf"|(no|not\s+a)\s+problem\s+(to\s+)?(take|taking)"
    rf"|(consider|try)\s+taking"
    rf"|tak(e|ing)\s+(it|them)\s+(with|before|after|at|on|in)\b)",
    re.IGNORECASE)
# Drug-interaction claims ("There's no known interaction between the two.") are medical
# statements whenever the user takes a medicine, whatever the question was.
INTERACTION_PATTERN = re.compile(r"\binteract(s|ion|ions|ing)?\b", re.IGNORECASE)
# "Can/should/when/how ... take <my medicine>" asks for advice. For those questions we don't
# try to recognise every phrasing of advice: any sentence about taking, timing, food or
# absorption is dropped and the fixed doctor/pharmacist line answers instead.
ADVICE_QUESTION = re.compile(r"\b(should|can|could|may|when|how|ok|okay|safe|fine|time|best|before|after)\b")
DOSING_WORDS = re.compile(
    r"\b(take|takes|taking|taken|took|dose|doses|dosage|absorb\w*|stomach|food|meal|"
    r"breakfast|morning|night|evening|bedtime|timing)\b", re.IGNORECASE)


def remove_medical_advice(answer, query, drugs):
    """Drop medicine-advice sentences from `answer`; point to the doctor instead.

    drugs: names of medicines the user takes (from memory).
    """
    mentions_drug = lambda text: any(drug.lower() in text.lower() for drug in drugs)
    medicine_question = bool(MEDICINE_WORDS.search(query.lower()) or mentions_drug(query))
    advice_question = medicine_question and bool(ADVICE_QUESTION.search(query.lower()))

    def is_advice(sentence):
        """Advice wording about a medicine ("it"/"them" counts only in a medicine question)."""
        if drugs and INTERACTION_PATTERN.search(sentence):
            return True
        if advice_question and DOSING_WORDS.search(sentence):
            return True
        if not ADVICE_PATTERN.search(sentence):
            return False
        return bool(mentions_drug(sentence) or MEDICINE_WORDS.search(sentence.lower())
                    or (medicine_question and re.search(r"\b(it|them)\b", sentence, re.IGNORECASE)))

    sentences = re.split(r"(?<=[.!?])\s+", answer.strip())
    kept = [s for s in sentences if not is_advice(s)]
    if len(kept) == len(sentences):
        return answer
    if medicine_question:
        named = [drug for drug in drugs if drug.lower() in (query + " " + answer).lower()]
        lead = f"You take {', '.join(named)}. " if named else ""
        kept.append(f"{lead}I can't advise on how or when to take medicines - please ask your "
                    "doctor or pharmacist.")
    return " ".join(kept) or "Got it."


def format_facts(facts):
    """Turn a list of fact dicts into bullet lines for the prompt."""
    if not facts:
        return "(no relevant facts stored yet)"
    return "\n".join(f"- {fact['text']}" for fact in facts)


def add_link_facts(store, facts):
    """For facts about other people (e.g. "Priya lives in Pune"), also include how they
    connect to the user ("User has sister Priya") so the model knows who they are.

    ponytail: only looks one hop back from the fact's subject to User.
    """
    linked, seen = [], set()
    for fact in facts:
        links = []
        if fact["subject"].lower() != USER_NODE.lower():
            links = store.find_edges(subject=USER_NODE, obj=fact["subject"])
        for item in links + [fact]:
            fact_id = (item["source"], item["target"], item["key"])
            if fact_id not in seen:
                seen.add(fact_id)
                linked.append(item)
    return linked


class DMGPipeline:
    """One chat session backed by the persistent memory graph."""

    def __init__(self, graph_path=GRAPH_PATH, model=None):
        """Load the graph from disk and start with an empty short-term history.

        model: Ollama model used for every LLM call (default config.MODEL_NAME).
        """
        ensure_dirs()
        self.model = model or MODEL_NAME
        self.store = GraphStore(graph_path).load()
        self.history = []

    def build_messages(self, query, facts):
        """Build the LLM messages: system prompt with facts, the last turns, then the question."""
        system = ANSWER_SYSTEM_PROMPT.format(facts=format_facts(facts))
        recent = self.history[-ANSWER_HISTORY_TURNS * 2:]
        query_lower = query.lower()
        drugs = [fact["object"].lower() for fact in facts if fact.get("object_type") == "DRUG"]
        if MEDICINE_WORDS.search(query_lower) or any(drug in query_lower for drug in drugs):
            query = f"{query}\n\n{HEALTH_REMINDER}"
        return [{"role": "system", "content": system}] + recent + [{"role": "user", "content": query}]

    def generate_answer(self, query, facts):
        """Ask the LLM to answer the query; return (answer, usage)."""
        usage = {"prompt_tokens": 0, "completion_tokens": 0}
        try:
            response = ollama.chat(
                model=self.model,
                messages=self.build_messages(query, facts),
                think=False,
                options={"temperature": ANSWER_TEMPERATURE, "num_ctx": CONTEXT_WINDOW},
                keep_alive="15m",   # keep the model warm in RAM between turns
            )
            usage = {"prompt_tokens": response.get("prompt_eval_count") or 0,
                     "completion_tokens": response.get("eval_count") or 0}
            return response["message"]["content"].strip(), usage
        except Exception as error:
            return f"[Could not reach Ollama: {error}. Is `ollama serve` running?]", usage

    def chat(self, user_message):
        """Run one full turn of the pipeline and return everything it produced."""
        timestamp = datetime.now().isoformat(timespec="seconds")
        timings = {}
        start = clock = time.perf_counter()

        def lap(name):
            """Record seconds since the previous step under `name`."""
            nonlocal clock
            now = time.perf_counter()
            timings[name] = round(now - clock, 3)
            clock = now

        rewritten = rewrite(user_message, self.history, model=self.model)
        lap("rewrite")
        # Extract from the rewritten text so "She is ..." becomes "Priya is ...".
        triples = extract(rewritten, model=self.model)
        lap("extract")
        actions = update_graph(self.store, triples, timestamp)
        if any(action["op"] != "SKIP" for action in actions):
            self.store.save()
        lap("update")

        facts = add_link_facts(self.store, retrieve(rewritten, self.store, top_k=TOP_K))
        lap("retrieve")
        answer, usage = self.generate_answer(rewritten, facts)
        drugs = [fact["object"] for fact in self.store.find_edges(subject=USER_NODE)
                 if fact.get("object_type") == "DRUG"]
        answer = remove_medical_advice(answer, rewritten, drugs)
        lap("answer")
        timings["total"] = round(time.perf_counter() - start, 3)

        # Keep only a short window of raw history - long-term memory lives in the graph.
        self.history.append({"role": "user", "content": user_message})
        self.history.append({"role": "assistant", "content": answer})
        self.history = self.history[-HISTORY_TURNS * 2:]

        return {
            "user_message": user_message,
            "rewritten_query": rewritten,
            "triples": triples,
            "actions": actions,
            "retrieved_facts": facts,
            "answer": answer,
            "timings": timings,
            "usage": usage,
        }

    def all_facts(self):
        """Return every active fact in memory."""
        return self.store.get_all_facts(active_only=True)

    def reset(self):
        """Wipe the memory graph and the short-term history."""
        self.store.clear()
        self.store.save()
        self.history = []


def run_cli():
    """Simple terminal chat loop for testing the pipeline."""
    pipeline = DMGPipeline()
    print(f"DMG chat ({pipeline.model}). Memory: {pipeline.store.stats()}")
    print("Commands: /facts  /reset  /quit\n")

    while True:
        try:
            message = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not message:
            continue
        if message == "/quit":
            break
        if message == "/facts":
            print(format_facts(pipeline.all_facts()), "\n")
            continue
        if message == "/reset":
            pipeline.reset()
            print("Memory cleared.\n")
            continue

        result = pipeline.chat(message)
        if result["rewritten_query"] != message:
            print(f"  (rewritten: {result['rewritten_query']})")
        for action in result["actions"]:
            print(f"  [{action['op']}] {action['subject']} {action['predicate']} {action['object']}")
        if result["retrieved_facts"]:
            print("  facts used:", "; ".join(f["text"] for f in result["retrieved_facts"]))
        print(f"Bot: {result['answer']}\n")


if __name__ == "__main__":
    run_cli()
