"""Streamlit UI: Candidate + Job ID dropdowns, Start Chat, automatic initial summary, follow-up chat."""

from __future__ import annotations

import streamlit as st

from hr_chatbot.ingest import list_candidates, list_jobs
from hr_chatbot.service import ChatSession, TurnResult, create_services

DISCLAIMER = "Analysis support only, not an automated hiring decision."

st.set_page_config(page_title="Agentic HR Chatbot", page_icon="🧭", layout="wide")


@st.cache_resource(show_spinner="Loading indexes and models...")
def get_services():
    return create_services()


def render_turn(turn: TurnResult) -> None:
    out = turn.output
    st.markdown(out["response"])
    evidence, context = out["source_evidence"], out["retrieved_context"]
    with st.expander(f"Source evidence ({len(evidence)})"):
        if not evidence:
            st.caption("No verified citations for this answer.")
        for item in evidence:
            st.markdown(f"**{item['source_file']}** - {item['page_or_section']}")
            st.markdown(f"> {item['content']}")
    with st.expander(f"Retrieved context ({len(context)})"):
        for item in context:
            score = f" - score {item['score']}" if item["score"] is not None else ""
            st.markdown(f"**{item['source_file']}** - {item['page_or_section']}{score}")
            st.text(item["content"])
    with st.expander(f"Agent trace ({len(turn.trace)} tool calls)"):
        for step in turn.trace:
            st.code(f"{step['name']}({step['args']})  ->  {step['results']} context item(s)" + ("" if step["ok"] else "  [error]"))
    with st.expander("Raw JSON output"):
        st.json(out)


def main() -> None:
    st.title("Agentic HR Chatbot")
    st.caption(DISCLAIMER)

    try:
        services = get_services()
    except Exception as exc:  # missing API key, embedder mismatch, ...
        st.error(f"Startup problem: {exc}")
        st.stop()

    data_dir = services.settings.data_dir
    candidates, jobs = list_candidates(data_dir), list_jobs(data_dir)
    if not candidates or not jobs:
        st.info(f"No data found in `{data_dir}`. Run `python -m hr_chatbot.sample_data` to create sample files.")
        st.stop()

    with st.sidebar:
        st.header("Selection")
        candidate_id = st.selectbox("Candidate", candidates)
        job_id = st.selectbox("Job ID", jobs)
        start = st.button("Start Chat", type="primary", use_container_width=True)
        stats = services.store.stats()
        st.caption(f"Index: {stats['documents']} documents, {stats['chunks']} chunks ({stats['embedder']})")

    if start:
        session = ChatSession(services, candidate_id, job_id)
        with st.spinner("Refreshing changed files and preparing the initial summary..."):
            try:
                turn = session.start()
            except Exception as exc:
                st.error(f"Could not start the chat: {type(exc).__name__}: {exc}")
                st.stop()
        st.session_state["session"] = session
        st.session_state["transcript"] = [("assistant", turn)]
        for warning in turn.ingest.warnings if turn.ingest else []:
            st.warning(warning)

    session: ChatSession | None = st.session_state.get("session")
    if session is None:
        st.info("Select a candidate and a job, then click **Start Chat**.")
        st.stop()

    st.subheader(f"{session.candidate_id}  x  {session.job_id}")
    for role, content in st.session_state["transcript"]:
        with st.chat_message(role):
            if role == "user":
                st.markdown(content)
            else:
                render_turn(content)

    question = st.chat_input("Ask a follow-up question about this candidate and job")
    if question:
        st.session_state["transcript"].append(("user", question))
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Searching the documents..."):
                try:
                    turn = session.ask(question)
                except Exception as exc:
                    st.error(f"{type(exc).__name__}: {exc}")
                    st.stop()
            render_turn(turn)
        st.session_state["transcript"].append(("assistant", turn))


main()
