# ---------------------------------------------------
VERSION ="11.03.2025"
# Author: M. Weber
# ---------------------------------------------------
# 12.02.2024 added test4
# 23.02.2025 added search_web
# 23.02.2025 added logbook
# 24.02.2025 added use dict for test queries
# ---------------------------------------------------

import streamlit as st

import ask_llm
import ask_mongo
import ask_web
import logbook

import os
from dotenv import load_dotenv
load_dotenv()

# Functions -------------------------------------------------------------
@st.dialog("Login")
def login_code_dialog() -> None:
    with st.form(key="login_code_form"):
        code = st.text_input(label="Code", type="password")
        if st.form_submit_button("Enter"):
            if code == os.environ.get('CODE_PVD'):
                st.success("Code is correct.")
                st.session_state.code = True
                st.rerun()
            else:
                st.error("Code is not correct.")
                st.session_state.code = True
                # st.rerun()

def write_history() -> None:
    for entry in st.session_state.history:
        with st.chat_message(entry["role"]):
            st.write(f"{entry['content']}")

# Main -----------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title='pvBuddy', layout="wide", initial_sidebar_state="expanded")

    # Initialize Session State -----------------------------------------
    if 'init' not in st.session_state:
        # Check if System-Prompt exists
        if ask_mongo.get_system_prompt() == {}:
            ask_mongo.add_system_prompt("Du bist ein hilfreicher Assistent.")
        st.session_state.init: bool = True
        st.session_state.code: bool = False
        st.session_state.model: str = "gemini"
        st.session_state.system_prompt: str = ask_mongo.get_system_prompt()
        st.session_state.search_status: bool = False
        st.session_state.search_type: str = "fulltext"
        st.session_state.search_db: bool = True
        st.session_state.search_web: bool = False
        st.session_state.sort_by: str = "score"
        st.session_state.history: list = []
        st.session_state.results_limit:int  = 20
        st.session_state.results_web: str = ""
        st.session_state.results_db: str = ""

    if st.session_state.code == False:
        login_code_dialog()

    # Define Sidebar ---------------------------------------------------
    with st.sidebar:
        st.header("pvBuddy")
        st.caption(f"Version: {VERSION} Status: POC")

        radio = st.radio(label="Model", options=ask_llm.MODELS, index=ask_llm.MODELS.index(st.session_state.model))
        if radio != st.session_state.model:
            st.session_state.model = radio
            st.rerun()

        checkbox = st.checkbox(label="WEB-Suche", value=st.session_state.search_web)
        if checkbox != st.session_state.search_web:
            st.session_state.search_web = checkbox
            st.rerun()

        checkbox = st.checkbox(label="DB-Suche", value=st.session_state.search_db)
        if checkbox != st.session_state.search_db:
            st.session_state.search_db = checkbox
            st.rerun()

        RADIO_OPTIONS = ["fulltext", "vector"]
        radio = st.radio(label="Suchtyp", options=RADIO_OPTIONS, index=RADIO_OPTIONS.index(st.session_state.search_type))
        if radio != st.session_state.search_type:
            st.session_state.search_type = radio
            st.rerun()

        RADIO_OPTIONS = ["score", "doknr"]
        radio = st.radio(label="Sortierung", options=RADIO_OPTIONS, index=RADIO_OPTIONS.index(st.session_state.sort_by))
        if radio != st.session_state.sort_by:
            st.session_state.sort_by = radio
            st.rerun()
        
        slider = st.slider("Search Results", min_value=0, max_value=50, value=st.session_state.results_limit, step=5)
        if slider != st.session_state.results_limit:
            st.session_state.results_limit = slider
            st.rerun()
        
        switch_SystemPrompt = st.text_area("System-Prompt", st.session_state.system_prompt, height=200)
        if switch_SystemPrompt != st.session_state.system_prompt:
            st.session_state.system_prompt = switch_SystemPrompt
            ask_mongo.update_system_prompt(switch_SystemPrompt)
            st.rerun()
        
        st.divider()
        if st.button("Clear History"):
            st.session_state.history = []
            st.session_state.results_web = ""
            st.session_state.results_db = ""
            st.rerun()

    # Define Search Form ----------------------------------------------
    question = st.chat_input("Frage oder test1, test2, test3, test4, test5 eingeben:")

    TEST_QUERIES = {
        "test1": "Erstelle ein Dossier zu El Pais?",
        "test2": "Erstelle ein Dossier zur Firma Readly. Welche Informationen sind relevant?", 
        "test3": "Was sind die Zeitungen mit den höchsten Digitalumsätzen?",
        "test4": "Wie funktioniert das deutsche Presse Grosso System?",
        "test5": "Wann hat The Economist seine Paywall installiert?"
    }
    
    if question:
        if question in TEST_QUERIES:
            question = TEST_QUERIES[question]
        st.session_state.search_status = True

    # Define Search & Search Results -------------------------------------------
    if st.session_state.search_status:

        # Web Search ------------------------------------------------
        if st.session_state.search_web and st.session_state.results_web == "":
            web_results_str = ""
            web_search_handler = ask_web.WebSearch()
            results = web_search_handler.search(query=question, score=0.5, limit=st.session_state.results_limit)
            with st.expander("WEB Suchergebnisse"):
                for result in results:
                    st.write(f"[{round(result['score'], 3)}] {result['title']} [{result['url']}]")
                    web_results_str += f"Titel: {result['title']}\nURL: {result['url']}\nText: {result['content']}\n\n"
            st.session_state.results_web = web_results_str

        # Database Search ------------------------------------------------
        if st.session_state.search_db and st.session_state.results_db == "":
            db_results_str = ""
            if st.session_state.search_type == "vector":
                results_list, suchworte = ask_mongo.vector_search(search_text=question, sort=st.session_state.sort_by, limit=st.session_state.results_limit)
            else:
                results_list, suchworte = ask_mongo.fulltext_search_artikel(search_text=question, gen_suchworte=True, sort=st.session_state.sort_by, limit=st.session_state.results_limit)
            if results_list != []:
                with st.expander("DB Suchergebnisse"):
                    st.write(f"Suchworte: {suchworte}")
                    for result in results_list:
                        st.write(f"{result['doknr']} [{result['score']:.2f}] {result['text'][:50].replace('\n', ' ')}...")
                        db_results_str += f"DokNr: {result['doknr']} Text: {result['text']}\n\n"
            else:
                st.warning("Keine Ergebnisse gefunden.")
            st.session_state.results_db = db_results_str
                
        # LLM Search ------------------------------------------------
        llm_handler = ask_llm.LLMHandler()
        summary = llm_handler.ask_llm(
            temperature=0.2,
            question=question,
            history=st.session_state.history,
            system_prompt=st.session_state.system_prompt,
            db_results_str=st.session_state.results_db,
            web_results_str=st.session_state.results_web
            )
        st.session_state.history.append({"role": "user", "content": question})
        st.session_state.history.append({"role": "assistant", "content": summary})
        st.session_state.search_status = False
        logbook.add_entry(app="pvBuddy", user="default", text=question)

        write_history()
        
if __name__ == "__main__":
    main()
