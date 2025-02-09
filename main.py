# ---------------------------------------------------
VERSION ="09.02.2025"
# Author: M. Weber
# ---------------------------------------------------
# ---------------------------------------------------

import streamlit as st

import ask_llm
import ask_mongo

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
        st.session_state.history: list = []
        st.session_state.results_limit:int  = 10
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

        checkbox = st.checkbox(label="DB-Suche", value=st.session_state.search_db)
        if checkbox != st.session_state.search_db:
            st.session_state.search_db = checkbox
            st.rerun()

        RADIO_OPTIONS = ["fulltext","vector"]
        radio = st.radio(label="Suchtyp", options=RADIO_OPTIONS, index=RADIO_OPTIONS.index(st.session_state.search_type))
        if radio != st.session_state.search_type:
            st.session_state.search_type = radio
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
        
        if st.session_state.search_web:
            st.text_area("Web Results", st.session_state.results_web, height=200)
            st.divider()
        
        # st.text_area("History", st.session_state.history, height=200)
        if st.button("Clear History"):
            st.session_state.history = []
            st.session_state.results_web = ""
            st.session_state.results_db = ""
            st.rerun()

    # Define Search Form ----------------------------------------------
    question = st.chat_input("Frage oder test1, test2, test3 eingeben:")

    if question:
    
        if question == "test1":
            question = "was sagen die ausgaben zu El Pais?"

        if question == "test2":
            question = "Erstelle ein Dossier zur Firma Readly. Welche Informationen sind relevant?"

        if question == "test3":
            question = "Was sind die Zeitungen mit den höchsten Digitalumsätzen?"
    
        if question == "reset":
            st.session_state.history = []
            st.session_state.web_results = ""
            st.session_state.db_results = ""
            st.rerun()
    
        st.session_state.search_status = True

    # Define Search & Search Results -------------------------------------------
    if st.session_state.search_status:

        # Database Search ------------------------------------------------
        db_results_str = ""
        if st.session_state.search_db and st.session_state.results_db == "":

            if st.session_state.search_type == "vector":
                results_list, suchworte = ask_mongo.vector_search(search_text=question, limit=st.session_state.results_limit)
            else:
                results_list, suchworte = ask_mongo.fulltext_search_artikel(search_text=question, gen_suchworte=True, limit=st.session_state.results_limit)
            
            if results_list != []:
                with st.expander("Entscheidungssuche"):
                    st.write(f"Suchworte: {suchworte}")
                    for result in results_list:
                        st.write(f"{result['doknr']} {result['text'][:50]}...")
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
        write_history()
        st.session_state.search_status = False

if __name__ == "__main__":
    main()
