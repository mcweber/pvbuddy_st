# ---------------------------------------------------
# Version:05.01.2025
# Author: M. Weber
# ---------------------------------------------------
# 30.08.2024 switched to class-based approach
# 12.10.2024 added source documents
# 05.01.2024 added o1, o1-mini, deepseek
# ---------------------------------------------------
# Description:
# llm: gemini, o1, o1-mini, gpt4o, gpt4omini, deepseek, llama
# local: True/False
# ---------------------------------------------------

from datetime import datetime
import os
from dotenv import load_dotenv
import psutil

import openai
import google.generativeai as gemini
from groq import Groq
import ollama

MODELS = ["gemini", "o1", "o1-mini", "gpt-4o", "gpt-4o-mini", "deepseek", "llama"]

# Define class ---------------------------------------------------
class LLMHandler:
    
    
    def __init__(self, llm: str = "gemini", local: bool = False):
        self.LLM = llm
        self.LOCAL = local
        load_dotenv()

        if self.LLM in ["o1", "o1-mini", "gpt-4o", "gpt4o-mini"]:
            self.openaiClient = openai.OpenAI(api_key=os.environ.get('OPENAI_API_KEY_PRIVAT'))
        elif self.LLM == "llama":
            self.groqClient = Groq(api_key=os.environ.get('GROQ_API_KEY_PRIVAT'))
        elif self.LLM == "gemini":
            self.geminiClient = openai.OpenAI(
                api_key=os.environ.get('GEMINI_API_KEY'),
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
        elif self.LLM == "deepseek":
            self.deepseekClient = openai.OpenAI(
                api_key=os.environ.get('DEEPSEEK_API_KEY'),
                base_url="https://api.deepseek.com")
        

    @staticmethod
    def is_ollama_running() -> bool:
        for proc in psutil.process_iter(['pid', 'name']):
            if 'ollama' in proc.info['name'].lower():
                return True
        return False

    @staticmethod
    def define_prompt(system_prompt: str = "", question: str = "", history: list = [], db_results_str: str = "", web_results_str: str = "", source_doc_str:str = "") -> list:
        prompt = [
            {"role": "system", "content": f"{system_prompt}\n Das heutige Datum ist {datetime.now().date()}."}
        ]
        # prompt.extend(history)
        if db_results_str:
            prompt.append({"role": "assistant", "content": f'Hier sind einige relevante Informationen aus der Datenbank:\n{db_results_str}'})
        if web_results_str:
            prompt.append({"role": "assistant", "content": f'Hier sind einige relevante Informationen aus dem Internet:\n{web_results_str}'})
        if source_doc_str:
            prompt.append({"role": "assistant", "content": f'Dies ist das Quelldokument:\n{source_doc_str}'})
        question_prefix = "Basierend auf den beigefügten Informationen, " if web_results_str or db_results_str else ""
        prompt.append({"role": "user", "content": f"{question_prefix}{question}"})

        return prompt

    def ask_llm(self, temperature: float = 0.2, question: str = "", history: list = [],
                system_prompt: str = "", db_results_str: str = "", web_results_str: str = "", source_doc_str: str = "") -> str:
        prompt = self.define_prompt(system_prompt, question, history, db_results_str, web_results_str, source_doc_str)
        if self.LOCAL:
            return self._handle_local_llm(prompt)
        else:
            return self._handle_remote_llm(temperature, prompt)

    def _handle_local_llm(self, input_messages: list) -> str:
        if self.LLM == "mistral":
            response = ollama.chat(model="mistral", messages=input_messages)
            return response['message']['content']
        elif self.LLM == "llama3.2":
            response = ollama.chat(model="llama3.2", messages=input_messages)
            return response['message']['content']
        else:
            return f"Error: No valid local LLM specified [{self.LLM}]."

    def _handle_remote_llm(self, temperature: float, input_messages: list) -> str:
        if self.LLM in ["o1", "o1-mini", "gpt-4o", "gpt4o-mini"]:
            response = self.openaiClient.chat.completions.create(model=self.LLM, temperature=temperature, messages=input_messages)
            return response.choices[0].message.content
        elif self.LLM == "llama":
            response = self.groqClient.chat.completions.create(model="llama-3.3-70b-versatile", messages=input_messages)
            return response.choices[0].message.content
        elif self.LLM == "gemini":
            response = self.geminiClient.chat.completions.create(model="gemini-1.5-flash-latest", temperature=temperature, messages=input_messages)
            return(response.choices[0].message.content)
        elif self.LLM == "deepseek":
            response = self.deepseekClient.chat.completions.create(model="deepseek-chat", temperature=temperature, messages=input_messages)
            return(response.choices[0].message.content)
        else:
            return "Error: No valid remote LLM specified."
