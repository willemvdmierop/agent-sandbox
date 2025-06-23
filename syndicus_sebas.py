from __future__ import annotations
from typing import List, Dict, Optional, Literal, Any
from pydantic import BaseModel
from datetime import datetime, timedelta
from pathlib import Path
import os
import json
import uuid
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.schema import Document
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool
from langgraph.graph import StateGraph, END
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnableConfig
from langdetect import detect
from document_processor import DocumentProcessor

# ───────────────────────────────────────────────────────────
# State and Memory Models
# ───────────────────────────────────────────────────────────
class KnowledgeTriple(BaseModel):
    subject: str
    predicate: str
    object: str

class AgentState(BaseModel):
    user_msg: str
    working_set: List[Dict] = []          # docs retrieved for this turn
    answer: Optional[str] = None
    critique: Optional[Dict] = None       # {"self_reflect": bool, "notes": str}
    feedback: Optional[Dict] = None       # user thumbs-up/down etc.
    memories_to_write: List[Dict] = []    # items destined for long-term store
    language: str = "nl"                  # default to Dutch
    context: Dict[str, List[Any]] = {}    # context from vector stores
    processed_documents: List[str] = []    # list of processed document names

class SyndicusSebas:
    """Expert Syndicus agent that uses vector stores for knowledge and memory."""
    
    def __init__(self, rules_folder: str, model_name: str = "gpt-4.1-mini", verbose: bool = False):
        """Initialize the Syndicus-Sebas agent.
        
        Args:
            rules_folder: Path to folder containing rules and regulations
            model_name: Name of the LLM to use
            verbose: Whether to print detailed logs
        """
        self.verbose = verbose
        self.rules_folder = Path(rules_folder)
        self.memory_dir = self.rules_folder / ".sebas_memory"
        self.memory_dir.mkdir(exist_ok=True)
        
        # Initialize LLM and embeddings
        self.llm = ChatOpenAI(model_name=model_name, temperature=0)
        self.embeddings = OpenAIEmbeddings()
        
        # Initialize vector stores
        self.rules_vectorstore = self._initialize_rules_vectorstore()
        self.memory_vectorstore = FAISS.from_texts(
            ["Initial memory"], self.embeddings
        )
        self.workflow_vectorstore = FAISS.from_texts(
            ["Initial workflow"], self.embeddings
        )
        
        # Initialize memory systems
        self.conversation_memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Create the workflow graph
        self.graph = self.create_graph()
        
    def _initialize_rules_vectorstore(self) -> FAISS:
        """Initialize or load the rules vector store."""
        rules_store_path = self.memory_dir / "rules_vectorstore"
        if rules_store_path.exists():
            return FAISS.load_local(str(rules_store_path), self.embeddings)
        return self._create_rules_vectorstore()
        
    def _create_rules_vectorstore(self) -> FAISS:
        """Create a new vector store from rules documents."""
        processor = DocumentProcessor(self.embeddings)
        vectorstore = processor.process_folder(str(self.rules_folder))
        
        # Save the vector store
        save_path = self.memory_dir / "rules_vectorstore"
        processor.save_vectorstore(vectorstore, str(save_path))
        
        return vectorstore
        
    def create_graph(self) -> StateGraph:
        """Create the LangGraph workflow."""
        # Initialize the graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("detect_language", self.detect_language)
        workflow.add_node("retrieve_memory", self.retrieve_memory)
        workflow.add_node("reason", self.reason)
        workflow.add_node("critique", self.critique)
        workflow.add_node("retrieve_docs", self.retrieve_docs)
        workflow.add_node("write_memory", self.write_memory)
        workflow.add_node("emit_answer", self.emit_answer)
        workflow.add_node("capture_feedback", self.capture_feedback)
        
        # Add edges
        workflow.add_edge("detect_language", "retrieve_memory")
        workflow.add_edge("retrieve_memory", "reason")
        workflow.add_edge("reason", "critique")
        workflow.add_edge("retrieve_docs", "reason")
        workflow.add_edge("critique", "write_memory")
        workflow.add_edge("critique", "emit_answer")
        workflow.add_edge("emit_answer", "capture_feedback")
        workflow.add_edge("capture_feedback", "write_memory")
        workflow.add_edge("write_memory", END)
        
        # Add conditional edge for self-reflection
        workflow.add_conditional_edges(
            "critique",
            self.should_reflect,
            {
                "reflect": "retrieve_docs",
                "done": None
            }
        )
        
        # Set entry point
        workflow.set_entry_point("detect_language")
        
        return workflow
        
    def detect_language(self, state: AgentState) -> AgentState:
        """Detect the language of the input message."""
        try:
            state.language = detect(state.user_msg)
        except:
            state.language = "nl"  # Default to Dutch
        return state
        
    def retrieve_memory(self, state: AgentState) -> AgentState:
        """Retrieve relevant memories and context."""
        # Search memory vector store
        memory_docs = self.memory_vectorstore.similarity_search(
            state.user_msg, k=3
        )
        state.working_set.extend([
            {"content": doc.page_content, "source": "memory"}
            for doc in memory_docs
        ])
        return state
        
    def reason(self, state: AgentState) -> AgentState:
        """Generate a response using context and memories."""
        # Format context for the LLM
        context = "\n\n".join([
            f"From {doc['source']}:\n{doc['content']}"
            for doc in state.working_set
        ])
        
        # Create the prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Je bent Sebas, een ervaren en behulpzame Syndicus expert. 
            Je communiceert altijd in het Nederlands, op een professionele maar vriendelijke manier.
            Je geeft duidelijke, beknopte antwoorden gebaseerd op de regels en je ervaring.
            Gebruik deze context om je antwoord te formuleren:
            
            {context}"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])
        
        # Generate response
        response = self.llm.invoke(prompt.format(
            context=context,
            chat_history=self.conversation_memory.chat_memory.messages,
            question=state.user_msg
        ))
        
        state.answer = response.content
        return state
        
    def critique(self, state: AgentState) -> AgentState:
        """Self-reflect on the current answer."""
        critique_prompt = """Evalueer dit antwoord kritisch:

Vraag: {question}
Antwoord: {answer}

Geef aan of we meer informatie nodig hebben en waarom:
"""
        
        response = self.llm.invoke(critique_prompt.format(
            question=state.user_msg,
            answer=state.answer
        ))
        
        state.critique = {
            "self_reflect": "meer informatie" in response.content.lower(),
            "notes": response.content
        }
        return state
        
    def should_reflect(self, state: AgentState) -> Literal["reflect", "done"]:
        """Decide whether to continue reflection."""
        return "reflect" if state.critique["self_reflect"] else "done"
        
    def retrieve_docs(self, state: AgentState) -> AgentState:
        """Retrieve additional documents based on critique."""
        docs = self.rules_vectorstore.similarity_search(
            state.user_msg, k=3
        )
        state.working_set.extend([
            {"content": doc.page_content, "source": "rules"}
            for doc in docs
        ])
        return state
        
    def write_memory(self, state: AgentState) -> AgentState:
        """Write important information to memory."""
        if state.feedback and state.feedback.get("rating") == "positive":
            # Store successful interaction
            self.memory_vectorstore.add_texts(
                [f"Q: {state.user_msg}\nA: {state.answer}"],
                metadatas=[{"timestamp": datetime.now().isoformat()}]
            )
            
            # Store workflow if it was successful
            self.workflow_vectorstore.add_texts(
                [json.dumps({
                    "question": state.user_msg,
                    "workflow": [doc["source"] for doc in state.working_set],
                    "successful": True
                })],
                metadatas=[{"timestamp": datetime.now().isoformat()}]
            )
            
        return state
        
    def emit_answer(self, state: AgentState) -> AgentState:
        """Format and emit the final answer."""
        # Add to conversation memory
        self.conversation_memory.chat_memory.add_user_message(state.user_msg)
        self.conversation_memory.chat_memory.add_ai_message(state.answer)
        return state
        
    def capture_feedback(self, state: AgentState) -> AgentState:
        """Capture and process user feedback."""
        # In a real implementation, this would wait for user feedback
        # For now, we'll assume positive feedback
        state.feedback = {"rating": "positive"}
        return state
        
    def run(self, question: str) -> str:
        """Run the agent with a question and return the response."""
        try:
            state = AgentState(user_msg=question)
            final_state = self.graph.invoke(state)
            return final_state.answer
        except Exception as e:
            return f"Er is een fout opgetreden: {str(e)}"

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Gebruik Sebas, de Syndicus expert agent.")
    parser.add_argument(
        "--rules_folder",
        type=str,
        default="/Users/willemvandemierop/Desktop/sandbox_owen/MYSebas bibliotheek",
        help="Pad naar de map met regels en documenten (PDF, TXT, DOC, DOCX). Standaard: '/Users/willemvandemierop/Desktop/sandbox_owen/MYSebas bibliotheek'"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt-4.1-mini",
        help="Naam van het OpenAI model (standaard: gpt-4.1-mini)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Toon extra loginformatie"
    )
    args = parser.parse_args()

    sebas = SyndicusSebas(
        rules_folder=args.rules_folder,
        model_name=args.model_name,
        verbose=args.verbose
    )

    print("Welkom bij Sebas, uw digitale Syndicus expert! (Typ 'stop' om te stoppen)")
    while True:
        vraag = input("\nUw vraag: ")
        if vraag.strip().lower() in {"stop", "exit", "quit"}:
            print("Tot ziens!")
            break
        antwoord = sebas.run(vraag)
        print(f"\nSebas: {antwoord}")

if __name__ == "__main__":
    main() 