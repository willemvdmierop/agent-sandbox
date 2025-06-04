import os
import traceback
from typing import List, Dict, Any, TypedDict, Optional, Set
from pathlib import Path
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from Processdocument import process_document, get_vectorstore_path, create_document_chunks

class VMEAgentState(TypedDict):
    messages: List[Dict[str, Any]]
    context: Dict[str, List[Document]]
    current_language: str
    processed_documents: List[str]

class VMEAgent:
    def __init__(self, document_folder: str, model_name: str = "gpt-4.1", max_depth: int = 3):
        """Initialize the VME agent with document processing and graph setup
        
        Args:
            document_folder: Root folder containing VME documents
            model_name: Name of the OpenAI model to use
            max_depth: Maximum folder depth to search for documents
        """
        self.document_folder = Path(document_folder)
        self.model = ChatOpenAI(model_name=model_name, temperature=0)
        self.embeddings = OpenAIEmbeddings()
        self.vectorstores = {}
        self.max_depth = max_depth
        self.processed_paths: Set[Path] = set()  # Track processed documents
        self.setup_documents()
        self.graph = self.create_graph()

    def find_documents(self, folder: Path, current_depth: int = 0) -> List[Path]:
        """Recursively find all PDF documents in the folder structure
        
        Args:
            folder: Current folder to search
            current_depth: Current depth in the folder structure
            
        Returns:
            List of paths to PDF documents
        """
        if current_depth > self.max_depth:
            print(f"Reached maximum depth {self.max_depth} at {folder}")
            return []

        documents = []
        try:
            # First, check if this is a valid directory
            if not folder.is_dir():
                print(f"Not a directory: {folder}")
                return []

            # Get all items in the directory
            for item in folder.iterdir():
                try:
                    if item.is_file() and item.suffix.lower() == '.pdf':
                        # Skip if already processed
                        if item in self.processed_paths:
                            print(f"Skipping already processed document: {item}")
                            continue
                        documents.append(item)
                        self.processed_paths.add(item)
                    elif item.is_dir():
                        # Recursively search subdirectories
                        sub_docs = self.find_documents(item, current_depth + 1)
                        documents.extend(sub_docs)
                except PermissionError:
                    print(f"Permission denied accessing: {item}")
                except Exception as e:
                    print(f"Error processing {item}: {str(e)}")

        except Exception as e:
            print(f"Error searching directory {folder}: {str(e)}")

        return documents

    def setup_documents(self):
        """Process all documents and create vector stores"""
        print(f"\nSearching for documents in {self.document_folder} (max depth: {self.max_depth})...")
        
        # Find all documents recursively
        document_paths = self.find_documents(self.document_folder)
        
        if not document_paths:
            raise ValueError(f"No PDF files found in: {self.document_folder}")

        print(f"\nFound {len(document_paths)} documents to process:")
        for doc_path in document_paths:
            print(f"- {doc_path.relative_to(self.document_folder)}")

        # Process each document
        successful = 0
        failed = 0
        
        for doc_path in document_paths:
            try:
                doc_name = doc_path.name
                node_name = self.sanitize_doc_name(doc_name)
                print(f"\nProcessing {doc_path.relative_to(self.document_folder)}...")

                # Try to load existing vector store
                vectorstore_path = get_vectorstore_path(str(doc_path))
                if os.path.exists(vectorstore_path):
                    print(f"Loading existing vector store...")
                    vectorstore = self.load_vectorstore(vectorstore_path)
                    if vectorstore:
                        self.vectorstores[node_name] = vectorstore
                        successful += 1
                        continue
                    print("Will create a new vector store...")

                # Process new document
                vectorstore = process_document(str(doc_path))
                if vectorstore:
                    self.vectorstores[node_name] = vectorstore
                    print(f"Successfully processed document")
                    successful += 1
                else:
                    print(f"Warning: Failed to process document")
                    failed += 1

            except Exception as e:
                print(f"Error processing {doc_path}: {str(e)}")
                traceback.print_exc()
                failed += 1

        # Print processing summary
        print("\nDocument Processing Summary:")
        print(f"Total documents found: {len(document_paths)}")
        print(f"Successfully processed: {successful}")
        print(f"Failed to process: {failed}")
        
        if not self.vectorstores:
            raise ValueError("No documents were successfully processed")

    def create_graph(self) -> StateGraph:
        """Create the agent's workflow graph"""
        workflow = StateGraph(VMEAgentState)

        # Add core nodes
        workflow.add_node("detect_language", self.detect_language)
        workflow.add_node("process_documents", self.process_documents)
        workflow.add_node("generate_response", self.generate_response)

        # Add edges
        workflow.add_edge("detect_language", "process_documents")
        workflow.add_edge("process_documents", "generate_response")
        workflow.add_edge("generate_response", END)

        # Set entry point
        workflow.set_entry_point("detect_language")

        return workflow.compile()

    def detect_language(self, state: VMEAgentState) -> VMEAgentState:
        """Detect the language of the question"""
        last_message = state["messages"][-1]["content"]
        response = self.model.invoke(
            f"Detect the language of this text and respond with only the language name: {last_message}"
        )
        return {**state, "current_language": response.content.strip()}

    def process_documents(self, state: VMEAgentState) -> VMEAgentState:
        """Process all documents in parallel"""
        last_message = state["messages"][-1]["content"]
        new_context = state["context"].copy()
        processed_docs = []

        for doc_name, vectorstore in self.vectorstores.items():
            try:
                context = vectorstore.similarity_search(last_message, k=3)
                if context:
                    new_context[doc_name] = context
                    processed_docs.append(doc_name)
                    print(f"Found {len(context)} relevant sections in {doc_name}")
            except Exception as e:
                print(f"Error processing {doc_name}: {str(e)}")

        return {
            **state,
            "context": new_context,
            "processed_documents": processed_docs
        }

    def generate_response(self, state: VMEAgentState) -> VMEAgentState:
        """Generate a comprehensive response based on document contexts"""
        if not any(state["context"].values()):
            response = "I couldn't find any relevant information in the available documents to answer your question."
            return {**state, "messages": state["messages"] + [{"role": "assistant", "content": response}]}

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """
        Ik ben de syndicus van het gebouw en handel in die hoedanigheid.  
        Ik beschik over diepgaande kennis van de Belgische mede-eigendom (art. 577-3 e.v. BW), gebouwbeheer, techniek, verzekering en boekhouding.

        Mijn opdracht is om:

        • elke vraag van mede-eigenaars, bewoners of derden **helder, volledig en onmiddellijk uitvoerbaar** te beantwoorden;  
        • uitsluitend te steunen op:  
        – de meegeleverde documenten (basisakte, reglement, offertes, verslagen, …), én  
        – algemeen geldende wet- en regelgeving of erkende technische normen;  
        • alle beweringen exact te staven met citaten en paginaverwijzingen.

        ================================================================
        🔍 **WERKPROCEDURE** (voer in gedachten uit vóór je antwoord)  
        1. **Begrijp de vraag** – Wat wil de gebruiker concreet weten of gedaan krijgen?  
        2. **Permission & Competence Check**  
        a. Is het een privatieve of gemeenschappelijke aangelegenheid?  
        b. Wie is bevoegd (AV, Raad van mede-eigendom, syndicus, vergunning…) en welke meerderheid geldt?  
        c. Zijn er wettelijke/technische randvoorwaarden (AREI, brandveiligheid, EPB, verzekering, …)?  
        3. **Actieplan** – Formuleer de meest efficiënte aanpak (stappen, timing, offertes, verzekering, agenda-punt AV, …).  
        4. **Controleer taal & helderheid** – Antwoord in de taal van de vraag, gebruik korte alinea’s en bullets, vermijd jargon of leg het in één zin uit.  

        ================================================================
        📑 **STRUCTUUR ANTWOORD**  

        **Besluit**  
        (één zin: “Toegelaten”, “Niet toegelaten”, of “Onbeslist – AV-stemming vereist”).  

        **Actieplan**  
        • Stap-voor-stap wat er moet gebeuren (met indicatieve data of deadlines).  
        • Kostentoewijzing (privatief/gemeenschappelijk) en verrekenmethode.  
        • Verzekering of waarborg inroepen? Vermeld concrete contact- of dossier-info indien bekend.  
        • Eventuele risico’s of aansprakelijkheid.  

        **Analyse van Bevoegdheid & Toestemming**  
        Toon in 3-5 genummerde punten hoe je tot het besluit kwam (privatief vs. gemeenschappelijk, wettelijke meerderheid, artikel uit basisakte, enz.).  

        **Bronnen**  
        - [Documenttitel], p./art. [nr] — “exact citaat…”  
        - […]  

        **Disclaimer**  
        Dit antwoord is opgesteld in mijn functie van syndicus en biedt praktische richtsnoeren. Voor bindend juridisch advies kan steeds een notaris of advocaat geraadpleegd worden.
        """
            ),
            MessagesPlaceholder(variable_name="messages"),
            (
                "system",
                "Context from the documents:\n{context}"
            ),
        ])




        # Format context
        formatted_context = []
        for doc_name, docs in state["context"].items():
            if docs:
                doc_context = f"\nFrom {doc_name}:\n"
                doc_context += "\n".join([
                    f"Page {doc.metadata.get('page', 'N/A')}:\n{doc.page_content}"
                    for doc in docs
                ])
                formatted_context.append(doc_context)

        # Generate response
        chain = prompt | self.model | StrOutputParser()
        response = chain.invoke({
            "messages": state["messages"],
            "context": "\n\n".join(formatted_context)
        })

        return {
            **state,
            "messages": state["messages"] + [{"role": "assistant", "content": response}]
        }

    @staticmethod
    def sanitize_doc_name(doc_name: str) -> str:
        """Convert a document name to a valid, hashable string"""
        # Remove file extension and convert to lowercase
        name = Path(doc_name).stem.lower()
        # Replace any non-alphanumeric characters with underscore
        name = ''.join(c if c.isalnum() else '_' for c in name)
        # Ensure it starts with a letter and is unique
        if not name[0].isalpha():
            name = 'doc_' + name
        return name

    def get_document_paths(self) -> List[Path]:
        """Get all PDF files from the specified folder and its subfolders"""
        if not self.document_folder.exists():
            raise ValueError(f"Folder does not exist: {self.document_folder}")
        return self.find_documents(self.document_folder)

    def load_vectorstore(self, vectorstore_path: str) -> Optional[FAISS]:
        """Safely load a vector store from disk"""
        try:
            return FAISS.load_local(
                vectorstore_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        except Exception as e:
            print(f"Error loading vector store: {str(e)}")
            return None

    def run(self, question: str) -> str:
        """Run the agent with a question and return the response"""
        state = {
            "messages": [{"role": "user", "content": question}],
            "context": {},
            "current_language": "en",
            "processed_documents": []
        }

        try:
            result = self.graph.invoke(state)
            return result["messages"][-1]["content"]
        except Exception as e:
            error_msg = f"Error processing question: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            return f"Error: {error_msg}"

def main():
    """Main function to run the VME agent"""
    folder_path = "/Users/willemvandemierop/Desktop/sandbox_owen/OI/lift-casus"
    
    try:
        # Create the agent with a maximum folder depth of 3
        print("\nInitializing VME agent...")
        agent = VMEAgent(folder_path, max_depth=3)
        
        print("\nVME Document Question Answering System")
        print("Type 'exit' to end")
        print("-" * 50)
        
        while True:
            question = input("\nYour question: ")
            if question.lower() == 'exit':
                break
            
            response = agent.run(question)
            print("\n" + response)
            
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main() 