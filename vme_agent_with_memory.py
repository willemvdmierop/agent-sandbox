import os
import traceback
import hashlib
import json
from typing import List, Dict, Any, Optional, Union, Literal, Callable, Tuple, Set
from typing_extensions import TypedDict
from pathlib import Path
from datetime import datetime
import PyPDF2
from tqdm import tqdm
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import InMemoryVectorStore
import uuid
from collections import defaultdict
from IPython.display import Image, display

# Import all document processors
from Processdocument import process_document as process_pdf
from ProcessWordDocument import process_document as process_word
from ProcessEmailDocument import process_document as process_email

# Define supported file types and their processors
SUPPORTED_EXTENSIONS = {
    '.pdf': process_pdf,
    '.docx': process_word,
    '.doc': process_word,
    '.msg': process_email
}

class DocumentMetadata(TypedDict):
    """Metadata for processed documents"""
    file_path: str
    file_size: int
    last_modified: float
    hash: str
    processing_date: str
    status: str
    error: Optional[str]

class ConversationMemory(TypedDict):
    """Memory for conversation history and user preferences"""
    chat_history: List[Dict[str, Any]]  # Full conversation history
    user_preferences: Dict[str, Any]    # User-specific preferences
    frequent_questions: Dict[str, int]  # Questions and their frequency
    decisions: List[Dict[str, Any]]     # Important decisions and rationales

class DocumentMemory(TypedDict):
    """Memory for document access patterns and relationships"""
    access_patterns: Dict[str, List[str]]  # Document -> frequently accessed sections
    document_relationships: Dict[str, List[str]]  # Document -> related documents
    relevance_scores: Dict[str, Dict[str, float]]  # Question type -> document relevance

class KnowledgeMemory(TypedDict):
    """Memory for extracted knowledge and facts"""
    knowledge_graph: Dict[str, List[Dict[str, Any]]]  # Concept -> related facts
    temporal_info: Dict[str, Dict[str, Any]]  # Time-based information
    source_tracking: Dict[str, List[str]]  # Fact -> source documents

class VMEAgentState(TypedDict):
    """State for the VME agent workflow"""
    messages: List[Dict[str, str]]
    context: Dict[str, List[Any]]
    question_analysis: Optional[Dict[str, Any]]
    response_evaluation: Optional[Dict[str, Any]]
    processed_documents: List[str]
    search_focus: Optional[str]
    user_id: str
    feedback: Optional[Dict[str, Any]]  # New field for feedback
    learning_points: Optional[List[Dict[str, Any]]]  # New field for learning points

class VMEAgent:
    def __init__(self, document_folder: str, model_name: str = "gpt-4.1-mini", max_depth: int = 5, verbose: bool = False, force_reprocess: bool = False):
        """Initialize the VME agent with enhanced memory capabilities"""
        self.verbose = verbose
        self.force_reprocess = force_reprocess
        self.memory_dir = Path(document_folder) / ".vme_memory"
        self.memory_dir.mkdir(exist_ok=True)
        
        # Initialize memory stores
        self.conversation_memory = self._load_memory("conversation_memory.json", ConversationMemory)
        self.document_memory = self._load_memory("document_memory.json", DocumentMemory)
        self.knowledge_memory = self._load_memory("knowledge_memory.json", KnowledgeMemory)
        
        # Initialize vector store for memory
        self.memory_vectorstore = InMemoryVectorStore(OpenAIEmbeddings())
        
        # Convert to absolute path and validate
        try:
            self.document_folder = Path(document_folder).resolve()
            if not self.document_folder.exists():
                raise ValueError(f"The specified folder does not exist: {self.document_folder}")
            if not self.document_folder.is_dir():
                raise ValueError(f"The specified path is not a directory: {self.document_folder}")
            if not os.access(self.document_folder, os.R_OK):
                raise ValueError(f"No permission to read the directory: {self.document_folder}")
        except Exception as e:
            if isinstance(e, ValueError):
                raise
            raise ValueError(f"Invalid document folder path: {document_folder}. Error: {str(e)}")

        self.model = ChatOpenAI(model_name=model_name, temperature=0)
        self.embeddings = OpenAIEmbeddings()
        self.vectorstores = {}
        self.max_depth = max_depth
        self.processed_paths: Set[Path] = set()
        self.metadata_file = self.document_folder / ".vme_metadata.json"
        self.document_metadata: Dict[str, DocumentMetadata] = self._load_metadata()
        
        if self.verbose:
            print(f"\nInitializing VME agent with:")
            print(f"- Document folder: {self.document_folder}")
            print(f"- Maximum folder depth: {self.max_depth}")
            print(f"- Model: {model_name}")
            print(f"- Memory directory: {self.memory_dir}")
        
        self.setup_documents()
        self.graph = self.create_graph()

        # Initialize feedback and learning memory
        self.feedback_memory = {
            "corrections": {},  # Store corrections for specific questions
            "learning_points": [],  # Store general learning points
            "question_patterns": {},  # Store patterns of questions and their correct answers
            "last_updated": datetime.now().isoformat()
        }
        
        # Load existing feedback memory if available
        self.feedback_memory_file = os.path.join(self.memory_dir, "feedback_memory.json")
        self._load_feedback_memory()

    def _load_metadata(self) -> Dict[str, DocumentMetadata]:
        """Load document processing metadata from cache"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load metadata cache: {e}")
        return {}

    def _save_metadata(self):
        """Save document processing metadata to cache"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.document_metadata, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save metadata cache: {e}")

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file contents"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def _validate_pdf(self, file_path: Path) -> Tuple[bool, Optional[str]]:
        """Validate PDF file and return (is_valid, error_message)
        Only checks basic PDF structure since we'll use OCR for all PDFs
        """
        try:
            # Check file size
            file_size = file_path.stat().st_size
            if file_size == 0:
                return False, "File is empty"
            if file_size > 100 * 1024 * 1024:  # 100MB limit
                return False, "File is too large (>100MB)"

            # Try to open and read PDF with basic validation
            with open(file_path, 'rb') as f:
                try:
                    # Basic PDF structure validation with strict=False to ignore minor PDF issues
                    pdf = PyPDF2.PdfReader(f, strict=False)
                    
                    # Check if PDF has pages
                    if len(pdf.pages) == 0:
                        return False, "PDF has no pages"
                        
                    # Check if PDF is encrypted
                    if pdf.is_encrypted:
                        return False, "PDF is encrypted/password protected"
                        
                    # Basic structure check - try to access first page
                    try:
                        _ = pdf.pages[0]
                    except Exception as e:
                        return False, f"Error reading PDF structure: {str(e)}"
                            
                    return True, None
                    
                except PyPDF2.PdfReadError as e:
                    return False, f"Invalid PDF format: {str(e)}"
                except Exception as e:
                    return False, f"Error reading PDF: {str(e)}"

        except Exception as e:
            return False, f"Error validating PDF: {str(e)}"

    def _validate_document(self, file_path: Path) -> Tuple[bool, Optional[str]]:
        """Validate a document based on its file type"""
        extension = file_path.suffix.lower()
        
        if extension not in SUPPORTED_EXTENSIONS:
            return False, f"Unsupported file type: {extension}"
            
        try:
            if extension == '.pdf':
                # Use existing PDF validation
                return self._validate_pdf(file_path)
            elif extension in ['.docx', '.doc']:
                # Basic validation for Word documents
                if not os.access(file_path, os.R_OK):
                    return False, "No permission to read file"
                if file_path.stat().st_size == 0:
                    return False, "File is empty"
                return True, None
            elif extension == '.msg':
                # Basic validation for email files
                if not os.access(file_path, os.R_OK):
                    return False, "No permission to read file"
                if file_path.stat().st_size == 0:
                    return False, "File is empty"
                return True, None
                    
        except Exception as e:
            return False, f"Error validating file: {str(e)}"

    def _should_reprocess_document(self, file_path: Path) -> Tuple[bool, Optional[str]]:
        """Check if document needs reprocessing based on metadata"""
        if self.force_reprocess:
            return True, "Force reprocessing enabled"
            
        file_hash = self._calculate_file_hash(file_path)
        metadata = self.document_metadata.get(str(file_path))
        
        if not metadata:
            return True, "No metadata found"
            
        if metadata['hash'] != file_hash:
            return True, "File has changed"
            
        if metadata['status'] != 'success':
            return True, f"Previous processing failed: {metadata.get('error', 'Unknown error')}"
            
        # If we get here, the file hasn't changed and was successfully processed before
        return False, None

    def _print(self, message: str, level: str = "info"):
        """Print message only if verbose is enabled or if it's an error/warning"""
        if self.verbose or level in ["error", "warning"]:
            if level == "error":
                print(f"Error: {message}")
            elif level == "warning":
                print(f"Warning: {message}")
            else:
                print(message)

    def find_documents(self, folder: Path, current_depth: int = 0) -> List[Path]:
        """Recursively find all supported documents in the folder structure"""
        if current_depth > self.max_depth:
            self._print(f"Reached maximum depth {self.max_depth} at {folder}")
            return []

        documents = []
        try:
            for item in folder.iterdir():
                try:
                    if item.is_file() and item.suffix.lower() in SUPPORTED_EXTENSIONS:
                        if item in self.processed_paths:
                            self._print(f"Skipping already processed document: {item}")
                            continue
                            
                        if not os.access(item, os.R_OK):
                            self._print(f"No permission to read file: {item}", "warning")
                            continue
                            
                        is_valid, error_msg = self._validate_document(item)
                        if not is_valid:
                            self._print(f"Invalid {item.suffix} file {item}: {error_msg}", "warning")
                            self.document_metadata[str(item)] = {
                                'file_path': str(item),
                                'file_size': item.stat().st_size,
                                'last_modified': item.stat().st_mtime,
                                'hash': self._calculate_file_hash(item),
                                'processing_date': datetime.now().isoformat(),
                                'status': 'failed',
                                'error': error_msg
                            }
                            continue
                            
                        documents.append(item)
                        self.processed_paths.add(item)
                        
                    elif item.is_dir():
                        if not os.access(item, os.R_OK):
                            self._print(f"No permission to read directory: {item}", "warning")
                            continue
                        sub_docs = self.find_documents(item, current_depth + 1)
                        documents.extend(sub_docs)
                        
                except PermissionError:
                    self._print(f"Permission denied accessing: {item}", "warning")
                except Exception as e:
                    self._print(f"Error processing {item}: {str(e)}", "error")

        except Exception as e:
            self._print(f"Error searching directory {folder}: {str(e)}", "error")

        return documents

    def setup_documents(self):
        """Process all documents and create vector stores"""
        self._print(f"Searching for documents in {self.document_folder} (max depth: {self.max_depth})...")
        
        document_paths = self.find_documents(self.document_folder)
        
        if not document_paths:
            raise ValueError(
                f"No valid documents found in: {self.document_folder}\n"
                "Please ensure:\n"
                "1. The folder contains supported files (PDF, Word, Email)\n"
                "2. The files are valid and readable\n"
                "3. You have read permissions for the folder and files\n"
                "4. The files are not empty or corrupted"
            )

        if self.verbose:
            print(f"\nFound {len(document_paths)} documents to process:")
            for doc_path in document_paths:
                print(f"- {doc_path.relative_to(self.document_folder)}")

        successful = 0
        failed = 0
        failed_files = []
        
        # Create progress bar with proper disable flag
        progress_bar = tqdm(
            total=len(document_paths),
            desc="Processing documents",
            disable=self.verbose,
            unit="doc"
        )
        
        for doc_path in document_paths:
            try:
                should_reprocess, reason = self._should_reprocess_document(doc_path)
                if not should_reprocess:
                    self._print(f"Skipping {doc_path.name} (unchanged)")
                    progress_bar.update(1)
                    continue
                    
                self._print(f"Processing {doc_path.relative_to(self.document_folder)}...")
                doc_name = doc_path.name
                node_name = self.sanitize_doc_name(doc_name)

                # Get the appropriate processor based on file extension
                extension = doc_path.suffix.lower()
                processor = SUPPORTED_EXTENSIONS.get(extension)
                if not processor:
                    self._print(f"Unsupported file type: {extension}", "warning")
                    failed += 1
                    failed_files.append(str(doc_path))
                    self._update_metadata(doc_path, 'failed', f"Unsupported file type: {extension}")
                    continue

                # Process the document using the appropriate processor
                vectorstore = processor(str(doc_path))
                if vectorstore:
                    self.vectorstores[node_name] = vectorstore
                    successful += 1
                    self._update_metadata(doc_path, 'success')
                else:
                    failed += 1
                    failed_files.append(str(doc_path))
                    self._update_metadata(doc_path, 'failed', "Processing failed")

            except Exception as e:
                self._print(f"Error processing {doc_path}: {str(e)}", "error")
                traceback.print_exc()
                failed += 1
                failed_files.append(str(doc_path))
                self._update_metadata(doc_path, 'failed', str(e))
            finally:
                progress_bar.update(1)

        progress_bar.close()
        self._save_metadata()

        # Always print summary
        print("\nDocument Processing Summary:")
        print(f"Total documents found: {len(document_paths)}")
        print(f"Successfully processed: {successful}")
        print(f"Failed to process: {failed}")
        
        if failed_files and self.verbose:
            print("\nFailed files:")
            for file in failed_files:
                print(f"- {file}")
        
        if not self.vectorstores:
            raise ValueError(
                "No documents were successfully processed.\n"
                "Please check:\n"
                "1. The files are not corrupted\n"
                "2. The files contain readable text\n"
                "3. You have sufficient permissions\n"
                "4. The files are not empty"
            )

    def _update_metadata(self, file_path: Path, status: str, error: Optional[str] = None):
        """Update metadata for a processed document"""
        self.document_metadata[str(file_path)] = {
            'file_path': str(file_path),
            'file_size': file_path.stat().st_size,
            'last_modified': file_path.stat().st_mtime,
            'hash': self._calculate_file_hash(file_path),
            'processing_date': datetime.now().isoformat(),
            'status': status,
            'error': error
        }

    def create_graph(self) -> StateGraph:
        """Create the agent's workflow graph with response quality evaluation"""
        from langgraph.graph import END
        
        workflow = StateGraph(VMEAgentState)

        # Add nodes
        workflow.add_node("detect_language", self.detect_language)
        workflow.add_node("analyze_question", self.analyze_question)
        workflow.add_node("process_documents", self.process_documents)
        workflow.add_node("generate_response", self.generate_response)
        workflow.add_node("generate_memory_response", self.generate_memory_response)
        workflow.add_node("evaluate_response", self.evaluate_response)
        workflow.add_node("should_continue_searching", self.should_continue_searching)
        workflow.add_node("should_search_documents", self.should_search_documents)

        # Add edges
        workflow.add_edge("detect_language", "analyze_question")
        workflow.add_edge("analyze_question", "should_search_documents")
        workflow.add_edge("process_documents", "generate_response")
        workflow.add_edge("generate_response", "evaluate_response")
        workflow.add_edge("generate_memory_response", "evaluate_response")
        workflow.add_edge("evaluate_response", "should_continue_searching")

        # Add conditional edges
        workflow.add_conditional_edges(
            "should_search_documents",
            lambda x: x["next_step"],
            {
                "search": "process_documents",
                "memory": "generate_memory_response"
            }
        )

        workflow.add_conditional_edges(
            "should_continue_searching",
            lambda x: x["next_step"],
            {
                "continue": "process_documents",
                "complete": END
            }
        )

        # Set entry point
        workflow.set_entry_point("detect_language")

        # Compile the graph
        graph = workflow.compile()

        # Save the graph using Mermaid format
        try:
            output_path = os.path.join(self.document_folder, "workflow_graph.png")
            with open(output_path, "wb") as f:
                f.write(graph.get_graph().draw_mermaid_png())
            self._print(f"Graph visualization saved to {output_path}", "info")
        except Exception as e:
            self._print(f"Could not save graph visualization: {str(e)}", "warning")

        return graph

    def detect_language(self, state: VMEAgentState) -> VMEAgentState:
        """Detect the language of the question"""
        last_message = state["messages"][-1]["content"]
        response = self.model.invoke(
            f"Detect the language of this text and respond with only the language name: {last_message}"
        )
        return {**state, "current_language": response.content.strip()}

    def analyze_question(self, state: VMEAgentState) -> VMEAgentState:
        """Analyze the question to determine if document search is needed"""
        try:
            last_message = state["messages"][-1]["content"]
            
            # Use the LLM to analyze the question with explicit JSON formatting
            prompt = f"""Analyze the following question and determine if it can be answered using previously extracted knowledge, conversation history, or if it requires a new document search.

Return ONLY a JSON object with these exact fields:
- requires_search (boolean): Whether a document search is needed
- reason (string): Brief explanation of the decision
- memory_type (string): Type of memory to use if no search needed ("conversation", "knowledge", or "document")
- confidence (number between 0 and 1): Confidence in the analysis
- question_type (string): Type of question ("factual", "procedural", "clarification", "general_inquiry")

Example response format:
{{
    "requires_search": false,
    "reason": "Question can be answered using existing knowledge about maintenance schedules",
    "memory_type": "knowledge",
    "confidence": 0.9,
    "question_type": "factual"
}}

Question to analyze:
{last_message}

Remember: Return ONLY the JSON object, nothing else."""

            # Add timeout to model invocation
            import signal
            from contextlib import contextmanager
            import time
            import re

            @contextmanager
            def timeout(seconds):
                def handler(signum, frame):
                    raise TimeoutError("Question analysis timed out")
                signal.signal(signal.SIGALRM, handler)
                signal.alarm(seconds)
                try:
                    yield
                finally:
                    signal.alarm(0)

            try:
                with timeout(10):  # 10 second timeout
                    response = self.model.invoke(prompt)
                    content = response.content.strip()
            except TimeoutError:
                self._print("Question analysis timed out, defaulting to document search", "warning")
                return {
                    **state,
                    "question_analysis": {
                        "requires_search": True,
                        "reason": "Analysis timed out, defaulting to search",
                        "memory_type": "document",
                        "confidence": 0.5,
                        "question_type": "general_inquiry"
                    }
                }

            # Try multiple approaches to parse the JSON
            try:
                # First attempt: direct JSON parsing
                analysis = json.loads(content)
            except json.JSONDecodeError:
                # Second attempt: try to extract JSON object if wrapped in other text
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    try:
                        analysis = json.loads(json_match.group(0))
                    except json.JSONDecodeError:
                        # Third attempt: try to fix common JSON formatting issues
                        cleaned_content = content.replace('\n', ' ').replace('\\', '\\\\')
                        # Remove any non-JSON text before or after the object
                        cleaned_content = re.sub(r'^[^{]*', '', cleaned_content)
                        cleaned_content = re.sub(r'[^}]*$', '', cleaned_content)
                        try:
                            analysis = json.loads(cleaned_content)
                        except json.JSONDecodeError:
                            self._print("Could not parse question analysis, defaulting to document search", "warning")
                            return {
                                **state,
                                "question_analysis": {
                                    "requires_search": True,
                                    "reason": "Failed to parse analysis, defaulting to search",
                                    "memory_type": "document",
                                    "confidence": 0.5,
                                    "question_type": "general_inquiry"
                                }
                            }

            # Validate the structure of the analysis
            required_fields = ['requires_search', 'reason', 'memory_type', 'confidence', 'question_type']
            if not all(field in analysis for field in required_fields):
                self._print("Invalid analysis structure, defaulting to document search", "warning")
                return {
                    **state,
                    "question_analysis": {
                        "requires_search": True,
                        "reason": "Invalid analysis structure, defaulting to search",
                        "memory_type": "document",
                        "confidence": 0.5,
                        "question_type": "general_inquiry"
                    }
                }

            # Validate field types
            if not isinstance(analysis['requires_search'], bool):
                analysis['requires_search'] = True
            if not isinstance(analysis['reason'], str):
                analysis['reason'] = "Invalid reason format"
            if not isinstance(analysis['memory_type'], str) or analysis['memory_type'] not in ['conversation', 'knowledge', 'document']:
                analysis['memory_type'] = 'document'
            if not isinstance(analysis['confidence'], (int, float)) or not 0 <= analysis['confidence'] <= 1:
                analysis['confidence'] = 0.5
            if not isinstance(analysis['question_type'], str):
                analysis['question_type'] = 'general_inquiry'

            return {
                **state,
                "question_analysis": analysis
            }

        except Exception as e:
            self._print(f"Error analyzing question: {str(e)}", "warning")
            return {
                **state,
                "question_analysis": {
                    "requires_search": True,
                    "reason": f"Error during analysis: {str(e)}",
                    "memory_type": "document",
                    "confidence": 0.5,
                    "question_type": "general_inquiry"
                }
            }

    def should_search_documents(self, state: VMEAgentState) -> VMEAgentState:
        """Determine if we should search documents or use memory"""
        analysis = state.get("question_analysis", {})
        
        # If analysis suggests no search needed and confidence is high
        if not analysis.get("requires_search", True) and analysis.get("confidence", 0) > 0.7:
            return {**state, "next_step": "memory"}
        return {**state, "next_step": "search"}

    def process_documents(self, state: VMEAgentState) -> VMEAgentState:
        """Process documents with a limit on the number of documents to search"""
        last_message = state["messages"][-1]["content"]
        new_context = state["context"].copy()
        processed_docs = state.get("processed_documents", [])

        # Get question type and search focus from analysis if available
        question_type = state.get("question_analysis", {}).get("question_type", "general_inquiry")
        search_focus = state.get("search_focus", last_message)
        
        # Get most relevant documents first, excluding already processed ones
        all_docs = set(self.vectorstores.keys())
        remaining_docs = list(all_docs - set(processed_docs))
        
        if not remaining_docs:
            return {
                **state,
                "context": new_context,
                "processed_documents": processed_docs,
                "reason": "No more documents to process"
            }
        
        # Get relevant documents from remaining ones - limit to 2
        relevant_docs = self.get_relevant_documents(search_focus, question_type, remaining_docs[:2])
        
        if self.verbose:
            print(f"\nProcessing {len(relevant_docs)} most relevant documents:")
            for doc in relevant_docs:
                print(f"- {doc}")

        for doc_name in relevant_docs:
            try:
                vectorstore = self.vectorstores.get(doc_name)
                if not vectorstore:
                    continue
                    
                # Only get 1 most relevant chunk per document
                context = vectorstore.similarity_search(search_focus, k=1)
                if context:
                    new_context[doc_name] = context
                    processed_docs.append(doc_name)
                    if self.verbose:
                        print(f"Found relevant section in {doc_name}")
            except Exception as e:
                self._print(f"Error processing {doc_name}: {str(e)}", "error")

        return {
            **state,
            "context": new_context,
            "processed_documents": processed_docs,
            "reason": f"Processed {len(relevant_docs)} documents"
        }

    def generate_response(self, state: VMEAgentState) -> VMEAgentState:
        """Generate response using context and learning memory"""
        try:
            question = state["messages"][-1]["content"]
            
            # Check learning memory for relevant patterns
            relevant_learning = []
            for pattern, responses in self.feedback_memory["question_patterns"].items():
                # Check if question matches pattern
                pattern_prompt = f"""Does this question match the pattern: "{pattern}"?
Question: {question}
Return ONLY true or false."""

                try:
                    with timeout(3):
                        match_response = self.model.invoke(pattern_prompt)
                        if "true" in match_response.content.lower():
                            # Get most recent correct response for this pattern
                            relevant_learning.extend(responses[-2:])  # Last 2 responses
                except:
                    continue

            # Prepare context with learning points
            formatted_context = []
            
            # Add document context
            for doc_name, docs in state["context"].items():
                if docs:
                    doc_context = f"\nFrom {doc_name}:\n"
                    doc_context += "\n".join([
                        f"Page {doc.metadata.get('page', 'N/A')}:\n{doc.page_content}"
                        for doc in docs
                    ])
                    formatted_context.append(doc_context)

            # Add learning context if available
            if relevant_learning:
                learning_context = "\nBased on previous feedback:\n"
                for learning in relevant_learning:
                    learning_context += f"- {learning['correct_response']}\n"
                formatted_context.append(learning_context)

            # Generate response with learning context
            prompt = f"""You are a helpful assistant. Use the following context to answer the question.
                If there are learning points from previous feedback, prioritize those over document context.
                Be precise and accurate in your response.

                Context:
                {"\n\n".join(formatted_context)}

                Question: {question}

                Remember to:
                1. Use learning from feedback if available
                2. Be clear and concise
                3. Cite sources when possible
                4. Acknowledge when using feedback-based learning"""

            chain = prompt | self.model | StrOutputParser()
            response = chain.invoke({
                "messages": state["messages"],
                "context": "\n\n".join(formatted_context)
            })

            return {
                **state,
                "messages": state["messages"] + [{"role": "assistant", "content": response}]
            }

        except Exception as e:
            self._print(f"Error generating response: {str(e)}", "error")
            return state

    def generate_memory_response(self, state: VMEAgentState) -> VMEAgentState:
        """Generate response using memory without document search"""
        question = state["messages"][-1]["content"]
        analysis = state["question_analysis"]
        memory_type = analysis.get("memory_type", "none")
        
        # Prepare memory context based on type
        memory_context = []
        
        if memory_type == "conversation":
            # Get relevant conversation history
            user_id = state.get("user_id", "default_user")
            history = self.get_historical_context(user_id, question)
            if history:
                memory_context.append("Relevant conversation history:\n" + history)
                
        elif memory_type == "knowledge":
            # Search knowledge graph for relevant facts
            relevant_facts = []
            for concept, facts in self.knowledge_memory["knowledge_graph"].items():
                # Use embeddings to find relevant concepts
                if self.memory_vectorstore.similarity_search(question, k=3):
                    relevant_facts.extend(facts)
            
            if relevant_facts:
                memory_context.append("Relevant facts from memory:\n" + 
                    "\n".join(f"- {fact['fact']} (Source: {', '.join(self.knowledge_memory['source_tracking'].get(str(i), ['Unknown']))})"
                             for i, fact in enumerate(relevant_facts)))
                
        elif memory_type == "document":
            # Use document access patterns
            question_type = analysis.get("question_type", "general_inquiry")
            relevant_docs = self.get_relevant_documents(question, question_type)
            if relevant_docs:
                memory_context.append("Based on document access patterns:\n" +
                    "\n".join(f"- {doc} has been frequently accessed for similar questions"
                             for doc in relevant_docs))

        # Generate response using memory context
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are a helpful assistant with access to memory of previous interactions and extracted knowledge.
                Use the provided memory context to answer the question. If the memory context is insufficient,
                acknowledge this and suggest what additional information might be needed."""
            ),
            MessagesPlaceholder(variable_name="messages"),
            (
                "system",
                "Memory context:\n{memory_context}"
            ),
        ])

        chain = prompt | self.model | StrOutputParser()
        response = chain.invoke({
            "messages": state["messages"],
            "memory_context": "\n\n".join(memory_context) if memory_context else "No relevant memory found."
        })

        return {
            **state,
            "messages": state["messages"] + [{"role": "assistant", "content": response}]
        }

    def evaluate_response(self, state: VMEAgentState) -> VMEAgentState:
        """Evaluate the quality of the generated response and determine if more searching is needed"""
        try:
            last_message = state["messages"][-1]["content"]
            last_response = state["messages"][-2]["content"] if len(state["messages"]) > 1 else ""
            
            # Add timeout context manager
            import signal
            from contextlib import contextmanager

            @contextmanager
            def timeout(seconds):
                def handler(signum, frame):
                    raise TimeoutError("Response evaluation timed out")
                signal.signal(signal.SIGALRM, handler)
                signal.alarm(seconds)
                try:
                    yield
                finally:
                    signal.alarm(0)
            
            # Simplified evaluation prompt for faster processing
            prompt = f"""Evaluate this response to the question. Return ONLY a JSON object:
{{
    "is_sufficient": boolean,
    "quality_score": number between 0 and 1,
    "needs_more_search": boolean
}}

Question: {last_message}
Response: {last_response}"""

            try:
                # Reduced timeout to 5 seconds
                with timeout(5):
                    response = self.model.invoke(prompt)
                    content = response.content.strip()
                    evaluation = json.loads(content)
            except (TimeoutError, json.JSONDecodeError):
                # Default to continuing search if evaluation fails
                return {
                    **state,
                    "response_evaluation": {
                        "is_sufficient": False,
                        "quality_score": 0.5,
                        "needs_more_search": True
                    }
                }

            return {
                **state,
                "response_evaluation": evaluation
            }

        except Exception as e:
            self._print(f"Error evaluating response: {str(e)}", "warning")
            return {
                **state,
                "response_evaluation": {
                    "is_sufficient": False,
                    "quality_score": 0.5,
                    "needs_more_search": True
                }
            }

    def should_continue_searching(self, state: VMEAgentState) -> VMEAgentState:
        """Determine if more document searching is needed based on response evaluation"""
        evaluation = state.get("response_evaluation", {})
        processed_docs = state.get("processed_documents", [])
        
        # If we've already processed more than 3 documents, stop to prevent loops
        if len(processed_docs) >= 3:
            return {**state, "next_step": "complete", "reason": "Maximum document limit reached"}
            
        # If we don't have an evaluation, default to one more search
        if not evaluation:
            return {**state, "next_step": "continue", "reason": "No evaluation available"}
            
        # If the response is sufficient, stop
        if evaluation.get("is_sufficient", False):
            return {**state, "next_step": "complete", "reason": "Response is sufficient"}
            
        # If quality score is high enough (0.8 or above), stop
        if evaluation.get("quality_score", 0) >= 0.8:
            return {**state, "next_step": "complete", "reason": "High quality response achieved"}
            
        # If we need more search and haven't exceeded the limit
        if evaluation.get("needs_more_search", True):
            # Check if we have any new documents to search
            all_docs = set(self.vectorstores.keys())
            remaining_docs = all_docs - set(processed_docs)
            
            if not remaining_docs:
                return {**state, "next_step": "complete", "reason": "No more documents to search"}
                
            # Update the search focus in the state
            return {
                **state,
                "next_step": "continue",
                "search_focus": evaluation.get("search_focus", "general information"),
                "reason": "Continuing search with focus on remaining documents"
            }
            
        return {**state, "next_step": "complete", "reason": "No more search needed"}

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
            # Use the same loading logic as the processors
            return FAISS.load_local(
                vectorstore_path,
                self.embeddings,
                allow_dangerous_deserialization=True  # Safe because we create these files
            )
        except Exception as e:
            self._print(f"Error loading vector store from {vectorstore_path}: {str(e)}", "error")
            return None

    def _load_memory(self, filename: str, memory_type: type) -> Any:
        """Load memory from disk or create new if not exists"""
        memory_path = self.memory_dir / filename
        if memory_path.exists():
            try:
                with open(memory_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self._print(f"Warning: Could not load memory from {filename}: {e}")
        
        # Return empty memory structure based on type
        if memory_type == ConversationMemory:
            return {"chat_history": [], "user_preferences": {}, "frequent_questions": {}, "decisions": []}
        elif memory_type == DocumentMemory:
            return {"access_patterns": {}, "document_relationships": {}, "relevance_scores": {}}
        elif memory_type == KnowledgeMemory:
            return {"knowledge_graph": {}, "temporal_info": {}, "source_tracking": {}}
        return {}

    def _save_memory(self, memory: Any, filename: str):
        """Save memory to disk"""
        try:
            with open(self.memory_dir / filename, 'w') as f:
                json.dump(memory, f, indent=2)
        except Exception as e:
            self._print(f"Warning: Could not save memory to {filename}: {e}")

    def update_conversation_memory(self, message: Dict[str, Any], user_id: str):
        """Update conversation memory with new message"""
        self.conversation_memory["chat_history"].append({
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id
        })
        
        # Update frequent questions
        if message["role"] == "user":
            question = message["content"]
            self.conversation_memory["frequent_questions"][question] = \
                self.conversation_memory["frequent_questions"].get(question, 0) + 1
        
        self._save_memory(self.conversation_memory, "conversation_memory.json")

    def update_document_memory(self, doc_name: str, accessed_sections: List[str], question_type: str):
        """Update document memory with access patterns"""
        if doc_name not in self.document_memory["access_patterns"]:
            self.document_memory["access_patterns"][doc_name] = []
        self.document_memory["access_patterns"][doc_name].extend(accessed_sections)
        
        # Update relevance scores
        if question_type not in self.document_memory["relevance_scores"]:
            self.document_memory["relevance_scores"][question_type] = {}
        self.document_memory["relevance_scores"][question_type][doc_name] = \
            self.document_memory["relevance_scores"][question_type].get(doc_name, 0) + 1
        
        self._save_memory(self.document_memory, "document_memory.json")

    def update_knowledge_memory(self, fact: Dict[str, Any], source_doc: str):
        """Update knowledge memory with new facts"""
        concept = fact.get("concept", "general")
        if concept not in self.knowledge_memory["knowledge_graph"]:
            self.knowledge_memory["knowledge_graph"][concept] = []
        self.knowledge_memory["knowledge_graph"][concept].append(fact)
        
        # Track source
        fact_id = str(uuid.uuid4())
        self.knowledge_memory["source_tracking"][fact_id] = [source_doc]
        
        self._save_memory(self.knowledge_memory, "knowledge_memory.json")

    def _extract_facts(self, text: str) -> List[Dict[str, Any]]:
        """Extract facts from text using the LLM with timeout and chunking"""
        try:
            # Limit text length to prevent timeouts
            MAX_TEXT_LENGTH = 2000
            if len(text) > MAX_TEXT_LENGTH:
                text = text[:MAX_TEXT_LENGTH] + "..."
            
            # Use the LLM to extract facts from the text with more explicit JSON formatting
            prompt = f"""Extract key facts from the following text. Return ONLY a JSON array of fact objects.
Each fact object must have these exact fields:
- concept (string): The main subject or category
- fact (string): The actual fact
- temporal_info (string or null): Any time-related information
- values (array of numbers or null): Any numerical values or measurements
- confidence (number between 0 and 1): Your confidence in this fact

Limit to maximum 3 most important facts.

Example response format:
[
    {{
        "concept": "maintenance",
        "fact": "Annual elevator inspection required",
        "temporal_info": "yearly",
        "values": null,
        "confidence": 0.95
    }}
]

Text to analyze:
{text}

Remember: Return ONLY the JSON array, nothing else."""

            # Add timeout to model invocation
            import signal
            from contextlib import contextmanager
            import time

            @contextmanager
            def timeout(seconds):
                def handler(signum, frame):
                    raise TimeoutError("Fact extraction timed out")
                signal.signal(signal.SIGALRM, handler)
                signal.alarm(seconds)
                try:
                    yield
                finally:
                    signal.alarm(0)

            try:
                with timeout(10):  # 10 second timeout
                    response = self.model.invoke(prompt)
                    content = response.content.strip()
            except TimeoutError:
                self._print("Fact extraction timed out, skipping...", "warning")
                return []

            # Try to clean the response if it's not valid JSON
            try:
                # First attempt: direct JSON parsing
                facts = json.loads(content)
            except json.JSONDecodeError:
                # Second attempt: try to extract JSON array if wrapped in other text
                import re
                json_match = re.search(r'\[.*\]', content, re.DOTALL)
                if json_match:
                    try:
                        facts = json.loads(json_match.group(0))
                    except json.JSONDecodeError:
                        # Third attempt: try to fix common JSON formatting issues
                        cleaned_content = content.replace('\n', ' ').replace('\\', '\\\\')
                        # Remove any non-JSON text before or after the array
                        cleaned_content = re.sub(r'^[^[]*', '', cleaned_content)
                        cleaned_content = re.sub(r'[^]]*$', '', cleaned_content)
                        try:
                            facts = json.loads(cleaned_content)
                        except json.JSONDecodeError:
                            self._print(f"Could not parse facts from response: {content}", "warning")
                            return []

            # Validate the structure of each fact
            valid_facts = []
            for fact in facts if isinstance(facts, list) else []:
                if not isinstance(fact, dict):
                    continue
                    
                # Ensure all required fields are present and of correct type
                if not all(key in fact for key in ['concept', 'fact', 'confidence']):
                    continue
                    
                if not isinstance(fact['concept'], str) or not isinstance(fact['fact'], str):
                    continue
                    
                if not isinstance(fact['confidence'], (int, float)) or not 0 <= fact['confidence'] <= 1:
                    continue
                    
                # Ensure optional fields are of correct type
                if 'temporal_info' in fact and fact['temporal_info'] is not None:
                    if not isinstance(fact['temporal_info'], str):
                        fact['temporal_info'] = str(fact['temporal_info'])
                        
                if 'values' in fact and fact['values'] is not None:
                    if not isinstance(fact['values'], list):
                        fact['values'] = [fact['values']] if fact['values'] is not None else None
                    # Convert all values to numbers where possible
                    try:
                        fact['values'] = [float(v) if isinstance(v, (int, float)) else v for v in fact['values']]
                    except (ValueError, TypeError):
                        fact['values'] = None
                
                valid_facts.append(fact)
            
            return valid_facts[:3]  # Limit to 3 facts
            
        except Exception as e:
            self._print(f"Error extracting facts: {str(e)}", "warning")
            return []

    def get_relevant_documents(self, query: str, question_type: str = "general_inquiry", doc_list: Optional[List[str]] = None) -> List[str]:
        """Get most relevant documents for a query, considering question type and document access patterns"""
        try:
            # If no specific document list is provided, use all documents
            if doc_list is None:
                doc_list = list(self.vectorstores.keys())
            
            # Get document relevance scores - limit to top 3 most relevant
            doc_scores = []
            for doc_name in doc_list[:3]:  # Only check top 3 documents
                try:
                    vectorstore = self.vectorstores.get(doc_name)
                    if not vectorstore:
                        continue
                        
                    # Get similarity score - only get 1 result
                    docs = vectorstore.similarity_search_with_score(query, k=1)
                    if docs:
                        score = 1 - docs[0][1]  # Convert distance to similarity
                        doc_scores.append((doc_name, score))
                except Exception as e:
                    self._print(f"Error scoring document {doc_name}: {str(e)}", "warning")
                    continue

            # Sort by score and return document names
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            return [doc for doc, _ in doc_scores]

        except Exception as e:
            self._print(f"Error getting relevant documents: {str(e)}", "error")
            return list(self.vectorstores.keys())[:3]  # Fallback to first 3 documents

    def get_historical_context(self, user_id: str, question: str) -> str:
        """Get relevant historical context from conversation memory"""
        relevant_history = []
        
        # Get recent conversations for this user
        user_history = [
            entry for entry in self.conversation_memory["chat_history"]
            if entry["user_id"] == user_id
        ][-5:]  # Last 5 conversations
        
        # Get similar past questions
        similar_questions = [
            q for q, _ in sorted(
                self.conversation_memory["frequent_questions"].items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]  # Top 3 most frequent similar questions
        ]
        
        if user_history:
            relevant_history.append("Recent conversation history:")
            for entry in user_history:
                relevant_history.append(f"- {entry['message']['role']}: {entry['message']['content']}")
        
        if similar_questions:
            relevant_history.append("\nSimilar past questions:")
            for q in similar_questions:
                relevant_history.append(f"- {q}")
        
        return "\n".join(relevant_history) if relevant_history else ""

    def _load_feedback_memory(self):
        """Load feedback memory from file"""
        if os.path.exists(self.feedback_memory_file):
            try:
                with open(self.feedback_memory_file, 'r') as f:
                    self.feedback_memory = json.load(f)
            except Exception as e:
                self._print(f"Could not load feedback memory: {str(e)}", "warning")

    def _save_feedback_memory(self):
        """Save feedback memory to file"""
        try:
            self.feedback_memory["last_updated"] = datetime.now().isoformat()
            with open(self.feedback_memory_file, 'w') as f:
                json.dump(self.feedback_memory, f, indent=2)
        except Exception as e:
            self._print(f"Could not save feedback memory: {str(e)}", "warning")

    def process_feedback(self, state: VMEAgentState) -> VMEAgentState:
        """Process user feedback and update learning memory"""
        try:
            feedback = state.get("feedback", {})
            if not feedback:
                return state

            question = state["messages"][-2]["content"]  # The original question
            response = state["messages"][-1]["content"]  # The response that got feedback
            correction = feedback.get("correction")
            is_correct = feedback.get("is_correct", False)
            explanation = feedback.get("explanation", "")

            # Generate a learning point
            learning_prompt = f"""Based on this feedback, create a learning point that can be used to improve future responses.
Question: {question}
Original Response: {response}
Feedback: {correction if correction else explanation}
Is Correct: {is_correct}

Return a JSON object with:
{{
    "learning_point": "Key learning from this feedback",
    "question_pattern": "Pattern of questions this applies to",
    "correct_response": "How similar questions should be answered",
    "keywords": ["relevant", "keywords", "for", "matching"]
}}"""

            try:
                with timeout(5):
                    learning_response = self.model.invoke(learning_prompt)
                    learning_point = json.loads(learning_response.content)
                    
                    # Add to learning memory
                    self.feedback_memory["learning_points"].append({
                        **learning_point,
                        "timestamp": datetime.now().isoformat(),
                        "source_question": question,
                        "source_feedback": feedback
                    })
                    
                    # Update question patterns
                    pattern = learning_point["question_pattern"]
                    if pattern not in self.feedback_memory["question_patterns"]:
                        self.feedback_memory["question_patterns"][pattern] = []
                    self.feedback_memory["question_patterns"][pattern].append({
                        "correct_response": learning_point["correct_response"],
                        "keywords": learning_point["keywords"],
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    # Save updated memory
                    self._save_feedback_memory()
                    
                    return {
                        **state,
                        "learning_points": self.feedback_memory["learning_points"][-5:],  # Last 5 learning points
                        "feedback_processed": True
                    }
            except Exception as e:
                self._print(f"Error processing learning point: {str(e)}", "warning")
                return state

        except Exception as e:
            self._print(f"Error processing feedback: {str(e)}", "warning")
            return state

    def run(self, question: str, user_id: str = "default_user", feedback: Optional[Dict[str, Any]] = None) -> str:
        """Run the agent with optional feedback"""
        try:
            # Initialize state with feedback if provided
            initial_state = {
                "messages": [{"role": "user", "content": question}],
                "context": {},
                "processed_documents": [],
                "user_id": user_id,
                "feedback": feedback
            }

            # Run the graph
            final_state = self.graph.invoke(initial_state)

            # Process feedback if provided
            if feedback:
                final_state = self.process_feedback(final_state)

            # Return the last message
            return final_state["messages"][-1]["content"]

        except Exception as e:
            self._print(f"Error processing question: {str(e)}", "error")
            return f"I apologize, but I encountered an error: {str(e)}"

def main():
    """Main function to run the VME agent"""
    folder_path = "/Users/willemvandemierop/Downloads/MYSebas bibliotheek"
    
    try:
        # Create the agent with increased folder depth and force reprocessing
        print("\nInitializing VME agent...")
        agent = VMEAgent(folder_path, max_depth=10, verbose=True, force_reprocess=True)  # Force reprocessing of all documents
        
        print("\nVME Document Question Answering System")
        print("Type 'exit' to end")
        print("-" * 50)
        
        while True:
            question = input("\nYour question: ")
            if question.lower() == 'exit':
                break
            
            response = agent.run(question)
            print("\n" + response)
            
    except ValueError as e:
        print(f"\nError: {str(e)}")
        print("\nPlease ensure:")
        print("1. The folder path is correct")
        print("2. The folder contains PDF files")
        print("3. You have read permissions for the folder and files")
        print("4. The PDFs are valid and not corrupted")
    except Exception as e:
        print(f"\nFatal error: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main() 