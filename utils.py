import os
import time
from typing import List
from dotenv import load_dotenv

import groq
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from pydantic import BaseModel, Field, field_validator
from types import SimpleNamespace

load_dotenv()


def _extract_json_payload(text: str):
    """Try to parse the first JSON object or array from text.
    Returns the decoded JSON structure or None if no valid payload is found.
    """
    import json

    decoder = json.JSONDecoder()
    for i, char in enumerate(text):
        if char not in '{[':
            continue
        try:
            payload, _ = decoder.raw_decode(text[i:])
            return payload
        except json.JSONDecodeError:
            continue

    # Fallback: grab the first bracketed JSON-like body and try parsing it.
    import re
    patterns = [r"\[.*?\]", r"\{.*?\}"]
    for pattern in patterns:
        m = re.search(pattern, text, re.DOTALL)
        if not m:
            continue
        candidate = m.group(0)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    return None


def _validate_mcq_list(payload, expected_count: int):
    if not isinstance(payload, list):
        raise ValueError("Expected a JSON array of MCQs")
    if len(payload) < expected_count:
        raise ValueError(f"Expected {expected_count} MCQs, got {len(payload)}")
    mcqs = []
    for item in payload:
        mcqs.append(MCQQstn.model_validate(item))
    return mcqs


def _validate_fill_blank_list(payload, expected_count: int):
    if not isinstance(payload, list):
        raise ValueError("Expected a JSON array of fill-in-the-blank questions")
    if len(payload) < expected_count:
        raise ValueError(f"Expected {expected_count} questions, got {len(payload)}")
    questions = []
    for item in payload:
        questions.append(FillBlankQstn.model_validate(item))
    return questions

# define a data model for multiple choice questions using pydantic
class MCQQstn(BaseModel):
    question: str = Field(..., description="The question text")
    options: List[str] = Field(..., description="List of 4 possible answers")
    correct_answer: str = Field(..., description="The correct answer from the options")


    @field_validator('question', mode='before')
    def clean_question(cls, v):
        if isinstance(v, dict):
            return v.get('description', str(v))
        return str(v)
    

# define a prompt template for generating fill in the blank questions
class FillBlankQstn(BaseModel):
    question: str = Field(description="The question text with a blank represented by '___'")
    answer : str = Field(description="The correct answer to fill in the blank")

    @field_validator('question', mode='before')
    def clean_question(cls, v):
        if isinstance(v, dict):
            return v.get('description', str(v))
        return str(v)
    

# generate a questions

class QuestionGenerator:
    def __init__(self):
        """
        Initialize question generator with groq api sets up the language model with specific
        parameters for generating questions.
        - tries a small list of supported Groq models
        - temperature of 0.9 for creativity
        """
        model_list = os.getenv('GROQ_MODEL_LIST', 'llama-3.3-70b-versatile,llama-3.3-70b').split(',')
        self.model_order = [model.strip() for model in model_list if model.strip()]
        self.llm = None
        self.active_model = None
        self._init_llm(self.model_order[0])

    def _init_llm(self, model_name: str):
        self.llm = ChatGroq(
            api_key=os.getenv('GROQ_API_KEY'),
            model=model_name,
            temperature=0.9
        )
        self.active_model = model_name

    def _invoke_with_model_fallback(self, prompt_text: str):
        last_exception = None
        for model_name in self.model_order:
            if self.active_model != model_name:
                self._init_llm(model_name)
            try:
                return self.llm.invoke(prompt_text)
            except groq.BadRequestError as e:
                message = str(e).lower()
                if 'decommissioned' in message or 'invalid_request_error' in message:
                    print(f"Model {model_name} is invalid/decommissioned; switching to next model.")
                    last_exception = e
                    continue
                raise
        if last_exception:
            raise last_exception
        raise RuntimeError(f"No available Groq model worked from configured list: {', '.join(self.model_order)}")


    def generate_mcq(self, topic: str, difficulty: str= 'medium') -> MCQQstn:
        """
        generate a multiple qstns with robust error handling
        includes:
        - output parsing using Pydantic
        - Structured prompt template
        - Multiple retry attempts on failure
        - Validation of  generated questions

        """
        # set up pydantic parser for type checking and validation
        mcq_parser= PydanticOutputParser(pydantic_object=MCQQstn)

        # define the prompt template with specific format requiremnts

        prompt = PromptTemplate(
            template=(
                "Generate a {difficulty} multiple-choice question about {topic}.\n\n"
                "Return ONLY a JSON object with these exact fields:\n"
                "- 'question': A clear, specific question\n"
                "- 'options': An array of exactly 4 possible answers\n"
                "- 'correct_answer': One of the options that is the correct answer\n\n"
                "Example format:\n"
                '{{\n'
                '    "question": "What is the capital of France?",\n'
                '    "options": ["London", "Berlin", "Paris", "Madrid"],\n'
                '    "correct_answer": "Paris"\n'
                '}}\n\n'
                "Your response:"
            ),
            input_variables=["topic", "difficulty"]
            
        )

        # Implement retry logic with maximum attempts
        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                # Generate the question using the language model
                response = self._invoke_with_model_fallback(prompt.format(topic=topic, difficulty=difficulty))
                # Extract text content robustly (support different LLM client return shapes)
                if hasattr(response, 'content'):
                    text = response.content
                elif isinstance(response, dict) and 'content' in response:
                    text = response['content']
                else:
                    # fallback to string conversion
                    text = str(response)
                # Parse the response using the pydantic parser
                # PydanticOutputParser may expect an object with a `.content` attribute,
                # so wrap the text into a simple object to be safe.
                # Prefer extracting a JSON object directly from the LLM text and validate
                json_obj = _extract_json_payload(text)
                if json_obj:
                    try:
                        mcq = MCQQstn.model_validate(json_obj)
                    except Exception as ve:
                        print("Direct JSON -> model_validate failed:", repr(ve))
                        # fall through to try parser
                        mcq = None
                else:
                    mcq = None

                if mcq is None:
                    # Try using the PydanticOutputParser as a fallback (pass the raw text)
                    try:
                        mcq_parsed = mcq_parser.parse(text)
                    except Exception as pe:
                        print("PydanticOutputParser.parse raised:", repr(pe))
                        print("Response repr:", repr(response)[:1000])
                        print("Text repr:", repr(text)[:1000])
                        raise

                    # The parser may return a pydantic model or a JSON string
                    if isinstance(mcq_parsed, str):
                        import json
                        mcq_data = json.loads(mcq_parsed)
                        mcq = MCQQstn.model_validate(mcq_data)
                    else:
                        mcq = mcq_parsed

                if not mcq.question or len(mcq.options) != 4 or not mcq.correct_answer:
                    raise ValueError("Generated question does not meet the required format or content criteria.")
                if mcq.correct_answer not in mcq.options:
                    raise ValueError("Correct answer must be one of the provided options.")
                
                return mcq
            
            except groq.RateLimitError as e:
                wait_secs = 2
                print(f"Rate limit reached, waiting {wait_secs}s before retrying ({attempt + 1}/{max_attempts})...")
                time.sleep(wait_secs)
                if attempt == max_attempts - 1:
                    raise RuntimeError(f"Failed to generate valid MCQ after {max_attempts} attempts due to rate limiting: {str(e)}")
                continue
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                import traceback
                traceback.print_exc()
                if attempt == max_attempts - 1:
                    raise RuntimeError(f"Failed to generate valid MCQ after {max_attempts} attempts: {str(e)}")
                continue
            


    def generate_mcq_set(self, topic: str, difficulty: str = 'medium', count: int = 5) -> List[MCQQstn]:
        """
        Generate a batch of unique MCQs in one request to reduce duplicate questions.
        """
        prompt = PromptTemplate(
            template=(
                "Generate {count} unique {difficulty} multiple-choice questions about {topic}.\n\n"
                "Return ONLY a JSON array of objects with these exact fields for each question:\n"
                "- 'question': A clear, specific question\n"
                "- 'options': An array of exactly 4 possible answers\n"
                "- 'correct_answer': One of the options that is the correct answer\n\n"
                "Example format:\n"
                "[\n"
                "  {{\n"
                "    \"question\": \"What is the capital of France?\",\n"
                "    \"options\": [\"London\", \"Berlin\", \"Paris\", \"Madrid\"],\n"
                "    \"correct_answer\": \"Paris\"\n"
                "  }},\n"
                "  {{\n"
                "    \"question\": \"Which planet is known as the Red Planet?\",\n"
                "    \"options\": [\"Earth\", \"Jupiter\", \"Mars\", \"Venus\"],\n"
                "    \"correct_answer\": \"Mars\"\n"
                "  }}\n"
                "]\n\n"
                "Your response:"
            ),
            input_variables=["topic", "difficulty", "count"]
        )

        max_attempts = 4
        for attempt in range(max_attempts):
            try:
                response = self._invoke_with_model_fallback(prompt.format(topic=topic, difficulty=difficulty, count=count))
                if hasattr(response, 'content'):
                    text = response.content
                elif isinstance(response, dict) and 'content' in response:
                    text = response['content']
                else:
                    text = str(response)

                payload = _extract_json_payload(text)
                if payload is None:
                    raise ValueError("No valid JSON array found in the model response.")
                mcqs = _validate_mcq_list(payload, count)
                questions = []
                seen = set()
                for mcq in mcqs:
                    question_text = mcq.question.strip()
                    if question_text.lower() in seen:
                        continue
                    seen.add(question_text.lower())
                    questions.append(mcq)
                if len(questions) < count:
                    raise ValueError("Generated MCQs were not unique enough. Retrying.")
                return questions[:count]
            except groq.RateLimitError as e:
                wait_secs = 2
                print(f"Rate limit reached during batch MCQ generation, waiting {wait_secs}s before retrying ({attempt + 1}/{max_attempts})...")
                time.sleep(wait_secs)
                if attempt == max_attempts - 1:
                    raise RuntimeError(f"Failed to generate batch MCQs after {max_attempts} attempts due to rate limiting: {str(e)}")
                continue
            except Exception as e:
                print(f"Batch MCQ attempt {attempt + 1} failed: {e}")
                if attempt == max_attempts - 1:
                    raise RuntimeError(f"Failed to generate batch MCQs after {max_attempts} attempts: {str(e)}")
                continue


    def generate_fill_blank_set(self, topic: str, difficulty: str = 'medium', count: int = 5) -> List[FillBlankQstn]:
        """
        Generate a batch of unique fill-in-the-blank questions in one request.
        """
        prompt = PromptTemplate(
            template=(
                "Generate {count} unique {difficulty} fill-in-the-blank questions about {topic}.\n\n"
                "Return ONLY a JSON array of objects with these exact fields for each question:\n"
                "- 'question': A sentence with '_____' marking the blank\n"
                "- 'answer': The correct word or phrase to fill in the blank\n\n"
                "Example format:\n"
                "[\n"
                "  {{\n"
                "    \"question\": \"The capital of France is _____.\",\n"
                "    \"answer\": \"Paris\"\n"
                "  }},\n"
                "  {{\n"
                "    \"question\": \"The largest planet in our solar system is _____.\",\n"
                "    \"answer\": \"Jupiter\"\n"
                "  }}\n"
                "]\n\n"
                "Your response:"
            ),
            input_variables=["topic", "difficulty", "count"]
        )

        max_attempts = 4
        for attempt in range(max_attempts):
            try:
                response = self._invoke_with_model_fallback(prompt.format(topic=topic, difficulty=difficulty, count=count))
                if hasattr(response, 'content'):
                    text = response.content
                elif isinstance(response, dict) and 'content' in response:
                    text = response['content']
                else:
                    text = str(response)

                payload = _extract_json_payload(text)
                if payload is None:
                    raise ValueError("No valid JSON array found in the model response.")
                questions = _validate_fill_blank_list(payload, count)
                unique_questions = []
                seen = set()
                for item in questions:
                    question_text = item.question.strip()
                    if question_text.lower() in seen:
                        continue
                    seen.add(question_text.lower())
                    unique_questions.append(item)
                if len(unique_questions) < count:
                    raise ValueError("Generated fill-in-the-blank questions were not unique enough. Retrying.")
                return unique_questions[:count]
            except groq.RateLimitError as e:
                wait_secs = 2
                print(f"Rate limit reached during batch fill-in-the-blank generation, waiting {wait_secs}s before retrying ({attempt + 1}/{max_attempts})...")
                time.sleep(wait_secs)
                if attempt == max_attempts - 1:
                    raise RuntimeError(f"Failed to generate batch fill-in-the-blank questions after {max_attempts} attempts due to rate limiting: {str(e)}")
                continue
            except Exception as e:
                print(f"Batch fill-in-the-blank attempt {attempt + 1} failed: {e}")
                if attempt == max_attempts - 1:
                    raise RuntimeError(f"Failed to generate batch fill-in-the-blank questions after {max_attempts} attempts: {str(e)}")
                continue


    def generate_fill_blank(self, topic: str, difficulty: str = 'medium') -> FillBlankQstn:
        """
        Generate Fill in the Blank Question with robust error handling
        Includes:
        - Output parsing using Pydantic
        - Structured prompt template
        - Multiple retry attempts on failure
        - Validation of blank marker format
        """
        # Set up Pydantic parser for type checking and validation
        fill_blank_parser = PydanticOutputParser(pydantic_object=FillBlankQstn)
        
        # Define the prompt template with specific format requirements
        prompt = PromptTemplate(
            template=(
                "Generate a {difficulty} fill-in-the-blank question about {topic}.\n\n"
                "Return ONLY a JSON object with these exact fields:\n"
                "- 'question': A sentence with '_____' marking where the blank should be\n"
                "- 'answer': The correct word or phrase that belongs in the blank\n\n"
                "Example format:\n"
                '{{\n'
                '    "question": "The capital of France is _____.",\n'
                '    "answer": "Paris"\n'
                '}}\n\n'
                "Your response:"
            ),
            input_variables=["topic", "difficulty"]
        )

        # Implement retry logic with maximum attempts
        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                # Generate response using LLM
                response = self._invoke_with_model_fallback(prompt.format(topic=topic, difficulty=difficulty))
                # Extract text content robustly (support different LLM client return shapes)
                if hasattr(response, 'content'):
                    text = response.content
                elif isinstance(response, dict) and 'content' in response:
                    text = response['content']
                else:
                    text = str(response)
                # First try extracting JSON directly
                json_obj = _extract_json_payload(text)
                if json_obj:
                    try:
                        parsed_response = FillBlankQstn.model_validate(json_obj)
                    except Exception as ve:
                        print("Direct JSON -> model_validate failed for fill-blank:", repr(ve))
                        parsed_response = None
                else:
                    parsed_response = None

                if parsed_response is None:
                    try:
                        parsed_parsed = fill_blank_parser.parse(text)
                    except Exception as pe:
                        print("Fill PydanticOutputParser.parse raised:", repr(pe))
                        print("Response repr:", repr(response)[:1000])
                        print("Text repr:", repr(text)[:1000])
                        raise

                    if isinstance(parsed_parsed, str):
                        import json
                        fb_data = json.loads(parsed_parsed)
                        parsed_response = FillBlankQstn.model_validate(fb_data)
                    else:
                        parsed_response = parsed_parsed
                
                # Validate the generated question meets requirements
                if not parsed_response.question or not parsed_response.answer:
                    raise ValueError("Invalid question format")
                if "_____" not in parsed_response.question:
                    parsed_response.question = parsed_response.question.replace("___", "_____")
                    if "_____" not in parsed_response.question:
                        raise ValueError("Question missing blank marker '_____'")
                
                return parsed_response
            except groq.RateLimitError as e:
                wait_secs = 2
                print(f"Rate limit reached for fill-in-the-blank, waiting {wait_secs}s before retrying ({attempt + 1}/{max_attempts})...")
                time.sleep(wait_secs)
                if attempt == max_attempts - 1:
                    raise RuntimeError(f"Failed to generate valid fill-in-the-blank question after {max_attempts} attempts due to rate limiting: {str(e)}")
                continue
            except Exception as e:
                # On final attempt, raise error; otherwise continue trying
                import traceback
                traceback.print_exc()
                if attempt == max_attempts - 1:
                    raise RuntimeError(f"Failed to generate valid fill-in-the-blank question after {max_attempts} attempts: {str(e)}")
                continue        