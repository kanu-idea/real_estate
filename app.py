from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import openai
import os
import json
import re
import csv
import langfuse
import logging
import pandas as pd
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from langfuse import get_client, observe, Langfuse


from dotenv import load_dotenv
load_dotenv()

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

class QuestionAnswer(BaseModel):
    question: str
    answer: str

class Question(BaseModel):
    question: str
    answer: str
    
class QuestionSection(BaseModel):
    section: str
    questions: list[QuestionAnswer]

class ContentRequest(BaseModel):
    agent_answers: list[QuestionSection]

class Section(BaseModel):
    section: str
    questions: List[Question]

class EmailGenerator(BaseModel):
    agent_answers: List[Section]


# === Load DataFrame ===
import json

JSON_INPUT_PATH = r"C:\Users\tarun\OneDrive\Desktop\workspace\IDEAFOUNDATION\doc_understand\email_generate\easy_templates_csv_variables.json"
with open(JSON_INPUT_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)  # data is a list of dicts with 'stage' and 'content'

# Initialize LangChain's OpenAI wrapper
llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY"),  # Use environment variable instead
    model="gpt-4o-mini",
    temperature=0.7,
    max_tokens=1500
)

@observe(name="build_agent_prompt")
def build_agent_prompt(agent_answers):
    """Build the complete prompt from agent answers"""
    prompt = prompt_template
    for section in agent_answers:
        prompt += f"\n## {section.section}\n"
        for qa in section.questions:
            prompt += f"- {qa.question}\n{qa.answer}\n"
    
    return prompt

def build_prompt(original_html, stage, agent_context=None):
    if agent_context is None:
        agent_context = YOUR_DEFAULT_AGENT_CONTEXT  # fallback

    # Fetch prompt template from Langfuse
    prompt_obj = langfuse_client.get_prompt("personalize_email_prompt", label="production")
    prompt_template = prompt_obj.prompt  # or .content, depending on SDK version

    # Render the prompt with variables
    return prompt_template.format(
        original_html=original_html,
        stage=stage,
        agent_context=agent_context
    )

# Initialize Langfuse
langfuse = get_client()
prompt_obj = langfuse.get_prompt("real_estate_content_generation", label="production")
prompt_template = prompt_obj.prompt  # or .content, depending on SDK version

langfuse_client = Langfuse(
    public_key=os.environ.get("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.environ.get("LANGFUSE_SECRET_KEY"),
    host=os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")
)

@observe(name="clean_json")
def clean_json(text):
    """Clean and format JSON response"""
    # Remove code fences
    if text.strip().startswith('```'):
        text = text.split('```')[1]
    # Remove any stray control characters (like \1)
    text = re.sub(r'\\[0-9]+,?', '', text)
    # Remove trailing commas before } or ]
    text = re.sub(r',([ \t\r\n]*[}\]])', r'\1', text)
    
    cleaned_text = text.strip()
    # Try to add a closing brace if missing
    if cleaned_text.count('{') > cleaned_text.count('}'):
        cleaned_text += '}'
    return cleaned_text

@observe
@app.post("/generate-content")
def generate_content(request: ContentRequest):
    """Generate real estate content using LLM"""
    with langfuse.start_as_current_span(
        name="llm_generation",
        input={"request": request.dict()},
        metadata={"model": "gpt-4o-mini"}
    ) as span:
        # Build the prompt
        prompt = build_agent_prompt(request.agent_answers)
        
        # Use LangChain to call the LLM
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content=prompt)
        ]
        
        response = llm(messages)
        content = response.content
        
        span.update(
            output={"raw_response": content[:500] + "..." if len(content) > 500 else content},
            metadata={"response_length": len(content)}
        )
        
        # Clean and parse JSON
        cleaned_content = clean_json(content)
        print("LLM Output:\n", cleaned_content)
        result = json.loads(cleaned_content)
        
        span.update(
            output={"parsed_successfully": True, "result_keys": list(result.keys())},
            metadata={"cleaned_content_length": len(cleaned_content)}
        )
        
        # Score the generation (optional - you can customize this)
        span.score(name="correctness", value=1.0, comment="Content generated successfully")
        
        section_scoring_functions = {
            "home_page": score_home_page,
            "three_steps_carousel": score_three_steps_carousel,
            "about_us_page": score_about_us_page,
            "contact_us_page": score_contact_us_page,
            "global_settings": score_global_settings,
            "call_to_action": score_call_to_action,
        }

        scores = {}
        for section, scoring_func in section_scoring_functions.items():
            if section in result:
                score, reason = scoring_func(llm, result[section])
                scores[section] = {"score": score, "reason": reason}
                span.score(name=section, value=score, comment=reason)
        
        langfuse.flush()
        
        return {
            "status": "ok",
            "result": "success",
            "data": result,
            "scores": scores
        }

@app.post("/generate-email")
async def post_agent_questionnaire(agent_questionnaire: EmailGenerator):
    custom_agent_context = agent_questionnaire.dict()
    personalized_emails = []

    # Start a single Langfuse span for the batch (or do one per row if you prefer)
    span = langfuse_client.start_span(
        name="agent_questionnaire_batch",
        input={"questionnaire": custom_agent_context}
    )
    try:
        for idx, row in enumerate(data):
            sample_html = row['template']
            stage = row['stage']
            prompt = build_prompt(sample_html, stage, agent_context=custom_agent_context)
            try:
                client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1
                )
                output = response.choices[0].message.content.strip()
                personalized_emails.append({
                    "row": idx,
                    "stage": stage,
                    "personalized_email": output
                })
            except Exception as e:
                personalized_emails.append({
                    "row": idx,
                    "stage": stage,
                    "error": str(e)
                })
        span.update(output=personalized_emails)
        return JSONResponse(content={"personalized_emails": personalized_emails})
    except Exception as e:
        span.update(output=str(e), level="ERROR")
        raise
    finally:
        span.end()

# === Call OpenAI ===
def personalize_content(html, stage):
    prompt = build_prompt(html, stage)
    span = langfuse_client.start_span(name="personalize_email", input=prompt)
    try:
        client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        output = response.choices[0].message.content.strip()
        span.update(output=output)
        return output
    except Exception as e:
        span.update(output=str(e), level="ERROR")
        print(f"⚠️ LLM Error: {e}")
        return html  # fallback to original
    finally:
        span.end()

# === Apply Personalization ===
def personalize_row(row):
    try:
        original_html = row.get('template', '')
        stage = row.get('Stage')  # Default to 'Unknown' if not present
        if not isinstance(original_html, str) or not original_html.strip():
            raise ValueError("Empty template")
        personalized_html = personalize_content(original_html, stage)
        return personalized_html
    except Exception as e:
        print(f"skipped row: {e}")
        return row['template']  # fallback to original content
    
def score_home_page(llm, content):
    return score_section_with_llm(llm, "home_page", content)

def score_three_steps_carousel(llm, content):
    return score_section_with_llm(llm, "three_steps_carousel", content)

def score_about_us_page(llm, content):
    return score_section_with_llm(llm, "about_us_page", content)

def score_contact_us_page(llm, content):
    return score_section_with_llm(llm, "contact_us_page", content)

def score_global_settings(llm, content):
    return score_section_with_llm(llm, "global_settings", content)

def score_call_to_action(llm, content):
    return score_section_with_llm(llm, "call_to_action", content)

def score_section_with_llm(llm, section_name, section_content):
    eval_prompt = f"""Evaluate the following section for quality, completeness, and clarity. Give a score from 0.0 to 1.0 and a short reason.

Section ("{section_name}"):
{json.dumps(section_content, indent=2)}

Respond in this JSON format:
{{"score": float, "reason": string}}"""
    eval_result = llm([HumanMessage(content=eval_prompt)])
    try:
        parsed = json.loads(eval_result.content.strip())
        return parsed.get("score", 1.0), parsed.get("reason", "No reason provided")
    except Exception as e:
        return 1.0, f"Failed to parse reason: {str(e)}"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)