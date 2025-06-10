from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import io
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

visa_screener = FastAPI(title="O-1A Visa Screener")

# task involves name entity recognition (ner) so more efficient to use an NLP model
# decided to use encoder transformer model that is strong at natural language processing
# using bert: strong encoder only model that is fast 
MODEL_NAME = "dbmdz/bert-large-cased-finetuned-conll03-english" 
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
bert_pre_trained = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)

# using hugging face's transformer pipeline
bert = pipeline("ner", model=bert_pre_trained, tokenizer=tokenizer)

# preprocess pdf files into string 
def preprocess(raw_files: bytes) -> str:
    try:
        return raw_files.decode("utf-8")
    except Exception as e:
        raise HTTPException(status_code=400, detail="Error extracting text from the file.")


# Bert outputs a dictionary with the detected entities, then filter based on the 8 criterias 
# map recognized entities (from BERT) to O-1A evidentiary criteria
# count the number of entities the applicant meets 
# ** can fine-tune pre-trained bert to increase accuracy"
def bert_entities_to_criteria(entities: list) -> dict:

    criteria = {
        "awards": [],
        "membership": [],
        "press": [],
        "judging": [],
        "original contribution": [],
        "scholarly articles": [],
        "critical employment": [],
        "high remuneration": []
    }
    
    for entity in entities:
        label = entity.get("entity", "")
        entity_text = entity.get("word", "")
        lower_text = entity_text.lower()
        
        # potential award mentions
        award_keywords = ["award", "prize", "honor", "medal", "fellowship"]
        if label.endswith("MISC") or any(k in lower_text for k in award_keywords):
            criteria["awards"].append(entity_text)
            
        # if the entity is ORG include common membership indicators
        org_keywords = ["ORG", "association", "society", "member"]
        if label.endswith("ORG") and any(k in lower_text for k in org_keywords):
            criteria["membership"].append(entity_text)
            
        # MISC with known media names or keywords
        press_keywords = ["press", "media", "interview", "coverage"]
        if any(k in lower_text for k in press_keywords):
            criteria["press"].append(entity_text)
            
        # Judging: look for judge, panel, reviewer, adjudicator, evaluator
        judging_keywords = ["judge", "panel", "reviewer", "adjudicator", "evaluator"]
        if any(k in lower_text for k in judging_keywords):
            criteria["judging"].append(entity_text)

        # Original contribution: contribution, innovation, developed, invention, breakthrough, discovery
        contribution_keywords = [
            "contribution", "innovation", "developed", "invention", "breakthrough", "discovery"
        ]
        if any(k in lower_text for k in contribution_keywords):
            criteria["original contribution"].append(entity_text)

        # Scholarly articles: article, journal, publication, paper, manuscript, proceedings
        scholarly_keywords = [
            "article", "journal", "publication", "paper", "manuscript", "proceedings"
        ]
        if any(k in lower_text for k in scholarly_keywords):
            criteria["scholarly articles"].append(entity_text)

        # Critical employment: look for orgs with high-level roles or company suffixes
        critical_roles = [
            "executive", "director", "chief", "president", "founder", "ceo", "cto", "cfo", "lead"
        ]
        company_suffixes = ["inc", "llc", "corporation", "corp", "ltd", "company", "plc"]
        if label.endswith("ORG") and (
            any(role in lower_text for role in critical_roles) or
            any(suffix in lower_text for suffix in company_suffixes)
        ):
            criteria["critical employment"].append(entity_text)

        # High remuneration: salary, compensation, remuneration, income, earnings, wage, stipend, pay
        remuneration_keywords = [
            "salary", "compensation", "remuneration", "income", "earnings", "wage", "stipend", "pay"
        ]
        if any(k in lower_text for k in remuneration_keywords):
            criteria["high remuneration"].append(entity_text)
            
    return criteria

from typing import Tuple

def assess_visa(text: str) -> Tuple[dict, str]:

    # pass string to bert for assessment 
    entities = bert(text)
    evidentiary = bert_entities_to_criteria(entities)
    
    # compute a simple rating based on number of criteria met: 
    # currently assigned the same weighting to all criteria, in the future can use past visa admissions
    # to see which criteria is weighted more 
    criteria_met = sum(1 for items in evidentiary.values() if items)
    if criteria_met >= 6:
        rating = "high"
    elif criteria_met >= 4:
        rating = "medium"
    else:
        rating = "low"
    
    return evidentiary, rating

@visa_screener.post("/assess")
async def assess_cv_endpoint(file: UploadFile = File(...)):
    file_bytes = await file.read()
    text = preprocess(file_bytes)
    evidentiary_items, rating = assess_visa(text)
    return JSONResponse(content={
        "evidentiary_items": evidentiary_items,
        "qualification_rating": rating
    })

