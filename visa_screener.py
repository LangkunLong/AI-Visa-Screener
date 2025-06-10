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
        if label.endswith("MISC") or "award" in lower_text or "prize" in lower_text:
            criteria["Awards"].append(entity_text)
            
        # if the entity is ORG include common membership indicators
        if label.endswith("ORG") and ("association" in lower_text or "society" in lower_text or "member" in lower_text):
            criteria["Membership"].append(entity_text)
            
        # MISC with known media names or keywords
        if ("press" in lower_text or "media" in lower_text or 
            (label.endswith("MISC") and any(media in entity_text for media in ["CNN", "BBC", "Reuters"]))):
            criteria["Press"].append(entity_text)
            
        # "judge" or "panel"
        if "judge" in lower_text or "panel" in lower_text:
            criteria["Judging"].append(entity_text)
            
        # imply innovation
        if "contribution" in lower_text or "innovation" in lower_text or "developed" in lower_text:
            criteria["Original contribution"].append(entity_text)
            
        # using keywords often found in academic publications
        if "article" in lower_text or "journal" in lower_text or "publication" in lower_text:
            criteria["Scholarly articles"].append(entity_text)
            
        # if the entity is ORG, suggests high-level roles or companies
        if label.endswith("ORG") and ("inc" in lower_text or "llc" in lower_text or 
                                        "corporation" in lower_text or "executive" in lower_text or "director" in lower_text):
            criteria["Critical employment"].append(entity_text)
            
        # using salary or compensation related keywords
        if "salary" in lower_text or "compensation" in lower_text or "remuneration" in lower_text:
            criteria["High remuneration"].append(entity_text)
            
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

