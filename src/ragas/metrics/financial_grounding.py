import re

def verify_financial_grounding(source_doc, ai_response):
    # Extracts all numbers from source and AI response to find lies
    source_numbers = set(re.findall(r'\d+', source_doc))
    ai_numbers = set(re.findall(r'\d+', ai_response))
    
    # Identify numbers the AI invented
    hallucinated = ai_numbers - source_numbers
    
    if not hallucinated:
        return "✅ GROUNDED: All numbers verified."
    else:
        return f"❌ HALLUCINATION: AI invented: {hallucinated}"

# Testing your logic
source = "School: 60000, Rent: 60000"
ai_msg = "Total is 150000"
print(verify_financial_grounding(source, ai_msg))
