def build_prompt(context, job_description):
    prompt = f"""
    You are an HR analysis system.

    Return ONLY valid JSON.
    No explanations.
    No markdown.
    No extra text.

    Schema:
    {{
    "match_score": int,
    "matching_skills": list,
    "missing_skills": list,
    "strengths": list,
    "weaknesses": list,
    "recommendations": list
    }}

    CV CONTEXT:
    {context}

    JOB DESCRIPTION:
    {job_description}
    """
    return prompt