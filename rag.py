def build_prompt(context, job_description):
    """
    This function creates a structured prompt for an LLM
    to evaluate how well a candidate CV matches a job description.
    """

    # We use an f-string to dynamically insert:
    # - context (retrieved CV content)
    # - job_description (job requirements)
    # into a single prompt string
    prompt = f"""
    <|begin_of_text|>
    <|start_header_id|>system<|end_header_id|>
    You are an expert HR assistant.

    <|start_header_id|>user<|end_header_id|>
    Context:
    {context}

    Job Description:
    {job_description}

    Tasks:
    - Give match score (0-100)
    - List missing skills
    - Suggest improvements

    <|start_header_id|>assistant<|end_header_id|>
    """

    # Return the final formatted prompt to be sent to the LLM
    return prompt


