import json
import logging
import os
import subprocess

import openai

from utils.pdf_utils import process_pdf
from utils.text_utils import get_or_create_vector_store, split_text_by_tokens
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter

def parse_json_garbage(s):
    s = s[next(idx for idx, c in enumerate(s) if c in "{["):]
    try:
        return json.loads(s)
    except json.JSONDecodeError as e:
        return json.loads(s[:e.pos])

def extract_json_from_images(filename, user_query):
     pdf_path = os.path.join("uploads", filename)    
    
     cmd = f"pdfgrep -Pn '^(?s:(?=.*consolidated results of operations)|(?=.*Consolidated Statements of Operations)|(?=.*Consolidated Statements of Cash Flows)|(?=.*CONSOLIDATED STATEMENTS OF CASH FLOWS)|(?=.*CONSOLIDATED STATEMENTS OF INCOME)|(?=.*Interest expenses and other bank charges)|(?=.*Depreciation and Amortization)|(?=.*CONSOLIDATED BALANCE SHEETS))' {pdf_path} | awk -F\":\" '$0~\":\"{{print $1}}' | tr '\n' ','"
     print(cmd)
     logging.info(cmd)
     pages = subprocess.check_output(cmd, shell=True).decode("utf-8")
     logging.info(f'count of pages {pages}')
     print(pages)
     pages_list = pages.split(",")
     del pages_list[-1]
     print(pages_list)
     if not pages:
        logging.warning(f"No matching pages found in {pdf_path}")
        return
     images = pdf_to_image(pdf_path, pages_list)
     image_payloads = [ {
           "type": "image_url",
           "image_url": {
             "url": f"data:image/jpeg;base64,{t}"
           }
         } for t in zip(images)]

     openai.api_key = os.environ["OPENAI_API_KEY"]
     completion = openai.ChatCompletion.create(
         model=model,
         messages=[
             {"role": "user", "content": image_payloads},
             {"role": "user", "content": user_query},
         ]
     )
     print(completion.choices[0].message.content.strip() )
     current_result = parse_json_garbage(completion.choices[0].message.content.strip() )
     return current_result

def extract_json(filename, user_query):
    text = ""
    pdf_path = os.path.join("uploads", filename)
    
    cmd = f"pdfgrep -Pn '^(?s:(?=.*consolidated results of operations)|(?=.*Consolidated Statements of Cash Flows)|(?=.*CONSOLIDATED STATEMENTS))' {pdf_path} | awk -F\":\" '$0~\":\"{{print $1}}' | tr '\n' ','"
    print(cmd)
    logging.info(cmd)
    pages = subprocess.check_output(cmd, shell=True).decode("utf-8")
    logging.info(f'count of pages {pages}')
    if not pages:
       logging.warning(f"No matching pages found in {pdf_path}")
       return
    processed_text = process_pdf(pdf_path, pages)
    if processed_text is not None:
        text += processed_text
    openai.api_key = os.environ["OPENAI_API_KEY"]
    print(text)
    completion = openai.ChatCompletion.create(
        model= model,
        messages=[
            {"role": "system", "content": "You are a highly experienced Business Analyst and Financial Expert with a rich history of over 30 years in the field. Your expertise is firmly grounded in data-driven insights and comprehensive analysis. When presented with unsorted data, your primary objective is to meticulously filter out any extraneous or irrelevant components, including elements containing symbols like # and $. Furthermore, you excel at identifying and eliminating any HTML or XML tags and syntax within the data, streamlining it into a refined and meaningful form."},
            {"role": "user", "content": text},
            {"role": "assistant", "content": f"Question: {user_query}\nAnswer:"},
        ]
    )
    print(completion.choices[0].message.content.strip() )
    current_result = parse_json_garbage(completion.choices[0].message.content.strip() )
    return current_result

def generate_guidance(data): #this is the function that will generate the guidance for the company
    openai.api_key = os.environ["OPENAI_API_KEY"]
    prompt = f"The company {data['name']} has the following financial data:\n"
    for key, value in data.items():
        if key != 'name':
            prompt += f"{key}: {value}\n"
    prompt += '''You are a highly experienced Business Analyst and Financial Expert with a rich history of over 30 years in the field. For your information, Guidance is an informal report a public company issues to shareholders detailing the earnings it expects to achieve in the upcoming fiscal quarter or year ahead. Based only from this data, what is the guidance for the company's financial performance for the next year? 
    Here are some example guidance formats that you can use as a reference:
    Example 1:
    "Revenues of $15.7 - $16.3 billion. Non-GAAP operating income of $4.0-$4.5 billion. Adjusted EBITDA of $4.5 - $5.0 billion. Non-GAAP diluted EPS of $2.20 - $2.50",
    Example 2:
    the company expects net revenues between $520 million and $542 million, Cortrophin Gel Net Revenue in the range of $170 million - $180 million.
    Example 3:
    Royalty Pharma expects 2024 Portfolio Receipts to be between $2,600 million and $2,700 million. 2024 Portfolio Receipts guidance includes expected growth in royalty receipts of 5% to 9%.
    The ideal length of the guidance is 1-3 sentences MAX and this is compulsory. Keep the information concise and to the point. The response should be based on the data provided. The response must also have quantitative values to support the guidance.
    Ensure the quantitative values are realistic and accurately calculated based on typical industry standards and historical performance.
     '''
    response = openai.Completion.create(
      model ="gpt-3.5-turbo-instruct",
      prompt=prompt,
      max_tokens=300,
      temperature=0,
      n = 1
    )
    
    return response.choices[0].text.strip(),

def generate_expanalysis(data): #this is the function that will generate the expert analysis for the company
    openai.api_key = os.environ["OPENAI_API_KEY"]
    prompt = f"The company {data['name']} has the following financial data:\n"
    for key, value in data.items():
        if key != 'name':
            prompt += f"{key}: {value}\n"
    prompt += '''You are a highly experienced Business Analyst and Financial Expert with a rich history of over 30 years in the field. It should also include steps for the upcoming future that can benefit the company realistically. Based only from this data, Can you write an expert analysis paragraph. 

    When writing the analysis, consider the following details:
    Keep the information concise and to the point. The response should be based on the data provided. The response must also have quantitative values to support the response. Ensure the quantitative values are realistic and accurately calculated based on typical industry standards and historical performance.
    Market Conditions: Discuss any market trends or economic factors that might influence the company's performance.
    Specific Financial Metrics: Include key financial metrics such as revenues, EBITDA, net income, and EPS.
    Factors Affecting Performance: Mention any significant factors such as new product launches, regulatory changes, cost management strategies, or investment plans.
    Comparative Analysis: Compare the companys projections with industry averages or competitors if applicable.
    
    Here are some example of formats  of how you can write the expert analysis that you can use as a reference and keep it under 100 words:
    Example 1: Based on analysts offering 12 month price targets for TEVA in the last 3 months. The average price target is $15.71 with a high estimate of $19 and a low estimate of $11
    Example 2: analysts expect ANI Pharmaceuticals to post earnings of $0.97 per share. This would mark a year-over-year decline of 17.09%. Meanwhile, the Zacks Consensus Estimate for revenue is projecting net sales of $124.38 million, up \"16.47%\" from the year-ago period.
    Example 3: Royalty Pharma's eight analysts are now forecasting revenues of US$2.68b in 2024. This would be a meaningful \"14%\" improvement in revenue compared to the last 12 months. Statutory earnings per share are expected to shrink 6.3% to US$2.38 in the same period
     '''
    response = openai.Completion.create(
      model ="gpt-3.5-turbo-instruct",
      prompt=prompt,
      max_tokens=300,
      temperature=0,
      n = 1
    )

    return response.choices[0].text.strip(),



def analysis_10k_json(data, scrapped_data, project_id):
    openai.api_key = os.environ["OPENAI_API_KEY"]
    prompt_context = f"The company {data['name']} has the following financial data:\n"
    for key, value in data.items():
        if key != 'name':
            prompt_context += f"{key}: {value}\n"

    prompt_context += '''
    We have also extracted following text information from few websites:\n
    '''

    prompt_context += scrapped_data

    prompt = '''
     provide the response in the below example JSON format
     
     example JSON format:
     {
       "guidance": "guidance of the company",
       "expert_analysis":  "expert analysis of the company",
       "countries": ["Country 1", "Country 2", "Country 3"],
       "products": ["Product 1", "Product 2", "Product 3"]
     }
     '''

    prompt += '''
    You are a highly experienced Business Analyst and Financial Expert with a rich history of over 30 years in the field. For your information, Guidance is an informal report a public company issues to shareholders detailing the earnings it expects to achieve in the upcoming fiscal quarter or year ahead. Based only from this data,
     what is the guidance for the company's financial performance for the next year? store the answer to this question in the "guidance" property of the response JSON as string 
     what is the expert analysis on company's performance ? store the answer to this question in the "expert_analysis" property of the response JSON as string
     what are the top performing countries of the company ? store the answer to this question in the "countries" property of the response JSON as array of strings
     what are the top performing products of the company ? store the answer to this question in the "products" property of the response JSON as array of strings
    '''


    prompt += '''
    When writing the guidance, consider the following instructions:
    Begin by reviewing the company's strategic objectives and financial goals. Understand what the company aims to achieve in terms of revenue, profitability, market share, etc.
    Utilize historical financial data, market trends, and internal forecasts to project future financial performance. This includes revenue growth, cost trends, margins, and cash flow expectations.
    Evaluate external factors that could impact financial performance, such as economic conditions, industry trends, regulatory changes, and competitive dynamics. Incorporate these into your projections.
    Based on your analysis, create detailed projections for key financial metrics. This typically includes revenue, earnings per share (EPS), operating income, margins, capital expenditures, and other relevant financial indicators.
    Clearly state the assumptions underlying your projections. This could include assumptions about market conditions, customer demand, pricing trends, production costs, currency fluctuations, and any other factors affecting financial outcomes.
    Consider providing a range of potential outcomes or scenarios rather than a single point estimate. This acknowledges uncertainty and helps investors understand the range of possible results based on different assumptions or market conditions.
    Avoid overly optimistic projections. It's crucial to be transparent and realistic about the challenges and risks the company may face in achieving its financial goals.
    Write the financial guidance in a clear, understandable manner. Avoid jargon or overly technical language that could confuse investors.
    Ensure that the financial guidance aligns with the company's overall strategic plan and is supported by reliable data and analysis. Review the guidance with key stakeholders, such as senior management and the board of directors, to validate assumptions and projections.
    Once finalized, communicate the financial guidance through appropriate channels, such as regulatory filings (e.g., SEC filings), investor presentations, press releases, or conference calls. Be prepared to address questions and provide additional context as needed.
    Continuously monitor actual performance against the guidance provided. If there are material changes or developments, consider updating the guidance to reflect new information or revised expectations.
    
    Here are some example guidance formats that you can use as a reference:
    Example 1:
    "Revenues of $15.7 - $16.3 billion. Non-GAAP operating income of $4.0-$4.5 billion. Adjusted EBITDA of $4.5 - $5.0 billion. Non-GAAP diluted EPS of $2.20 - $2.50",
    Example 2:
    the company expects net revenues between $520 million and $542 million, Cortrophin Gel Net Revenue in the range of $170 million - $180 million.
    Example 3:
    Royalty Pharma expects 2024 Portfolio Receipts to be between $2,600 million and $2,700 million. 2024 Portfolio Receipts guidance includes expected growth in royalty receipts of 5% to 9%.
    The ideal length of the guidance is 1-3 sentences MAX and this is compulsory. Keep the information concise and to the point. The response should be based on the data provided. The response must also have quantitative values to support the guidance.
    Ensure the quantitative values are realistic and accurately calculated based on typical industry standards and historical performance.
    
    '''

    prompt += '''
    When writing the analysis, consider the following instructions:
    Keep the information concise and to the point. The response should be based on the data provided. The response must also have quantitative values to support the response. Ensure the quantitative values are realistic and accurately calculated based on typical industry standards and historical performance.
    Market Conditions: Discuss any market trends or economic factors that might influence the company's performance.
    Specific Financial Metrics: Include key financial metrics such as revenues, EBITDA, net income, and EPS.
    Factors Affecting Performance: Mention any significant factors such as new product launches, regulatory changes, cost management strategies, or investment plans.
    Comparative Analysis: Compare the companys projections with industry averages or competitors if applicable.
    Collect and review the company's financial statements, including income statements, balance sheets, and cash flow statements. Analyze trends over multiple periods (e.g., quarterly, annual) to understand the company's financial health.
    Consider non-financial metrics such as market share, customer growth, operational efficiency, and competitive positioning. These metrics provide a broader view of the company's performance beyond financial numbers.
    Benchmark the company's performance against its industry peers. Use industry reports, financial databases, and market research to assess how the company stacks up in terms of profitability, growth rates, and other key metrics.
    Determine the factors driving the company's performance. This may include product innovation, cost management, pricing strategies, expansion into new markets, acquisitions, or changes in consumer behavior.
    Calculate and analyze financial ratios such as profitability ratios (e.g., gross profit margin, operating margin), liquidity ratios (e.g., current ratio, quick ratio), and leverage ratios (e.g., debt-to-equity ratio). These ratios provide insights into the company's financial structure and efficiency.
    Evaluate the effectiveness of the company's management team in executing its strategy. Consider factors such as strategic decision-making, capital allocation, operational efficiency, and corporate governance practices.
    Assess external factors impacting the company's performance, such as economic conditions, industry trends, regulatory changes, and geopolitical events. Analyze how these factors have influenced the company's results and outlook.
    Summarize the company's strengths and weaknesses based on your analysis. Focus on key areas where the company excels and areas that present challenges or risks to future performance.
    Offer insights into what the company is doing well and where improvements could be made. Recommend strategic actions or initiatives that could enhance the company's performance, mitigate risks, or capitalize on opportunities.
    Structure your analysis in a clear and logical manner. Use concise language and avoid unnecessary technical jargon. Ensure that your analysis is accessible to readers who may not have a deep financial background.
    Summarize your findings and conclusions in a succinct manner. Provide a balanced view of the company's performance, acknowledging both strengths and areas for improvement.
    Be transparent about the assumptions and limitations of your analysis. Acknowledge any uncertainties or data gaps that may affect the accuracy of your conclusions.
    Keep your analysis up to date by monitoring the company's performance and reviewing new financial disclosures or market developments. Update your analysis as needed to reflect changes in the company's outlook or industry dynamics.
    Here are some example of formats  of how you can write the expert analysis that you can use as a reference and keep it under 100 words:
    Example 1: Based on analysts offering 12 month price targets for TEVA in the last 3 months. The average price target is $15.71 with a high estimate of $19 and a low estimate of $11
    Example 2: analysts expect ANI Pharmaceuticals to post earnings of $0.97 per share. This would mark a year-over-year decline of 17.09%. Meanwhile, the Zacks Consensus Estimate for revenue is projecting net sales of $124.38 million, up \"16.47%\" from the year-ago period.
    Example 3: Royalty Pharma's eight analysts are now forecasting revenues of US$2.68b in 2024. This would be a meaningful \"14%\" improvement in revenue compared to the last 12 months. Statutory earnings per share are expected to shrink 6.3% to US$2.38 in the same period
   
    '''


    print(prompt, "prompt")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text=prompt_context)
    print("chunks ready")
    vector_store = get_or_create_vector_store(chunks, project_id)
    print("vector store ready")
    docs = vector_store.similarity_search(query=prompt, k=3)
    print("docs ready")
    llm = OpenAI(temperature=0.7, model="gpt-3.5-turbo-instruct")
    print("llm ready")
    chain = load_qa_chain(llm=llm, chain_type="stuff")
    print("chain loaded")

    with get_openai_callback() as cb:
        response = chain.run(input_documents=docs, question=prompt)

    print(response, 'AI response')
    current_result = parse_json_garbage(response)
    print("returning extracted json", current_result)

    return current_result


def analysis_from_html(data, prompt):
    # Query OpenAI's Davinci Chat Model
    openai.api_key = os.environ["OPENAI_API_KEY"]
    ai_response = openai.ChatCompletion.create(
        model="gpt-4-turbo-preview",
        messages=[
            {"role": "system",
             "content": "You are a highly experienced Business Analyst and Financial Expert with a rich history of over 30 years in the field. Your expertise is firmly grounded in data-driven insights and comprehensive analysis. When presented with unsorted data, your primary objective is to meticulously filter out any extraneous or irrelevant components, including elements containing symbols like # and $. Furthermore, you excel at identifying and eliminating any HTML or XML tags and syntax within the data, streamlining it into a refined and meaningful form."},
            {"role": "user", "content": data},
            {"role": "assistant", "content": f"Question: {prompt}\nAnswer:"},
        ],
    )
    answer = ai_response['choices'][0]['message']['content']
    return answer;


def append_guidance_analysis(project, company_index, existing_guidance, new_guidance_from_user, project_id):
    openai.api_key = os.environ["OPENAI_API_KEY"]
    current_company_report = project['report'][company_index]
    prompt_context = f"The company {current_company_report['name']} has the following financial data:\n"
    for key, value in current_company_report.items():
        if key != 'scrapped_data' and key != 'name':
            prompt_context += f"{key}: {value}\n"
    
    prompt_context += '''
    We have also extracted following text information from few websites:\n
    '''

    prompt_context += current_company_report['scrapped_data']

    prompt = '''
       You are a highly experienced Business Analyst and Financial Expert with a rich history of over 30 years in the field. For your information, Guidance is an informal report a public company issues to shareholders detailing the earnings it expects to achieve in the upcoming fiscal quarter or year ahead. Based only from this data,
       what is the guidance for the company's financial performance for the next year? store the answer to this question in the "guidance" property of the response JSON as string 
       '''
    prompt += '''
    When writing the guidance, consider the following instructions:
    Begin by reviewing the company's strategic objectives and financial goals. Understand what the company aims to achieve in terms of revenue, profitability, market share, etc.
    Utilize historical financial data, market trends, and internal forecasts to project future financial performance. This includes revenue growth, cost trends, margins, and cash flow expectations.
    Evaluate external factors that could impact financial performance, such as economic conditions, industry trends, regulatory changes, and competitive dynamics. Incorporate these into your projections.
    Based on your analysis, create detailed projections for key financial metrics. This typically includes revenue, earnings per share (EPS), operating income, margins, capital expenditures, and other relevant financial indicators.
    Clearly state the assumptions underlying your projections. This could include assumptions about market conditions, customer demand, pricing trends, production costs, currency fluctuations, and any other factors affecting financial outcomes.
    Consider providing a range of potential outcomes or scenarios rather than a single point estimate. This acknowledges uncertainty and helps investors understand the range of possible results based on different assumptions or market conditions.
    Avoid overly optimistic projections. It's crucial to be transparent and realistic about the challenges and risks the company may face in achieving its financial goals.
    Write the financial guidance in a clear, understandable manner. Avoid jargon or overly technical language that could confuse investors.
    Ensure that the financial guidance aligns with the company's overall strategic plan and is supported by reliable data and analysis. Review the guidance with key stakeholders, such as senior management and the board of directors, to validate assumptions and projections.
    Once finalized, communicate the financial guidance through appropriate channels, such as regulatory filings (e.g., SEC filings), investor presentations, press releases, or conference calls. Be prepared to address questions and provide additional context as needed.
    Continuously monitor actual performance against the guidance provided. If there are material changes or developments, consider updating the guidance to reflect new information or revised expectations.

    Here are some example guidance formats that you can use as a reference:
    Example 1:
    "Revenues of $15.7 - $16.3 billion. Non-GAAP operating income of $4.0-$4.5 billion. Adjusted EBITDA of $4.5 - $5.0 billion. Non-GAAP diluted EPS of $2.20 - $2.50",
    Example 2:
    the company expects net revenues between $520 million and $542 million, Cortrophin Gel Net Revenue in the range of $170 million - $180 million.
    Example 3:
    Royalty Pharma expects 2024 Portfolio Receipts to be between $2,600 million and $2,700 million. 2024 Portfolio Receipts guidance includes expected growth in royalty receipts of 5% to 9%.
    The ideal length of the guidance is 1-3 sentences MAX and this is compulsory. Keep the information concise and to the point. The response should be based on the data provided. The response must also have quantitative values to support the guidance.
    Ensure the quantitative values are realistic and accurately calculated based on typical industry standards and historical performance.

    '''


    prompt += f"In a previous analysis from openAI, we have recieved the following guidance {existing_guidance}, but we have found some errors in the response and we would like to reitirate the guidance with the following feedback {new_guidance_from_user}"
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text=prompt_context)
    print("chunks ready")
    vector_store = get_or_create_vector_store(chunks, project_id)
    print("vector store ready")
    docs = vector_store.similarity_search(query=prompt, k=3)
    print("docs ready")
    llm = OpenAI(temperature=0.7, model="gpt-3.5-turbo-instruct")
    print("llm ready")
    chain = load_qa_chain(llm=llm, chain_type="stuff")
    print("chain loaded")

    with get_openai_callback() as cb:
        response = chain.run(input_documents=docs, question=prompt)

    print(response, 'AI response')
    return response